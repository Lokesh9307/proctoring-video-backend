# app/services/detection_service.py
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
from typing import List, Dict, Any
import asyncio
from ..db import events_col

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

FACE_DET_CONF = 0.5
FACE_MESH_CONF = 0.5
HAND_CONF = 0.5

LEFT_EYE_IDX = [33, 133, 160, 159, 158, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 380]

LOOK_AWAY_X_THRESHOLD = 0.25
EYE_CLOSED_EAR_THRESHOLD = 0.18
EYE_CLOSED_CONSEC_FRAMES = 3

_candidate_state = {}

def _init_state(candidate_id: str):
    if candidate_id not in _candidate_state:
        _candidate_state[candidate_id] = {
            "consec_eye_closed": 0,
            "last_hand_near_face_ts": None,
            "hand_near_face_count": 0,
        }

def _eye_aspect_ratio(landmarks, idxs, img_w, img_h):
    try:
        pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in idxs]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    except Exception:
        ear = 0.0
    return ear

def _to_iso(frame_time: Any) -> datetime:
    """
    Convert incoming frame_time (maybe float seconds) to datetime object (UTC).
    If invalid, return current UTC datetime.
    """
    try:
        if frame_time is None:
            return datetime.utcnow()
        # if numeric (seconds)
        if isinstance(frame_time, (int, float)):
            return datetime.utcfromtimestamp(float(frame_time))
        # try parse iso string
        if isinstance(frame_time, str):
            # attempt to parse ISO-like string
            try:
                return datetime.fromisoformat(frame_time.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
            except Exception:
                return datetime.utcnow()
    except Exception:
        return datetime.utcnow()
    return datetime.utcnow()

async def process_frame(candidate_id: str, frame_bgr: np.ndarray, frame_time: Any = None) -> List[Dict]:
    _init_state(candidate_id)
    events = []
    img_h, img_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=FACE_DET_CONF) as face_det, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=FACE_MESH_CONF) as face_mesh, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=HAND_CONF) as hands:

        face_results = face_det.process(frame_rgb)
        # convert frame_time to datetime
        ft_dt = _to_iso(frame_time)

        if not face_results.detections:
            ev = {
                "candidate_id": candidate_id,
                "event_type": "NO_FACE",
                "confidence": 0.0,
                "frame_time": ft_dt,
                "extra": {"note": "no face detected"}
            }
            events.append(ev)
            await _store_events(events)
            return events

        if len(face_results.detections) > 1:
            events.append({
                "candidate_id": candidate_id,
                "event_type": "MULTIPLE_FACES",
                "confidence": 1.0,
                "frame_time": ft_dt,
                "extra": {"count": len(face_results.detections)}
            })

        mesh_results = face_mesh.process(frame_rgb)
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            det = face_results.detections[0]
            bbox = det.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2.0
            dx = cx - 0.5
            if abs(dx) > LOOK_AWAY_X_THRESHOLD:
                events.append({
                    "candidate_id": candidate_id,
                    "event_type": "LOOK_AWAY",
                    "confidence": float(abs(dx)),
                    "frame_time": ft_dt,
                    "extra": {"dx_norm": float(dx)}
                })

            ear_left = _eye_aspect_ratio(landmarks.landmark, LEFT_EYE_IDX, img_w, img_h)
            ear_right = _eye_aspect_ratio(landmarks.landmark, RIGHT_EYE_IDX, img_w, img_h)
            ear = (ear_left + ear_right) / 2.0
            if ear < EYE_CLOSED_EAR_THRESHOLD:
                _candidate_state[candidate_id]["consec_eye_closed"] += 1
            else:
                _candidate_state[candidate_id]["consec_eye_closed"] = 0

            if _candidate_state[candidate_id]["consec_eye_closed"] >= EYE_CLOSED_CONSEC_FRAMES:
                events.append({
                    "candidate_id": candidate_id,
                    "event_type": "EYE_CLOSED",
                    "confidence": ear,
                    "frame_time": ft_dt,
                    "extra": {"ear": ear}
                })
                _candidate_state[candidate_id]["consec_eye_closed"] = 0

        hands_res = hands.process(frame_rgb)
        if hands_res.multi_hand_landmarks:
            det = face_results.detections[0]
            bbox = det.location_data.relative_bounding_box
            face_xmin = bbox.xmin
            face_xmax = bbox.xmin + bbox.width
            face_ymin = bbox.ymin
            face_ymax = bbox.ymin + bbox.height

            hand_near = False
            for hld in hands_res.multi_hand_landmarks:
                xs = [lm.x for lm in hld.landmark]
                ys = [lm.y for lm in hld.landmark]
                hxmin = min(xs); hxmax = max(xs)
                hymin = min(ys); hymax = max(ys)
                overlap_x = not (hxmax < face_xmin or hxmin > face_xmax)
                overlap_y = not (hymax < face_ymin or hymin > face_ymax)
                if overlap_x and overlap_y:
                    hand_near = True
                    break

            if hand_near:
                now_ts = datetime.utcnow().timestamp()
                _candidate_state[candidate_id]["last_hand_near_face_ts"] = now_ts
                _candidate_state[candidate_id]["hand_near_face_count"] += 1
                events.append({
                    "candidate_id": candidate_id,
                    "event_type": "HAND_NEAR_FACE",
                    "confidence": 0.9,
                    "frame_time": ft_dt,
                    "extra": {"hand_near_count": _candidate_state[candidate_id]["hand_near_face_count"]}
                })
                if _candidate_state[candidate_id]["hand_near_face_count"] >= 3:
                    events.append({
                        "candidate_id": candidate_id,
                        "event_type": "PHONE_SUSPECTED",
                        "confidence": 0.8,
                        "frame_time": ft_dt,
                        "extra": {"hand_near_count": _candidate_state[candidate_id]["hand_near_face_count"]}
                    })
                    _candidate_state[candidate_id]["hand_near_face_count"] = 0

    if events:
        await _store_events(events)
    return events

async def _store_events(events: List[Dict[str, Any]]):
    docs = []
    for e in events:
        # ensure frame_time and timestamp are datetime objects (pydantic will serialize)
        frame_time_val = e.get("frame_time")
        timestamp_val = datetime.utcnow()
        doc = {
            "candidate_id": e["candidate_id"],
            "event_type": e["event_type"],
            "confidence": e.get("confidence"),
            "frame_time": frame_time_val,   # datetime
            "extra": e.get("extra"),
            "timestamp": timestamp_val
        }
        docs.append(doc)
    if docs:
        await events_col.insert_many(docs)
