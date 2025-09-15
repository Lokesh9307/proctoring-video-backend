# app/services/detection_service.py

import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
from typing import List, Dict, Any
from ..db import events_col
from ultralytics import YOLO

mp_face_detection = mp.solutions.face_detection

# thresholds
LOOK_AWAY_X_THRESHOLD = 0.25
LOOK_AWAY_MIN_SECONDS = 5
NO_FACE_MIN_SECONDS = 10

# candidate state
_candidate_state = {}

# Load YOLOv5 (pretrained COCO model includes "cell phone" = class 67)
yolo_model = YOLO("yolov5s.pt")


def _init_state(candidate_id: str):
    """Initialize candidate tracking state if not exists."""
    if candidate_id not in _candidate_state:
        _candidate_state[candidate_id] = {
            "look_away_start": None,
            "no_face_start": None,
        }


async def process_frame(candidate_id: str, frame_bgr, frame_time: float = None) -> List[Dict]:
    """
    Process a single frame:
    - Detect faces (no face, multiple faces, looking away).
    - Detect suspicious devices (cell phone).
    - Log events to DB.
    """
    _init_state(candidate_id)
    events = []
    now = datetime.utcnow()

    img_h, img_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------
    # Face detection with Mediapipe
    # ---------------------------
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_det:
        results = face_det.process(frame_rgb)

        # Case 1: No face
        if not results.detections:
            if _candidate_state[candidate_id]["no_face_start"] is None:
                _candidate_state[candidate_id]["no_face_start"] = now
            else:
                delta = (now - _candidate_state[candidate_id]["no_face_start"]).total_seconds()
                if delta >= NO_FACE_MIN_SECONDS:
                    events.append({
                        "candidate_id": candidate_id,
                        "event_type": "NO_FACE",
                        "confidence": 1.0,
                        "frame_time": frame_time,
                        "timestamp": now,
                        "extra": {"duration_sec": delta}
                    })
                    _candidate_state[candidate_id]["no_face_start"] = None
        else:
            _candidate_state[candidate_id]["no_face_start"] = None

            # Multiple faces
            if len(results.detections) > 1:
                events.append({
                    "candidate_id": candidate_id,
                    "event_type": "MULTIPLE_FACES",
                    "confidence": 1.0,
                    "frame_time": frame_time,
                    "timestamp": now,
                    "extra": {"count": len(results.detections)}
                })

            # Looking away
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2.0
            dx = cx - 0.5
            if abs(dx) > LOOK_AWAY_X_THRESHOLD:
                if _candidate_state[candidate_id]["look_away_start"] is None:
                    _candidate_state[candidate_id]["look_away_start"] = now
                else:
                    delta = (now - _candidate_state[candidate_id]["look_away_start"]).total_seconds()
                    if delta >= LOOK_AWAY_MIN_SECONDS:
                        events.append({
                            "candidate_id": candidate_id,
                            "event_type": "LOOK_AWAY",
                            "confidence": float(abs(dx)),
                            "frame_time": frame_time,
                            "timestamp": now,
                            "extra": {"duration_sec": delta}
                        })
                        _candidate_state[candidate_id]["look_away_start"] = None
            else:
                _candidate_state[candidate_id]["look_away_start"] = None

    # ---------------------------
    # Phone detection with YOLOv5
    # ---------------------------
    results = yolo_model(frame_bgr, verbose=False)
    best_phone = None

    for r in results:
        for box in r.boxes.cpu().numpy():  # convert to CPU numpy
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 67 and conf > 0.5:  # 67 = "cell phone"
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [round(float(x1), 2), round(float(y1), 2),
                        round(float(x2), 2), round(float(y2), 2)]
                if best_phone is None or conf > best_phone["confidence"]:
                    best_phone = {
                        "candidate_id": candidate_id,
                        "event_type": "PHONE_DETECTED",
                        "confidence": conf,
                        "frame_time": frame_time,
                        "timestamp": now,
                        "extra": {"bbox": bbox}
                    }

    if best_phone:
        events.append(best_phone)

    return await _store_events(events)


async def _store_events(events: List[Dict[str, Any]]):
    """Insert events into DB if any."""
    if events:
        await events_col.insert_many(events)
    return events
