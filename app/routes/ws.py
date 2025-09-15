# app/routes/ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import base64
import numpy as np
import cv2
import asyncio
from ..services import detection_service

router = APIRouter(prefix="/ws", tags=["ws"])

@router.websocket("/stream")
async def stream_ws(websocket: WebSocket):
    """
    Receives JSON messages:
    {
      "candidate_id": "cand_001",
      "frame_b64": "<base64 jpeg>",
      "frame_time": 169xxxxxx (optional)
    }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                candidate_id = payload.get("candidate_id")
                frame_b64 = payload.get("frame_b64")
                frame_time = payload.get("frame_time")
                if not candidate_id or not frame_b64:
                    await websocket.send_text(json.dumps({"error": "candidate_id and frame_b64 required"}))
                    continue

                img_bytes = base64.b64decode(frame_b64.split(",", -1)[-1])
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send_text(json.dumps({"error": "could not decode frame"}))
                    continue

                # process frame asynchronously (detection_service handles DB writes)
                async def run_and_send():
                    events = await detection_service.process_frame(candidate_id, frame, frame_time)
                    if events:
                        out = []
                        for e in events:
                            # convert datetimes to iso strings for WS message
                            ev = {
                                "candidate_id": e["candidate_id"],
                                "event_type": e["event_type"],
                                "confidence": e.get("confidence"),
                                "frame_time": e.get("frame_time").isoformat() + "Z" if hasattr(e.get("frame_time"), "isoformat") else e.get("frame_time"),
                                "extra": e.get("extra"),
                            }
                            out.append(ev)
                        await websocket.send_text(json.dumps({"candidate_id": candidate_id, "events": out}))
                    else:
                        await websocket.send_text(json.dumps({"candidate_id": candidate_id, "events": []}))

                asyncio.create_task(run_and_send())

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid json"}))
    except WebSocketDisconnect:
        return
