# app/services/storage.py
import os
from pathlib import Path
from fastapi import UploadFile
from uuid import uuid4
from ..config import settings
import cv2

async def save_upload_file(upload_file: UploadFile, destination_folder: str = None) -> dict:
    dest_root = Path(destination_folder or settings.VIDEO_STORAGE_PATH)
    dest_root.mkdir(parents=True, exist_ok=True)

    ext = Path(upload_file.filename).suffix or ""
    filename = f"{uuid4().hex}{ext}"
    full_path = dest_root / filename

    size = 0
    with full_path.open("wb") as out_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
            size += len(chunk)
    await upload_file.close()
    return {"filename": filename, "path": str(full_path), "size_bytes": size}

def probe_video_metadata(filepath: str) -> dict:
    """
    Use OpenCV to extract duration, fps, resolution.
    Returns: {duration_seconds, frame_rate, resolution: {width, height}}
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps and frame_count else None
    return {
        "duration_seconds": float(duration) if duration else None,
        "frame_rate": float(fps) if fps else None,
        "resolution": {"width": width, "height": height}
    }
