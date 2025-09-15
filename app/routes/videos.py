# app/routes/videos.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.storage import save_upload_file, probe_video_metadata
from ..schemas import VideoMetaOut
from ..db import videos_col
from datetime import datetime
import os

router = APIRouter(prefix="/api", tags=["videos"])

@router.post("/upload_video", status_code=201)
async def upload_video(candidate_id: str, file: UploadFile = File(...)):
    """
    Upload recorded video file. Returns saved meta and creates video_meta record.
    """
    saved = await save_upload_file(file)
    meta_info = probe_video_metadata(saved["path"])
    doc = {
        "candidate_id": candidate_id,
        "filename": saved["filename"],
        "uploaded_at": datetime.utcnow(),
        "duration_seconds": meta_info.get("duration_seconds"),
        "frame_rate": meta_info.get("frame_rate"),
        "resolution": meta_info.get("resolution"),
        "size_bytes": saved["size_bytes"],
        "status": "processed",
        "events_count": 0,
        "notes": None,
    }
    res = await videos_col.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    return {"status": "ok", "video_meta": doc}

@router.get("/videos/{filename}")
async def get_video(filename: str):
    from fastapi.responses import FileResponse
    storage_folder = os.path.abspath("./data/videos")
    path = os.path.join(storage_folder, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)

