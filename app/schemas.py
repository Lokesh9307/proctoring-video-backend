# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class EventOut(BaseModel):
    candidate_id: str
    event_type: str
    confidence: Optional[float] = None
    frame_time: Optional[datetime] = None   # will be serialized to ISO
    extra: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    class Config:
        orm_mode = True

class VideoMetaOut(BaseModel):
    candidate_id: str
    filename: str
    uploaded_at: datetime
    duration_seconds: Optional[float] = None
    frame_rate: Optional[float] = None
    resolution: Optional[Dict[str, int]] = None
    size_bytes: Optional[int] = None
    status: Optional[str] = "processed"
    events_count: Optional[int] = 0
    notes: Optional[str] = None

    class Config:
        orm_mode = True
