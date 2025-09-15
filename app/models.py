# app/models.py
from pydantic import BaseModel
from typing import Optional, Dict, Any

class EventIn(BaseModel):
    candidate_id: str
    event_type: str
    confidence: Optional[float] = None
    frame_time: Optional[float] = None  # client may send UNIX timestamp in seconds
    extra: Optional[Dict[str, Any]] = None
