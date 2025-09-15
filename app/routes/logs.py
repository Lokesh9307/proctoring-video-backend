# app/routes/logs.py
from fastapi import APIRouter, HTTPException, Response, Query
from typing import List, Optional
from ..models import EventIn
from ..schemas import EventOut
from ..db import events_col, videos_col
from ..services.report_service import make_csv_report, make_pdf_report
from ..utils.scoring import compute_integrity_score
from datetime import datetime
import uuid

router = APIRouter(prefix="/api", tags=["logs"])


@router.post("/events", status_code=201)
async def post_event(event: EventIn, session_id: Optional[str] = Query(default=None)):
    """
    Accept an event posted by client.
    - Stores in `events_col`.
    - Updates `video_meta` in `videos_col` (per candidate + session).
    """
    # Generate session_id if not passed
    if not session_id:
        session_id = str(uuid.uuid4())

    # Normalize frame_time to datetime
    frame_time = None
    if event.frame_time is not None:
        try:
            frame_time = datetime.utcfromtimestamp(float(event.frame_time))
        except Exception:
            frame_time = datetime.utcnow()

    # Event document
    doc = {
        "candidate_id": event.candidate_id,
        "session_id": session_id,
        "event_type": event.event_type,
        "confidence": event.confidence,
        "frame_time": frame_time,
        "extra": event.extra,
        "timestamp": datetime.utcnow(),
    }
    res = await events_col.insert_one(doc)

    # Update session metadata in videos_col
    await videos_col.update_one(
        {"candidate_id": event.candidate_id, "session_id": session_id},
        {
            "$setOnInsert": {
                "candidate_id": event.candidate_id,
                "session_id": session_id,
                "session_start": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "events_count": 0,
            },
            "$set": {
                "updated_at": datetime.utcnow(),
                "last_event_at": datetime.utcnow(),
            },
            "$inc": {"events_count": 1},
        },
        upsert=True,
    )

    return {"status": "ok", "event_id": str(res.inserted_id), "session_id": session_id}


@router.get("/events/{candidate_id}", response_model=List[EventOut])
async def get_events(candidate_id: str, session_id: Optional[str] = None, limit: int = 1000):
    """
    Get candidate's events.
    - If session_id provided → return events for that session only.
    - Otherwise → return all events for candidate.
    """
    query = {"candidate_id": candidate_id}
    if session_id:
        query["session_id"] = session_id

    cursor = events_col.find(query).sort("timestamp", 1).limit(limit)
    out = []
    async for d in cursor:
        d["_id"] = str(d["_id"])
        out.append(d)
    return out


@router.get("/report/{candidate_id}")
async def get_report(candidate_id: str, session_id: Optional[str] = None, format: str = "json"):
    """
    Get candidate report in JSON | CSV | PDF.
    Includes events + metadata (per session).
    """
    query = {"candidate_id": candidate_id}
    if session_id:
        query["session_id"] = session_id

    # Fetch events
    cursor = events_col.find(query).sort("timestamp", 1)
    events = []
    async for d in cursor:
        d["_id"] = str(d["_id"])
        events.append(d)

    if not events:
        raise HTTPException(status_code=404, detail="No events found")

    # Integrity score
    summary = compute_integrity_score(events)

    # Metadata from video_meta
    metadata = await videos_col.find_one(query, sort=[("updated_at", -1)])
    if metadata:
        metadata["_id"] = str(metadata["_id"])

    if format == "json":
        return {"candidate_id": candidate_id, "summary": summary, "events": events, "metadata": metadata}
    elif format == "csv":
        csv_bytes = make_csv_report(candidate_id, events)
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={candidate_id}_report.csv"},
        )
    elif format == "pdf":
        pdf_bytes = make_pdf_report(candidate_id, events, summary, metadata)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={candidate_id}_report.pdf"},
        )
    else:
        raise HTTPException(status_code=400, detail="format must be json|csv|pdf")
