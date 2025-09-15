# app/db.py
from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings

_client = AsyncIOMotorClient(settings.MONGODB_URI)
db = _client[settings.DATABASE_NAME]

events_col = db["events"]
videos_col = db["video_meta"]

async def create_indexes():
    # create indexes to speed up common queries
    await events_col.create_index("candidate_id")
    await events_col.create_index("timestamp")
    await videos_col.create_index("candidate_id")
