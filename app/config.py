# app/config.py
from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    MONGODB_URI: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "proctoring_db"
    VIDEO_STORAGE_PATH: str = "./data/videos"
    MAX_UPLOAD_SIZE_MB: int = 200
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
Path(settings.VIDEO_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
