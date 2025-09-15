# app/main.py
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .db import create_indexes
from .config import settings
from .routes import logs, videos, ws
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    await create_indexes()
    print("✅ DB indexes created")
    yield
    # shutdown
    print("👋 Shutting down...")

app = FastAPI(title="Proctoring Backend", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(logs.router)
app.include_router(videos.router)
app.include_router(ws.router)

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
