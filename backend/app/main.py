"""
Application entry point.

Run from the `backend/` directory:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.utils.logger import setup_logging
from app.routes import lectures, quiz, proctoring, ml_routes

# ── Logging ───────────────────────────────────────────────────────────────────
setup_logging(os.getenv("LOG_LEVEL", "INFO"))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Smart Learning System",
    description=(
        "Real-time exam proctoring, lecture management, "
        "quiz generation and AI-powered lecture analysis."
    ),
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(lectures.router,   prefix="/api")
app.include_router(quiz.router,       prefix="/api")
app.include_router(ml_routes.router,  prefix="/api")
app.include_router(proctoring.router)   # WebSocket – no /api prefix

# ── Health endpoints ──────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "service": "AI Smart Learning System", "version": "2.0.0"}


@app.get("/api/health", tags=["Health"])
async def health():
    return {
        "status":          "healthy",
        "mediapipe":       "initialized",
        "websocket":       "ready",
        "quiz_generation": "enabled",
    }


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
