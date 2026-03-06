"""
ML inference routes.
These endpoints expose the ML layer (backend/ml/) as REST endpoints
so the frontend can call them directly when needed.

Currently provides:
  POST /ml/analyze-frame  – single-frame proctoring analysis (HTTP alternative to WS)

Extend with additional inference endpoints (voice, gesture, highlight detection, etc.)
"""
import cv2
import base64
import logging
import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ml.inference.face_infer import ProctoringEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ML Inference"])

# One shared engine instance for HTTP requests (stateless analysis only)
_engine: Optional[ProctoringEngine] = None


def get_engine() -> ProctoringEngine:
    global _engine
    if _engine is None:
        _engine = ProctoringEngine()
    return _engine


class FrameRequest(BaseModel):
    frame: str  # base64-encoded JPEG


@router.post("/analyze-frame")
async def analyze_frame(request: FrameRequest):
    """
    Analyze a single base64-encoded JPEG frame and return proctoring status.
    Primarily for testing; use the WebSocket endpoint for real-time monitoring.
    """
    try:
        frame_bytes = base64.b64decode(request.frame)
        np_arr      = np.frombuffer(frame_bytes, np.uint8)
        frame       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode frame image.")

        alert, status = get_engine().process_frame(frame)
        return {
            "status": status.model_dump(),
            "alert":  alert.model_dump() if alert else None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ML analyze-frame error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
