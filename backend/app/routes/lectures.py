"""
Routes for lecture video upload and listing.
"""
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.lecture_service import save_lecture, list_lectures

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lectures", tags=["Lectures"])


@router.post("/upload")
async def upload_lecture(file: UploadFile = File(...)):
    """Upload a lecture video recording (webm, mp4, avi, mov, mkv)."""
    try:
        result = await save_lecture(file)
        return JSONResponse(status_code=200, content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Lecture upload error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")


@router.get("")
async def get_lectures():
    """Return metadata for all stored lecture videos."""
    try:
        return JSONResponse(status_code=200, content=list_lectures())
    except Exception as exc:
        logger.error("Error listing lectures: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to list lectures: {exc}")
