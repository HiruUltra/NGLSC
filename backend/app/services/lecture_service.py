"""
Lecture upload and listing service.
Handles file I/O so that the route layer stays thin.
"""
import logging
from pathlib import Path
from datetime import datetime
from fastapi import UploadFile, HTTPException

from app.core.config import LECTURE_GALLERY_DIR

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".webm", ".mp4", ".avi", ".mov", ".mkv"}


async def save_lecture(file: UploadFile) -> dict:
    """
    Persist an uploaded lecture video and return metadata.

    Raises HTTPException 400 for disallowed file types.
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename  = f"lecture_{timestamp}{file_ext}"
    file_path = LECTURE_GALLERY_DIR / filename

    contents = await file.read()
    file_path.write_bytes(contents)

    size_mb = len(contents) / (1024 * 1024)
    logger.info("Lecture saved: %s (%.2f MB)", filename, size_mb)

    return {
        "success":   True,
        "message":   "Lecture uploaded successfully",
        "filename":  filename,
        "file_path": str(file_path),
        "size_mb":   round(size_mb, 2),
    }


def list_lectures() -> dict:
    """Return metadata for all stored lecture videos, newest first."""
    lectures = []
    for fp in LECTURE_GALLERY_DIR.glob("lecture_*.*"):
        if fp.is_file():
            stat = fp.stat()
            lectures.append({
                "filename":    fp.name,
                "size_mb":     round(stat.st_size / (1024 * 1024), 2),
                "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path":        str(fp),
            })

    lectures.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return {"total": len(lectures), "lectures": lectures}
