"""
Route for quiz generation.
"""
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.quiz import QuizResponse
from app.services.quiz_service import generate_quiz

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["Quiz"])


@router.get("/generate", response_model=QuizResponse)
async def generate_quiz_endpoint(topic: str, count: int = 5, duration: int = 10):
    """
    Generate a randomised quiz.

    Parameters
    ----------
    topic    : Subject area (Mathematics, Science, History, General Knowledge).
    count    : Number of questions (1–50, default 5).
    duration : Duration in minutes (1–180, default 10).
    """
    count    = max(1, min(count, 50))
    duration = max(1, min(duration, 180))

    try:
        questions = generate_quiz(topic, count)
        return QuizResponse(
            topic=topic,
            total_questions=len(questions),
            duration_minutes=duration,
            questions=questions,
        )
    except Exception as exc:
        logger.error("Quiz generation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {exc}")
