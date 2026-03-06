"""
Pydantic schemas for quiz generation.
"""
from pydantic import BaseModel, Field
from typing import List


class QuizQuestion(BaseModel):
    """A single quiz question with multiple-choice options."""
    id:             int
    question:       str
    options:        List[str]
    correct_answer: int = Field(..., description="Zero-based index of the correct option")


class QuizConfig(BaseModel):
    """Input configuration for generating a quiz."""
    topic:            str
    num_questions:    int = Field(5,  ge=1, le=50)
    duration_minutes: int = Field(10, ge=1, le=180)


class QuizResponse(BaseModel):
    """Full quiz payload returned to the frontend."""
    topic:            str
    total_questions:  int
    duration_minutes: int
    questions:        List[QuizQuestion]
