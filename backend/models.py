"""
Pydantic models for WebSocket communication
"""
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional, List


class AlertType(str, Enum):
    """Types of alerts that can be triggered"""
    NO_FACE = "NO_FACE"
    HEAD_TURN_LEFT = "HEAD_TURN_LEFT"
    HEAD_TURN_RIGHT = "HEAD_TURN_RIGHT"
    TALKING = "TALKING"
    ALL_CLEAR = "ALL_CLEAR"


class Severity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertEvent(BaseModel):
    """Alert event sent to frontend via WebSocket"""
    alert_type: AlertType
    message_en: str
    message_si: str
    timestamp: str
    severity: Severity
    metadata: Optional[dict] = None
    
    class Config:
        use_enum_values = True


class FrameData(BaseModel):
    """Incoming frame data from frontend"""
    frame: str  # base64 encoded image
    timestamp: Optional[str] = None


class StatusUpdate(BaseModel):
    """Status update for monitoring"""
    status: str
    face_detected: bool
    head_pose: Optional[dict] = None
    mouth_status: Optional[str] = None
    timestamp: str

# --- Authentication Models ---

class User(BaseModel):
    """User model for database"""
    username: str
    email: str
    full_name: Optional[str] = None
    role: str # 'admin' or 'student'
    disabled: Optional[bool] = None

class UserCreate(User):
    """User creation model"""
    password: str

class UserInDB(User):
    """User in database with hashed password"""
    hashed_password: str

class Token(BaseModel):
    """Token model for response"""
    access_token: str
    token_type: str
    role: str

class TokenData(BaseModel):
    """Token data for decoding"""
    username: Optional[str] = None
    role: Optional[str] = None


# --- Analytics Models ---

class QuizSubmission(BaseModel):
    """Model for student quiz result submission"""
    username: Optional[str] = None
    topic: str
    score: int
    total: int
    percentage: float
    duration_seconds: int
    timestamp: Optional[datetime] = None

class ViolationRecord(BaseModel):
    """Model for recording a proctoring violation"""
    username: Optional[str] = None
    violation_type: AlertType
    severity: Severity
    timestamp: Optional[datetime] = None
    details: Optional[str] = None
