"""
Pydantic schemas for proctoring alerts and WebSocket communication.
"""
from pydantic import BaseModel, ConfigDict
from enum import Enum
from datetime import datetime
from typing import Optional


class AlertType(str, Enum):
    """Types of alerts that can be triggered by the proctoring engine."""
    NO_FACE        = "NO_FACE"
    HEAD_TURN_LEFT  = "HEAD_TURN_LEFT"
    HEAD_TURN_RIGHT = "HEAD_TURN_RIGHT"
    TALKING        = "TALKING"
    ALL_CLEAR      = "ALL_CLEAR"


class Severity(str, Enum):
    """Alert severity levels."""
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


class AlertEvent(BaseModel):
    """Alert event sent to the frontend via WebSocket."""
    model_config = ConfigDict(use_enum_values=True)

    alert_type: AlertType
    message_en: str
    message_si: str
    timestamp:  str
    severity:   Severity
    metadata:   Optional[dict] = None


class FrameData(BaseModel):
    """Incoming frame data from the frontend (base64-encoded JPEG)."""
    frame:     str             # base64 encoded image
    timestamp: Optional[str] = None


class StatusUpdate(BaseModel):
    """Periodic status update sent to the frontend."""
    status:       str
    face_detected: bool
    head_pose:    Optional[dict] = None
    mouth_status: Optional[str]  = None
    timestamp:    str
