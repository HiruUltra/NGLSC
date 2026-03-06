"""
ML inference service – bridges the route layer and the ml/ module.
"""
from app.core.config import ALERT_MESSAGES
from app.schemas.alerts import AlertEvent, AlertType, Severity, StatusUpdate


def build_alert(alert_type: AlertType, current_time: float) -> AlertEvent:
    """
    Create an AlertEvent from a raw alert type.
    Used by the proctoring WebSocket route after receiving results
    from the ML inference layer.
    """
    severity_map = {
        AlertType.ALL_CLEAR:       Severity.INFO,
        AlertType.NO_FACE:         Severity.WARNING,
        AlertType.HEAD_TURN_LEFT:  Severity.CRITICAL,
        AlertType.HEAD_TURN_RIGHT: Severity.CRITICAL,
        AlertType.TALKING:         Severity.CRITICAL,
    }
    from datetime import datetime
    messages = ALERT_MESSAGES[alert_type.value]
    return AlertEvent(
        alert_type=alert_type,
        message_en=messages["en"],
        message_si=messages["si"],
        timestamp=datetime.now().isoformat(),
        severity=severity_map.get(alert_type, Severity.WARNING),
        metadata={"triggered_at": current_time},
    )
