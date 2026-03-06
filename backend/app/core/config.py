"""
Configuration settings for the AI Exam Proctoring System
"""
import os
from pathlib import Path

# ── Detection Thresholds ──────────────────────────────────────────────────────
YAW_THRESHOLD = 30    # degrees – head turn angle to trigger detection
MAR_THRESHOLD = 0.6   # Mouth Aspect Ratio threshold for talking detection

# ── Timer Settings (seconds) ─────────────────────────────────────────────────
HEAD_TURN_DURATION = 5   # seconds user must turn head before alert fires
ALERT_COOLDOWN     = 3   # seconds between repeated alerts

# ── MediaPipe Settings ────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5
MAX_NUM_FACES            = 1    # only track primary face for exam proctoring

# ── Camera Settings ───────────────────────────────────────────────────────────
DEFAULT_CAMERA_WIDTH  = 640
DEFAULT_CAMERA_HEIGHT = 480

# ── WebSocket Settings ────────────────────────────────────────────────────────
WEBSOCKET_PING_INTERVAL = 30  # seconds
WEBSOCKET_PING_TIMEOUT  = 10  # seconds

# ── Storage Paths ─────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parents[2]          # backend/
LECTURE_GALLERY_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "data" / "uploads"))
OUTPUT_DIR          = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "data" / "outputs"))

# Create directories on import
LECTURE_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# ── External APIs (optional) ──────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL   = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")

# ── Alert Messages (EN / SI bilingual) ───────────────────────────────────────
ALERT_MESSAGES = {
    "NO_FACE": {
        "en": "Warning. Face not detected. Please stay in view.",
        "si": "අවවාදයයි. මුහුණ හඳුනාගත නොහැක. කරුණාකර දර්ශනයේ සිටින්න."
    },
    "HEAD_TURN_LEFT": {
        "en": "Warning. Please focus on the exam. Do not look left.",
        "si": "අවවාදයයි. විභාගයට අවධානය යොමු කරන්න. වමට නොබලන්න."
    },
    "HEAD_TURN_RIGHT": {
        "en": "Warning. Please focus on the exam. Do not look right.",
        "si": "අවවාදයයි. විභාගයට අවධානය යොමු කරන්න. දකුණට නොබලන්න."
    },
    "TALKING": {
        "en": "Warning. Please stop talking during the exam.",
        "si": "අවවාදයයි. විභාගය අතරතුර කතා නොකරන්න."
    },
    "ALL_CLEAR": {
        "en": "All clear. Continue with your exam.",
        "si": "සියල්ල හරි. ඔබේ විභාගය දිගටම කරන්න."
    }
}
