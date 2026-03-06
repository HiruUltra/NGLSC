"""
WebSocket route for real-time exam proctoring.
"""
import cv2
import base64
import json
import logging
import numpy as np

from fastapi import APIRouter
from fastapi.websockets import WebSocket, WebSocketDisconnect

from ml.inference.face_infer import ProctoringEngine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Proctoring"])


@router.websocket("/ws/proctoring")
async def websocket_proctoring(websocket: WebSocket):
    """
    WebSocket endpoint for real-time frame processing.

    Protocol
    --------
    Client sends:  JSON  { "frame": "<base64-jpeg>" }
    Server sends:  JSON  { "type": "status"|"alert"|"error", "data": {...} }
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    engine = ProctoringEngine()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload    = json.loads(raw)
                frame_data = base64.b64decode(payload["frame"])
                np_arr     = np.frombuffer(frame_data, np.uint8)
                frame      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Invalid frame data"})
                    continue

                alert, status = engine.process_frame(frame)

                await websocket.send_json({"type": "status", "data": status.model_dump()})

                if alert:
                    logger.info("Alert triggered: %s", alert.alert_type)
                    await websocket.send_json({"type": "alert", "data": alert.model_dump()})

            except (json.JSONDecodeError, KeyError) as exc:
                logger.error("Bad frame payload: %s", exc)
                await websocket.send_json({"type": "error", "message": "Invalid JSON or missing 'frame' key"})
            except Exception as exc:
                logger.error("Frame processing error: %s", exc)
                await websocket.send_json({"type": "error", "message": f"Processing error: {exc}"})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        engine.cleanup()
        logger.info("Proctoring engine cleaned up")
