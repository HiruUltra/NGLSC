"""
FastAPI WebSocket server for real-time exam proctoring
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from proctoring_engine import ProctoringEngine
from models import AlertEvent, FrameData
from quiz_generator import generate_quiz, QuizConfig, QuizResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Exam Proctoring System",
    description="Real-time exam monitoring using computer vision",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI Exam Proctoring System",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "mediapipe": "initialized",
        "websocket": "ready",
        "quiz_generation": "enabled"
    }


# Lecture gallery directory
LECTURE_GALLERY_DIR = Path("lecture_gallery")
LECTURE_GALLERY_DIR.mkdir(exist_ok=True)


@app.post("/upload-lecture")
async def upload_lecture(file: UploadFile = File(...)):
    """
    Upload a lecture video recording
    
    Args:
        file: Video file (webm, mp4, etc.)
        
    Returns:
        JSON with success status and file path
    """
    try:
        # Validate file type
        allowed_extensions = {".webm", ".mp4", ".avi", ".mov", ".mkv"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"lecture_{timestamp}{file_ext}"
        file_path = LECTURE_GALLERY_DIR / filename
        
        # Save file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        file_size_mb = len(contents) / (1024 * 1024)
        
        logger.info(f"Lecture uploaded: {filename} ({file_size_mb:.2f} MB)")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Lecture uploaded successfully",
                "filename": filename,
                "file_path": str(file_path),
                "size_mb": round(file_size_mb, 2)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lecture upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/lectures")
async def list_lectures():
    """
    Get list of all uploaded lecture videos
    
    Returns:
        JSON array of lecture files with metadata
    """
    try:
        lectures = []
        
        for file_path in LECTURE_GALLERY_DIR.glob("lecture_*.*"):
            if file_path.is_file():
                stat = file_path.stat()
                lectures.append({
                    "filename": file_path.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(file_path)
                })
        
        # Sort by upload time, newest first
        lectures.sort(key=lambda x: x["uploaded_at"], reverse=True)
        
        return JSONResponse(
            status_code=200,
            content={
                "total": len(lectures),
                "lectures": lectures
            }
        )
    
    except Exception as e:
        logger.error(f"Error listing lectures: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list lectures: {str(e)}")


@app.get("/generate-quiz", response_model=QuizResponse)
async def generate_quiz_endpoint(topic: str, count: int = 5, duration: int = 10):
    """
    Generate a quiz with random questions
    
    Args:
        topic: Topic/subject for the quiz (e.g., "Mathematics", "Science")
        count: Number of questions (default: 5)
        duration: Duration in minutes (default: 10)
        
    Returns:
        QuizResponse with generated questions
    """
    try:
        # Validate inputs
        if count < 1 or count > 50:
            count = 5
        if duration < 1 or duration > 180:
            duration = 10
        
        # Generate questions
        questions = generate_quiz(topic, count)
        
        return QuizResponse(
            topic=topic,
            total_questions=len(questions),
            duration_minutes=duration,
            questions=questions
        )
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI Exam Proctoring System",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "mediapipe": "initialized",
        "websocket": "ready"
    }


@app.websocket("/ws/proctoring")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time frame processing
    
    Protocol:
    - Client sends: JSON with base64 encoded frame
    - Server sends: JSON with alert events and status updates
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Initialize proctoring engine for this connection
    engine = ProctoringEngine()
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            
            try:
                # Parse incoming data
                frame_data = json.loads(data)
                
                # Decode base64 frame
                frame_bytes = base64.b64decode(frame_data["frame"])
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning("Failed to decode frame")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid frame data"
                    })
                    continue
                
                # Process frame with proctoring engine
                alert, status = engine.process_frame(frame)
                
                # Send status update
                await websocket.send_json({
                    "type": "status",
                    "data": status.dict()
                })
                
                # Send alert if triggered
                if alert:
                    logger.info(f"Alert triggered: {alert.alert_type}")
                    await websocket.send_json({
                        "type": "alert",
                        "data": alert.dict()
                    })
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup resources
        engine.cleanup()
        logger.info("Proctoring engine cleaned up")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
