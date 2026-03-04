"""
FastAPI WebSocket server for real-time exam proctoring
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Depends
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
from models import AlertEvent, FrameData, User, UserCreate, Token, UserInDB
from quiz_generator import generate_quiz, QuizConfig, QuizResponse
import auth
from database import connect_to_mongo, close_mongo_connection, get_database
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from jwt.exceptions import InvalidTokenError

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

# Authentication setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

# Auth Helpers
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = auth.decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    db = get_database()
    user_dict = await db.users.find_one({"username": username})
    if user_dict is None:
        raise credentials_exception
    return User(**user_dict)

def check_role(role: str):
    async def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != role and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Not enough permissions")
        return current_user
    return role_checker


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


# --- Auth Routes ---

@app.post("/auth/register", response_model=User)
async def register(user_in: UserCreate):
    db = get_database()
    # Check if user exists
    existing_user = await db.users.find_one({"username": user_in.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash password
    hashed_password = auth.get_password_hash(user_in.password)
    user_dict = user_in.dict()
    del user_dict["password"]
    user_dict["hashed_password"] = hashed_password
    
    # Save to DB
    await db.users.insert_one(user_dict)
    return user_in

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = get_database()
    user_dict = await db.users.find_one({"username": form_data.username})
    if not user_dict or not auth.verify_password(form_data.password, user_dict["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = auth.create_access_token(data={"sub": user_dict["username"], "role": user_dict["role"]})
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "role": user_dict["role"]
    }

@app.get("/auth/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Lecture gallery directory
LECTURE_GALLERY_DIR = Path("lecture_gallery")
LECTURE_GALLERY_DIR.mkdir(exist_ok=True)


@app.post("/upload-lecture")
async def upload_lecture(file: UploadFile = File(...), current_user: User = Depends(check_role("admin"))):
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
async def list_lectures(current_user: User = Depends(check_role("admin"))):
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
async def generate_quiz_endpoint(topic: str, count: int = 5, duration: int = 10, current_user: User = Depends(check_role("student"))):
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
