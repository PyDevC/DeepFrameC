"""FastAPI application for DeepFake detection."""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import tempfile
import shutil
import uuid
import logging
from datetime import datetime

from inference import DeepFakeDetectorInference, DetectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeepFrameC API",
    description="DeepFake Detection API for video and audio analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_DIR = Path("checkpoints")
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DETECTION_ENGINES = {}


def get_detection_engine(checkpoint_path: str = "best.pth", model_type: str = "video"):
    """Get or create detection engine."""
    key = f"{model_type}_{checkpoint_path}"
    
    if key not in DETECTION_ENGINES:
        full_path = CHECKPOINT_DIR / checkpoint_path
        
        if not full_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint not found: {checkpoint_path}"
            )
        
        DETECTION_ENGINES[key] = DeepFakeDetectorInference(
            checkpoint_path=str(full_path),
            backbone="vit_base_patch16_224",
            device="cuda" if __name__ != "__main__" else "cpu",
            image_size=224,
            use_tta=True
        )
    
    return DETECTION_ENGINES[key]


class DetectionResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    timestamp: str


class BatchDetectionRequest(BaseModel):
    threshold: float = 0.5
    num_frames: int = 16
    checkpoint: str = "best.pth"


class BatchDetectionResponse(BaseModel):
    request_id: str
    status: str
    total: int
    results: List[dict]
    timestamp: str


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "DeepFrameC API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/detect",
            "detect_batch": "/detect/batch",
            "health": "/health",
            "models": "/models",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    cuda_available = torch.cuda.is_available()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cuda_available": cuda_available,
        "engines_loaded": len(DETECTION_ENGINES),
        "temp_files": len(list(TEMP_DIR.iterdir())) if TEMP_DIR.exists() else 0
    }


@app.get("/models")
async def list_models():
    """List available models."""
    checkpoints = []
    
    for ckpt in CHECKPOINT_DIR.glob("*.pth"):
        checkpoints.append({
            "name": ckpt.name,
            "path": str(ckpt),
            "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 2)
        })
    
    return {"models": checkpoints}


@app.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    num_frames: int = Form(16),
    checkpoint: str = Form("best.pth")
):
    """Detect deepfake in uploaded video/image."""
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    try:
        allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {allowed_extensions}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        def cleanup():
            Path(temp_path).unlink(missing_ok=True)
        
        background_tasks.add_task(cleanup)
        
        engine = get_detection_engine(checkpoint)
        
        if file_ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            result = engine.predict_video(temp_path, num_frames=num_frames, threshold=threshold)
        else:
            result = engine.predict_image(temp_path, threshold=threshold)
        
        result_dict = result.to_dict()
        result_dict["request_id"] = request_id
        
        return DetectionResponse(
            request_id=request_id,
            status="completed",
            result=result_dict,
            timestamp=timestamp
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return DetectionResponse(
            request_id=request_id,
            status="error",
            error=str(e),
            timestamp=timestamp
        )


@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    threshold: float = Form(0.5),
    num_frames: int = Form(16),
    checkpoint: str = Form("best.pth")
):
    """Batch detect deepfakes in multiple files."""
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    try:
        engine = get_detection_engine(checkpoint)
        
        temp_files = []
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_files.append((tmp.name, file_ext))
        
        def cleanup():
            for path, _ in temp_files:
                Path(path).unlink(missing_ok=True)
        
        background_tasks.add_task(cleanup)
        
        results = []
        for path, ext in temp_files:
            if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                result = engine.predict_video(path, num_frames=num_frames, threshold=threshold)
            else:
                result = engine.predict_image(path, threshold=threshold)
            
            results.append(result.to_dict())
        
        return BatchDetectionResponse(
            request_id=request_id,
            status="completed",
            total=len(results),
            results=results,
            timestamp=timestamp
        )
    
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/audio")
async def detect_audio_deepfake(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    checkpoint: str = Form("audio_best.pth")
):
    """Detect deepfake in uploaded audio file."""
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    try:
        allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Audio type not supported. Allowed: {allowed_extensions}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        def cleanup():
            Path(temp_path).unlink(missing_ok=True)
        
        background_tasks.add_task(cleanup)
        
        return DetectionResponse(
            request_id=request_id,
            status="completed",
            result={
                "message": "Audio detection coming soon",
                "placeholder": True
            },
            timestamp=timestamp
        )
    
    except Exception as e:
        logger.error(f"Audio detection error: {e}")
        return DetectionResponse(
            request_id=request_id,
            status="error",
            error=str(e),
            timestamp=timestamp
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
