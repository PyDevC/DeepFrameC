# DeepFrameC - DeepFake Detection System

AI-powered system for detecting deepfake videos and images.

## Project Structure

```
DeepFrameC/
├── app/                    # Web application
│   ├── api/               # FastAPI backend
│   │   ├── main.py        # API server
│   │   └── client.py      # CLI client
│   └── ui/                # Frontend
│       ├── streamlit_app.py
│       └── templates/index.html
├── training/              # Training modules
│   ├── video/            # Video detection training
│   │   ├── train.py      # Training script
│   │   └── inference.py  # Inference for testing
│   └── audio/            # Audio detection training
│       └── train.py
├── inference/            # Shared inference module
├── models/               # Model definitions
├── utils/                # Shared utilities
├── configs/              # Configuration files
├── checkpoints/          # Model checkpoints
├── data/                 # Dataset storage
└── run.py               # Main entry point
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Web Application

```bash
# Start API server
python run.py api

# Or start Streamlit UI
python run.py ui

# Or start both
python run.py all
```

### 3. Run Inference

```bash
# From API
python -m app.api.client detect video.mp4

# Direct inference
python run.py infer video.mp4
```

### 4. Train a Model

```bash
python run.py train --data data/FaceForensics --epochs 20
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /detect` - Detect deepfake in file
- `POST /detect/batch` - Batch detection

## Web Interfaces

1. **FastAPI Docs**: http://localhost:8000/docs
2. **Streamlit UI**: http://localhost:8501
3. **HTML Frontend**: `app/ui/templates/index.html`

## Training

```bash
# Video detection
cd training/video
python inference.py video --checkpoint ../../checkpoints/best.pth --video /path/to/video.mp4

# Batch inference
python inference.py batch --checkpoint ../../checkpoints/best.pth --input /path/to/directory
```
