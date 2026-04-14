"""Main entry point for DeepFrameC application."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_api():
    """Run the FastAPI server."""
    print("Starting FastAPI server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


def run_streamlit():
    """Run the Streamlit UI."""
    print("Starting Streamlit UI on http://localhost:8501")
    subprocess.run([
        "streamlit", "run",
        "app/ui/streamlit_app.py",
        "--server.port", "8501"
    ])


def run_frontend():
    """Open the HTML frontend."""
    html_path = Path(__file__).parent / "app" / "ui" / "templates" / "index.html"
    import webbrowser
    webbrowser.open(f"file://{html_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="DeepFrameC - DeepFake Detection")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    api_parser = subparsers.add_parser("api", help="Run FastAPI server")
    streamlit_parser = subparsers.add_parser("ui", help="Run Streamlit UI")
    frontend_parser = subparsers.add_parser("frontend", help="Open HTML frontend")
    
    all_parser = subparsers.add_parser("all", help="Run API and UI together")
    
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", default="data/FaceForensics", help="Data directory")
    train_parser.add_argument("--backbone", default="vit_base_patch16_224", help="Model backbone")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("file", help="Video or image file")
    infer_parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Model checkpoint")
    infer_parser.add_argument("--frames", type=int, default=16, help="Number of frames")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api()
    elif args.command == "ui":
        run_streamlit()
    elif args.command == "frontend":
        run_frontend()
    elif args.command == "all":
        print("Starting both API and UI...")
        import threading
        api_thread = threading.Thread(target=run_api)
        ui_thread = threading.Thread(target=run_streamlit)
        
        api_thread.start()
        ui_thread.start()
        
        api_thread.join()
        ui_thread.join()
    elif args.command == "train":
        from training.video.train import train_model
        train_model(
            data_dir=Path(args.data),
            checkpoint_dir=Path("checkpoints"),
            backbone=args.backbone,
            epochs=args.epochs
        )
    elif args.command == "infer":
        from inference import DeepFakeDetectorInference
        engine = DeepFakeDetectorInference(checkpoint_path=args.checkpoint)
        result = engine.predict_video(args.file, num_frames=args.frames)
        print(f"\n{'='*60}")
        print(f"Verdict: {'FAKE' if result.is_fake else 'REAL'}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"{'='*60}\n")
    else:
        print("DeepFrameC - DeepFake Detection System\n")
        print("Usage:")
        print("  python run.py api       - Start API server (http://localhost:8000)")
        print("  python run.py ui        - Start Streamlit UI (http://localhost:8501)")
        print("  python run.py frontend  - Open HTML frontend")
        print("  python run.py all       - Start both API and UI")
        print("  python run.py train     - Train a model")
        print("  python run.py infer     - Run inference on a file")
        print("\nExamples:")
        print("  python run.py api")
        print("  python run.py infer video.mp4")
        print("  python run.py train --data data/ --epochs 10")


if __name__ == "__main__":
    main()
