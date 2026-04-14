"""CLI client for DeepFake detection API."""

import requests
import argparse
from pathlib import Path
from typing import Optional
import sys
import json


class DeepFakeAPI:
    """Client for DeepFrameC API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> dict:
        """List available models."""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def detect(
        self,
        file_path: str,
        threshold: float = 0.5,
        num_frames: int = 16,
        checkpoint: str = "best.pth"
    ) -> dict:
        """Detect deepfake in file."""
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f)}
            data = {
                "threshold": threshold,
                "num_frames": num_frames,
                "checkpoint": checkpoint
            }
            response = requests.post(
                f"{self.base_url}/detect",
                files=files,
                data=data
            )
        
        response.raise_for_status()
        return response.json()
    
    def detect_batch(
        self,
        file_paths: list,
        threshold: float = 0.5,
        num_frames: int = 16,
        checkpoint: str = "best.pth"
    ) -> dict:
        """Batch detect deepfakes."""
        files = []
        for path in file_paths:
            files.append(("files", (Path(path).name, open(path, "rb"))))
        
        data = {
            "threshold": threshold,
            "num_frames": num_frames,
            "checkpoint": checkpoint
        }
        
        response = requests.post(
            f"{self.base_url}/detect/batch",
            files=files,
            data=data
        )
        
        for _, (_, f) in files:
            f.close()
        
        response.raise_for_status()
        return response.json()


def print_result(result: dict):
    """Print detection result nicely."""
    if result["status"] != "completed":
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    data = result["result"]
    
    print(f"\n{'='*60}")
    print("DEEPFAKE DETECTION RESULTS")
    print(f"{'='*60}\n")
    
    verdict = "FAKE" if data["is_fake"] else "REAL"
    symbol = "❌" if data["is_fake"] else "✅"
    
    print(f"{symbol} Verdict: {verdict}")
    print(f"   Confidence: {data['confidence']:.2%}")
    print(f"   Fake Prob:  {data['fake_probability']:.4f}")
    print(f"   Real Prob:  {data['real_probability']:.4f}")
    print(f"\n   Frames: {data['num_frames']}")
    print(f"   Time:   {data['processing_time']:.2f}s")
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="DeepFrameC API Client")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    health = subparsers.add_parser("health", help="Check API health")
    models = subparsers.add_parser("models", help="List available models")
    
    detect = subparsers.add_parser("detect", help="Detect deepfake in file")
    detect.add_argument("file", help="Path to video/image file")
    detect.add_argument("--threshold", "-t", type=float, default=0.5)
    detect.add_argument("--frames", "-f", type=int, default=16)
    detect.add_argument("--checkpoint", "-c", default="best.pth")
    
    batch = subparsers.add_parser("batch", help="Batch detect deepfakes")
    batch.add_argument("files", nargs="+", help="Path to files")
    batch.add_argument("--threshold", "-t", type=float, default=0.5)
    batch.add_argument("--frames", "-f", type=int, default=16)
    batch.add_argument("--checkpoint", "-c", default="best.pth")
    
    parser.add_argument("--url", "-u", default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    
    client = DeepFakeAPI(args.url)
    
    try:
        if args.command == "health":
            result = client.health_check()
            print(json.dumps(result, indent=2))
        
        elif args.command == "models":
            result = client.list_models()
            print(json.dumps(result, indent=2))
        
        elif args.command == "detect":
            print(f"Detecting: {args.file}")
            result = client.detect(
                args.file,
                threshold=args.threshold,
                num_frames=args.frames,
                checkpoint=args.checkpoint
            )
            print_result(result)
        
        elif args.command == "batch":
            print(f"Processing {len(args.files)} files...")
            result = client.detect_batch(
                args.files,
                threshold=args.threshold,
                num_frames=args.frames,
                checkpoint=args.checkpoint
            )
            
            print(f"\n{'='*60}")
            print(f"BATCH RESULTS ({result['total']} files)")
            print(f"{'='*60}\n")
            
            for i, res in enumerate(result["results"]):
                verdict = "FAKE" if res["is_fake"] else "REAL"
                symbol = "❌" if res["is_fake"] else "✅"
                print(f"{symbol} {verdict} ({res['confidence']:.2%})")
            
            print(f"\n{'='*60}\n")
        
        else:
            parser.print_help()
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to API at {args.url}")
        print("Make sure the server is running: python -m app.api.main")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
