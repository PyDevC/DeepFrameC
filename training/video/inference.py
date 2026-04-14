"""Inference module for training - test models during/after training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference import DeepFakeDetectorInference, DetectionResult
from utils import FrameExtractor


def test_on_video(
    checkpoint_path: str,
    video_path: str,
    backbone: str = "vit_base_patch16_224",
    num_frames: int = 16,
    threshold: float = 0.5,
    device: str = "cuda"
):
    """Test inference on a single video."""
    print(f"\n{'='*60}")
    print("DeepFake Detection - Video Inference (Training Module)")
    print(f"{'='*60}\n")
    
    engine = DeepFakeDetectorInference(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        device=device,
        image_size=224,
        use_tta=True,
        use_face_detection=False
    )
    
    print(f"Model: {backbone}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Frames to process: {num_frames}")
    print(f"TTA enabled: True")
    print(f"Threshold: {threshold}\n")
    
    result = engine.predict_video(video_path, num_frames=num_frames, threshold=threshold)
    
    print(f"{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Verdict: {'❌ FAKE' if result.is_fake else '✅ REAL'}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Fake Probability: {result.fake_probability:.4f}")
    print(f"Real Probability: {result.real_probability:.4f}")
    print(f"\nFrames processed: {result.num_frames}")
    print(f"Processing time: {result.processing_time:.2f}s")
    if result.num_frames > 0:
        print(f"Average time per frame: {result.processing_time/result.num_frames*1000:.1f}ms")
    
    print(f"\nPer-frame probabilities:")
    for i, fr in enumerate(result.frame_results):
        bar_len = int(fr['fake_prob'] * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        print(f"  Frame {i+1:2d}: [{bar}] {fr['fake_prob']:.3f}")
    
    print(f"\n{'='*60}\n")
    
    return result


def test_on_batch(
    checkpoint_path: str,
    input_dir: str,
    backbone: str = "vit_base_patch16_224",
    num_frames: int = 16,
    threshold: float = 0.5,
    device: str = "cuda"
):
    """Test inference on a directory of videos."""
    print(f"\n{'='*60}")
    print("DeepFake Detection - Batch Inference (Training Module)")
    print(f"{'='*60}\n")
    
    engine = DeepFakeDetectorInference(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        device=device,
        image_size=224,
        use_tta=True
    )
    
    input_path = Path(input_dir)
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    video_files = [f for f in input_path.rglob("*") if f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files\n")
    
    results = []
    for video_file in video_files:
        print(f"Processing: {video_file.name}")
        result = engine.predict_video(video_file, num_frames=num_frames, threshold=threshold)
        results.append((video_file.name, result))
        print(f"  -> {'FAKE' if result.is_fake else 'REAL'} "
              f"(conf: {result.confidence:.2%}, fake: {result.fake_probability:.3f})\n")
    
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}\n")
    
    fake_count = sum(1 for _, r in results if r.is_fake)
    real_count = len(results) - fake_count
    
    print(f"Total videos: {len(results)}")
    print(f"Detected FAKE: {fake_count}")
    print(f"Detected REAL: {real_count}")
    print(f"\nAverage confidence: {sum(r.confidence for _, r in results) / len(results):.2%}")
    
    print(f"\nDetailed results:")
    print(f"{'Filename':<40} {'Verdict':<8} {'Confidence':<12} {'Fake Prob':<10}")
    print("-" * 70)
    for name, result in sorted(results, key=lambda x: -x[1].confidence):
        verdict = "FAKE" if result.is_fake else "REAL"
        print(f"{name:<40} {verdict:<8} {result.confidence:.2%}       {result.fake_probability:.4f}")
    
    print(f"\n{'='*60}\n")


def evaluate_model(checkpoint_path: str, data_dir: str, backbone: str = "vit_base_patch16_224"):
    """Evaluate model on a test set."""
    print(f"\n{'='*60}")
    print("DeepFake Detection - Model Evaluation (Training Module)")
    print(f"{'='*60}\n")
    
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
    import numpy as np
    
    engine = DeepFakeDetectorInference(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        device="cuda" if __name__ == "__main__" else "cpu",
        image_size=224,
        use_tta=False
    )
    
    data_path = Path(data_dir)
    all_probs = []
    all_labels = []
    
    for label_name in ["REAL", "FAKE"]:
        label_dir = data_path / label_name
        if not label_dir.exists():
            continue
        
        label = 0 if label_name == "REAL" else 1
        
        for img_path in label_dir.rglob("*.jpg"):
            result = engine.predict_image(img_path)
            all_probs.append(result.fake_probability)
            all_labels.append(label)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    
    print(f"\nEvaluation Results:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, preds, target_names=["REAL", "FAKE"]))
    
    print(f"Confusion Matrix:")
    cm = confusion_matrix(all_labels, preds)
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    return {"auc": auc, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser(
        description="DeepFake Detection Inference (Training Module)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    video_parser = subparsers.add_parser("video", help="Test on single video")
    video_parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    video_parser.add_argument("--video", "-v", required=True, help="Path to video file")
    video_parser.add_argument("--backbone", "-b", default="vit_base_patch16_224", help="Model backbone")
    video_parser.add_argument("--frames", "-f", type=int, default=16, help="Number of frames to sample")
    video_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Decision threshold")
    video_parser.add_argument("--device", "-d", default="cuda", help="Device (cuda/cpu)")
    
    batch_parser = subparsers.add_parser("batch", help="Test on directory of videos")
    batch_parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    batch_parser.add_argument("--input", "-i", required=True, help="Input directory")
    batch_parser.add_argument("--backbone", "-b", default="vit_base_patch16_224", help="Model backbone")
    batch_parser.add_argument("--frames", "-f", type=int, default=16, help="Number of frames to sample")
    batch_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Decision threshold")
    batch_parser.add_argument("--device", "-d", default="cuda", help="Device (cuda/cpu)")
    
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on test set")
    eval_parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--data", "-d", required=True, help="Test data directory")
    eval_parser.add_argument("--backbone", "-b", default="vit_base_patch16_224", help="Model backbone")
    
    args = parser.parse_args()
    
    if args.command == "video":
        test_on_video(
            checkpoint_path=args.checkpoint,
            video_path=args.video,
            backbone=args.backbone,
            num_frames=args.frames,
            threshold=args.threshold,
            device=args.device
        )
    elif args.command == "batch":
        test_on_batch(
            checkpoint_path=args.checkpoint,
            input_dir=args.input,
            backbone=args.backbone,
            num_frames=args.frames,
            threshold=args.threshold,
            device=args.device
        )
    elif args.command == "evaluate":
        evaluate_model(
            checkpoint_path=args.checkpoint,
            data_dir=args.data,
            backbone=args.backbone
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
