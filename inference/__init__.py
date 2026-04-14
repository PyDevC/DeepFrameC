"""Shared inference module for DeepFake detection."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from PIL import Image
from tqdm import tqdm

from models import create_model
from utils import FrameExtractor, ImageTransforms, FaceDetector, BlurDetector, load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of deepfake detection."""
    is_fake: bool
    fake_probability: float
    real_probability: float
    confidence: float
    frame_results: List[Dict[str, float]]
    processing_time: float
    num_frames: int
    model_name: str
    metadata: Dict

    def to_dict(self) -> dict:
        return {
            "verdict": "FAKE" if self.is_fake else "REAL",
            "is_fake": self.is_fake,
            "fake_probability": self.fake_probability,
            "real_probability": self.real_probability,
            "confidence": self.confidence,
            "frame_results": self.frame_results,
            "processing_time": self.processing_time,
            "num_frames": self.num_frames,
            "model_name": self.model_name,
            "metadata": self.metadata
        }


class DeepFakeDetectorInference:
    """Inference engine for DeepFake detection."""
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        backbone: str = "vit_base_patch16_224",
        device: Optional[str] = None,
        image_size: int = 224,
        use_tta: bool = True,
        use_face_detection: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.backbone = backbone
        self.image_size = image_size
        self.use_tta = use_tta
        
        self.model = create_model(
            backbone=backbone,
            num_classes=2,
            pretrained=False,
            dropout=0.5
        )
        
        if self.checkpoint_path.exists():
            self.model, _, _ = load_checkpoint(
                self.checkpoint_path,
                self.model,
                device=self.device
            )
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {self.checkpoint_path}. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = ImageTransforms.get_inference_transforms(image_size)
        self.tta_transforms = ImageTransforms.get_tta_transforms(image_size) if use_tta else None
        
        self.frame_extractor = FrameExtractor(target_size=(image_size, image_size))
        self.face_detector = FaceDetector() if use_face_detection else None
        self.blur_detector = BlurDetector()
        
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame."""
        if self.face_detector:
            frame = self.face_detector.detect_face(frame)
        
        image = Image.fromarray(frame)
        return self.transform(image)
    
    def preprocess_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of frames."""
        tensors = [self.preprocess_frame(f) for f in frames]
        return torch.stack(tensors)
    
    def predict_frame(self, frame: np.ndarray, use_tta: bool = True) -> np.ndarray:
        """Predict on a single frame."""
        if use_tta and self.tta_transforms:
            predictions = []
            for name, transform in self.tta_transforms.items():
                if self.face_detector:
                    processed = self.face_detector.detect_face(frame)
                else:
                    processed = frame
                image = Image.fromarray(processed)
                tensor = transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=1)
                    predictions.append(probs.cpu().numpy())
            
            return np.mean(predictions, axis=0)[0]
        else:
            tensor = self.preprocess_frame(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()[0]
    
    def predict_video(
        self,
        video_path: Union[str, Path],
        num_frames: int = 16,
        threshold: float = 0.5
    ) -> DetectionResult:
        """Predict on a video file."""
        import time
        start_time = time.time()
        
        frames = self.frame_extractor.extract_frames(video_path, num_frames)
        
        if not frames:
            return DetectionResult(
                is_fake=False,
                fake_probability=0.0,
                real_probability=1.0,
                confidence=0.0,
                frame_results=[],
                processing_time=time.time() - start_time,
                num_frames=0,
                model_name=self.backbone,
                metadata={"error": "Could not extract frames from video"}
            )
        
        frame_results = []
        all_probs = []
        
        for frame in tqdm(frames, desc="Processing frames", leave=False):
            probs = self.predict_frame(frame, use_tta=self.use_tta)
            all_probs.append(probs)
            frame_results.append({
                "fake_prob": float(probs[1]),
                "real_prob": float(probs[0])
            })
        
        all_probs = np.array(all_probs)
        avg_probs = all_probs.mean(axis=0)
        
        fake_prob = float(avg_probs[1])
        real_prob = float(avg_probs[0])
        is_fake = fake_prob > threshold
        confidence = max(fake_prob, real_prob)
        
        return DetectionResult(
            is_fake=is_fake,
            fake_probability=fake_prob,
            real_probability=real_prob,
            confidence=confidence,
            frame_results=frame_results,
            processing_time=time.time() - start_time,
            num_frames=len(frames),
            model_name=self.backbone,
            metadata={
                "video_path": str(video_path),
                "threshold": threshold,
                "use_tta": self.use_tta
            }
        )
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.5
    ) -> DetectionResult:
        """Predict on a single image."""
        import time
        start_time = time.time()
        
        image = Image.open(image_path).convert("RGB")
        frame = np.array(image)
        
        probs = self.predict_frame(frame, use_tta=self.use_tta)
        
        fake_prob = float(probs[1])
        real_prob = float(probs[0])
        is_fake = fake_prob > threshold
        confidence = max(fake_prob, real_prob)
        
        return DetectionResult(
            is_fake=is_fake,
            fake_probability=fake_prob,
            real_probability=real_prob,
            confidence=confidence,
            frame_results=[{"fake_prob": fake_prob, "real_prob": real_prob}],
            processing_time=time.time() - start_time,
            num_frames=1,
            model_name=self.backbone,
            metadata={
                "image_path": str(image_path),
                "threshold": threshold
            }
        )
    
    def predict_batch(
        self,
        file_paths: List[Union[str, Path]],
        threshold: float = 0.5
    ) -> List[DetectionResult]:
        """Predict on a batch of files."""
        results = []
        for path in file_paths:
            path = Path(path)
            if path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                result = self.predict_video(path, threshold=threshold)
            else:
                result = self.predict_image(path, threshold=threshold)
            results.append(result)
        return results


def create_inference_engine(
    checkpoint_path: Union[str, Path],
    **kwargs
) -> DeepFakeDetectorInference:
    """Factory function to create inference engine."""
    return DeepFakeDetectorInference(checkpoint_path=checkpoint_path, **kwargs)
