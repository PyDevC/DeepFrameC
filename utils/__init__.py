"""Shared utilities for DeepFake detection."""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video files."""
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        sample_strategy: str = "uniform"
    ):
        self.target_size = target_size
        self.sample_strategy = sample_strategy
        
    def extract_frames(
        self,
        video_path: Union[str, Path],
        num_frames: int = 16
    ) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        if self.sample_strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.random.choice(total_frames, min(num_frames, total_frames), replace=False)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_CUBIC)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def extract_frames_from_bytes(self, video_bytes: bytes, num_frames: int = 16) -> List[np.ndarray]:
        """Extract frames from video bytes."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
        
        try:
            return self.extract_frames(temp_path, num_frames)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class FaceDetector:
    """Simple face detection using OpenCV Haar Cascade."""
    
    def __init__(
        self,
        margin: float = 0.2,
        min_size: int = 50,
        scale_factor: float = 1.1
    ):
        self.margin = margin
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.min_size = min_size
        self.scale_factor = scale_factor
        
    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop face from frame."""
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=5,
            minSize=(self.min_size, self.min_size)
        )
        
        if len(faces) == 0:
            h, w = frame.shape[:2]
            crop_size = min(h, w)
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
            return frame[top:top+crop_size, left:left+crop_size]
        
        x, y, fw, fh = faces[0]
        
        margin_x = int(fw * self.margin)
        margin_y = int(fh * self.margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + fw + margin_x)
        y2 = min(frame.shape[0], y + fh + margin_y)
        
        return frame[y1:y2, x1:x2]


class ImageTransforms:
    """Standard image transforms for DeepFake detection."""
    
    @staticmethod
    def get_train_transforms(image_size: int = 224):
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms(image_size: int = 224):
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_inference_transforms(image_size: int = 224):
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_tta_transforms(image_size: int = 224):
        return {
            "original": T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "flip": T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "brightness_up": T.Compose([
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "brightness_down": T.Compose([
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=-0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }


class BlurDetector:
    """Detect blurry images using Laplacian variance."""
    
    def __init__(self, threshold: float = 80.0):
        self.threshold = threshold
        
    def is_blurry(self, image: Union[np.ndarray, Image.Image]) -> bool:
        """Check if image is blurry."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.threshold
    
    def get_blur_score(self, image: Union[np.ndarray, Image.Image]) -> float:
        """Get blur score (higher = sharper)."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> Tuple[torch.nn.Module, Optional[dict]]:
    """Load model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model, checkpoint.get("epoch"), checkpoint.get("metrics")


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None
):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics:
        checkpoint["metrics"] = metrics
    
    torch.save(checkpoint, checkpoint_path)


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
