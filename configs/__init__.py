"""Shared configuration for DeepFake detection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ModelConfig:
    backbone: str = "vit_base_patch16_224"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.5
    freeze_backbone: bool = False


@dataclass
class VideoConfig:
    face_size: int = 224
    frames_per_video: int = 16
    batch_size: int = 8
    use_mtcnn: bool = False
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.3
    face_margin: float = 0.2


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration: float = 4.0
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    top_db: float = 80.0


@dataclass
class TrainingConfig:
    data_root: Path = field(default_factory=lambda: Path("data"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    warmup_epochs: int = 3
    scheduler: str = "cosine"
    early_stopping_patience: int = 5
    mixed_precision: bool = True


@dataclass
class InferenceConfig:
    checkpoint_path: Path = field(default_factory=lambda: Path("checkpoints/best.pth"))
    device: str = "cuda"
    num_frames: int = 16
    batch_size: int = 8
    threshold: float = 0.5
    use_tta: bool = True
    tta_transforms: List[str] = field(default_factory=lambda: ["original", "flip", "brightness"])


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    max_upload_size: int = 500 * 1024 * 1024
    allowed_extensions: List[str] = field(default_factory=lambda: ["mp4", "avi", "mov", "mkv", "webm", "wav", "mp3"])
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
