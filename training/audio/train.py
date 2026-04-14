"""Training module for audio DeepFake detection."""

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
import numpy as np

from models import create_model
from utils import set_seed, AverageMeter, save_checkpoint

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Dataset for audio deepfake detection."""
    
    def __init__(
        self,
        data_dir: Path,
        manifest_path: Optional[Path] = None,
        sample_rate: int = 16000,
        duration: float = 4.0,
        is_train: bool = True,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_train = is_train
        self.augment = augment and is_train
        self.target_length = int(sample_rate * duration)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        if manifest_path and manifest_path.exists():
            self.df = pd.read_csv(manifest_path)
        else:
            self.df = self._build_manifest()
    
    def _build_manifest(self) -> pd.DataFrame:
        """Build manifest from directory structure."""
        records = []
        
        for split in ["train", "val", "test"]:
            split_path = self.data_dir / split
            if not split_path.exists():
                continue
            
            for label_name in ["REAL", "FAKE"]:
                label_dir = split_path / label_name
                if not label_dir.exists():
                    continue
                
                label = 0 if label_name == "REAL" else 1
                
                for audio_path in label_dir.rglob("*.wav"):
                    records.append({"path": str(audio_path), "label": label, "split": split})
                for audio_path in label_dir.rglob("*.mp3"):
                    records.append({"path": str(audio_path), "label": label, "split": split})
        
        return pd.DataFrame(records)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file."""
        try:
            waveform, sr = torchaudio.load(path)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return waveform.squeeze(0), self.sample_rate
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return torch.zeros(self.target_length), self.sample_rate
    
    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or crop to target length."""
        length = waveform.shape[0]
        
        if length < self.target_length:
            padding = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif length > self.target_length:
            start = np.random.randint(0, length - self.target_length) if self.is_train else 0
            waveform = waveform[start:start + self.target_length]
        
        return waveform
    
    def _spectrogram_augment(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment."""
        freq_mask = torchaudio.transforms.FrequencyMasking(20)
        time_mask = torchaudio.transforms.TimeMasking(40)
        
        mel_spec = freq_mask(mel_spec)
        mel_spec = time_mask(mel_spec)
        
        return mel_spec
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = row["path"]
        label = row["label"]
        
        waveform, sr = self._load_audio(path)
        waveform = self._pad_or_crop(waveform)
        
        mel_spec = self.mel_transform(waveform.unsqueeze(0))
        mel_spec = self.amplitude_to_db(mel_spec)
        
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        mel_spec = torch.clamp(mel_spec, -5, 5)
        
        if self.augment:
            mel_spec = self._spectrogram_augment(mel_spec)
        
        return mel_spec, label


class AudioTrainer:
    """Trainer for audio deepfake detection."""
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer=None,
        criterion=None,
        device: str = "cuda",
        config: Optional[Dict] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        
        self.model.to(device)
        
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get("lr", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-4)
        )
        
        self.criterion = criterion or nn.CrossEntropyLoss(
            label_smoothing=self.config.get("label_smoothing", 0.1)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("epochs", 20)
        )
        
        self.best_auc = 0
        self.scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()
        correct = 0
        total = 0
        
        for mel_specs, labels in self.train_loader:
            mel_specs = mel_specs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(mel_specs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(mel_specs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            losses.update(loss.item(), mel_specs.size(0))
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            "train_loss": losses.avg,
            "train_acc": 100.0 * correct / total
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for mel_specs, labels in self.val_loader:
                mel_specs = mel_specs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(mel_specs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        
        return {
            "val_acc": 100.0 * acc,
            "val_auc": auc
        }
    
    def train(self, epochs: int, checkpoint_dir: Path, early_stopping_patience: int = 5):
        """Full training loop."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        no_improve = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                        f"Train Acc: {train_metrics['train_acc']:.2f}%")
            
            if val_metrics:
                logger.info(f"  Val Acc: {val_metrics['val_acc']:.2f}%, Val AUC: {val_metrics['val_auc']:.4f}")
                
                current_auc = val_metrics.get("val_auc", 0)
                
                if current_auc > self.best_auc:
                    self.best_auc = current_auc
                    save_checkpoint(
                        checkpoint_dir / "best.pth",
                        self.model,
                        epoch=epoch,
                        metrics={**train_metrics, **val_metrics}
                    )
                    no_improve = 0
                else:
                    no_improve += 1
            
            save_checkpoint(checkpoint_dir / "last.pth", self.model, self.optimizer, epoch=epoch)
            self.scheduler.step()
            
            if no_improve >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        return self.best_auc
