"""Training module for video DeepFake detection."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import logging

from models import create_model
from utils import ImageTransforms, set_seed, AverageMeter, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class VideoFrameDataset(Dataset):
    """Dataset for video frame classification."""
    
    def __init__(
        self,
        data_dir: Path,
        manifest_path: Optional[Path] = None,
        transform=None,
        is_train: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform or ImageTransforms.get_train_transforms()
        self.is_train = is_train
        
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
                
                for img_path in label_dir.rglob("*.jpg"):
                    records.append({
                        "path": str(img_path),
                        "label": label,
                        "split": split
                    })
                for img_path in label_dir.rglob("*.png"):
                    records.append({
                        "path": str(img_path),
                        "label": label,
                        "split": split
                    })
        
        return pd.DataFrame(records)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = row["label"]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Trainer:
    """Trainer for DeepFake detection model."""
    
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
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            losses.update(loss.item(), images.size(0))
            
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
        losses = AverageMeter()
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                losses.update(loss.item(), images.size(0))
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
        acc = accuracy_score(all_labels, [p > 0.5 for p in all_probs])
        
        return {
            "val_loss": losses.avg,
            "val_acc": 100.0 * correct / total,
            "val_auc": auc
        }
    
    def train(
        self,
        epochs: int,
        checkpoint_dir: Path,
        early_stopping_patience: int = 5
    ):
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
                logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                           f"Val Acc: {val_metrics['val_acc']:.2f}%, "
                           f"Val AUC: {val_metrics['val_auc']:.4f}")
                
                current_auc = val_metrics.get("val_auc", 0)
                
                if current_auc > self.best_auc:
                    self.best_auc = current_auc
                    save_checkpoint(
                        checkpoint_dir / "best.pth",
                        self.model,
                        epoch=epoch,
                        metrics={**train_metrics, **val_metrics}
                    )
                    logger.info(f"  Saved best model (AUC: {self.best_auc:.4f})")
                    no_improve = 0
                else:
                    no_improve += 1
            
            save_checkpoint(
                checkpoint_dir / "last.pth",
                self.model,
                self.optimizer,
                epoch=epoch,
                metrics={**train_metrics, **val_metrics}
            )
            
            self.scheduler.step()
            
            if no_improve >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        logger.info(f"Training complete. Best AUC: {self.best_auc:.4f}")
        return self.best_auc


def train_model(
    data_dir: Path,
    checkpoint_dir: Path,
    backbone: str = "vit_base_patch16_224",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: int = 224,
    device: str = "cuda"
):
    """Convenience function to train a model."""
    set_seed(42)
    
    model = create_model(
        backbone=backbone,
        num_classes=2,
        pretrained=True,
        dropout=0.5
    )
    
    train_dataset = VideoFrameDataset(
        data_dir / "train",
        transform=ImageTransforms.get_train_transforms(image_size),
        is_train=True
    )
    
    val_dataset = VideoFrameDataset(
        data_dir / "val",
        transform=ImageTransforms.get_val_transforms(image_size),
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    config = {
        "lr": lr,
        "epochs": epochs,
        "label_smoothing": 0.1,
        "weight_decay": 1e-4
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    return trainer.train(epochs, checkpoint_dir)
