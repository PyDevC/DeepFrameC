"""Shared model definitions for DeepFake detection."""

import torch
import torch.nn as nn
import timm


class DeepFakeDetector(nn.Module):
    """Vision Transformer based DeepFake detector."""
    
    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        
        feature_dim = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetDetector(nn.Module):
    """EfficientNet based DeepFake detector."""
    
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        
        feature_dim = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class XceptionDetector(nn.Module):
    """Xception based detector with frequency analysis."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.spatial_backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        
        spatial_dim = self.spatial_backbone.num_features
        freq_dim = 32
        
        self.freq_head = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, freq_dim),
            nn.ReLU()
        )
        
        combined_dim = spatial_dim + freq_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_backbone(x)
        
        x_gray = x.mean(dim=1, keepdim=True)
        dct = torch.fft.rfft2(x_gray)
        dct_mag = torch.abs(dct)
        freq_features = self.freq_head(dct_mag)
        
        combined = torch.cat([spatial_features, freq_features], dim=1)
        return self.classifier(combined)


class EnsembleDetector(nn.Module):
    """Ensemble of multiple detectors."""
    
    def __init__(self, detectors: list):
        super().__init__()
        self.detectors = nn.ModuleList(detectors)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [detector(x) for detector in self.detectors]
        return torch.stack(outputs).mean(dim=0)


MODEL_REGISTRY = {
    "vit": DeepFakeDetector,
    "vit_base_patch16_224": DeepFakeDetector,
    "efficientnet_b4": EfficientNetDetector,
    "efficientnet_b0": EfficientNetDetector,
    "xception": XceptionDetector,
}


def create_model(
    backbone: str = "vit_base_patch16_224",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """Create a model by name."""
    model_cls = MODEL_REGISTRY.get(backbone, DeepFakeDetector)
    return model_cls(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
