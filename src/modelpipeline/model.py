import torch
import torch.nn as nn
import timm

class DeepFakeDetector(nn.Module):
    """
    Backbone: EfficientNet-B4 or Xception via timm.
    Head: dropout → linear → 2-class logits.
    """
    def __init__(self, backbone_name="efficientnet_b4", pretrained=True,
                 dropout=0.5, num_classes=2):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,       # remove classifier head
            global_pool="avg",
        )
        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)          # (B, in_features)
        logits   = self.classifier(features) # (B, num_classes)
        return logits

    def get_feature_maps(self, x):
        """For GradCAM visualization."""
        return self.backbone.forward_features(x)

class XceptionDetector(nn.Module):
    """
    Xception backbone + frequency-domain attention branch.
    Frequency branch captures DCT artifacts specific to GAN generation.
    """
    def __init__(self, pretrained=True, num_classes=2, dropout=0.5):
        super().__init__()
        self.spatial_branch = timm.create_model(
            "xception", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        spatial_features = self.spatial_branch.num_features  # 2048

        # Frequency branch: apply DCT, treat as a 3-channel "image"
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        freq_features = 32

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(spatial_features + freq_features, num_classes),
        )

    def _dct_magnitude(self, x):
        """Grayscale → per-image DCT magnitude (log-scaled)."""
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B, H, W)
        # torch.fft.rfft2 for frequency domain
        freq = torch.fft.rfft2(gray)
        magnitude = torch.log1p(torch.abs(freq)).unsqueeze(1)          # (B, 1, H, W//2+1)
        return magnitude

    def forward(self, x):
        sp_feat   = self.spatial_branch(x)                   # (B, 2048)
        freq_map  = self._dct_magnitude(x)
        freq_feat = self.freq_branch(freq_map).flatten(1)    # (B, 32)
        combined  = torch.cat([sp_feat, freq_feat], dim=1)
        return self.classifier(combined)
