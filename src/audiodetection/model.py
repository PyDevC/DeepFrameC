import torch
import torch.nn as nn
import timm

class AudioDeepFakeDetector(nn.Module):
    def __init__(self, backbone_name="efficientnet_b4", pretrained=True, dropout=0.4, num_classes=2):
        super().__init__()
        # in_chans=1 enables processing of single-channel Mel-spectrograms
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            in_chans=1, 
            num_classes=0, 
            global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512), 
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
