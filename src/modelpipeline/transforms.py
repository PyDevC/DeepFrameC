import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(split: str, face_size: int = 224):
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(face_size, face_size), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ImageCompression(quality_range=(60, 100)),   # fixed
                A.GaussNoise(noise_scale_factor=0.1),
            ], p=0.4),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.CoarseDropout(                                    # fixed
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.3
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=face_size, width=face_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
