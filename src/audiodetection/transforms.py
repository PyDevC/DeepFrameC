import torchaudio.transforms as T
import torch.nn as nn

def get_audio_transforms(split: str):
    if split == "train":
        return nn.Sequential(
            T.TimeMasking(time_mask_param=30),
            T.FrequencyMasking(freq_mask_param=15)
        )
    return nn.Identity()
