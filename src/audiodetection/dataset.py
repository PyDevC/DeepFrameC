import torchaudio
import torchaudio.transforms as T
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class WaveFakeDataset(Dataset):
    def __init__(self, data_dir: str, df: pd.DataFrame, cfg, transform=None):
        self.data_dir = Path(data_dir)
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=cfg.SAMPLE_RATE,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS
        )
        self.label_map = {"REAL": 0, "FAKE": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.data_dir / row["File Path"]
        label = self.label_map[row["Label"]]

        waveform, sr = torchaudio.load(str(audio_path))
        
        # Resample if needed
        if sr != self.cfg.SAMPLE_RATE:
            resampler = T.Resample(sr, self.cfg.SAMPLE_RATE)
            waveform = resampler(waveform)

        # Truncate or pad to MAX_SAMPLES
        if waveform.shape[1] > self.cfg.MAX_SAMPLES:
            waveform = waveform[:, :self.cfg.MAX_SAMPLES]
        else:
            pad_amount = self.cfg.MAX_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Convert to Mel-Spectrogram (1, N_MELS, Time)
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale
        mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, label
