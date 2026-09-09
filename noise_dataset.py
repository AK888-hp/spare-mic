"""Speech/background-noise mixtures for magnitude-mel denoising."""
import os
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NoiseSuppressionDataset(Dataset):
    def __init__(self, root="./data", subset="training", n_mels=16,
                 target_sr=16000, hop_length=160, max_items=None):
        base = Path(root) / "SpeechCommands" / "speech_commands_v0.02"
        if not base.is_dir():
            raise FileNotFoundError(f"SpeechCommands is not present at {base}; refusing to download")
        self.sc = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, url="speech_commands_v0.02", folder_in_archive="SpeechCommands",
            download=False, subset=subset)
        self.items = list(range(len(self.sc)))
        if max_items is not None:
            self.items = self.items[:max_items]
        noise_dir = base / "_background_noise_"
        self.noise_files = list(noise_dir.glob("*.wav"))
        if not self.noise_files:
            raise FileNotFoundError(f"No .wav files found in {noise_dir}")
        self.target_sr, self.n_mels, self.hop_length = target_sr, n_mels, hop_length
        self.frames = 1 + target_sr // hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=400, hop_length=hop_length,
            n_mels=n_mels, power=2.0)
        print(f"Noise {subset}: {len(self.items)} speech clips, {len(self.noise_files)} noise clips")

    def __len__(self):
        return len(self.items)

    def _fixed_length(self, waveform):
        waveform = waveform.mean(dim=0, keepdim=True)
        length = self.target_sr
        if waveform.shape[1] < length:
            waveform = torch.nn.functional.pad(waveform, (0, length - waveform.shape[1]))
        return waveform[:, :length]

    def __getitem__(self, index):
        clean, sr, *_ = self.sc[self.items[index]]
        if sr != self.target_sr:
            clean = torchaudio.functional.resample(clean, sr, self.target_sr)
        clean = self._fixed_length(clean)
        noise, noise_sr = torchaudio.load(random.choice(self.noise_files))
        if noise_sr != self.target_sr:
            noise = torchaudio.functional.resample(noise, noise_sr, self.target_sr)
        noise = noise.mean(dim=0, keepdim=True)
        start = random.randrange(max(1, noise.shape[1] - self.target_sr + 1))
        noise = noise[:, start:start + self.target_sr]
        noise = self._fixed_length(noise)
        snr_db = random.uniform(-5.0, 15.0)
        clean_rms = clean.pow(2).mean().sqrt().clamp_min(1e-5)
        noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-5)
        noise = noise * (clean_rms / (noise_rms * (10.0 ** (snr_db / 20.0))))
        noisy = clean + noise
        clean_mel = self.mel(clean).clamp_min(1e-10).sqrt().squeeze(0).transpose(0, 1)
        noisy_mel = self.mel(noisy).clamp_min(1e-10).sqrt().squeeze(0).transpose(0, 1)
        return noisy_mel, clean_mel



def get_noise_dataloaders(batch_size=16, n_mels=32, root="./data", num_workers=0, **kwargs):
    train_dataset = NoiseSuppressionDataset(root=root, subset="training", n_mels=n_mels, **kwargs)
    val_dataset = NoiseSuppressionDataset(root=root, subset="validation", n_mels=n_mels, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader