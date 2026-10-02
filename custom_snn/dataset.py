import torch
from torch.utils.data import Dataset, DataLoader
import math

CLASSES = ["silence", "noise_clean", "speaker_verify"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class CustomCommandDataset(Dataset):
    def __init__(self, num_samples=1000, target_sr=16000, max_len_s=1.0, n_mels=16, hop_length=160, delta_threshold=5.0):
        self.num_samples = num_samples
        self.target_sr = target_sr
        self.max_len_samples = int(max_len_s * target_sr)
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.delta_threshold = delta_threshold

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Evenly distribute the classes
        label_idx = idx % 3
        label = CLASSES[label_idx]

        t = torch.linspace(0, 1.0, self.max_len_samples)
        waveform = torch.zeros(1, self.max_len_samples)

        # Base background noise
        noise = torch.randn(1, self.max_len_samples) * 0.01
        
        if label == "silence":
            waveform = noise
        elif label == "noise_clean":
            # High frequency sweep (simulate "noise clean" sound)
            # Sweep from 2000Hz to 4000Hz
            f0 = 2000
            f1 = 4000
            phase = 2 * math.pi * (f0 * t + (f1 - f0) / 2 * t**2)
            waveform = 0.5 * torch.sin(phase).unsqueeze(0) + noise
        elif label == "speaker_verify":
            # Low frequency pulses (simulate "speaker verify" sound)
            # 500 Hz base with amplitude modulation (3 Hz)
            base_freq = torch.sin(2 * math.pi * 500 * t)
            modulator = torch.sin(2 * math.pi * 3 * t)
            waveform = 0.5 * (base_freq * modulator).unsqueeze(0) + noise

        # Import mel_delta_modulate from the parent directory
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from audio_encoder import mel_delta_modulate

        spikes = mel_delta_modulate(
            waveform,
            sample_rate=self.target_sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            threshold=self.delta_threshold
        )

        return spikes, label_idx

def collate_fn_time_first(batch):
    spikes_list, labels_list = zip(*batch)
    spikes_tensor = torch.stack(spikes_list, dim=0).transpose(0, 1)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    return spikes_tensor, labels_tensor

def get_dataloaders(batch_size=32, target_sr=16000, delta_threshold=5.0, n_mels=16, hop_length=160):
    train_ds = CustomCommandDataset(num_samples=1000, target_sr=target_sr, delta_threshold=delta_threshold, n_mels=n_mels, hop_length=hop_length)
    val_ds = CustomCommandDataset(num_samples=200, target_sr=target_sr, delta_threshold=delta_threshold, n_mels=n_mels, hop_length=hop_length)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_time_first)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_time_first)

    return train_dl, val_dl
