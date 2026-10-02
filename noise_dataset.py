import torch
from torch.utils.data import Dataset, DataLoader

class NoiseSuppressionDataset(Dataset):
    def __init__(self, root="./data", subset="training", n_mels=16, target_sr=16000, hop_length=160, max_items=100):
        self.n_mels = n_mels
        self.max_items = 200 if subset == "training" else 40
        self.frames = 101

    def __len__(self):
        return self.max_items

    def __getitem__(self, index):
        t = torch.linspace(0, 10, self.frames).unsqueeze(1).repeat(1, self.n_mels)
        clean_mel = (torch.sin(t + index) * 0.5 + 0.5)
        noise = torch.randn(self.frames, self.n_mels) * 0.1
        noisy_mel = clean_mel + noise
        return noisy_mel, clean_mel

def get_noise_dataloaders(batch_size=16, n_mels=16, root="./data", num_workers=0, **kwargs):
    train_dataset = NoiseSuppressionDataset(subset="training", n_mels=n_mels)
    val_dataset = NoiseSuppressionDataset(subset="validation", n_mels=n_mels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader