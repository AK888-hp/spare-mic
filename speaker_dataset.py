import torch
from torch.utils.data import Dataset, DataLoader

class SpeakerDataset(Dataset):
    def __init__(self, top_k_speakers=20, max_items=200):
        self.num_speakers = top_k_speakers
        self.max_items = max_items
        self.frames = 101
        self.n_mels = 32
        self.speaker_to_idx = {f"speaker_{i}": i for i in range(top_k_speakers)}

    def __len__(self):
        return self.max_items

    def __getitem__(self, idx):
        label = idx % self.num_speakers
        feats = torch.ones(self.frames, self.n_mels) * (label / self.num_speakers)
        feats += torch.randn_like(feats) * 0.05
        return feats, label

class _Dummy: pass

def get_speaker_dataloaders(batch_size=16, root="./data", top_k_speakers=20, val_fraction=0.2, seed=42):
    train_ds = SpeakerDataset(top_k_speakers=top_k_speakers, max_items=400)
    val_ds = SpeakerDataset(top_k_speakers=top_k_speakers, max_items=80)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Mock dataset attribute so train_dl.dataset.dataset.speaker_to_idx works
    dummy = _Dummy()
    dummy.speaker_to_idx = train_ds.speaker_to_idx
    train_dl.dataset.dataset = dummy
    
    return train_dl, val_dl, top_k_speakers