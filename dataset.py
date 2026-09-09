import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.datasets as datasets
from audio_encoder import mel_delta_modulate

# Desired subset of commands.
CLASSES = ["up", "down", "on", "off", "silence"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

class SpikingSpeechCommands(Dataset):
    def __init__(self, root="./data", subset="training", target_sr=16000, delta_threshold=5.0, max_len_s=1.0, n_mels=16, hop_length=160):
        """
        Args:
            root (str): Where to download/find the dataset.
            subset (str): 'training', 'validation', or 'testing'.
            target_sr (int): Target sampling rate (16kHz is standard for SpeechCommands).
            delta_threshold (float): Threshold in dB for Mel delta modulation.
            max_len_s (float): Maximum length of audio in seconds.
            n_mels (int): Number of Mel frequency bins.
            hop_length (int): Hop length for the Mel spectrogram.
        """
        self.root = root
        self.subset = subset
        self.target_sr = target_sr
        self.delta_threshold = delta_threshold
        self.max_len_samples = int(max_len_s * target_sr)
        self.n_mels = n_mels
        self.hop_length = hop_length
        
        # Ensure the root directory exists before downloading
        os.makedirs(self.root, exist_ok=True)
        
        self.sc_dataset = datasets.SPEECHCOMMANDS(
            root=self.root,
            url="speech_commands_v0.02",
            folder_in_archive="SpeechCommands",
            download=True,
            subset=self.subset
        )
        
        self.data_items = []
        self._filter_dataset()
        
    def _filter_dataset(self):
        print(f"Filtering {self.subset} dataset for classes: {CLASSES}...")
        # IMPORTANT: use get_metadata() here, NOT self.sc_dataset[i] -- the latter
        # decodes the full waveform for every single file just to read its label,
        # which is why this used to take a very long time on the full dataset.
        # get_metadata() reads only the filepath/label/speaker_id, no audio decode.
        for i in range(len(self.sc_dataset)):
            _, sample_rate, label, speaker_id, utterance_number = self.sc_dataset.get_metadata(i)
            if label in CLASS_TO_IDX:
                self.data_items.append({
                    "index": i,
                    "label": label,
                    "label_idx": CLASS_TO_IDX[label]
                })
                
        # Count current items
        counts = {c: 0 for c in CLASSES}
        for item in self.data_items:
            counts[item["label"]] += 1
            
        avg_count = int(sum(counts.values()) / max(1, len([c for c in counts.values() if c > 0])))
        
        if counts["silence"] == 0 and avg_count > 0:
            print(f"Synthesizing 'silence' class with {avg_count} samples...")
            for _ in range(avg_count):
                self.data_items.append({
                    "index": -1, # indicates synthetic silence
                    "label": "silence",
                    "label_idx": CLASS_TO_IDX["silence"]
                })
                
        print(f"Dataset {self.subset} size: {len(self.data_items)}")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        if item["index"] == -1:
            # Generate synthetic silence with small background noise
            waveform = torch.randn(1, self.max_len_samples) * 0.005 # very quiet
        else:
            # Load real audio
            waveform, sample_rate, _, _, _ = self.sc_dataset[item["index"]]
            
            # Resample if needed
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
                waveform = resampler(waveform)
                
            # Standardize length (pad or truncate)
            if waveform.shape[1] > self.max_len_samples:
                waveform = waveform[:, :self.max_len_samples]
            elif waveform.shape[1] < self.max_len_samples:
                padding = self.max_len_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
        # Apply Mel-filterbank delta modulation (convert to spikes)
        # spikes is (time_steps, n_mels * 2)
        spikes = mel_delta_modulate(
            waveform, 
            sample_rate=self.target_sr, 
            n_mels=self.n_mels, 
            hop_length=self.hop_length, 
            threshold=self.delta_threshold
        )
        
        return spikes, item["label_idx"]

def collate_fn_time_first(batch):
    """
    Collate function to reshape batch to (time_steps, batch_size, input_features).
    snnTorch expects time-first format for most surrogate training.
    """
    spikes_list, labels_list = zip(*batch)
    
    # Stack along batch dimension (dim 1) -> shape: (batch, time, features)
    spikes_tensor = torch.stack(spikes_list, dim=0)
    # Transpose to (time, batch, features)
    spikes_tensor = spikes_tensor.transpose(0, 1)
    
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    
    return spikes_tensor, labels_tensor

def get_dataloaders(batch_size=32, target_sr=16000, delta_threshold=5.0, n_mels=16, hop_length=160):
    """
    Returns train and validation dataloaders.
    """
    print("Setting up training dataset...")
    train_ds = SpikingSpeechCommands(subset="training", target_sr=target_sr, delta_threshold=delta_threshold, n_mels=n_mels, hop_length=hop_length)
    
    print("Setting up validation dataset...")
    val_ds = SpikingSpeechCommands(subset="validation", target_sr=target_sr, delta_threshold=delta_threshold, n_mels=n_mels, hop_length=hop_length)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_time_first)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_time_first)
    
    return train_dl, val_dl

if __name__ == "__main__":
    # Test dataset loading
    import torchaudio
    torchaudio.set_audio_backend("soundfile") # Just in case it's run standalone
    
    print("Testing dataset loading...")
    train_dl, val_dl = get_dataloaders(batch_size=4)
    
    for spikes, labels in train_dl:
        print(f"Spikes batch shape (time, batch, channels): {spikes.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        break