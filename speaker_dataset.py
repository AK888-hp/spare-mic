"""
Dataset for training the speaker-verification embedding network.

Speech Commands' metadata already includes a speaker_id per clip (the
dataset's __getitem__ returns waveform, sample_rate, label, speaker_id,
utterance_number) -- so no extra recording/download is needed to get
many-speaker training data.

We train a classifier over the N most frequent speakers (closed-set
classification), then at inference time discard the final classification
layer and use the penultimate layer as a fixed-length speaker embedding
(the standard "d-vector" approach) -- compare new embeddings to an
enrolled one via cosine similarity.
"""
import os
import random
from collections import Counter
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.datasets as datasets

SR = 16000
N_MELS = 32
N_FFT = 400
HOP_LENGTH = 160
MAX_LEN_S = 1.0


def mel_db(waveform, sample_rate=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=N_FFT, hop_length=hop_length
    )
    mel = mel_transform(waveform).squeeze(0)
    return torchaudio.transforms.AmplitudeToDB()(mel)  # (n_mels, T)


class SpeakerDataset(Dataset):
    def __init__(self, root="./data", subset="training", top_k_speakers=20, max_len_s=MAX_LEN_S):
        self.root = root
        self.max_len_samples = int(max_len_s * SR)

        self.sc_dataset = datasets.SPEECHCOMMANDS(
            root=root, url="speech_commands_v0.02", folder_in_archive="SpeechCommands",
            download=True, subset=subset
        )

        # Count speaker frequency, keep only the most common top_k.
        # IMPORTANT: use get_metadata() here, NOT self.sc_dataset[i] -- the latter
        # decodes every waveform just to read speaker_id, which is why this used
        # to be extremely slow on the full ~85k-clip training split.
        print(f"Scanning speakers in {subset} split...")
        speaker_counts = Counter()
        for i in range(len(self.sc_dataset)):
            _, _, _, speaker_id, _ = self.sc_dataset.get_metadata(i)
            speaker_counts[speaker_id] += 1

        top_speakers = [sid for sid, _ in speaker_counts.most_common(top_k_speakers)]
        self.speaker_to_idx = {sid: i for i, sid in enumerate(top_speakers)}
        self.num_speakers = len(top_speakers)

        self.indices = []
        for i in range(len(self.sc_dataset)):
            _, _, _, speaker_id, _ = self.sc_dataset.get_metadata(i)
            if speaker_id in self.speaker_to_idx:
                self.indices.append(i)

        print(f"SpeakerDataset[{subset}]: {self.num_speakers} speakers, {len(self.indices)} clips")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        waveform, sr, _, speaker_id, _ = self.sc_dataset[real_idx]
        if sr != SR:
            waveform = torchaudio.transforms.Resample(sr, SR)(waveform)

        if waveform.shape[1] > self.max_len_samples:
            waveform = waveform[:, :self.max_len_samples]
        elif waveform.shape[1] < self.max_len_samples:
            pad = self.max_len_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        features = mel_db(waveform)  # (n_mels, T)
        label = self.speaker_to_idx[speaker_id]
        return features.T, label  # (T, n_mels)


def collate_fn(batch):
    feats, labels = zip(*batch)
    feats = torch.stack(feats, dim=0)  # (batch, T, n_mels)
    labels = torch.tensor(labels, dtype=torch.long)
    return feats, labels


def get_speaker_dataloaders(batch_size=16, root="./data", top_k_speakers=20, val_fraction=0.2, seed=42):
    """
    IMPORTANT: Speech Commands assigns each speaker to exactly ONE official
    split (train/validation/test never share a speaker) -- that's fine for
    word classification, but wrong for speaker verification, since it means
    the training split's top speakers essentially never appear in the
    official validation split (0 overlap).

    Instead, we build everything from the "training" split (by far the
    largest), pick the top-K most frequent speakers there, and manually
    hold out ~20% of EACH speaker's own utterances for validation -- so
    validation checks generalization to new utterances from known speakers,
    which is the actually-relevant question for verification.
    """
    random.seed(seed)
    full_ds = SpeakerDataset(root=root, subset="training", top_k_speakers=top_k_speakers)

    # group this dataset's own indices by speaker so we can split per-speaker
    by_speaker = {}
    for local_idx, real_idx in enumerate(full_ds.indices):
        _, _, _, speaker_id, _ = full_ds.sc_dataset.get_metadata(real_idx)
        by_speaker.setdefault(speaker_id, []).append(local_idx)

    train_local_idx, val_local_idx = [], []
    for speaker_id, local_idxs in by_speaker.items():
        random.shuffle(local_idxs)
        n_val = max(1, int(len(local_idxs) * val_fraction))
        val_local_idx.extend(local_idxs[:n_val])
        train_local_idx.extend(local_idxs[n_val:])

    print(f"Per-speaker split: {len(train_local_idx)} train clips, {len(val_local_idx)} val clips "
          f"across {len(by_speaker)} speakers")

    train_ds = torch.utils.data.Subset(full_ds, train_local_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_local_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dl, val_dl, full_ds.num_speakers