import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Update import to use your robust dataloader function
from speaker_dataset import get_speaker_dataloaders
from train_noise_suppression import ExplicitGRUCell


class SpeakerVerificationModel(nn.Module):
    def __init__(self, n_mels=16, hidden_size=32, embedding_size=32, n_speakers=20):
        super().__init__()
        self.gru = ExplicitGRUCell(n_mels, hidden_size)
        self.embedding = nn.Linear(hidden_size, embedding_size)
        self.classifier = nn.Linear(embedding_size, n_speakers)

    def forward(self, features):
        hidden = features.new_zeros(features.size(0), self.gru.hidden_size)
        for frame in features.transpose(0, 1):
            hidden = self.gru(frame, hidden)
        embedding = torch.nn.functional.normalize(self.embedding(hidden), dim=-1)
        return embedding, self.classifier(embedding)


def run(epochs=3, max_items=None, batch_size=32, out="speaker_verification.pth"):
    # 2. Automatically get properly partitioned dataloaders
    train_dl, valid_dl, num_speakers = get_speaker_dataloaders(batch_size=batch_size)
    
    model = SpeakerVerificationModel(n_mels=32, hidden_size=32, embedding_size=32, n_speakers=num_speakers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        model.train(); loss_sum = correct = total = 0
        for features, labels in train_dl:
            _, logits = model(features)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item(); correct += (logits.argmax(1) == labels).sum().item(); total += labels.numel()
        
        model.eval(); val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for features, labels in valid_dl:
                _, logits = model(features); val_loss += nn.functional.cross_entropy(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item(); val_total += labels.numel()
        
        history["train_loss"].append(loss_sum / len(train_dl)); history["val_loss"].append(val_loss / len(valid_dl))
        history["train_acc"].append(correct / total); history["val_acc"].append(val_correct / val_total)
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={history['train_loss'][-1]:.4f} "
              f"val_loss={history['val_loss'][-1]:.4f} train_acc={history['train_acc'][-1]*100:.2f}% "
              f"val_acc={history['val_acc'][-1]*100:.2f}%")
    
    # 3. Extract the speaker dictionary from the underlying dataset subset
    speaker_map = train_dl.dataset.dataset.speaker_to_idx
    torch.save({"state_dict": model.state_dict(), "speaker_to_idx": speaker_map}, out)
    print(f"Saved {out}; accuracy/loss curve: {history}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", default="speaker_verification.pth")
    args = parser.parse_args()
    run(args.epochs, args.max_items, out=args.out)