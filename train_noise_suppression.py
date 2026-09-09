import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from noise_dataset import NoiseSuppressionDataset


class ExplicitGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)

    def forward(self, x, hidden):
        gi = torch.nn.functional.linear(x, self.weight_ih, self.bias_ih)
        gh = torch.nn.functional.linear(hidden, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, -1)
        h_r, h_z, h_n = gh.chunk(3, -1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        candidate = torch.tanh(i_n + reset * h_n)
        return (1.0 - update) * candidate + update * hidden


class NoiseSuppressionModel(nn.Module):
    def __init__(self, n_mels=16, hidden_size=32):
        super().__init__()
        self.gru = ExplicitGRUCell(n_mels, hidden_size)
        self.mask = nn.Linear(hidden_size, n_mels)

    def forward(self, features):
        hidden = features.new_zeros(features.size(0), self.gru.hidden_size)
        outputs = []
        for frame in features.transpose(0, 1):
            hidden = self.gru(frame, hidden)
            outputs.append(torch.sigmoid(self.mask(hidden)))
        return torch.stack(outputs, dim=1)


def run(epochs=3, max_items=None, batch_size=32, out="noise_suppression.pth"):
    train = NoiseSuppressionDataset(subset="training", max_items=max_items)
    valid = NoiseSuppressionDataset(subset="validation", max_items=max_items)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid, batch_size=batch_size)
    model = NoiseSuppressionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train(); total = 0.0
        for noisy, clean in train_dl:
            loss = ((model(noisy) * noisy - clean) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        model.eval(); val_total = 0.0
        with torch.no_grad():
            for noisy, clean in valid_dl:
                val_total += ((model(noisy) * noisy - clean) ** 2).mean().item()
        history["train_loss"].append(total / len(train_dl))
        history["val_loss"].append(val_total / len(valid_dl))
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={history['train_loss'][-1]:.6f} "
              f"val_loss={history['val_loss'][-1]:.6f}")
    torch.save(model.state_dict(), out)
    print(f"Saved {out}; loss curve: {history}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", default="noise_suppression.pth")
    args = parser.parse_args()
    run(args.epochs, args.max_items, out=args.out)
