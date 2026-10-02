import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
import numpy as np
from dataset import get_dataloaders, CLASSES
from tqdm import tqdm
import os

class SpikeRouterSNN(nn.Module):
    def __init__(self, num_inputs=32, num_hidden=128, num_outputs=3, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []
        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_drop = self.dropout(spk1)
            cur2 = self.fc2(spk1_drop)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return torch.stack(spk2_rec), torch.stack(mem2_rec)

def train_model(model, train_dl, val_dl, num_epochs=10, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SF.ce_count_loss()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_dl, desc="Training")
        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)
            spk_rec, mem_rec = model(data)
            loss = loss_fn(spk_rec, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, idx = spk_rec.sum(dim=0).max(1)
            correct += (idx == targets).sum().item()
            total += targets.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data, targets in val_dl:
                data, targets = data.to(device), targets.to(device)
                spk_rec, _ = model(data)
                _, idx = spk_rec.sum(dim=0).max(1)
                val_correct += (idx == targets).sum().item()
                val_total += targets.size(0)
        print(f"Train Acc: {correct/total*100:.2f}% | Val Acc: {val_correct/val_total*100:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_mels = 16
    train_dl, val_dl = get_dataloaders(batch_size=64, n_mels=n_mels)
    
    model = SpikeRouterSNN(num_inputs=n_mels*2, num_hidden=128, num_outputs=len(CLASSES)).to(device)
    print("Starting training on custom words (Noise Clean, Speaker Verify, Silence)...")
    train_model(model, train_dl, val_dl, num_epochs=5, lr=2e-3, device=device)
    
    torch.save(model.state_dict(), "custom_router_snn.pth")
    print("Model saved to custom_router_snn.pth")
