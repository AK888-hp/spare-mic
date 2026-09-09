import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import get_dataloaders, CLASSES
from tqdm import tqdm
import os
import torchaudio

# --- 1. SNN Network Definition ---
class SpikeRouterSNN(nn.Module):
    def __init__(self, num_inputs=32, num_hidden=128, num_outputs=5, beta=0.9):
        super().__init__()
        
        # Initialize surrogate gradient (fast sigmoid is common and effective)
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Multi-channel input (Mel-bins * 2)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        """
        Forward pass.
        x shape: (time_steps, batch_size, num_inputs)
        """
        # Initialize hidden states (membrane potentials)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record output spikes and membrane potentials
        spk2_rec = []
        mem2_rec = []
        
        # Loop over time steps
        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            spk1_drop = self.dropout(spk1)
            
            cur2 = self.fc2(spk1_drop)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            
        # Convert lists to tensors: (time_steps, batch_size, num_outputs)
        return torch.stack(spk2_rec), torch.stack(mem2_rec)


# --- 2. Training and Evaluation Functions ---
def train_model(model, train_dl, val_dl, num_epochs=10, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # Cross entropy loss on the total spike count
    loss_fn = SF.ce_count_loss()
    
    # Store history
    loss_hist = []
    acc_hist = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_dl, desc="Training")
        
        for data, targets in progress_bar:
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            spk_rec, mem_rec = model(data)
            
            # Loss calculation
            loss = loss_fn(spk_rec, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            train_loss += loss.item()
            # Predict by max spike count over time
            _, idx = spk_rec.sum(dim=0).max(1)
            correct += (idx == targets).sum().item()
            total += targets.size(0)
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        # Validation phase
        val_acc = evaluate(model, val_dl, device)
        train_acc = correct / total
        
        loss_hist.append(train_loss / len(train_dl))
        acc_hist.append(val_acc)
        
        print(f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%\n")
        
    return loss_hist, acc_hist

def evaluate(model, dataloader, device="cpu", return_predictions=False):
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            
            spk_rec, _ = model(data)
            _, idx = spk_rec.sum(dim=0).max(1)
            
            correct += (idx == targets).sum().item()
            total += targets.size(0)
            
            if return_predictions:
                all_preds.extend(idx.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
    if return_predictions:
        return correct / total, all_preds, all_targets
    return correct / total

def plot_confusion_matrix(targets, preds, classes):
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# --- 3. Main Execution ---
if __name__ == "__main__":
    # Ensure torchaudio uses soundfile on Windows to avoid backend errors
    torchaudio.set_audio_backend("soundfile")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Get Dataloaders 
    # With 100 time steps per sample, we can easily increase batch size to 64
    n_mels = 16
    train_dl, val_dl = get_dataloaders(
        batch_size=64, 
        target_sr=16000, 
        delta_threshold=5.0, 
        n_mels=n_mels, 
        hop_length=160
    )
    
    # 2. Initialize Model
    # num_inputs = n_mels * 2 (UP and DOWN spike channels per mel bin)
    model = SpikeRouterSNN(num_inputs=n_mels*2, num_hidden=128, num_outputs=len(CLASSES)).to(device)
    
    # 3. Train Model
    print("Starting training...")
    loss_hist, acc_hist = train_model(model, train_dl, val_dl, num_epochs=10, lr=2e-3, device=device)
    
    # 4. Save the trained model for quantization check
    torch.save(model.state_dict(), "spikerouter_snn.pth")
    print("Model saved to spikerouter_snn.pth")
    
    # 5. Final Evaluation & Confusion Matrix
    print("Generating Confusion Matrix on Validation Set...")
    val_acc, preds, targets = evaluate(model, val_dl, device=device, return_predictions=True)
    
    plot_confusion_matrix(targets, preds, CLASSES)
    print("Training script finished.")
