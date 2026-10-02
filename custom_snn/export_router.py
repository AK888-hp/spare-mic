import torch
import struct
from train_router import SpikeRouterSNN
import os

def export_weights_to_bin(model, filepath):
    with open(filepath, 'wb') as f:
        # Write fc1 weights
        fc1_w = model.fc1.weight.data.cpu().numpy().flatten()
        for w in fc1_w:
            f.write(struct.pack('f', w))
        # Write fc1 bias
        if model.fc1.bias is not None:
            fc1_b = model.fc1.bias.data.cpu().numpy().flatten()
            for b in fc1_b:
                f.write(struct.pack('f', b))
                
        # Write fc2 weights
        fc2_w = model.fc2.weight.data.cpu().numpy().flatten()
        for w in fc2_w:
            f.write(struct.pack('f', w))
        # Write fc2 bias
        if model.fc2.bias is not None:
            fc2_b = model.fc2.bias.data.cpu().numpy().flatten()
            for b in fc2_b:
                f.write(struct.pack('f', b))
    print(f"Weights exported to {filepath}")

if __name__ == "__main__":
    device = torch.device("cpu")
    model = SpikeRouterSNN(num_inputs=32, num_hidden=128, num_outputs=3).to(device)
    if os.path.exists("custom_router_snn.pth"):
        model.load_state_dict(torch.load("custom_router_snn.pth", map_location=device))
        export_weights_to_bin(model, "custom_router_weights.bin")
    else:
        print("Model file custom_router_snn.pth not found! Run train_router.py first.")
