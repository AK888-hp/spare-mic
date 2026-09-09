"""
Exports a trained SpikeRouterSNN's weights to a raw binary format the ESP32
firmware can read directly off the SD card -- no PyTorch needed on-device.

Per-tensor int8 quantization with a float32 scale factor per layer, which is
simple enough to dequantize in a few lines of C on the ESP32:
    float_value = int8_value * scale

Binary layout per file (little-endian):
    [4 bytes]  magic = b"SNN1"
    [4 bytes]  int32  num_layers
    for each layer:
        [4 bytes]  int32  layer_type      (0 = Linear)
        [4 bytes]  int32  in_features
        [4 bytes]  int32  out_features
        [4 bytes]  float32 weight_scale
        [in_features * out_features bytes]  int8 quantized weights (row-major: out x in)
        [out_features * 4 bytes]            float32 biases (kept fp32 -- small, cheap)

Usage:
    python export_weights.py --model spikerouter_snn.pth --out router_weights.bin
"""
import argparse
import torch
from train_snn import SpikeRouterSNN
from dataset import CLASSES
from export_utils import export_layers


def export_model(model: torch.nn.Module, out_path: str):
    layers = []
    # Walk the model's named Linear layers in forward-pass order
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers.append((name, module))

    print(f"Found {len(layers)} Linear layers to export: {[n for n, _ in layers]}")

    export_layers([(name, 0, layer.weight, layer.bias) for name, layer in layers], out_path)
    print(f"Classes (for firmware label mapping, in order): {CLASSES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spikerouter_snn.pth")
    parser.add_argument("--out", default="router_weights.bin")
    parser.add_argument("--num_inputs", type=int, default=32)
    parser.add_argument("--num_hidden", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cpu")
    model = SpikeRouterSNN(num_inputs=args.num_inputs, num_hidden=args.num_hidden, num_outputs=len(CLASSES))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    export_model(model, args.out)