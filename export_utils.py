"""Shared SNN1 binary export and int8 quantization helpers.

Each layer is: int32 type (0 Linear, 1 GRU matrix), int32 input size,
int32 output size, float32 scale, row-major int8 weights, and float32 bias.
GRU matrices use the framework gate order reset/update/new.
"""
import os
import struct
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

MAGIC = b"SNN1"
LINEAR = 0
GRU = 1


def quantize_tensor_int8(weight: torch.Tensor) -> Tuple[np.ndarray, float]:
    """Per-tensor symmetric int8 quantization."""
    values = weight.detach().cpu().numpy().astype(np.float32)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0.0 else 1e-8
    quantized = np.round(values / scale).clip(-128, 127).astype(np.int8)
    return quantized, scale


def _write_layer(handle, layer_type: int, weight: torch.Tensor,
                 bias: Optional[torch.Tensor], name: str) -> None:
    out_features, in_features = weight.shape
    quantized, scale = quantize_tensor_int8(weight)
    handle.write(struct.pack("<iiif", layer_type, in_features, out_features, scale))
    handle.write(quantized.tobytes())
    bias_values = (bias.detach().cpu().numpy().astype(np.float32)
                   if bias is not None else np.zeros(out_features, dtype=np.float32))
    if bias_values.shape != (out_features,):
        raise ValueError(f"{name}: expected bias shape {(out_features,)}, got {bias_values.shape}")
    handle.write(bias_values.tobytes())
    error = float(np.max(np.abs(quantized.astype(np.float32) * scale -
                              weight.detach().cpu().numpy())))
    magnitude = float(np.max(np.abs(weight.detach().cpu().numpy())))
    print(f"  {name}: {out_features}x{in_features}, scale={scale:.6g}, "
          f"max dequant error={error:.6g} (max weight={magnitude:.6g})")


def export_layers(layers: Iterable[Tuple[str, int, torch.Tensor, Optional[torch.Tensor]]],
                  out_path: str) -> None:
    """Write named (name, type, weight, bias) tensors in the common SNN1 format."""
    layers = list(layers)
    with open(out_path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<i", len(layers)))
        for name, layer_type, weight, bias in layers:
            _write_layer(handle, layer_type, weight, bias, name)
    print(f"\nWrote {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")
