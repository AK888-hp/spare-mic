import argparse
import torch

from export_utils import GRU, LINEAR, export_layers
from train_noise_suppression import NoiseSuppressionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="noise_suppression.pth")
    parser.add_argument("--out", default="noise_suppression.bin")
    args = parser.parse_args()
    model = NoiseSuppressionModel()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    gru = model.gru
    layers = [
        ("gru.weight_ih", GRU, gru.weight_ih, gru.bias_ih),
        ("gru.weight_hh", GRU, gru.weight_hh, gru.bias_hh),
        ("mask", LINEAR, model.mask.weight, model.mask.bias),
    ]
    export_layers(layers, args.out)


if __name__ == "__main__":
    main()
