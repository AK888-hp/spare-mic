import argparse
import torch

from export_utils import GRU, LINEAR, export_layers
from train_speaker_verification import SpeakerVerificationModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="speaker_verification.pth")
    parser.add_argument("--out", default="speaker_verification.bin")
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    model = SpeakerVerificationModel(n_speakers=len(checkpoint["speaker_to_idx"]),n_mels=32)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    gru = model.gru
    export_layers([
        ("gru.weight_ih", GRU, gru.weight_ih, gru.bias_ih),
        ("gru.weight_hh", GRU, gru.weight_hh, gru.bias_hh),
        ("embedding", LINEAR, model.embedding.weight, model.embedding.bias),
    ], args.out)


if __name__ == "__main__":
    main()
