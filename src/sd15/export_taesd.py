"""SD1/2 TAESD decoder 가중치를 고정 512px ONNX로 내보낸다.

구조: https://github.com/madebyollin/taesd (MIT, Ollin Boer Bohan).
입력은 diffusion latent 그대로, 출력은 대략 [0, 1]이다.
"""

import argparse
from pathlib import Path

import torch
from torch import nn


def conv(inputs, outputs, **kwargs):
    return nn.Conv2d(inputs, outputs, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, values):
        return torch.tanh(values / 3) * 3


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            conv(64, 64), nn.ReLU(), conv(64, 64), nn.ReLU(), conv(64, 64)
        )
        self.skip = nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, values):
        return self.fuse(self.conv(values) + self.skip(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decoder = nn.Sequential(
        Clamp(),
        conv(4, 64),
        nn.ReLU(),
        Block(),
        Block(),
        Block(),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(),
        Block(),
        Block(),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(),
        Block(),
        Block(),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(),
        conv(64, 3),
    ).eval()
    decoder.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            torch.zeros(1, 4, 64, 64),
            str(args.output),
            input_names=["latent_sample"],
            output_names=["sample"],
            opset_version=17,
            dynamo=False,
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
