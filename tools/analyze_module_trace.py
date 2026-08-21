import argparse
from pathlib import Path

import torch


def error(actual, expected):
    delta = actual.float() - expected.float()
    return delta.abs().max().item(), delta.square().mean().sqrt().item()


def main():
    parser = argparse.ArgumentParser(description="Compare upstream and native decoder-layer traces")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()

    upstream = torch.load(args.upstream, map_location="cpu", weights_only=True)["layers"]
    native = torch.load(args.native, map_location="cpu", weights_only=True)["layers"]
    layer_count = len(native)
    if len(upstream) != layer_count * 2:
        raise ValueError("CFG1 one-step upstream trace must contain one prefix and one generation call per layer")

    print("layer prefix_input prefix_output image_input image_output")
    for layer in range(layer_count):
        prefix = upstream[layer]
        image = upstream[layer_count + layer]
        current = native[layer]
        values = (
            error(current["prefix_input"], prefix["input"]),
            error(current["prefix_output"], prefix["output"]),
            error(current["image_input"], image["input"]),
            error(current["image_output"], image["output"]),
        )
        print(layer, *(f"{maximum:.6g}/{rmse:.6g}" for maximum, rmse in values))


if __name__ == "__main__":
    main()
