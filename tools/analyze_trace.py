import argparse
from pathlib import Path

import torch


def unpatchify(patches, height, width):
    token_height = height // 32
    token_width = width // 32
    batch = patches.shape[0]
    return patches.view(batch, token_height, token_width, 32, 32, 3).permute(0, 5, 1, 3, 2, 4).reshape(batch, 3, height, width)


def error(actual, expected):
    delta = actual.float() - expected.float()
    return delta.abs().max().item(), delta.square().mean().sqrt().item()


def main():
    parser = argparse.ArgumentParser(description="Analyze upstream and native Euler traces")
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()

    oracle = torch.load(args.oracle, map_location="cpu", weights_only=True)
    native = torch.load(args.native, map_location="cpu", weights_only=True)
    branches = oracle["branch_predictions"]
    steps = oracle["steps"]
    if branches.shape[0] != steps * 2:
        raise ValueError("T2I CFG trace must contain positive and negative predictions for every step")

    base = torch.linspace(0.0, 1.0, steps + 1)
    sigma = 1.0 - base
    shifted_sigma = oracle["shift"] * sigma / (1.0 + (oracle["shift"] - 1.0) * sigma)
    timesteps = 1.0 - shifted_sigma
    upstream_current = oracle["initial"].clone()
    report_steps = {0, 1, 2, 4, 9, 19, 29, 39, steps - 1}

    print("step state_max state_rmse velocity_max velocity_rmse")
    for step in range(steps):
        positive = unpatchify(branches[step * 2], oracle["height"], oracle["width"])
        negative = unpatchify(branches[step * 2 + 1], oracle["height"], oracle["width"])
        upstream_velocity = negative + oracle["cfg"] * (positive - negative)
        native_step = native["steps"][step]
        native_current = native_step["current"]
        native_velocity = (native_step["denoised"] - native_current) / shifted_sigma[step]
        state_error = error(native_current, upstream_current)
        velocity_error = error(native_velocity, upstream_velocity)
        if step in report_steps:
            print(step, *state_error, *velocity_error)
        delta_t = timesteps[step + 1] - timesteps[step]
        upstream_current = (upstream_current + delta_t * upstream_velocity).to(torch.bfloat16)

    print("reconstructed_upstream", *error(upstream_current, oracle["output"]))
    print("native_output", *error(native["output"], oracle["output"]))


if __name__ == "__main__":
    main()
