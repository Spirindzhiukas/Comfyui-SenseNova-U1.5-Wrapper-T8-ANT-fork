import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import psutil
import torch
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]


def load_package():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    spec = importlib.util.spec_from_file_location(
        "comfyui_sensenova_u15_t8",
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return package


def encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text), show_pbar=False)


def attach_reference(conditioning, image, mode):
    import node_helpers

    return node_helpers.conditioning_set_values(
        conditioning,
        {
            "sensenova_reference_image": image,
            "sensenova_reference_mode": mode,
        },
        append=True,
    )


def compare(actual, expected):
    delta = actual.float().cpu() - expected.float().cpu()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "close_0.02": torch.allclose(actual.float().cpu(), expected.float().cpu(), rtol=0.02, atol=0.02),
    }


def save_image(path, tensor):
    pixels = tensor[0].float().clamp(-1.0, 1.0).add(1.0).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def main():
    parser = argparse.ArgumentParser(description="Compare the native ComfyUI path with a saved upstream oracle")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--custom-edit-guider", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--output-image", type=Path)
    parser.add_argument("--module-trace-output", type=Path)
    args = parser.parse_args()

    load_package()
    import comfy.sample
    import comfy.samplers
    from comfyui_sensenova_u15_t8.nodes import SenseNovaEditGuiderImpl
    from comfyui_sensenova_u15_t8.sensenova_u15.loader import load_sensenova_clip, load_sensenova_model
    from comfyui_sensenova_u15_t8.sensenova_u15.sampling import SenseNovaModelSampling, resolution_noise_scale

    oracle = torch.load(args.oracle, map_location="cpu", weights_only=True)
    model = load_sensenova_model(str(args.model), torch.bfloat16)
    layer_trace = []
    trace_handles = []
    if args.module_trace_output:
        def trace_layer(module, inputs, output):
            layer_trace.append(
                {
                    "prefix_input": inputs[0].detach().cpu(),
                    "image_input": inputs[1].detach().cpu(),
                    "prefix_output": output[0].detach().cpu(),
                    "image_output": output[1].detach().cpu(),
                }
            )

        trace_handles = [
            layer.register_forward_hook(trace_layer)
            for layer in model.model.diffusion_model.language_model.model.layers
        ]
    model_sampling = SenseNovaModelSampling(model.model.model_config)
    model_sampling.set_parameters(shift=oracle.get("shift", 1.0))
    model.add_object_patch("model_sampling", model_sampling)
    clip = load_sensenova_clip()
    positive = encode(clip, oracle["prompt"])
    negative = encode(clip, "")
    width = args.width or oracle["width"]
    height = args.height or oracle["height"]
    steps = args.steps or oracle["steps"]
    if args.width is None and args.height is None and args.steps is None:
        scale = resolution_noise_scale(height, width)
        noise = oracle["initial"].float() / scale
    else:
        noise = torch.randn((1, 3, height, width), generator=torch.manual_seed(oracle["seed"]))
    latent = torch.zeros_like(noise)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trace = []

    def callback(step, denoised, current, total_steps):
        trace.append(
            {
                "step": step,
                "denoised": denoised.detach().cpu(),
                "current": current.detach().cpu(),
            }
        )

    if "reference_image" not in oracle:
        actual = comfy.sample.sample(
            model,
            noise,
            steps,
            oracle.get("cfg", 1.0),
            "euler",
            "normal",
            positive,
            negative,
            latent,
            callback=callback if args.trace_output else None,
            disable_pbar=True,
            seed=oracle["seed"],
        )
    else:
        positive = attach_reference(positive, oracle["reference_image"], "condition")
        image_condition = attach_reference(negative, oracle["reference_image"], "image_only")
        img_cfg = oracle.get("img_cfg", 1.0)
        if args.custom_edit_guider or not math.isclose(img_cfg, 1.0):
            guider = SenseNovaEditGuiderImpl(model)
            guider.set_conds(positive, image_condition, negative)
            guider.set_cfg(oracle.get("cfg", 1.0), img_cfg)
            sigmas = comfy.samplers.calculate_sigmas(
                model.get_model_object("model_sampling"), "normal", steps
            )
            actual = guider.sample(
                noise,
                latent,
                comfy.samplers.sampler_object("euler"),
                sigmas,
                callback=callback if args.trace_output else None,
                disable_pbar=True,
                seed=oracle["seed"],
            )
        else:
            actual = comfy.sample.sample(
                model,
                noise,
                steps,
                oracle.get("cfg", 1.0),
                "euler",
                "normal",
                positive,
                image_condition,
                latent,
                callback=callback if args.trace_output else None,
                disable_pbar=True,
                seed=oracle["seed"],
            )

    if args.width is not None or args.height is not None or args.steps is not None:
        print(f"actual_shape={tuple(actual.shape)} finite={torch.isfinite(actual).all().item()}")
    else:
        print(f"actual_shape={tuple(actual.shape)} expected_shape={tuple(oracle['output'].shape)}")
        for name, value in compare(actual, oracle["output"]).items():
            print(f"{name}={value}")
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"steps": trace, "output": actual.cpu()}, args.trace_output)
        print(f"trace={args.trace_output}")
    if args.output_image:
        save_image(args.output_image, actual)
        print(f"image={args.output_image}")
    for handle in trace_handles:
        handle.remove()
    if args.module_trace_output:
        args.module_trace_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"layers": layer_trace}, args.module_trace_output)
        print(f"module_trace={args.module_trace_output}")
    memory = psutil.Process(os.getpid()).memory_info()
    print(f"rss_bytes={memory.rss} peak_working_set_bytes={getattr(memory, 'peak_wset', memory.rss)}")
    if torch.cuda.is_available():
        print(f"cuda_peak_allocated_bytes={torch.cuda.max_memory_allocated()} cuda_peak_reserved_bytes={torch.cuda.max_memory_reserved()}")


if __name__ == "__main__":
    main()
