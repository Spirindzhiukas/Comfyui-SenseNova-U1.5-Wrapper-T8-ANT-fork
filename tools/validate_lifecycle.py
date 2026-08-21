import argparse
import gc
import os
from pathlib import Path

import psutil
import torch

from validate_native import encode, load_package


def snapshot(label):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    memory = psutil.Process(os.getpid()).memory_info()
    values = {
        "label": label,
        "rss": memory.rss,
        "peak_working_set": getattr(memory, "peak_wset", memory.rss),
    }
    if torch.cuda.is_available():
        values.update(
            cuda_allocated=torch.cuda.memory_allocated(),
            cuda_reserved=torch.cuda.memory_reserved(),
            cuda_peak_allocated=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        )
    print(" ".join(f"{key}={value}" for key, value in values.items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Validate repeat, unload, cancellation, and reload behavior")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    load_package()
    import comfy.model_management
    import comfy.sample
    from comfyui_sensenova_u15_t8.sensenova_u15.loader import load_sensenova_clip, load_sensenova_model
    from comfyui_sensenova_u15_t8.sensenova_u15.sampling import SenseNovaModelSampling, resolution_noise_scale

    oracle = torch.load(args.oracle, map_location="cpu", weights_only=True)
    model = load_sensenova_model(str(args.model), torch.bfloat16)
    model_sampling = SenseNovaModelSampling(model.model.model_config)
    model_sampling.set_parameters(shift=oracle.get("shift", 1.0))
    model.add_object_patch("model_sampling", model_sampling)
    clip = load_sensenova_clip()
    positive = encode(clip, oracle["prompt"])
    negative = encode(clip, "")
    scale = resolution_noise_scale(oracle["height"], oracle["width"])
    noise = oracle["initial"].float() / scale
    latent = torch.zeros_like(noise)

    def sample():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return comfy.sample.sample(
            model,
            noise,
            args.steps,
            oracle.get("cfg", 1.0),
            "euler",
            "normal",
            positive,
            negative,
            latent,
            disable_pbar=True,
            seed=oracle["seed"],
        )

    reference = None
    for run in range(args.repeats):
        output = sample()
        if not torch.isfinite(output).all():
            raise ValueError(f"repeat {run} produced non-finite output")
        if reference is None:
            reference = output.clone()
        else:
            print(f"repeat_{run}_max_abs={(output - reference).abs().max().item()}", flush=True)
        snapshot(f"repeat_{run}")
        del output
        gc.collect()

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    snapshot("after_unload")

    output = sample()
    print(f"reload_max_abs={(output - reference).abs().max().item()}", flush=True)
    snapshot("after_reload")
    del output

    comfy.model_management.interrupt_current_processing(True)
    cancelled = False
    try:
        sample()
    except comfy.model_management.InterruptProcessingException:
        cancelled = True
    finally:
        comfy.model_management.interrupt_current_processing(False)
    if not cancelled:
        raise RuntimeError("sampling did not honor the cancellation request")
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()
    snapshot("after_cancel")

    output = sample()
    if not torch.isfinite(output).all():
        raise ValueError("post-cancellation reload produced non-finite output")
    print(f"post_cancel_max_abs={(output - reference).abs().max().item()}", flush=True)
    snapshot("post_cancel_reload")


if __name__ == "__main__":
    main()
