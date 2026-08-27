import argparse
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Run the pinned upstream SenseNova-U1.5 edit oracle")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="change the sky to blue")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--img-cfg", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--reference", type=Path, action="append")
    parser.add_argument("--output-image", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source.resolve() / "src"))
    import sensenova_u1
    from sensenova_u1.models.neo_unify import modeling_qwen3
    from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN, load_image_native
    from sensenova_u1.utils import DEFAULT_LAYERS_ATTR, load_model_and_tokenizer, offload_layers_sync
    from transformers.utils.generic import check_model_inputs

    closure = inspect.getclosurevars(modeling_qwen3.Qwen3Model.forward).nonlocals
    original_forward = closure.get("tie_last_hidden_states")
    if callable(original_forward):
        modeling_qwen3.Qwen3Model.forward = check_model_inputs()(original_forward)

    sensenova_u1.set_attn_backend("sdpa")
    model, tokenizer = load_model_and_tokenizer(
        str(args.model.resolve()),
        dtype=torch.bfloat16,
        device="cuda",
        for_offload=True,
    )

    if args.reference:
        references = [Image.open(path).convert("RGB") for path in args.reference]
        arrays = [np.asarray(reference, dtype=np.uint8) for reference in references]
    else:
        axis = np.arange(512, dtype=np.uint16)
        red = np.broadcast_to((axis % 256)[None, :], (512, 512))
        green = np.broadcast_to((axis % 256)[:, None], (512, 512))
        blue = ((red.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(np.uint8)
        array = np.stack((red.astype(np.uint8), green.astype(np.uint8), blue), axis=-1)
        arrays = [array]
        references = [Image.fromarray(array, mode="RGB")]

    grid_h = args.height // model.patch_size
    grid_w = args.width // model.patch_size
    merge_size = int(1 / model.downsample_ratio)
    scale = ((grid_h * grid_w) / (merge_size**2) / model.noise_scale_base_image_seq_len) ** 0.5
    noise_scale = min(scale * model.noise_scale, model.noise_scale_max_value)
    generator = torch.Generator("cuda").manual_seed(args.seed)
    initial = noise_scale * torch.randn(
        (1, 3, args.height, args.width),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    with offload_layers_sync(model, DEFAULT_LAYERS_ATTR, torch.device("cuda")) as offloaded:
        patch_list = []
        grid_list = []
        max_pixels = min(2048 * 2048, (4096 * 4096) // len(references))
        for reference in references:
            reference_patches, reference_grid = load_image_native(
                reference,
                model.patch_size,
                model.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=max_pixels,
                upscale=False,
            )
            patch_list.append(reference_patches)
            grid_list.append(reference_grid)
        reference_patches = torch.cat(patch_list)
        reference_grid = torch.cat(grid_list)
        reference_embedding = offloaded.extract_feature(
            reference_patches.to(device="cuda", dtype=torch.bfloat16),
            grid_hw=reference_grid.to("cuda"),
        )
        output = offloaded.it2i_generate(
            tokenizer,
            args.prompt,
            references,
            cfg_scale=args.cfg,
            img_cfg_scale=args.img_cfg,
            timestep_shift=args.shift,
            image_size=(args.width, args.height),
            num_steps=args.steps,
            batch_size=1,
            seed=args.seed,
            think_mode=False,
        )

    prompt_with_image = (
        "".join(f"Image-{index + 1}:<image>\n" for index in range(len(references))) + args.prompt
        if len(references) > 1
        else "<image>\n" + args.prompt
    )
    query = model._build_t2i_query(
        prompt_with_image,
        system_message=SYSTEM_MESSAGE_FOR_GEN,
        append_text="<think>\n\n</think>\n\n<img>",
    )
    for grid in reference_grid:
        context_count = int(grid[0] * grid[1] * model.downsample_ratio**2)
        query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * context_count + "</img>", 1)
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"]

    payload = {
        "input_ids": input_ids.cpu(),
        "prefix_indexes": model.get_thw_indexes(input_ids[0], reference_grid).cpu(),
        "reference_embedding": reference_embedding.cpu(),
        "reference_patches": reference_patches.cpu(),
        "reference_images": [torch.from_numpy(array.copy()).unsqueeze(0).float().div(255.0) for array in arrays],
        "initial": initial.cpu(),
        "output": output.cpu(),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "cfg": args.cfg,
        "img_cfg": args.img_cfg,
        "shift": args.shift,
    }
    if len(arrays) == 1:
        payload["reference_image"] = payload["reference_images"][0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    if args.output_image:
        pixels = output[0].float().clamp(-1.0, 1.0).add(1.0).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, mode="RGB").save(args.output_image)
    digest = hashlib.sha256(payload["output"].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    print(f"oracle={args.output}")
    print(f"shape={tuple(output.shape)} dtype={output.dtype} sha256={digest}")
    print(f"mean={output.float().mean().item()} max_abs={output.float().abs().max().item()}")


if __name__ == "__main__":
    main()
