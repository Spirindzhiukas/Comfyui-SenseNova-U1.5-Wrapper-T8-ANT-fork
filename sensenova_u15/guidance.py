import math

import torch
import torch.nn.functional as F


CFG_NORM_MODES = ("none", "global", "channel")
GUIDANCE_PATCH_SIZE = 32


def edit_guidance(positive, image_condition, negative, cfg, img_cfg):
    return negative + cfg * (positive - image_condition) + img_cfg * (image_condition - negative)


def rescale_guidance(guided, positive, mode="none", patch_size=GUIDANCE_PATCH_SIZE):
    """Match SenseNova's optional CFG norm on pixel-space flow predictions.

    The upstream model applies ``global`` over the whole prediction and
    ``channel`` over each generation token. SenseNova-U1.5 uses 32x32 pixel
    tokens, so the four-dimensional ComfyUI prediction is temporarily grouped
    into the same token layout for channel mode.
    """
    if mode not in CFG_NORM_MODES:
        raise ValueError(f"unsupported SenseNova CFG norm mode: {mode}")
    if mode == "none":
        return guided
    if guided.shape != positive.shape:
        raise ValueError("guided and positive predictions must have the same shape")
    if guided.ndim < 2:
        raise ValueError("SenseNova CFG norm requires a batched prediction")

    if mode == "global":
        dims = tuple(range(1, guided.ndim))
        positive_norm = torch.linalg.vector_norm(positive.float(), dim=dims, keepdim=True)
        guided_norm = torch.linalg.vector_norm(guided.float(), dim=dims, keepdim=True)
        scale = (positive_norm / (guided_norm + 1e-8)).clamp(min=0.0, max=1.0)
        return guided * scale.to(guided.dtype)

    if guided.ndim != 4:
        positive_norm = torch.linalg.vector_norm(positive.float(), dim=-1, keepdim=True)
        guided_norm = torch.linalg.vector_norm(guided.float(), dim=-1, keepdim=True)
        scale = (positive_norm / (guided_norm + 1e-8)).clamp(min=0.0, max=1.0)
        return guided * scale.to(guided.dtype)

    original_height, original_width = guided.shape[-2:]
    padded_height = math.ceil(original_height / patch_size) * patch_size
    padded_width = math.ceil(original_width / patch_size) * patch_size
    pad = (0, padded_width - original_width, 0, padded_height - original_height)
    guided_padded = F.pad(guided, pad)
    positive_padded = F.pad(positive, pad)
    batch, channels, _, _ = guided_padded.shape
    grid_height = padded_height // patch_size
    grid_width = padded_width // patch_size

    def to_tokens(value):
        return (
            value.reshape(batch, channels, grid_height, patch_size, grid_width, patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, grid_height, grid_width, -1)
        )

    guided_tokens = to_tokens(guided_padded)
    positive_tokens = to_tokens(positive_padded)
    positive_norm = torch.linalg.vector_norm(positive_tokens.float(), dim=-1, keepdim=True)
    guided_norm = torch.linalg.vector_norm(guided_tokens.float(), dim=-1, keepdim=True)
    scale = (positive_norm / (guided_norm + 1e-8)).clamp(min=0.0, max=1.0).to(guided.dtype)
    scale = scale.reshape(batch, grid_height, grid_width, 1, 1, 1).permute(0, 3, 1, 4, 2, 5)
    scale = scale.expand(batch, channels, grid_height, patch_size, grid_width, patch_size)
    scale = scale.reshape(batch, channels, padded_height, padded_width)
    return (guided_padded * scale)[..., :original_height, :original_width]


def rescale_denoised_guidance(guided, positive, latent, sigma, mode="none"):
    """Apply CFG norm in velocity space and convert back to Comfy denoised space."""
    if mode == "none":
        return guided
    sigma = torch.as_tensor(sigma, device=guided.device, dtype=guided.dtype)
    if sigma.numel() == 1:
        sigma = sigma.reshape((1,) + (1,) * (guided.ndim - 1))
    else:
        sigma = sigma.reshape((sigma.shape[0],) + (1,) * (guided.ndim - 1))
    safe_sigma = sigma.clamp_min(torch.finfo(guided.dtype).eps)
    guided_velocity = (latent - guided) / safe_sigma
    positive_velocity = (latent - positive) / safe_sigma
    guided_velocity = rescale_guidance(guided_velocity, positive_velocity, mode=mode)
    return latent - sigma * guided_velocity


def build_structured_edit_prompt(instruction, image_roles="", preserve="", avoid=""):
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("SenseNova edit instruction cannot be empty")

    sections = [f"【主要修改】\n{instruction}"]
    for title, value in (
        ("参考图职责", image_roles),
        ("必须保持", preserve),
        ("禁止出现", avoid),
    ):
        value = value.strip()
        if value:
            sections.append(f"【{title}】\n{value}")
    sections.append("【执行要求】\n只修改上面明确指定的内容；未要求修改的区域保持原图一致。")
    return "\n\n".join(sections)
