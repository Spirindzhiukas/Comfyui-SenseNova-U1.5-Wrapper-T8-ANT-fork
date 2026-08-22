import hashlib
from pathlib import Path

import torch

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths

from .loader import MODEL_REPO, MODEL_REVISION


LORA_REPO = "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"
LORA_REVISION = "e909f4636d119d65fe4cba8770c19daff2ac102e"
LORA_SOURCE_SHA256 = "3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708"
LORA_COMFY_SIZE = 814881652
LORA_COMFY_SHA256 = "dd5320f06986688dd41b0a4a2cb6ebd0036308f8a8a2d0c349ca22875a805aa1"
LORA_TENSOR_COUNT = 882
LORA_MODULE_COUNT = 294
LORA_RANK = 128
LORA_ALPHA = 8.0
LORA_PREFIX = "diffusion_model."
LORA_SUFFIXES = (".alpha", ".lora_down.weight", ".lora_up.weight")
_VERIFIED_LORA_FILES = set()


def _validate_final_model(model):
    checkpoint = model.get_attachment("sensenova_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("SenseNova U1.5 8-step LoRA requires a model from the SenseNova U1.5 loader")
    if (
        checkpoint.get("variant") != "final"
        or checkpoint.get("source_repo") != MODEL_REPO
        or checkpoint.get("source_revision") != MODEL_REVISION
    ):
        raise ValueError(
            "SenseNova U1.5 8-step LoRA requires its fixed official Final checkpoint; SFT uses 50 steps"
        )
    existing = model.get_attachment("sensenova_lora")
    if isinstance(existing, dict) and existing.get("type") == "8step":
        raise ValueError("SenseNova U1.5 8-step LoRA is already applied; do not stack it twice")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_lora_file(path):
    path = Path(path)
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cache_key in _VERIFIED_LORA_FILES:
        return
    if stat.st_size != LORA_COMFY_SIZE or _sha256_file(path) != LORA_COMFY_SHA256:
        raise ValueError("LoRA file size/SHA256 does not match the verified SenseNova U1.5 8-step conversion")
    _VERIFIED_LORA_FILES.add(cache_key)


def _validate_lora_metadata(metadata):
    expected = {
        "tensor_kind": "neo_hf_lora",
        "comfyui_format": "model_lora",
        "comfyui_key_prefix": LORA_PREFIX,
        "source_repo": LORA_REPO,
        "source_revision": LORA_REVISION,
        "source_sha256": LORA_SOURCE_SHA256,
        "conversion": "raw-data-key-prefix-v1",
    }
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("LoRA is not the verified official SenseNova U1.5 8-step ComfyUI conversion")


def _validate_lora_tensors(lora):
    if len(lora) != LORA_TENSOR_COUNT:
        raise ValueError(f"SenseNova U1.5 8-step LoRA tensor count mismatch: {len(lora)}")
    grouped = {}
    for name, tensor in lora.items():
        if not name.startswith(LORA_PREFIX):
            raise ValueError(f"SenseNova U1.5 8-step LoRA key has no ComfyUI prefix: {name}")
        suffix = next((value for value in LORA_SUFFIXES if name.endswith(value)), None)
        if suffix is None:
            raise ValueError(f"Unexpected SenseNova U1.5 8-step LoRA tensor: {name}")
        grouped.setdefault(name[: -len(suffix)], {})[suffix] = tensor
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"SenseNova U1.5 8-step LoRA tensor must be BF16: {name}")

    if len(grouped) != LORA_MODULE_COUNT:
        raise ValueError(f"SenseNova U1.5 8-step LoRA module count mismatch: {len(grouped)}")
    for base, tensors in grouped.items():
        if set(tensors) != set(LORA_SUFFIXES):
            raise ValueError(f"Incomplete SenseNova U1.5 8-step LoRA module: {base}")
        alpha = tensors[".alpha"]
        down = tensors[".lora_down.weight"]
        up = tensors[".lora_up.weight"]
        if alpha.ndim != 0 or float(alpha.float()) != LORA_ALPHA:
            raise ValueError(f"SenseNova U1.5 8-step LoRA alpha mismatch: {base}")
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != LORA_RANK or up.shape[1] != LORA_RANK:
            raise ValueError(f"SenseNova U1.5 8-step LoRA rank mismatch: {base}")


def apply_eight_step_lora(model, lora_name, strength_model):
    _validate_final_model(model)
    if strength_model == 0:
        return model

    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    _validate_lora_file(lora_path)
    lora, metadata = comfy.utils.load_torch_file(
        lora_path,
        safe_load=True,
        return_metadata=True,
    )
    _validate_lora_metadata(metadata)
    _validate_lora_tensors(lora)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    converted = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(converted, key_map)
    if len(loaded) != LORA_MODULE_COUNT:
        raise ValueError(
            f"SenseNova U1.5 8-step LoRA matched {len(loaded)} of {LORA_MODULE_COUNT} model modules"
        )

    patched = model.clone()
    applied = set(patched.add_patches(loaded, strength_model))
    if applied != set(loaded):
        raise ValueError(
            f"SenseNova U1.5 8-step LoRA applied {len(applied)} of {len(loaded)} model patches"
        )
    patched.set_attachments("lora_metadata", metadata)
    patched.set_attachments(
        "sensenova_lora",
        {"type": "8step", "source_repo": LORA_REPO, "source_revision": LORA_REVISION},
    )
    return patched
