import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils

from .model_config import SenseNovaModelConfig


CONFIG_SHA256 = "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b"
MODEL_REPO = "sensenova/SenseNova-U1.5-8B-MoT"
MODEL_REVISION = "19bc874ef6ffc97fda9837b40fc1d1301806158a"
LEGACY_MODEL_REVISION = "1f6ec60423d29939dde4202fd82ae340b144e280"
FINAL_MODEL_REVISIONS = (MODEL_REVISION, LEGACY_MODEL_REVISION)
SFT_MODEL_REVISION = "661834c5b5aee0f89958353511d6ac0ccaacb646"
SFT_MODEL_REPO = "sensenova/SenseNova-U1.5-8B-MoT-SFT"
MODEL_FORMAT = "sensenova-u1.5-mot"
CHECKPOINT_CONTRACT_SHA256 = "947b59973a054a3efa29dead737baae0bfbbacbe9158500a391a8de27a11f53c"
CHECKPOINT_PROFILES = {
    "final": {
        "variant": "final",
        "source_repo": MODEL_REPO,
        "source_revision": MODEL_REVISION,
    },
    "final_legacy": {
        "variant": "final",
        "source_repo": MODEL_REPO,
        "source_revision": LEGACY_MODEL_REVISION,
    },
    "sft": {
        "variant": "sft",
        "source_repo": SFT_MODEL_REPO,
        "source_revision": SFT_MODEL_REVISION,
    },
}
TOKENIZER_ASSET_SHA256 = {
    "config.json": CONFIG_SHA256,
    "tokenizer_config.json": "7433b95cec590c7d687259e81bca1bc4630ff39773dbf7f30f7df27a99748077",
    "special_tokens_map.json": "529306ff26be5cf190b4d96781e63c7dccd03ef0a39f87c0f1289d2d5a67a02f",
    "added_tokens.json": "d0ff3acec259fabfafc1ffa67638aeaf58203e5e604648fb44f072e4efe040c4",
    "vocab.json": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
    "merges.txt": "455e0caaa06abffc663e9282dfe71dde07fd1991eaf24146bf08793c4dba4497",
}


def _validate_metadata(metadata):
    if metadata.get("format") != MODEL_FORMAT:
        raise ValueError("SenseNova-U1.5 checkpoint format does not match this node version")
    if metadata.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("SenseNova-U1.5 config digest does not match this node version")
    source_repo = metadata.get("source_repo")
    source_revision = metadata.get("source_revision")
    matching_profiles = [
        (profile, contract)
        for profile, contract in CHECKPOINT_PROFILES.items()
        if source_repo == contract["source_repo"]
    ]
    for profile, contract in matching_profiles:
        if source_revision == contract["source_revision"]:
            return profile
    if matching_profiles:
        variant = matching_profiles[0][1]["variant"]
        raise ValueError(
            f"SenseNova-U1.5 {variant.upper()} model revision does not match this node version"
        )
    raise ValueError("SenseNova-U1.5 checkpoint is not a supported Final or SFT model")


@lru_cache(maxsize=1)
def _checkpoint_contract_data():
    path = Path(__file__).with_name("checkpoint_contract.json")
    raw = path.read_bytes() if path.is_file() else b""
    canonical_raw = raw.replace(b"\r\n", b"\n")
    digest = hashlib.sha256(canonical_raw).hexdigest() if canonical_raw else None
    if digest != CHECKPOINT_CONTRACT_SHA256:
        raise ValueError(
            "SenseNova-U1.5 bundled checkpoint contract is missing or has been modified: "
            f"{path}"
        )
    data = json.loads(raw)
    if data.get("format_version") != 1:
        raise ValueError("SenseNova-U1.5 checkpoint contract version is not supported")
    if data.get("model_format") != MODEL_FORMAT or data.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("SenseNova-U1.5 checkpoint contract metadata does not match this node")

    variants = data.get("variants")
    tensors = data.get("tensors")
    if not isinstance(variants, dict) or set(variants) != set(CHECKPOINT_PROFILES):
        raise ValueError("SenseNova-U1.5 checkpoint contract variants are invalid")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("SenseNova-U1.5 checkpoint contract has no tensors")
    for profile, expected in CHECKPOINT_PROFILES.items():
        actual = variants[profile]
        if actual.get("source_repo") != expected["source_repo"]:
            raise ValueError(f"SenseNova-U1.5 {profile.upper()} contract repository is invalid")
        if actual.get("source_revision") != expected["source_revision"]:
            raise ValueError(f"SenseNova-U1.5 {profile.upper()} contract revision is invalid")
        if actual.get("tensor_count") != len(tensors):
            raise ValueError(f"SenseNova-U1.5 {profile.upper()} contract tensor count is invalid")
    return data


@lru_cache(maxsize=len(CHECKPOINT_PROFILES))
def _checkpoint_contract(profile):
    if profile not in CHECKPOINT_PROFILES:
        raise ValueError(f"unsupported SenseNova-U1.5 checkpoint profile: {profile}")
    contract = {}
    for name, tensor in _checkpoint_contract_data()["tensors"].items():
        shape = tensor.get("shape")
        dtypes = tensor.get("dtypes")
        if not isinstance(shape, list) or not shape or not all(isinstance(value, int) for value in shape):
            raise ValueError(f"SenseNova-U1.5 checkpoint contract shape is invalid for {name}")
        if not isinstance(dtypes, dict) or dtypes.get(profile) not in {"BF16", "F32"}:
            raise ValueError(f"SenseNova-U1.5 checkpoint contract dtype is invalid for {name}")
        contract[name] = (tuple(shape), dtypes[profile])
    return contract


def _validate_checkpoint_header(checkpoint, model_path=None):
    profile = _validate_metadata(checkpoint.metadata() or {})
    contract = _checkpoint_contract(profile)
    actual_keys = set(checkpoint.keys())
    expected_keys = set(contract)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:5]
        unexpected = sorted(actual_keys - expected_keys)[:5]
        location = f", model={Path(model_path).resolve()}" if model_path is not None else ""
        raise ValueError(
            "SenseNova-U1.5 checkpoint key mismatch: "
            f"missing={missing}, unexpected={unexpected}{location}, loader={Path(__file__).resolve()}. "
            "Restart ComfyUI after updating the node and remove duplicate or stale node copies."
        )
    for name, (shape, expected_dtype) in contract.items():
        tensor = checkpoint.get_slice(name)
        actual_shape = tuple(tensor.get_shape())
        if actual_shape != shape:
            raise ValueError(f"SenseNova-U1.5 checkpoint shape mismatch for {name}: {actual_shape} != {shape}")
        if tensor.get_dtype() != expected_dtype:
            raise ValueError(
                f"SenseNova-U1.5 checkpoint dtype mismatch for {name}: {tensor.get_dtype()} != {expected_dtype}"
            )
    return expected_keys, profile


def _validate_tokenizer_assets():
    asset_dir = Path(__file__).resolve().parent / "tokenizer"
    for name, expected in TOKENIZER_ASSET_SHA256.items():
        path = asset_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if digest != expected:
            raise ValueError(f"SenseNova-U1.5 tokenizer asset digest mismatch: {name}")


def load_sensenova_model(model_path, dtype=torch.bfloat16, disable_dynamic=False):
    model_path = Path(model_path)
    if model_path.suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError("SenseNova-U1.5 loader accepts safetensors files only")
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        expected_keys, profile = _validate_checkpoint_header(checkpoint, model_path)
    expected_size = _checkpoint_contract_data()["variants"][profile]["file_size"]
    actual_size = model_path.stat().st_size
    if actual_size != expected_size:
        variant = CHECKPOINT_PROFILES[profile]["variant"]
        raise ValueError(
            f"SenseNova-U1.5 {variant.upper()} file size mismatch: {actual_size} != {expected_size}"
        )
    state_dict, metadata = comfy.utils.load_torch_file(str(model_path), return_metadata=True)
    loaded_profile = _validate_metadata(metadata)
    if loaded_profile != profile:
        raise ValueError("SenseNova-U1.5 checkpoint metadata changed while loading")
    if set(state_dict) != expected_keys:
        raise ValueError("SenseNova-U1.5 loaded state dict does not match the validated header")

    load_device = comfy.model_management.get_torch_device()
    model_config = SenseNovaModelConfig({})
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(dtype, manual_cast_dtype, device=load_device)

    parameters = comfy.utils.calculate_parameters(state_dict)
    initial_load_device = comfy.model_management.unet_inital_load_device(parameters, dtype)
    model = model_config.get_model(state_dict, device=initial_load_device)
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
    )
    model.load_model_weights(state_dict, assign=patcher.is_dynamic())
    if state_dict:
        raise ValueError(f"SenseNova-U1.5 unused checkpoint keys after load: {sorted(state_dict)[:5]}")
    patcher.cached_patcher_init = (load_sensenova_model, (model_path, dtype))
    patcher.set_attachments(
        "sensenova_checkpoint",
        {
            "variant": CHECKPOINT_PROFILES[profile]["variant"],
            "profile": profile,
            "source_repo": CHECKPOINT_PROFILES[profile]["source_repo"],
            "source_revision": CHECKPOINT_PROFILES[profile]["source_revision"],
        },
    )
    return patcher


def load_sensenova_clip():
    _validate_tokenizer_assets()
    target = SenseNovaModelConfig({}).clip_target()
    return comfy.sd.CLIP(target, parameters=0, state_dict=[])


def load_pixel_vae():
    return comfy.sd.VAE(sd={"pixel_space_vae": torch.tensor(1.0)})
