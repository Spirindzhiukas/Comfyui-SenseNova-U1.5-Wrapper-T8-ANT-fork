import hashlib
import json
import os
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
# Bundled tokenizer directory. Module level so tests can point it at a copy
# with foreign line endings.
TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"
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


# ---------------------------------------------------------------------------
# SenseNova fork addendum: optional ConvRot quantized checkpoint support.
#
# Everything below this marker only runs for checkpoints that carry per-layer
# `*.comfy_quant` sidecars (int8-ConvRot, ConvRot W4A4 and W4A8 conversions of
# an official Final/SFT file, see tools/convert_sensenova_int4_convrot.py).
# The official bf16 files have no such key, so they keep taking the strict
# upstream JSON contract untouched, which keeps future `git merge` of T8mars
# changes on this file small and local.
# ---------------------------------------------------------------------------
QUANT_METADATA_SUFFIX = ".comfy_quant"
QUANT_FORMATS = ("int8_tensorwise", "convrot_w4a4", "asym_w4a8_int8")
# Rank-2 linear weights the converter quantizes. Norms, embeddings and the LM
# head always stay in the floating dtype of the source checkpoint.
QUANT_EXCLUDED_SUBSTRINGS = ("norm", "embed_tokens", "lm_head")
# comfy-kitchen writes the W4A8 group scale as native fp8; files exported by
# older tooling keep it as raw bytes, which ComfyUI core views back to fp8.
QUANT_GROUP_SCALE_DTYPES = ("F8_E4M3", "U8")
# Sidecars a quantized layer must carry (required) or may carry (optional),
# as (shape kind, storage dtype) relative to the layer stem. The W4A8 channel
# scale and codebook depend on the conversion recipe, so they are only
# validated when the file actually has them.
QUANT_REQUIRED_SIDECARS = {
    "int8_tensorwise": {".weight_scale": ("rows_x_1", "F32")},
    "convrot_w4a4": {".weight_scale": ("rows", "F32")},
    "asym_w4a8_int8": {".weight_s_rel": ("rows_x_groups", QUANT_GROUP_SCALE_DTYPES)},
}
QUANT_OPTIONAL_SIDECARS = {
    "int8_tensorwise": {},
    "convrot_w4a4": {},
    "asym_w4a8_int8": {
        ".weight_s_channel": ("rows", "F32"),
        ".weight_codebook": ((16,), "F32"),
    },
}
# Packed weight shapes: 4-bit formats store two nibbles per int8 element.
QUANT_PACKED_WEIGHT = {"convrot_w4a4": 2, "asym_w4a8_int8": 2}
QUANT_GROUP_SIZE = 16


def _quant_support_enabled():
    """Allow users to switch the quantized-checkpoint path off completely."""
    return not os.environ.get("SENSENOVA_NO_QUANT")


def _is_quant_candidate(name, shape):
    """Rank-2 linear weights that the converter's SenseNova policy quantizes."""
    return (
        len(shape) == 2
        and name.endswith(".weight")
        and not any(token in name for token in QUANT_EXCLUDED_SUBSTRINGS)
    )


def _read_quant_formats(checkpoint):
    """Per-layer format map from every `comfy_quant` payload; empty means bf16."""
    formats = {}
    for key in checkpoint.keys():
        if not key.endswith(QUANT_METADATA_SUFFIX):
            continue
        try:
            payload = checkpoint.get_tensor(key)
            config = json.loads(bytes(payload.numpy()).decode("utf-8"))
            formats[key[: -len(QUANT_METADATA_SUFFIX)]] = config.get("format")
        except Exception:
            # A checkpoint we cannot parse here is handed back to the strict
            # upstream contract, which reports the key mismatch with the full
            # model and loader path.
            return {}
    return formats


def _quant_sidecar_shape(kind, rows, columns):
    if kind == "rows":
        return (rows,)
    if kind == "rows_x_1":
        return (rows, 1)
    if kind == "rows_x_groups":
        return (rows, columns // QUANT_GROUP_SIZE)
    if isinstance(kind, tuple):  # a literal shape from the table above
        return kind
    raise ValueError(f"unknown SenseNova quantized sidecar shape: {kind!r}")


def _quant_checkpoint_contract(profile, quant_formats):
    """Extend the bundled bf16 contract with the quantized per-layer layout.

    Returns ``(contract, optional_keys)``. The base key set, shapes and
    floating dtypes stay the ones pinned in `checkpoint_contract.json`, so a
    new upstream contract revision automatically carries the quant path too.
    Only the layers that actually carry a `comfy_quant` payload are rewritten,
    which also accepts partially quantized or hand-mixed checkpoints, and
    recipe-dependent sidecars land in ``optional_keys``: validated when the
    file has them, not required.
    """
    unsupported = sorted({fmt for fmt in quant_formats.values() if fmt not in QUANT_FORMATS})
    if unsupported:
        raise ValueError(
            f"SenseNova-U1.5 checkpoint uses unsupported quantization format(s) "
            f"{unsupported} (supported: {', '.join(QUANT_FORMATS)})"
        )
    quantized = {stem for stem, quant_format in quant_formats.items() if quant_format in QUANT_FORMATS}
    contract = {}
    optional_keys = set()
    for name, (shape, dtype) in _checkpoint_contract(profile).items():
        stem = name[: -len(".weight")]
        if not _is_quant_candidate(name, shape) or stem not in quantized:
            contract[name] = (shape, dtype)
            continue
        quant_format = quant_formats[stem]
        rows, columns = shape[0], shape[1]
        packed = QUANT_PACKED_WEIGHT.get(quant_format)
        contract[name] = ((rows, columns // packed) if packed else tuple(shape), "I8")
        for suffix, (kind, sidecar_dtype) in QUANT_REQUIRED_SIDECARS[quant_format].items():
            contract[stem + suffix] = (_quant_sidecar_shape(kind, rows, columns), sidecar_dtype)
        for suffix, (kind, sidecar_dtype) in QUANT_OPTIONAL_SIDECARS[quant_format].items():
            contract[stem + suffix] = (_quant_sidecar_shape(kind, rows, columns), sidecar_dtype)
            optional_keys.add(stem + suffix)
        # The JSON payload length varies per layer, so only its dtype is pinned.
        contract[stem + QUANT_METADATA_SUFFIX] = (None, "U8")
    return contract, optional_keys


def _validate_quant_header(checkpoint, profile, quant_formats, model_path=None):
    contract, optional_keys = _quant_checkpoint_contract(profile, quant_formats)
    actual_keys = set(checkpoint.keys())
    expected_keys = (set(contract) - optional_keys) | (actual_keys & optional_keys)
    # Conversion recipes may add their own per-layer tensors next to a
    # quantized weight (for example a W4A8 correction term). They are inert for
    # the runtime, so an unknown `weight_*` sidecar is accepted and the shapes
    # and dtypes that actually steer inference stay strict.
    quantized_stems = tuple(quant_formats)
    expected_keys |= {
        name
        for name in actual_keys - expected_keys
        if any(name.startswith(f"{stem}.weight_") for stem in quantized_stems)
    }
    location = f", model={Path(model_path).resolve()}" if model_path is not None else ""
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:5]
        unexpected = sorted(actual_keys - expected_keys)[:5]
        raise ValueError(
            "SenseNova-U1.5 quantized checkpoint key mismatch: "
            f"formats={sorted({value for value in quant_formats.values() if value})}, "
            f"contract_keys={len(expected_keys)}, file_keys={len(actual_keys)}, "
            f"missing={missing}, unexpected={unexpected}{location}. "
            "Re-run tools/convert_sensenova_int4_convrot.py and "
            "tools/inject_sensenova_metadata.py on the official single file."
        )
    for name, (shape, expected_dtype) in contract.items():
        if name not in actual_keys:
            continue  # an optional sidecar the conversion recipe did not emit
        tensor = checkpoint.get_slice(name)
        if shape is not None:
            actual_shape = tuple(tensor.get_shape())
            if actual_shape != shape:
                raise ValueError(
                    f"SenseNova-U1.5 quantized checkpoint shape mismatch for {name}: "
                    f"{actual_shape} != {shape}"
                )
        actual_dtype = tensor.get_dtype()
        accepted = expected_dtype if isinstance(expected_dtype, tuple) else (expected_dtype,)
        if actual_dtype not in accepted:
            raise ValueError(
                f"SenseNova-U1.5 quantized checkpoint dtype mismatch for {name}: "
                f"{actual_dtype} != {'/'.join(accepted)}"
            )
    return expected_keys, profile


def _validate_checkpoint_header(checkpoint, model_path=None, quant_formats=None):
    profile = _validate_metadata(checkpoint.metadata() or {})
    # >>> SenseNova fork: quantized checkpoints branch off here, the bf16 path
    # below is unchanged from upstream T8mars. <<<
    if quant_formats is None:
        quant_formats = _read_quant_formats(checkpoint) if _quant_support_enabled() else {}
    if quant_formats:
        return _validate_quant_header(checkpoint, profile, quant_formats, model_path)
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


def _tokenizer_digest_kind(raw, expected):
    """Classify a bundled tokenizer asset against its pinned digest.

    Returns ``"raw"`` for a byte-identical file, ``"normalized"`` for a file
    that only differs by CRLF line endings, and ``None`` otherwise. Windows
    clones made with ``core.autocrlf=true`` rewrite the packaged text files,
    which used to abort the load even though the tokenizer content is intact.
    """
    if hashlib.sha256(raw).hexdigest() == expected:
        return "raw"
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected:
        return "normalized"
    return None


def _validate_tokenizer_assets():
    asset_dir = TOKENIZER_DIR
    line_endings = []
    for name, expected in TOKENIZER_ASSET_SHA256.items():
        path = asset_dir / name
        if not path.is_file():
            raise ValueError(f"SenseNova-U1.5 tokenizer asset missing: {name}")
        raw = path.read_bytes()
        kind = _tokenizer_digest_kind(raw, expected)
        if kind == "raw":
            continue
        if kind == "normalized":
            # Same content, CRLF checkout. transformers reads these files with
            # universal newlines / JSON parsing, so they still work.
            line_endings.append(name)
            continue
        # A fully unexpected digest is a real integrity problem, but it is not
        # proof of a broken tokenizer, so it warns with both hashes instead of
        # refusing to load; compare the value with `sha256sum` by hand.
        digest = hashlib.sha256(raw).hexdigest()
        canonical = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        print(
            f"[SenseNova-U1.5] WARNING: tokenizer asset digest mismatch: {name} "
            f"expected={expected} got={digest} lf_normalized={canonical}. "
            "Continuing with the packaged file. If this was not intentional, re-clone "
            "the node with `git config --global core.autocrlf false`."
        )
    if line_endings:
        print(
            "[SenseNova-U1.5] NOTE: tokenizer assets were checked out with CRLF line "
            f"endings and matched after normalising: {', '.join(sorted(line_endings))}. "
            "Loading continues normally; use `git config --global core.autocrlf false` "
            "and re-clone to silence this note."
        )


def load_sensenova_model(model_path, dtype=torch.bfloat16, disable_dynamic=False):
    model_path = Path(model_path)
    if model_path.suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError("SenseNova-U1.5 loader accepts safetensors files only")
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        quant_formats = _read_quant_formats(checkpoint) if _quant_support_enabled() else {}
        expected_keys, profile = _validate_checkpoint_header(checkpoint, model_path, quant_formats)
    # >>> SenseNova fork: the pinned file size belongs to the bf16 contract;
    # converted quantized files legitimately differ in size. <<<
    if not quant_formats:
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
    attachment = {
        "variant": CHECKPOINT_PROFILES[profile]["variant"],
        "profile": profile,
        "source_repo": CHECKPOINT_PROFILES[profile]["source_repo"],
        "source_revision": CHECKPOINT_PROFILES[profile]["source_revision"],
    }
    if quant_formats:
        # >>> SenseNova fork: expose the conversion recipe to downstream nodes. <<<
        attachment["quant_formats"] = sorted({value for value in quant_formats.values() if value})
    patcher.set_attachments("sensenova_checkpoint", attachment)
    return patcher


def load_sensenova_clip():
    _validate_tokenizer_assets()
    target = SenseNovaModelConfig({}).clip_target()
    return comfy.sd.CLIP(target, parameters=0, state_dict=[])


def load_pixel_vae():
    return comfy.sd.VAE(sd={"pixel_space_vae": torch.tensor(1.0)})
