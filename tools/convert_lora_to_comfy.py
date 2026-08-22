from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path


CHUNK_SIZE = 16 * 1024 * 1024
COMFY_PREFIX = "diffusion_model."
LORA_SUFFIXES = (".alpha", ".lora_down.weight", ".lora_up.weight")
OFFICIAL_REPO = "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"
OFFICIAL_REVISION = "e909f4636d119d65fe4cba8770c19daff2ac102e"
OFFICIAL_FILE_SIZE = 814867236
OFFICIAL_SHA256 = "3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708"
OFFICIAL_MODULE_COUNT = 294
OFFICIAL_TENSOR_COUNT = 882
OFFICIAL_RANK = 128
OFFICIAL_ALPHA = 8.0


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path):
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        file_size = path.stat().st_size
        if header_size == 0 or header_size > file_size - 8:
            raise ValueError("invalid safetensors header size")
        header_bytes = handle.read(header_size)
    header = json.loads(header_bytes, object_pairs_hook=_reject_duplicate_keys)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("safetensors metadata must be an object")
    if metadata.get("tensor_kind") != "neo_hf_lora":
        raise ValueError("expected the official SenseNova neo_hf_lora tensor format")
    return header, metadata, 8 + header_size, file_size - 8 - header_size


def _validate_lora(header: dict, data_size: int):
    if not header:
        raise ValueError("LoRA contains no tensors")
    if any(name.startswith(COMFY_PREFIX) for name in header):
        raise ValueError("LoRA keys already use the ComfyUI diffusion_model prefix")

    grouped = {}
    intervals = []
    for name, info in header.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            raise ValueError("invalid safetensors tensor entry")
        suffix = next((item for item in LORA_SUFFIXES if name.endswith(item)), None)
        if suffix is None:
            raise ValueError(f"unexpected SenseNova LoRA tensor name: {name}")
        grouped.setdefault(name[: -len(suffix)], set()).add(suffix)

        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if dtype != "BF16" or not isinstance(shape, list):
            raise ValueError(f"unexpected dtype or shape for {name}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid data offsets for {name}")
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError(f"invalid data offsets for {name}")
        if end > data_size:
            raise ValueError(f"data offset outside file for {name}")
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise ValueError(f"invalid tensor shape for {name}")
            elements *= dimension
        if end - start != elements * 2:
            raise ValueError(f"BF16 shape/data size mismatch for {name}")
        intervals.append((start, end, name))

    expected_suffixes = set(LORA_SUFFIXES)
    incomplete = [base for base, suffixes in grouped.items() if suffixes != expected_suffixes]
    if incomplete:
        raise ValueError(f"incomplete LoRA module: {incomplete[0]}")

    intervals.sort()
    previous_end = 0
    for start, end, name in intervals:
        if start != previous_end:
            raise ValueError(f"gap or overlap before {name}")
        previous_end = end
    if previous_end != data_size:
        raise ValueError("unreferenced data tail in LoRA")
    return len(grouped)


def _validate_official_layout(source_path: Path, header: dict, data_start: int, module_count: int) -> None:
    if module_count != OFFICIAL_MODULE_COUNT or len(header) != OFFICIAL_TENSOR_COUNT:
        raise ValueError(
            f"official LoRA layout mismatch: modules={module_count}, tensors={len(header)}"
        )
    alpha_offsets = []
    bases = sorted(name[: -len(".alpha")] for name in header if name.endswith(".alpha"))
    for base in bases:
        alpha = header[f"{base}.alpha"]
        down = header[f"{base}.lora_down.weight"]
        up = header[f"{base}.lora_up.weight"]
        if alpha["shape"] != []:
            raise ValueError(f"official LoRA alpha is not scalar: {base}")
        if len(down["shape"]) != 2 or len(up["shape"]) != 2:
            raise ValueError(f"official LoRA matrix rank mismatch: {base}")
        if down["shape"][0] != OFFICIAL_RANK or up["shape"][1] != OFFICIAL_RANK:
            raise ValueError(f"official LoRA rank is not {OFFICIAL_RANK}: {base}")
        alpha_offsets.append((base, alpha["data_offsets"][0]))

    with source_path.open("rb") as handle:
        for base, offset in alpha_offsets:
            handle.seek(data_start + offset)
            raw = handle.read(2)
            if len(raw) != 2:
                raise EOFError(f"official LoRA alpha ended early: {base}")
            bits = struct.unpack("<H", raw)[0]
            value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            if value != OFFICIAL_ALPHA:
                raise ValueError(f"official LoRA alpha is not {OFFICIAL_ALPHA}: {base}")


def _copy_exact(source, target, size: int, digest) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(CHUNK_SIZE, remaining))
        if not chunk:
            raise EOFError("source LoRA data ended early")
        target.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def convert_lora(
    source_path: Path,
    output_path: Path,
    source_repo: str,
    revision: str,
    *,
    verify_official: bool = True,
):
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if sys.maxsize <= 2**32:
        raise RuntimeError("64-bit Python is required")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".safetensors" or output_path.suffix.lower() != ".safetensors":
        raise ValueError("source and output must be .safetensors files")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if verify_official and (source_repo != OFFICIAL_REPO or revision != OFFICIAL_REVISION):
        raise ValueError("source repo/revision does not match the pinned official U1.5 8-step LoRA")

    partial_path = Path(f"{output_path}.partial")
    manifest_path = Path(f"{output_path}.manifest.json")
    manifest_partial = Path(f"{manifest_path}.partial")
    for path in (partial_path, manifest_path, manifest_partial):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {path}")

    header, metadata, data_start, data_size = _read_header(source_path)
    module_count = _validate_lora(header, data_size)
    source_sha256 = _sha256_file(source_path)
    if verify_official:
        if source_path.stat().st_size != OFFICIAL_FILE_SIZE or source_sha256 != OFFICIAL_SHA256:
            raise ValueError("source size/SHA256 does not match the pinned official U1.5 8-step LoRA")
        _validate_official_layout(source_path, header, data_start, module_count)

    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "comfyui_format": "model_lora",
            "comfyui_key_prefix": COMFY_PREFIX,
            "source_repo": source_repo,
            "source_revision": revision,
            "source_sha256": source_sha256,
            "conversion": "raw-data-key-prefix-v1",
        }
    )
    output_header = {"__metadata__": output_metadata}
    output_header.update({f"{COMFY_PREFIX}{name}": info for name, info in header.items()})
    output_header_bytes = json.dumps(output_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    output_header_bytes += b" " * (-len(output_header_bytes) % 8)

    expected_size = 8 + len(output_header_bytes) + data_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    free_space = shutil.disk_usage(output_path.parent).free
    safety_margin = max(256 * 1024**2, expected_size // 20)
    if free_space < expected_size + safety_margin:
        raise OSError(
            f"insufficient free space: need {expected_size + safety_margin} bytes, have {free_space} bytes"
        )

    source_data_digest = hashlib.sha256()
    with source_path.open("rb") as source, partial_path.open("xb") as target:
        source.seek(data_start)
        target.write(struct.pack("<Q", len(output_header_bytes)))
        target.write(output_header_bytes)
        _copy_exact(source, target, data_size, source_data_digest)
        if source.read(1):
            raise ValueError("unexpected source data after final tensor")
        target.flush()
        os.fsync(target.fileno())

    output_header_check, output_metadata_check, output_data_start, output_data_size = _read_header(partial_path)
    if output_metadata_check != output_metadata or output_header_check != {
        f"{COMFY_PREFIX}{name}": info for name, info in header.items()
    }:
        raise ValueError("converted LoRA header verification failed")
    if output_data_size != data_size or partial_path.stat().st_size != expected_size:
        raise ValueError("converted LoRA size verification failed")

    output_data_digest = hashlib.sha256()
    with partial_path.open("rb") as handle:
        handle.seek(output_data_start)
        while chunk := handle.read(CHUNK_SIZE):
            output_data_digest.update(chunk)
    if output_data_digest.hexdigest() != source_data_digest.hexdigest():
        raise ValueError("source/output LoRA tensor data hash mismatch")

    output_sha256 = _sha256_file(partial_path)
    manifest = {
        "format_version": 1,
        "source": {
            "repo": source_repo,
            "revision": revision,
            "file": source_path.name,
            "size": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "conversion": {
            "type": "raw-data-key-prefix-v1",
            "key_prefix": COMFY_PREFIX,
            "tensor_count": len(header),
            "module_count": module_count,
            "tensor_data_bytes": data_size,
            "tensor_data_sha256": source_data_digest.hexdigest(),
        },
        "output": {
            "file": output_path.name,
            "size": expected_size,
            "sha256": output_sha256,
        },
    }
    with manifest_partial.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(partial_path, output_path)
    os.replace(manifest_partial, manifest_path)
    print(f"verified {module_count} LoRA modules and {len(header)} tensors", flush=True)
    print(f"verified output sha256: {output_sha256}", flush=True)
    print(output_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the official SenseNova U1.5 LoRA to native ComfyUI keys")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=OFFICIAL_REVISION, help="immutable Hugging Face revision")
    parser.add_argument("--source-repo", default=OFFICIAL_REPO)
    args = parser.parse_args()
    convert_lora(args.source, args.output, args.source_repo, args.revision)


if __name__ == "__main__":
    main()
