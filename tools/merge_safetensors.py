from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


CHUNK_SIZE = 16 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F32": 4}
SIDECAR_FILES = (
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    source_start: int
    source_end: int
    output_start: int = 0
    output_end: int = 0

    @property
    def size(self) -> int:
        return self.source_end - self.source_start


@dataclass(frozen=True)
class ShardHeader:
    path: Path
    data_start: int
    data_size: int
    tensors: tuple[TensorRecord, ...]


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_keys)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_size(dtype: str, shape: tuple[int, ...]) -> int:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype!r}; expected BF16/F32 SenseNova weights")
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError(f"invalid tensor shape: {shape!r}")
        elements *= dimension
    return elements * DTYPE_BYTES[dtype]


def _read_shard_header(path: Path, shard_name: str, expected_keys: set[str]) -> ShardHeader:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {shard_name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header size: {shard_name}")
        header_bytes = handle.read(header_size)
    header = json.loads(header_bytes, object_pairs_hook=_reject_duplicate_keys)
    header.pop("__metadata__", None)
    actual_keys = set(header)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:5]
        extra = sorted(actual_keys - expected_keys)[:5]
        raise ValueError(f"index/header key mismatch in {shard_name}: missing={missing}, extra={extra}")

    data_start = 8 + header_size
    data_size = path.stat().st_size - data_start
    tensors = []
    for name, info in header.items():
        if not isinstance(info, dict):
            raise ValueError(f"invalid tensor header for {name}")
        dtype = info.get("dtype")
        shape_value = info.get("shape")
        offsets = info.get("data_offsets")
        if not isinstance(shape_value, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid tensor metadata for {name}")
        shape = tuple(shape_value)
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError(f"invalid data offsets for {name}")
        if end > data_size:
            raise ValueError(f"data offset outside shard for {name}")
        if end - start != _tensor_size(dtype, shape):
            raise ValueError(f"shape/dtype byte mismatch for {name}")
        tensors.append(TensorRecord(name, shard_name, dtype, shape, start, end))

    tensors.sort(key=lambda item: item.source_start)
    previous_end = 0
    for tensor in tensors:
        if tensor.source_start != previous_end:
            raise ValueError(f"gap or overlap before {tensor.name} in {shard_name}")
        previous_end = tensor.source_end
    if previous_end != data_size:
        raise ValueError(f"unreferenced data tail in {shard_name}")
    return ShardHeader(path, data_start, data_size, tuple(tensors))


def inspect_source(source_dir: Path):
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing index: {index_path}")
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no weight_map")

    expected_by_shard = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard_name, str):
            raise ValueError("weight_map keys and values must be strings")
        if Path(shard_name).name != shard_name:
            raise ValueError(f"unsafe shard path in index: {shard_name}")
        expected_by_shard.setdefault(shard_name, set()).add(name)

    residuals = [
        path
        for path in source_dir.rglob("*")
        if (
            path.is_file()
            and path.name.endswith((".incomplete", ".partial", ".lock"))
            and path.relative_to(source_dir).parts[:2] != (".cache", "huggingface")
        )
    ]
    if residuals:
        raise ValueError(f"source contains incomplete or locked files: {residuals[0]}")

    shard_headers = []
    seen_keys = set()
    for shard_name, expected_keys in expected_by_shard.items():
        path = source_dir / shard_name
        if not path.is_file():
            raise FileNotFoundError(f"missing shard: {path}")
        shard = _read_shard_header(path, shard_name, expected_keys)
        overlap = seen_keys.intersection(expected_keys)
        if overlap:
            raise ValueError(f"duplicate tensor across shards: {next(iter(overlap))}")
        seen_keys.update(expected_keys)
        shard_headers.append(shard)

    if seen_keys != set(weight_map):
        raise ValueError("source tensor set does not match model index")
    return index_path, tuple(shard_headers)


def _build_output_header(shards: tuple[ShardHeader, ...], metadata: dict[str, str]):
    header = {"__metadata__": metadata}
    records = []
    output_offset = 0
    for shard in shards:
        for tensor in shard.tensors:
            output_end = output_offset + tensor.size
            record = TensorRecord(
                tensor.name,
                tensor.shard,
                tensor.dtype,
                tensor.shape,
                tensor.source_start,
                tensor.source_end,
                output_offset,
                output_end,
            )
            records.append(record)
            header[tensor.name] = {
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "data_offsets": [output_offset, output_end],
            }
            output_offset = output_end

    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    return header_bytes, tuple(records), output_offset


def _copy_exact(source, target, size: int, source_digest) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(CHUNK_SIZE, remaining))
        if not chunk:
            raise EOFError("source tensor data ended early")
        target.write(chunk)
        source_digest.update(chunk)
        remaining -= len(chunk)


def _copy_tensors(partial_path: Path, header_bytes: bytes, shards, records):
    source_hashes = {}
    records_by_shard = {}
    for record in records:
        records_by_shard.setdefault(record.shard, []).append(record)

    with partial_path.open("xb") as target:
        target.write(struct.pack("<Q", len(header_bytes)))
        target.write(header_bytes)
        for shard in shards:
            with shard.path.open("rb") as source:
                for record in records_by_shard[shard.path.name]:
                    source.seek(shard.data_start + record.source_start)
                    digest = hashlib.sha256()
                    _copy_exact(source, target, record.size, digest)
                    source_hashes[record.name] = digest.hexdigest()
            print(f"copied {shard.path.name} ({len(shard.tensors)} tensors)", flush=True)
        target.flush()
        os.fsync(target.fileno())
    return source_hashes


def _verify_output(partial_path: Path, header_bytes: bytes, records, source_hashes):
    output_digest = hashlib.sha256()
    target_hashes = {}
    with partial_path.open("rb") as handle:
        prefix = handle.read(8 + len(header_bytes))
        expected_prefix = struct.pack("<Q", len(header_bytes)) + header_bytes
        if prefix != expected_prefix:
            raise ValueError("output header changed while writing")
        output_digest.update(prefix)
        for record in records:
            digest = hashlib.sha256()
            remaining = record.size
            while remaining:
                chunk = handle.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise EOFError(f"output tensor data ended early: {record.name}")
                digest.update(chunk)
                output_digest.update(chunk)
                remaining -= len(chunk)
            target_hash = digest.hexdigest()
            if target_hash != source_hashes[record.name]:
                raise ValueError(f"source/output hash mismatch: {record.name}")
            target_hashes[record.name] = target_hash
        if handle.read(1):
            raise ValueError("unexpected data after final tensor")
    return output_digest.hexdigest(), target_hashes


def merge_model(source_dir: Path, output_path: Path, source_repo: str, revision: str):
    source_dir = source_dir.resolve()
    output_path = output_path.resolve()
    if sys.maxsize <= 2**32:
        raise RuntimeError("64-bit Python is required")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    partial_path = Path(f"{output_path}.partial")
    manifest_path = Path(f"{output_path}.manifest.json")
    manifest_partial = Path(f"{manifest_path}.partial")
    for path in (partial_path, manifest_path, manifest_partial):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {path}")

    index_path, shards = inspect_source(source_dir)
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config: {config_path}")
    missing_sidecars = [name for name in SIDECAR_FILES if not (source_dir / name).is_file()]
    if missing_sidecars:
        raise FileNotFoundError(f"missing sidecar files: {missing_sidecars}")

    config_sha256 = _sha256_file(config_path)
    metadata = {
        "format": "sensenova-u1.5-mot",
        "source_repo": source_repo,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "license": "Apache-2.0",
        "conversion": "raw-stream-repack-v1",
    }
    header_bytes, records, tensor_data_size = _build_output_header(shards, metadata)
    expected_size = 8 + len(header_bytes) + tensor_data_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    free_space = shutil.disk_usage(output_path.parent).free
    safety_margin = max(1024**3, expected_size // 20)
    if free_space < expected_size + safety_margin:
        raise OSError(
            f"insufficient free space: need {expected_size + safety_margin} bytes, have {free_space} bytes"
        )

    print(f"validated {len(shards)} shards and {len(records)} tensors", flush=True)
    print(f"writing {expected_size} bytes to {partial_path}", flush=True)
    source_hashes = _copy_tensors(partial_path, header_bytes, shards, records)
    if partial_path.stat().st_size != expected_size:
        raise ValueError(f"output size mismatch: {partial_path.stat().st_size} != {expected_size}")
    output_sha256, target_hashes = _verify_output(partial_path, header_bytes, records, source_hashes)

    sidecar_hashes = {name: _sha256_file(source_dir / name) for name in SIDECAR_FILES}
    manifest = {
        "format_version": 1,
        "source": {
            "repo": source_repo,
            "revision": revision,
            "index_sha256": _sha256_file(index_path),
            "sidecar_sha256": sidecar_hashes,
            "shards": [shard.path.name for shard in shards],
        },
        "output": {
            "file": output_path.name,
            "size": expected_size,
            "sha256": output_sha256,
            "tensor_count": len(records),
            "tensor_data_bytes": tensor_data_size,
        },
        "tensors": {
            record.name: {
                "source_shard": record.shard,
                "dtype": record.dtype,
                "shape": list(record.shape),
                "data_offsets": [record.output_start, record.output_end],
                "sha256": target_hashes[record.name],
            }
            for record in records
        },
    }
    with manifest_partial.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(partial_path, output_path)
    os.replace(manifest_partial, manifest_path)
    print(f"verified output sha256: {output_sha256}", flush=True)
    print(output_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream SenseNova U1.5 shards into one safetensors file")
    parser.add_argument("--source", type=Path, required=True, help="downloaded Hugging Face model directory")
    parser.add_argument("--output", type=Path, required=True, help="final .safetensors path")
    parser.add_argument("--revision", required=True, help="immutable Hugging Face revision")
    parser.add_argument("--source-repo", default="sensenova/SenseNova-U1.5-8B-MoT")
    args = parser.parse_args()
    if args.output.suffix != ".safetensors":
        parser.error("--output must end with .safetensors")
    merge_model(args.source, args.output, args.source_repo, args.revision)


if __name__ == "__main__":
    main()
