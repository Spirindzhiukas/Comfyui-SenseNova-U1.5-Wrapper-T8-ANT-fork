"""Inject SenseNova provenance metadata into a converted safetensors file.

``tools/convert_sensenova_int4_convrot.py`` (and ComfyUI's own quantizers) write
the quantized tensors plus a ``_quantization_metadata`` blob, but not the
SenseNova tags this node's strict loader requires. This tool rewrites ONLY the
JSON header and streams the data section unchanged: safetensors buffer offsets
are relative to the start of the data section, so a different header size never
invalidates tensor offsets.

Usage (any python with the pinned digests; no ComfyUI import needed):
    python tools/inject_sensenova_metadata.py -i in.safetensors -o out.safetensors \
        [--variant auto|final|final_legacy|sft]

``--variant auto`` (the default) keeps the source tags the file already carries,
which matters because the node validates the *non*-quantized tensors against the
dtype profile of the official file the conversion started from.

Keep the values below in sync with ``sensenova_u15/loader.py``;
``tests/test_fork_quant_checkpoint.py`` fails if they drift.
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

# Pinned in sync with sensenova_u15/loader.py.
MODEL_FORMAT = "sensenova-u1.5-mot"
CONFIG_SHA256 = "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b"
VARIANTS = {
    "final": {
        "source_repo": "sensenova/SenseNova-U1.5-8B-MoT",
        "source_revision": "19bc874ef6ffc97fda9837b40fc1d1301806158a",
    },
    "final_legacy": {
        "source_repo": "sensenova/SenseNova-U1.5-8B-MoT",
        "source_revision": "1f6ec60423d29939dde4202fd82ae340b144e280",
    },
    "sft": {
        "source_repo": "sensenova/SenseNova-U1.5-8B-MoT-SFT",
        "source_revision": "661834c5b5aee0f89958353511d6ac0ccaacb646",
    },
}
# Written only when the input has no tag yet; it is informational (the loader
# reads the per-layer comfy_quant payloads, not this value), so it covers every
# ConvRot format instead of claiming int8 for a W4A8 file.
QUANTIZATION_TAG = "convrot"


def metadata_for(variant):
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}, expected one of {sorted(VARIANTS)}")
    return {
        "format": MODEL_FORMAT,
        "config_sha256": CONFIG_SHA256,
        **VARIANTS[variant],
    }


def read_header(path):
    with Path(path).open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len))


def detect_variant(metadata):
    """Map existing ``source_repo`` / ``source_revision`` tags to a variant.

    The quantized dtypes the node validates depend on which official file the
    conversion started from: the all-BF16 Final (``final``), the legacy
    mixed-precision Final (``final_legacy``, which is what the community ConvRot
    releases were made from) and SFT. Re-tagging a file with the wrong variant
    makes unrelated tensors fail their dtype check, so the current tags win.
    """
    if not metadata:
        return None
    repo = metadata.get("source_repo")
    revision = metadata.get("source_revision")
    for variant, pinned in VARIANTS.items():
        if repo == pinned["source_repo"] and revision == pinned["source_revision"]:
            return variant
    return None


def rewrite_header(src, dst, metadata):
    src = Path(src)
    dst = Path(dst)
    if dst.resolve() == src.resolve():
        raise SystemExit("output must differ from input")
    with src.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))

    merged = dict(header.get("__metadata__") or {})
    merged.update(metadata)
    header["__metadata__"] = merged

    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with dst.open("wb") as out:
        out.write(struct.pack("<Q", len(new_header)))
        out.write(new_header)
        with src.open("rb") as f:
            f.seek(8 + header_len)
            shutil.copyfileobj(f, out, length=64 * 1024 * 1024)
    return dst.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--variant",
        choices=["auto", *sorted(VARIANTS)],
        default="auto",
        help="auto keeps the tags already in the file (a conversion of the legacy "
        "mixed-precision Final must stay final_legacy); use an explicit value to retag",
    )
    args = parser.parse_args()

    existing = read_header(args.input).get("__metadata__") or {}
    variant = args.variant
    if variant == "auto":
        variant = detect_variant(existing)
        if variant is None:
            raise SystemExit(
                "no SenseNova source tags found in the input file; pass --variant "
                "explicitly (final, final_legacy or sft)"
            )
        print(f"keeping the checkpoint's own tags: variant={variant}")

    metadata = metadata_for(variant)
    if "quantization" not in existing:
        metadata["quantization"] = QUANTIZATION_TAG
    size = rewrite_header(args.input, args.output, metadata)
    print(f"tagged {Path(args.input).name} -> {Path(args.output).name} "
          f"({size} bytes, variant={variant})")


if __name__ == "__main__":
    sys.exit(main())
