"""Build hybrid int8+w4a8 checkpoints by merging two existing converted files.

Takes the validated all-int8 and all-w4a8 checkpoints and emits rungs where a
range of transformer layers uses w4a8 storage while everything else keeps the
int8-convrot format. Pure byte-level remix of already-validated tensors: no
requantization happens here.

Usage (ComfyUI venv python):
    python tools/make_hybrid_ladder.py --int8 <int8.safetensors> --w4a8 <w4a8.safetensors>
        [--outdir DIR] [--rungs hybw4a8-L18-41 ...] [--dry-run] [--no-validate]

Both inputs must already carry the SenseNova metadata tags (see
tools/inject_sensenova_metadata.py). ``--outdir`` defaults to the input
directory; without ``--int8``/``--w4a8`` the tool falls back to
``$SENSENOVA_MODEL_DIR`` and to
``<ComfyUI>/models/diffusion_models/SenseNovaU1.5`` when the node is installed
under ``ComfyUI/custom_nodes``.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

try:  # these tools run inside the ComfyUI venv
    from safetensors import safe_open
except ModuleNotFoundError:  # pragma: no cover
    raise SystemExit("this tool needs the ComfyUI environment (pip install safetensors)")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INT8_NAME = "SenseNova-U1.5-8B-MoT-T8-int8-convrot-tagged.safetensors"
W4A8_NAME = "SenseNova-U1.5-8B-MoT-T8-w4a8-convrot-tagged.safetensors"

SIDECAR_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_s_rel",
    ".weight_s_channel",
    ".weight_codebook",
    ".comfy_quant",
    ".bias",
)

RUNGS = [
    # (tag, predicate(layer_idx) -> bool for w4a8 migration)
    ("hybw4a8-L00-07", lambda i: i < 8),
    ("hybw4a8-L34-41", lambda i: i >= 34),
    ("hybw4a8-L26-41", lambda i: i >= 26),
    ("hybw4a8-L18-41", lambda i: i >= 18),
    ("hybw4a8-L10-41", lambda i: i >= 10),
]


def default_model_dir():
    """$SENSENOVA_MODEL_DIR, else the ComfyUI install this node lives in."""
    env = os.environ.get("SENSENOVA_MODEL_DIR")
    if env:
        return Path(env)
    for candidate in (
        PACKAGE_ROOT.parents[2] / "models" / "diffusion_models" / "SenseNovaU1.5",
        Path.cwd() / "models" / "diffusion_models" / "SenseNovaU1.5",
    ):
        if candidate.is_dir():
            return candidate
    return None


def read_header(path):
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    meta = header.pop("__metadata__", None)
    return header, meta, 8 + header_len


def stem_of(key):
    for suffix in SIDECAR_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def group_by_stem(header):
    groups = {}
    for key, entry in header.items():
        groups.setdefault(stem_of(key), {})[key] = entry
    return groups


def quant_format_of(int8_groups, stem):
    entry = int8_groups.get(stem, {}).get(stem + ".comfy_quant")
    if entry is None:
        return None
    return entry  # placeholder; real JSON parsed lazily by caller if needed


def layer_index(stem):
    marker = "language_model.model.layers."
    if not stem.startswith(marker):
        return None
    rest = stem[len(marker):]
    token = rest.split(".", 1)[0]
    return int(token) if token.isdigit() else None


def build_rung(tag, predicate, int8_file, w4a8_file, outdir, dry_run=False):
    out_path = Path(outdir) / f"SenseNova-U1.5-8B-MoT-T8-{tag}.safetensors"
    header_i, meta, data_start_i = read_header(int8_file)
    header_w, _, data_start_w = read_header(w4a8_file)
    groups_i = group_by_stem(header_i)
    groups_w = group_by_stem(header_w)

    migrated, kept = [], []
    plan = []  # (from_w4a8, absolute_offset, length)
    new_header = {}
    if meta:
        new_header["__metadata__"] = meta

    cursor = 0
    for stem in sorted(groups_i):
        idx = layer_index(stem)
        use_w = (
            idx is not None
            and predicate(idx)
            and stem + ".comfy_quant" in groups_w.get(stem, {})
        )
        src_start = data_start_w if use_w else data_start_i
        src_groups = groups_w if use_w else groups_i
        if use_w:
            migrated.append(stem)
        else:
            kept.append(stem)
        for key in sorted(src_groups[stem]):
            entry = src_groups[stem][key]
            rel_start, rel_end = entry["data_offsets"]
            new_header[key] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [cursor, cursor + (rel_end - rel_start)],
            }
            plan.append((use_w, src_start + rel_start, rel_end - rel_start))
            cursor += rel_end - rel_start
        # sanity: migrated stems must expose their w4a8 sidecar set
        if use_w:
            needed = {stem + s for s in (".weight", ".weight_s_rel", ".weight_s_channel", ".comfy_quant")}
            missing = needed - set(groups_w[stem])
            if missing:
                raise SystemExit(f"{tag}: w4a8 source missing sidecars for {stem}: {missing}")

    print(f"[{tag}] migrated={len(migrated)} layers -> w4a8 | kept={len(kept)} stems on int8")
    print(f"[{tag}] tensors={len(plan)} bytes={cursor / 2**30:.2f}GiB -> {out_path.name}")
    if dry_run:
        return out_path

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    with open(out_path, "wb") as fo, open(int8_file, "rb") as fi, open(w4a8_file, "rb") as fw:
        fo.write(struct.pack("<Q", len(header_bytes) + pad))
        fo.write(header_bytes)
        fo.write(b"\x20" * pad)
        written = 0
        for is_w4a8_src, offset, length in plan:
            src = fw if is_w4a8_src else fi
            src.seek(offset)
            remaining = length
            while remaining > 0:
                chunk = src.read(min(remaining, 1 << 24))
                if not chunk:
                    raise SystemExit(f"{tag}: short read at offset {offset}")
                fo.write(chunk)
                remaining -= len(chunk)
                written += len(chunk)
    assert written == cursor
    return out_path


def validate(path, expect_migrated_count, comfy_root=None):
    """Re-run the node's own header validation on a generated rung."""
    if comfy_root:
        sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(PACKAGE_ROOT))
    from sensenova_u15.loader import _read_quant_formats, _validate_checkpoint_header

    with safe_open(str(path), framework="pt", device="cpu") as f:
        formats = _read_quant_formats(f)
        _validate_checkpoint_header(f)
    counts = {}
    for fmt in formats.values():
        counts[fmt] = counts.get(fmt, 0) + 1
    assert counts.get("asym_w4a8_int8", 0) == expect_migrated_count * 14, (
        f"{path.name}: expected {expect_migrated_count * 14} w4a8 layers, got {counts}"
    )
    print(f"VALIDATED {path.name}: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--int8", type=Path, default=None, help=f"default: <model dir>/{INT8_NAME}")
    parser.add_argument("--w4a8", type=Path, default=None, help=f"default: <model dir>/{W4A8_NAME}")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--rungs",
        nargs="*",
        default=None,
        help=f"subset of {', '.join(tag for tag, _ in RUNGS)} (default: all)",
    )
    parser.add_argument("--comfy-root", type=Path, default=None, help="ComfyUI folder for validation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    models = default_model_dir()
    if args.int8 is None or args.w4a8 is None:
        if models is None:
            raise SystemExit(
                "no checkpoint paths given and no model directory found; pass "
                "--int8/--w4a8 explicitly or set SENSENOVA_MODEL_DIR"
            )
        args.int8 = args.int8 or models / INT8_NAME
        args.w4a8 = args.w4a8 or models / W4A8_NAME
    outdir = args.outdir or models or args.int8.parent
    outdir.mkdir(parents=True, exist_ok=True)
    selected = [
        (tag, predicate)
        for tag, predicate in RUNGS
        if args.rungs is None or tag in args.rungs
    ]
    if not selected:
        raise SystemExit(f"no rungs matched {args.rungs}")
    for path in (args.int8, args.w4a8):
        if not path.is_file():
            raise SystemExit(f"missing input checkpoint: {path}")

    for tag, predicate in selected:
        count = sum(1 for i in range(42) if predicate(i))
        path = outdir / f"SenseNova-U1.5-8B-MoT-T8-{tag}.safetensors"
        if path.exists():
            print(f"[{tag}] already exists, skipping build")
        else:
            build_rung(tag, predicate, args.int8, args.w4a8, outdir, dry_run=args.dry_run)
        if not args.no_validate:
            validate(path, count, comfy_root=args.comfy_root)


if __name__ == "__main__":
    main()
