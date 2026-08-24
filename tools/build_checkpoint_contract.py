import argparse
import json
from pathlib import Path


MODEL_FORMAT = "sensenova-u1.5-mot"
CONFIG_SHA256 = "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b"
VARIANTS = {
    "final": {
        "source_repo": "sensenova/SenseNova-U1.5-8B-MoT",
        "source_revision": "1f6ec60423d29939dde4202fd82ae340b144e280",
    },
    "sft": {
        "source_repo": "sensenova/SenseNova-U1.5-8B-MoT-SFT",
        "source_revision": "661834c5b5aee0f89958353511d6ac0ccaacb646",
    },
}


def _load_manifest(path, variant):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = VARIANTS[variant]
    source = manifest.get("source", {})
    if source.get("repo") != expected["source_repo"]:
        raise ValueError(f"{variant} source repo mismatch: {source.get('repo')}")
    if source.get("revision") != expected["source_revision"]:
        raise ValueError(f"{variant} source revision mismatch: {source.get('revision')}")
    if source.get("sidecar_sha256", {}).get("config.json") != CONFIG_SHA256:
        raise ValueError(f"{variant} config digest mismatch")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError(f"{variant} manifest has no tensors")
    return manifest


def build_contract(final_path, sft_path):
    manifests = {
        "final": _load_manifest(final_path, "final"),
        "sft": _load_manifest(sft_path, "sft"),
    }
    final_tensors = manifests["final"]["tensors"]
    sft_tensors = manifests["sft"]["tensors"]
    if set(final_tensors) != set(sft_tensors):
        missing = sorted(set(final_tensors) - set(sft_tensors))[:5]
        unexpected = sorted(set(sft_tensors) - set(final_tensors))[:5]
        raise ValueError(f"Final/SFT tensor keys differ: missing={missing}, unexpected={unexpected}")

    tensors = {}
    for name in sorted(final_tensors):
        final = final_tensors[name]
        sft = sft_tensors[name]
        if final["shape"] != sft["shape"]:
            raise ValueError(f"Final/SFT shape differs for {name}")
        tensors[name] = {
            "shape": final["shape"],
            "dtypes": {
                "final": final["dtype"],
                "sft": sft["dtype"],
            },
        }

    variants = {}
    for variant, manifest in manifests.items():
        output = manifest["output"]
        if output["tensor_count"] != len(tensors):
            raise ValueError(f"{variant} tensor count does not match its manifest")
        variants[variant] = {
            **VARIANTS[variant],
            "file_size": output["size"],
            "file_sha256": output["sha256"],
            "tensor_count": output["tensor_count"],
        }

    return {
        "format_version": 1,
        "model_format": MODEL_FORMAT,
        "config_sha256": CONFIG_SHA256,
        "variants": variants,
        "tensors": tensors,
    }


def main():
    parser = argparse.ArgumentParser(description="Build the runtime SenseNova checkpoint contract")
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--sft-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = build_contract(args.final_manifest, args.sft_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(contract['tensors'])} tensors to {args.output}")


if __name__ == "__main__":
    main()
