import argparse
import json
from pathlib import Path


MODEL_FORMAT = "sensenova-u1.5-mot"
CONFIG_SHA256 = "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b"
PROFILES = {
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


def _load_manifest(path, profile):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = PROFILES[profile]
    source = manifest.get("source", {})
    if source.get("repo") != expected["source_repo"]:
        raise ValueError(f"{profile} source repo mismatch: {source.get('repo')}")
    if source.get("revision") != expected["source_revision"]:
        raise ValueError(f"{profile} source revision mismatch: {source.get('revision')}")
    if source.get("sidecar_sha256", {}).get("config.json") != CONFIG_SHA256:
        raise ValueError(f"{profile} config digest mismatch")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError(f"{profile} manifest has no tensors")
    return manifest


def build_contract(final_path, final_legacy_path, sft_path):
    manifests = {
        "final": _load_manifest(final_path, "final"),
        "final_legacy": _load_manifest(final_legacy_path, "final_legacy"),
        "sft": _load_manifest(sft_path, "sft"),
    }
    profile_tensors = {profile: manifest["tensors"] for profile, manifest in manifests.items()}
    final_tensors = profile_tensors["final"]
    for profile, tensors in profile_tensors.items():
        if set(final_tensors) != set(tensors):
            missing = sorted(set(final_tensors) - set(tensors))[:5]
            unexpected = sorted(set(tensors) - set(final_tensors))[:5]
            raise ValueError(f"Final/{profile} tensor keys differ: missing={missing}, unexpected={unexpected}")

    tensors = {}
    for name in sorted(final_tensors):
        shape = final_tensors[name]["shape"]
        for profile, values in profile_tensors.items():
            if values[name]["shape"] != shape:
                raise ValueError(f"Final/{profile} shape differs for {name}")
        tensors[name] = {
            "shape": shape,
            "dtypes": {profile: values[name]["dtype"] for profile, values in profile_tensors.items()},
        }

    variants = {}
    for profile, manifest in manifests.items():
        output = manifest["output"]
        if output["tensor_count"] != len(tensors):
            raise ValueError(f"{profile} tensor count does not match its manifest")
        variants[profile] = {
            "source_repo": PROFILES[profile]["source_repo"],
            "source_revision": PROFILES[profile]["source_revision"],
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
    parser.add_argument("--final-legacy-manifest", type=Path, required=True)
    parser.add_argument("--sft-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = build_contract(args.final_manifest, args.final_legacy_manifest, args.sft_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(contract['tensors'])} tensors to {args.output}")


if __name__ == "__main__":
    main()
