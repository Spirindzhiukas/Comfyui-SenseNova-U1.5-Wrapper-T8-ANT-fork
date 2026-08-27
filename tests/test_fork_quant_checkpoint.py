"""Quantized (ConvRot) checkpoint validation for the SenseNova fork.

The strict upstream contract only describes the official bf16 files. Quantized
conversions keep the same base keys but pack rank-2 linear weights and add
per-layer sidecars, so ``_validate_checkpoint_header`` switches to a derived
contract. These tests cover that branch and prove the bf16 branch is untouched.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import torch

import sensenova_u15.loader as loader  # noqa: E402
import sensenova_u15.quant_bridge as quant_bridge


Q_STEM = "language_model.model.layers.0.self_attn.q_proj"
K_STEM = "language_model.model.layers.0.self_attn.k_proj"
BASE_KEYS = {
    f"{Q_STEM}.weight": ((4096, 4096), "BF16"),
    f"{K_STEM}.weight": ((1024, 4096), "BF16"),
    "language_model.model.layers.0.input_layernorm.weight": ((4096,), "BF16"),
    "language_model.model.embed_tokens.weight": ((151936, 4096), "BF16"),
    "language_model.lm_head.weight": ((151936, 4096), "F32"),
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": ((1024, 3, 16, 16), "F32"),
}
METADATA = {
    "format": loader.MODEL_FORMAT,
    "config_sha256": loader.CONFIG_SHA256,
    "source_repo": loader.MODEL_REPO,
    "source_revision": loader.MODEL_REVISION,
}


class FakeSlice:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def get_shape(self):
        return tuple(self.shape)

    def get_dtype(self):
        return self.dtype


class FakeCheckpoint:
    def __init__(self, tensors, metadata=None):
        self._tensors = tensors
        self._metadata = METADATA if metadata is None else metadata

    def keys(self):
        return self._tensors.keys()

    def metadata(self):
        return self._metadata

    def get_slice(self, name):
        return self._tensors[name]

    def get_tensor(self, name):
        return self._tensors[name].payload


def _dtype(dtype):
    # a contract entry may allow several storage dtypes; the fake tensor picks one
    return dtype[0] if isinstance(dtype, tuple) else dtype


def quant_contract(formats):
    return loader._quant_checkpoint_contract("final", formats)[0]


def build_checkpoint(formats, **breakage):
    contract = quant_contract(formats)
    tensors = {}
    for name, (shape, dtype) in contract.items():
        if name.endswith(loader.QUANT_METADATA_SUFFIX):
            payload = json.dumps({"format": formats[name[: -len(loader.QUANT_METADATA_SUFFIX)]]}).encode()
            tensors[name] = FakeSlice((len(payload),), "U8")
            tensors[name].payload = torch.tensor(list(payload), dtype=torch.uint8)
            continue
        tensors[name] = FakeSlice(shape if shape else (1,), _dtype(dtype))
    if breakage.get("missing"):
        tensors.pop(breakage["missing"])
    if breakage.get("dtype"):
        tensors[breakage["dtype"]] = FakeSlice(
            tensors[breakage["dtype"]].get_shape(), "BF16"
        )
    if breakage.get("shape"):
        tensors[breakage["shape"]] = FakeSlice((1, 1), tensors[breakage["shape"]].get_dtype())
    return FakeCheckpoint(tensors)


class QuantContractTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(loader, "_checkpoint_contract", lambda profile: dict(BASE_KEYS))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_quant_candidate_policy_matches_the_converter(self):
        self.assertTrue(loader._is_quant_candidate(f"{Q_STEM}.weight", (4096, 4096)))
        for excluded in ("input_layernorm", "embed_tokens", "lm_head"):
            for name in BASE_KEYS:
                if excluded in name:
                    with self.subTest(name=name):
                        self.assertFalse(loader._is_quant_candidate(name, BASE_KEYS[name][0]))
        self.assertFalse(
            loader._is_quant_candidate(
                "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
                (1024, 3, 16, 16),
            )
        )

    def test_int8_contract_keeps_shape_and_adds_sidecars(self):
        contract = quant_contract({Q_STEM: "int8_tensorwise"})
        self.assertEqual(contract[f"{Q_STEM}.weight"], ((4096, 4096), "I8"))
        self.assertEqual(contract[f"{Q_STEM}.weight_scale"], ((4096, 1), "F32"))
        self.assertEqual(contract[f"{Q_STEM}.comfy_quant"], (None, "U8"))
        # untouched keys keep the dtype of the bundled JSON contract
        self.assertEqual(contract["language_model.lm_head.weight"], ((151936, 4096), "F32"))
        self.assertEqual(contract[f"{K_STEM}.weight"], ((1024, 4096), "BF16"))

    def test_w4a4_contract_packs_two_nibbles_per_byte(self):
        contract = quant_contract({Q_STEM: "convrot_w4a4"})
        self.assertEqual(contract[f"{Q_STEM}.weight"], ((4096, 2048), "I8"))
        self.assertEqual(contract[f"{Q_STEM}.weight_scale"], ((4096,), "F32"))

    def test_w4a8_contract_uses_group_and_channel_scales(self):
        contract = quant_contract({Q_STEM: "asym_w4a8_int8"})
        self.assertEqual(contract[f"{Q_STEM}.weight"], ((4096, 2048), "I8"))
        self.assertEqual(contract[f"{Q_STEM}.weight_s_rel"], ((4096, 256), loader.QUANT_GROUP_SCALE_DTYPES))
        self.assertEqual(contract[f"{Q_STEM}.weight_s_channel"], ((4096,), "F32"))
        self.assertEqual(contract[f"{Q_STEM}.weight_codebook"], ((16,), "F32"))

    def test_unsupported_format_is_reported_before_validation(self):
        with self.assertRaisesRegex(ValueError, "unsupported quantization format"):
            quant_contract({Q_STEM: "nvfp4"})

    def test_every_format_validates_through_the_header(self):
        for quant_format in loader.QUANT_FORMATS:
            with self.subTest(format=quant_format):
                keys, profile = loader._validate_checkpoint_header(build_checkpoint({Q_STEM: quant_format}))
                self.assertEqual(profile, "final")
                self.assertIn(f"{Q_STEM}.comfy_quant", keys)

    def test_mixed_layers_are_supported(self):
        formats = {Q_STEM: "convrot_w4a4", K_STEM: "int8_tensorwise"}
        contract = quant_contract(formats)
        self.assertEqual(contract[f"{Q_STEM}.weight"], ((4096, 2048), "I8"))
        self.assertEqual(contract[f"{K_STEM}.weight"], ((1024, 4096), "I8"))
        self.assertEqual(contract[f"{K_STEM}.weight_scale"], ((1024, 1), "F32"))
        loader._validate_checkpoint_header(build_checkpoint(formats))

    def test_optional_sidecars_may_be_absent(self):
        formats = {Q_STEM: "asym_w4a8_int8"}
        checkpoint = build_checkpoint(formats, missing=f"{Q_STEM}.weight_codebook")
        keys, _ = loader._validate_checkpoint_header(checkpoint)
        self.assertNotIn(f"{Q_STEM}.weight_codebook", keys)
        self.assertIn(f"{Q_STEM}.comfy_quant", keys)

    def test_corrupted_quant_header_is_rejected(self):
        formats = {Q_STEM: "int8_tensorwise"}
        with self.assertRaisesRegex(ValueError, "quantized checkpoint key mismatch"):
            loader._validate_checkpoint_header(build_checkpoint(formats, missing=f"{Q_STEM}.weight_scale"))
        with self.assertRaisesRegex(ValueError, "quantized checkpoint dtype mismatch"):
            loader._validate_checkpoint_header(build_checkpoint(formats, dtype=f"{Q_STEM}.weight"))
        with self.assertRaisesRegex(ValueError, "quantized checkpoint shape mismatch"):
            loader._validate_checkpoint_header(build_checkpoint(formats, shape=f"{Q_STEM}.weight_scale"))

    def test_unreadable_payload_falls_back_to_the_strict_contract(self):
        formats = {Q_STEM: "int8_tensorwise"}
        checkpoint = build_checkpoint(formats)
        key = f"{Q_STEM}.comfy_quant"
        checkpoint._tensors[key].payload = torch.tensor(list(b"{not json"), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "checkpoint key mismatch"):
            loader._validate_checkpoint_header(checkpoint)

    def test_bf16_header_check_never_reads_sidecars(self):
        # No `comfy_quant` key means the upstream bf16 branch runs untouched,
        # including its dtype and shape strictness.
        checkpoint = FakeCheckpoint({"a.weight": FakeSlice((2,), "BF16")})
        with mock.patch.object(loader, "_checkpoint_contract", lambda profile: {"a.weight": ((2,), "BF16")}):
            self.assertEqual(loader._validate_checkpoint_header(checkpoint), ({"a.weight"}, "final"))
            with self.assertRaisesRegex(ValueError, "dtype mismatch"):
                loader._validate_checkpoint_header(FakeCheckpoint({"a.weight": FakeSlice((2,), "F32")}))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                loader._validate_checkpoint_header(FakeCheckpoint({"a.weight": FakeSlice((3,), "BF16")}))

    def test_quant_support_env_switch(self):
        formats = {Q_STEM: "int8_tensorwise"}
        checkpoint = build_checkpoint(formats)
        with mock.patch.dict(__import__("os").environ, {"SENSENOVA_NO_QUANT": "1"}):
            with self.assertRaisesRegex(ValueError, "checkpoint key mismatch"):
                loader._validate_checkpoint_header(checkpoint)
        keys, _ = loader._validate_checkpoint_header(checkpoint)
        self.assertIn(f"{Q_STEM}.comfy_quant", keys)


class QuantRealFileTests(unittest.TestCase):
    """Exercise the header reader against an actual safetensors file."""

    def setUp(self):
        patcher = mock.patch.object(loader, "_checkpoint_contract", lambda profile: dict(BASE_KEYS))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_formats_are_read_from_a_written_file(self):
        from safetensors.torch import save_file

        formats = {Q_STEM: "int8_tensorwise"}
        contract = quant_contract(formats)
        tensors = {}
        for name, (shape, dtype) in contract.items():
            if name.endswith(loader.QUANT_METADATA_SUFFIX):
                payload = json.dumps({"format": formats[Q_STEM]}).encode("utf-8")
                tensors[name] = torch.tensor(list(payload), dtype=torch.uint8)
                continue
            storage = {
                "I8": torch.int8,
                "U8": torch.uint8,
                "F8_E4M3": torch.uint8,  # legacy raw-byte scale
                "BF16": torch.bfloat16,
                "F32": torch.float32,
            }[_dtype(dtype)]
            tensors[name] = torch.zeros(shape or (1,), dtype=storage)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quant.safetensors"
            save_file(tensors, str(path), metadata=dict(METADATA))
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                self.assertEqual(loader._read_quant_formats(checkpoint), formats)
                keys, profile = loader._validate_checkpoint_header(checkpoint, path)
                self.assertEqual(profile, "final")
                self.assertEqual(set(keys), set(tensors))

    def test_bf16_file_has_no_quant_formats(self):
        from safetensors.torch import save_file

        tensors = {
            "a.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.safetensors"
            save_file(tensors, str(path), metadata=dict(METADATA))
            from safetensors import safe_open

            with mock.patch.object(loader, "_checkpoint_contract", lambda profile: {"a.weight": ((2, 2), "BF16")}):
                with safe_open(path, framework="pt", device="cpu") as checkpoint:
                    self.assertEqual(loader._read_quant_formats(checkpoint), {})
                    self.assertEqual(loader._validate_checkpoint_header(checkpoint), ({"a.weight"}, "final"))


class ConverterToolTests(unittest.TestCase):
    """The shipped tools must describe the same checkpoint contract as the node."""

    def test_metadata_injector_matches_the_loader_pins(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "inject_sensenova_metadata", PACKAGE_ROOT / "tools" / "inject_sensenova_metadata.py"
        )
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)

        self.assertEqual(tool.MODEL_FORMAT, loader.MODEL_FORMAT)
        self.assertEqual(tool.CONFIG_SHA256, loader.CONFIG_SHA256)
        self.assertEqual(set(tool.VARIANTS), set(loader.CHECKPOINT_PROFILES))
        for profile, pinned in tool.VARIANTS.items():
            with self.subTest(profile=profile):
                self.assertEqual(pinned["source_repo"], loader.CHECKPOINT_PROFILES[profile]["source_repo"])
                self.assertEqual(
                    pinned["source_revision"], loader.CHECKPOINT_PROFILES[profile]["source_revision"]
                )

    def test_converter_policy_mirrors_the_loader(self):
        source = (PACKAGE_ROOT / "tools" / "convert_sensenova_int4_convrot.py").read_text(encoding="utf-8")
        for token in loader.QUANT_EXCLUDED_SUBSTRINGS:
            with self.subTest(token=token):
                self.assertIn(f'"{token}"', source)
        self.assertIn("loader.py::_is_quant_candidate", source)


class QuantBridgeGateTests(unittest.TestCase):
    def test_plain_state_dict_needs_no_bridge(self):
        self.assertEqual(quant_bridge.state_dict_quant_formats({"a.weight": torch.zeros(2)}), {})
        self.assertFalse(quant_bridge.quant_bridge_needed({"a.weight": torch.zeros(2)}))

    def test_bridge_requires_kitchen_before_touching_ops(self):
        if importlib.util.find_spec("comfy_kitchen") is not None:
            self.skipTest("comfy-kitchen installed; the guard cannot fire here")
        payload = json.dumps({"format": "convrot_w4a4"}).encode("utf-8")
        state_dict = {f"{Q_STEM}.comfy_quant": torch.tensor(list(payload), dtype=torch.uint8)}
        with self.assertRaisesRegex(ValueError, "comfy-kitchen"):
            quant_bridge.quant_bridge_needed(state_dict)

    def test_format_map_is_read_from_state_dict(self):
        payload = json.dumps({"format": "convrot_w4a4"}).encode("utf-8")
        state_dict = {
            f"{Q_STEM}.comfy_quant": torch.tensor(list(payload), dtype=torch.uint8),
            f"{Q_STEM}.weight": torch.zeros(4, dtype=torch.int8),
        }
        self.assertEqual(quant_bridge.state_dict_quant_formats(state_dict), {Q_STEM: "convrot_w4a4"})

    @unittest.skipUnless(
        importlib.util.find_spec("comfy_kitchen") is not None, "quantized loads need comfy-kitchen"
    )
    def test_environment_overrides_win(self):
        payload = json.dumps({"format": "int8_tensorwise"}).encode("utf-8")
        state_dict = {f"{Q_STEM}.comfy_quant": torch.tensor(list(payload), dtype=torch.uint8)}
        with mock.patch.dict(os.environ, {"SENSENOVA_NO_BRIDGE": "1"}):
            self.assertFalse(quant_bridge.quant_bridge_needed(state_dict))
        with mock.patch.dict(os.environ, {"SENSENOVA_FORCE_BRIDGE": "1"}):
            self.assertTrue(quant_bridge.quant_bridge_needed(state_dict))

    def test_capability_probe_is_conservative(self):
        # The bridge is skipped only when the running ComfyUI understands every
        # format in the checkpoint AND comfy-kitchen accepts the convrot kwargs,
        # so any missing piece falls back to the (validated) bridge.
        with mock.patch.object(quant_bridge, "QuantizedTensor", None):
            self.assertFalse(quant_bridge.core_supports_convrot(("int8_tensorwise",)))
        with mock.patch.object(quant_bridge, "QUANT_ALGOS", {}):
            self.assertFalse(quant_bridge.core_supports_convrot(("convrot_w4a4",)))

    def test_recipe_specific_extra_sidecars_are_accepted(self):
        formats = {Q_STEM: "asym_w4a8_int8"}
        checkpoint = build_checkpoint(formats)
        key = f"{Q_STEM}.weight_correction"
        checkpoint._tensors[key] = FakeSlice((4096, 256), "F32")
        keys, _ = loader._validate_checkpoint_header(checkpoint)
        self.assertIn(key, keys)

    def test_guards_respect_their_switch(self):
        import sensenova_u15.qt_guards as qt_guards

        with mock.patch.dict(os.environ, {"SENSENOVA_NO_QT_GUARDS": "1"}):
            self.assertFalse(qt_guards.install_quant_guards())


if __name__ == "__main__":
    unittest.main()
