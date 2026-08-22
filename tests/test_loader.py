import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import sensenova_u15.loader as loader


class FakeSlice:
    def __init__(self, shape=(2,), dtype="BF16"):
        self.shape = shape
        self.dtype = dtype

    def get_shape(self):
        return self.shape

    def get_dtype(self):
        return self.dtype


class FakeCheckpoint:
    def __init__(self, keys=("a",), shape=(2,), dtype="BF16", metadata=None):
        self._keys = keys
        self.tensor = FakeSlice(shape, dtype)
        self._metadata = metadata or {
            "format": loader.MODEL_FORMAT,
            "source_repo": loader.MODEL_REPO,
            "config_sha256": loader.CONFIG_SHA256,
            "source_revision": loader.MODEL_REVISION,
        }

    def keys(self):
        return self._keys

    def metadata(self):
        return self._metadata

    def get_slice(self, _name):
        return self.tensor


class LoaderTests(unittest.TestCase):
    def test_bundled_tokenizer_assets_match_pinned_revision(self):
        loader._validate_tokenizer_assets()

    def test_header_contract_rejects_key_shape_dtype_and_metadata_changes(self):
        with patch.object(loader, "_checkpoint_contract", return_value={"a": (2,)}):
            self.assertEqual(loader._validate_checkpoint_header(FakeCheckpoint()), ({"a"}, "final"))
            with self.assertRaisesRegex(ValueError, "key mismatch"):
                loader._validate_checkpoint_header(FakeCheckpoint(keys=("b",)))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                loader._validate_checkpoint_header(FakeCheckpoint(shape=(3,)))
            with self.assertRaisesRegex(ValueError, "dtype mismatch"):
                loader._validate_checkpoint_header(FakeCheckpoint(dtype="F32"))
            with self.assertRaisesRegex(ValueError, "revision"):
                loader._validate_checkpoint_header(FakeCheckpoint(metadata={
                    "format": loader.MODEL_FORMAT,
                    "source_repo": loader.MODEL_REPO,
                    "config_sha256": loader.CONFIG_SHA256,
                    "source_revision": "wrong",
                }))
            with self.assertRaisesRegex(ValueError, "supported Final or SFT"):
                loader._validate_checkpoint_header(FakeCheckpoint(metadata={
                    "format": loader.MODEL_FORMAT,
                    "source_repo": "sensenova/SenseNova-U1.5-8B-MoT-Preview",
                    "config_sha256": loader.CONFIG_SHA256,
                    "source_revision": loader.MODEL_REVISION,
                }))

    def test_sft_metadata_and_all_bf16_storage_are_supported(self):
        metadata = {
            "format": loader.MODEL_FORMAT,
            "source_repo": loader.SFT_MODEL_REPO,
            "config_sha256": loader.CONFIG_SHA256,
            "source_revision": loader.SFT_MODEL_REVISION,
        }
        f32_in_final = "fm_modules.timestep_embedder.mlp.0.weight"
        with patch.object(loader, "_checkpoint_contract", return_value={f32_in_final: (2,)}):
            self.assertEqual(
                loader._validate_checkpoint_header(
                    FakeCheckpoint(keys=(f32_in_final,), dtype="BF16", metadata=metadata)
                ),
                ({f32_in_final}, "sft"),
            )
            with self.assertRaisesRegex(ValueError, "revision"):
                loader._validate_checkpoint_header(FakeCheckpoint(
                    keys=(f32_in_final,),
                    dtype="BF16",
                    metadata={**metadata, "source_revision": loader.MODEL_REVISION},
                ))

    def test_final_mixed_precision_contract_is_unchanged(self):
        f32_in_final = "fm_modules.timestep_embedder.mlp.0.weight"
        with patch.object(loader, "_checkpoint_contract", return_value={f32_in_final: (2,)}):
            self.assertEqual(
                loader._validate_checkpoint_header(
                    FakeCheckpoint(keys=(f32_in_final,), dtype="F32")
                ),
                ({f32_in_final}, "final"),
            )

    def test_loader_rejects_non_safetensors_before_reading(self):
        with self.assertRaisesRegex(ValueError, "safetensors"):
            loader.load_sensenova_model("model.ckpt", torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
