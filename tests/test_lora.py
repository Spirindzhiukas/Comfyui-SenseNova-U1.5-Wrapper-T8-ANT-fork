import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import sensenova_u15.lora as lora


class FakeModel:
    def __init__(self, variant, *, repo=lora.MODEL_REPO, revision=lora.MODEL_REVISION, applied=False):
        self.variant = variant
        self.repo = repo
        self.revision = revision
        self.applied = applied

    def get_attachment(self, key):
        if key == "sensenova_checkpoint":
            return {"variant": self.variant, "source_repo": self.repo, "source_revision": self.revision}
        if key == "sensenova_lora" and self.applied:
            return {"type": "8step"}
        return None


class EightStepLoRATests(unittest.TestCase):
    def test_final_only_guard_runs_even_when_strength_is_zero(self):
        final = FakeModel("final")
        self.assertIs(lora.apply_eight_step_lora(final, "unused.safetensors", 0), final)
        with self.assertRaisesRegex(ValueError, "fixed official Final checkpoint"):
            lora.apply_eight_step_lora(FakeModel("sft"), "unused.safetensors", 0)

    def test_final_guard_checks_revision_and_rejects_double_application(self):
        with self.assertRaisesRegex(ValueError, "fixed official Final checkpoint"):
            lora.apply_eight_step_lora(FakeModel("final", revision="wrong"), "unused.safetensors", 0)
        with self.assertRaisesRegex(ValueError, "already applied"):
            lora.apply_eight_step_lora(FakeModel("final", applied=True), "unused.safetensors", 0)

    def test_metadata_is_pinned_to_official_u15_lora(self):
        metadata = {
            "tensor_kind": "neo_hf_lora",
            "comfyui_format": "model_lora",
            "comfyui_key_prefix": lora.LORA_PREFIX,
            "source_repo": lora.LORA_REPO,
            "source_revision": lora.LORA_REVISION,
            "source_sha256": lora.LORA_SOURCE_SHA256,
            "conversion": "raw-data-key-prefix-v1",
        }
        lora._validate_lora_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "verified official"):
            lora._validate_lora_metadata({**metadata, "source_revision": "wrong"})

    def test_converted_file_hash_is_verified_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.safetensors"
            path.write_bytes(b"verified")
            lora._VERIFIED_LORA_FILES.clear()
            with patch.object(lora, "LORA_COMFY_SIZE", path.stat().st_size), patch.object(
                lora, "LORA_COMFY_SHA256", lora._sha256_file(path)
            ), patch.object(lora, "_sha256_file", wraps=lora._sha256_file) as digest:
                lora._validate_lora_file(path)
                lora._validate_lora_file(path)
                self.assertEqual(digest.call_count, 1)
            path.write_bytes(b"changed")
            with patch.object(lora, "LORA_COMFY_SIZE", path.stat().st_size), self.assertRaisesRegex(
                ValueError, "size/SHA256"
            ):
                lora._validate_lora_file(path)


if __name__ == "__main__":
    unittest.main()
