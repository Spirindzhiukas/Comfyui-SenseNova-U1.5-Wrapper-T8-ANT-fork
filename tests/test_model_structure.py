import json
import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import comfy.ops

from sensenova_u15.model import SenseNovaU15


class ModelStructureTests(unittest.TestCase):
    def test_state_dict_matches_single_file_manifest(self):
        manifest_path = COMFY_ROOT / "models" / "diffusion_models" / "SenseNova-U1.5-8B-MoT.safetensors.manifest.json"
        if not manifest_path.exists():
            self.skipTest("converted model manifest is not present")

        model = SenseNovaU15(
            device=torch.device("meta"),
            dtype=torch.bfloat16,
            operations=comfy.ops.disable_weight_init,
        )
        self.assertIs(model.dtype, torch.bfloat16)
        actual = model.state_dict()
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))["tensors"]
        expected.pop("language_model.lm_head.weight")

        self.assertEqual(set(actual), set(expected))
        for name, tensor in actual.items():
            self.assertEqual(list(tensor.shape), expected[name]["shape"], name)


if __name__ == "__main__":
    unittest.main()
