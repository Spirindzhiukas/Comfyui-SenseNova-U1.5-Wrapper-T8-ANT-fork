import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import comfy.ops

from sensenova_u15.model import SenseNovaU15
from sensenova_u15.loader import _checkpoint_contract


class ModelStructureTests(unittest.TestCase):
    def test_state_dict_always_matches_bundled_checkpoint_contract(self):
        model = SenseNovaU15(
            device=torch.device("meta"),
            dtype=torch.bfloat16,
            operations=comfy.ops.disable_weight_init,
        )
        self.assertIs(model.dtype, torch.bfloat16)
        actual = model.state_dict()
        expected = {
            name: shape
            for name, (shape, _dtype) in _checkpoint_contract("final").items()
            if name != "language_model.lm_head.weight"
        }
        self.assertEqual(set(actual), set(expected))
        for name, tensor in actual.items():
            self.assertEqual(tuple(tensor.shape), expected[name], name)


if __name__ == "__main__":
    unittest.main()
