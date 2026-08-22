import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from sensenova_u15.model import _expand_prefix_batch, _generation_batch_size
from sensenova_u15.model_config import CONDSharedList, CONDSharedRegular


class SharedPrefixConditionTests(unittest.TestCase):
    def test_regular_condition_is_not_repeated_to_generation_batch(self):
        tensor = torch.arange(6).reshape(1, 2, 3)
        processed = CONDSharedRegular(tensor).process_cond(batch_size=4)
        self.assertEqual(tuple(processed.cond.shape), (1, 2, 3))
        self.assertIs(processed.cond, tensor)

    def test_reference_list_is_not_repeated_to_generation_batch(self):
        image = torch.zeros((1, 3, 32, 32))
        processed = CONDSharedList([image]).process_cond(batch_size=4)
        self.assertEqual(tuple(processed.cond[0].shape), (1, 3, 32, 32))
        self.assertIs(processed.cond[0], image)

    def test_guidance_branches_still_concat_in_branch_order(self):
        first = CONDSharedRegular(torch.tensor([[1.0]]))
        second = CONDSharedRegular(torch.tensor([[2.0]]))
        self.assertTrue(first.can_concat(second))
        self.assertEqual(first.concat([second]).tolist(), [[1.0], [2.0]])


class PrefixKVExpansionTests(unittest.TestCase):
    def test_batch_multiplier_is_derived_from_prefix_branches(self):
        self.assertEqual(_generation_batch_size(total_batch=6, prefix_batch=2), 3)
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            _generation_batch_size(total_batch=5, prefix_batch=2)

    def test_each_branch_is_repeated_for_its_generated_variants(self):
        values = torch.tensor([[[[1.0]]], [[[2.0]]]])
        expanded = _expand_prefix_batch(values, generation_batch=3)
        self.assertEqual(tuple(expanded.shape), (6, 1, 1, 1))
        self.assertEqual(expanded.flatten().tolist(), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def test_single_variant_returns_original_tensor(self):
        values = torch.zeros((2, 1, 4, 8))
        self.assertIs(_expand_prefix_batch(values, generation_batch=1), values)


if __name__ == "__main__":
    unittest.main()
