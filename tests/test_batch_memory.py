import sys
import unittest
from pathlib import Path


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from sensenova_u15.model_config import scale_batched_condition_shapes


class BatchMemoryEstimateTests(unittest.TestCase):
    def test_full_estimate_scales_each_reference_branch_to_generation_batch(self):
        shapes = {
            "reference_images": [[1, 3, 4096], [1, 3, 4096]],
            "prefix_mask": [[1, 1, 100, 100], [1, 1, 100, 100]],
        }
        actual = scale_batched_condition_shapes([8, 3, 512, 512], shapes)
        self.assertEqual(actual["reference_images"], [[4, 3, 4096], [4, 3, 4096]])
        self.assertEqual(actual["prefix_mask"], [[4, 1, 100, 100], [4, 1, 100, 100]])

    def test_minimum_estimate_uses_the_real_latent_batch(self):
        shapes = {
            "reference_images": [[1, 3, 4096]],
            "prefix_mask": [[1, 1, 100, 100]],
        }
        actual = scale_batched_condition_shapes([4, 3, 512, 512], shapes)
        self.assertEqual(actual["reference_images"][0][0], 4)
        self.assertEqual(actual["prefix_mask"][0][0], 4)


if __name__ == "__main__":
    unittest.main()
