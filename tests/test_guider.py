import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from sensenova_u15.guidance import edit_guidance


class EditGuiderTests(unittest.TestCase):
    def test_three_branch_formula(self):
        positive = torch.tensor([3.0, 5.0])
        image = torch.tensor([2.0, 1.0])
        negative = torch.tensor([-1.0, 0.0])
        actual = edit_guidance(positive, image, negative, cfg=4.0, img_cfg=1.5)
        expected = negative + 4.0 * (positive - image) + 1.5 * (image - negative)
        torch.testing.assert_close(actual, expected)

    def test_img_cfg_one_reduces_to_two_branches(self):
        positive = torch.randn(4)
        image = torch.randn(4)
        negative_a = torch.randn(4)
        negative_b = torch.randn(4)
        first = edit_guidance(positive, image, negative_a, cfg=4.0, img_cfg=1.0)
        second = edit_guidance(positive, image, negative_b, cfg=4.0, img_cfg=1.0)
        torch.testing.assert_close(first, image + 4.0 * (positive - image))
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
