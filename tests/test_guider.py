import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from sensenova_u15.guidance import build_structured_edit_prompt, edit_guidance, rescale_denoised_guidance, rescale_guidance


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

    def test_global_cfg_norm_only_reduces_over_guidance(self):
        positive = torch.tensor([[3.0, 4.0]])
        over_guided = torch.tensor([[6.0, 8.0]])
        under_guided = torch.tensor([[1.5, 2.0]])
        torch.testing.assert_close(rescale_guidance(over_guided, positive, "global"), positive)
        torch.testing.assert_close(rescale_guidance(under_guided, positive, "global"), under_guided)

    def test_channel_cfg_norm_matches_32_pixel_generation_tokens(self):
        positive = torch.ones((1, 3, 32, 64))
        guided = positive.clone()
        guided[..., :32] *= 2
        guided[..., 32:] *= 0.5
        actual = rescale_guidance(guided, positive, "channel")
        torch.testing.assert_close(actual[..., :32], positive[..., :32])
        torch.testing.assert_close(actual[..., 32:], guided[..., 32:])

    def test_denoised_cfg_norm_is_applied_in_flow_velocity_space(self):
        latent = torch.zeros((1, 1, 1, 2))
        sigma = torch.tensor([0.5])
        positive_velocity = torch.tensor([[[[3.0, 4.0]]]])
        guided_velocity = positive_velocity * 2
        positive_denoised = latent - sigma.reshape(1, 1, 1, 1) * positive_velocity
        guided_denoised = latent - sigma.reshape(1, 1, 1, 1) * guided_velocity
        actual = rescale_denoised_guidance(
            guided_denoised,
            positive_denoised,
            latent,
            sigma,
            mode="global",
        )
        torch.testing.assert_close(actual, positive_denoised)

    def test_batch_two_cfg_norm_matches_official_velocity_formula(self):
        generator = torch.Generator().manual_seed(20260822)
        latent = torch.randn((2, 3, 32, 64), generator=generator)
        positive_velocity = torch.randn((2, 3, 32, 64), generator=generator)
        guided_velocity = positive_velocity * 1.7 + torch.randn(
            (2, 3, 32, 64), generator=generator
        ) * 0.1
        sigma = torch.tensor([0.9, 0.35]).reshape(2, 1, 1, 1)
        positive_denoised = latent + sigma * positive_velocity
        guided_denoised = latent + sigma * guided_velocity

        for mode in ("global", "channel"):
            with self.subTest(mode=mode):
                expected_velocity = rescale_guidance(guided_velocity, positive_velocity, mode)
                expected = latent + sigma * expected_velocity
                actual = rescale_denoised_guidance(
                    guided_denoised,
                    positive_denoised,
                    latent,
                    sigma.flatten(),
                    mode=mode,
                )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_structured_edit_prompt_makes_roles_and_preservation_explicit(self):
        actual = build_structured_edit_prompt(
            "让人物穿上 Image-2 的外套",
            "Image-1 是人物主图；Image-2 只提供服装。",
            "保持 Image-1 的脸、姿势和背景。",
            "不要复制 Image-2 的人物。",
        )
        # English section titles in this fork; upstream uses
        # 【主要修改】/【参考图职责】/【必须保持】/【执行要求】.
        self.assertIn("[Main Edit]", actual)
        self.assertIn("[Reference Image Roles]", actual)
        self.assertIn("[Must Preserve]", actual)
        self.assertIn("[Must Avoid]", actual)
        self.assertIn("Only modify what is explicitly requested above", actual)

    def test_structured_edit_prompt_rejects_empty_instruction(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            build_structured_edit_prompt("  ")


if __name__ == "__main__":
    unittest.main()
