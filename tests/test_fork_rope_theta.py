"""Per-axis RoPE base thetas: defaults, overrides, and the call-site plumbing.

The fork keeps upstream's pure-PyTorch RoPE (Blackwell-safe, see CHANGELOG
1.3.4) but reads the three bases through ``transformer_options`` so an external
context-scaling suite (ANT RoPE_Lab: NTK/YaRN/SEGA style methods) can rescale a
single sampling pass without monkey-patching ``sensenova_u15.model``.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import torch

import sensenova_u15.model as model_module
from sensenova_u15.model import (
    HEAD_DIM,
    NUM_HEADS,
    NUM_KV_HEADS,
    ROPE_THETA_SPATIAL,
    ROPE_THETA_TIME,
    ROPE_THETA_VISION,
    Attention,
    _apply_llm_rope,
    resolve_rope_thetas,
)


class RopeThetaDefaultsTests(unittest.TestCase):
    def test_official_bases_are_the_defaults(self):
        self.assertEqual((ROPE_THETA_TIME, ROPE_THETA_SPATIAL, ROPE_THETA_VISION), (5000000.0, 10000.0, 10000.0))
        self.assertEqual(resolve_rope_thetas(), (5000000.0, 10000.0, 10000.0))
        self.assertEqual(resolve_rope_thetas({}), (5000000.0, 10000.0, 10000.0))
        self.assertEqual(resolve_rope_thetas(None), (5000000.0, 10000.0, 10000.0))

    def test_each_axis_can_be_scaled_independently(self):
        options = {
            "sensenova_rope_theta_t": 1e7,
            "sensenova_rope_theta_hw": 2e4,
            "sensenova_rope_theta_vision": 4e4,
        }
        self.assertEqual(resolve_rope_thetas(options), (10000000.0, 20000.0, 40000.0))
        self.assertEqual(resolve_rope_thetas({"sensenova_rope_theta_hw": 2e4}), (5000000.0, 20000.0, 10000.0))

    def test_tensors_and_exact_scales_are_accepted(self):
        self.assertEqual(resolve_rope_thetas({"sensenova_rope_theta_t": torch.tensor(2.5)}), (2.5, 10000.0, 10000.0))

    def test_nonsense_scales_fail_loudly(self):
        for key in ("sensenova_rope_theta_t", "sensenova_rope_theta_hw", "sensenova_rope_theta_vision"):
            for value in (0, -1.0, float("nan"), float("inf"), "10k", None, object()):
                with self.subTest(key=key, value=type(value).__name__), self.assertRaises(ValueError):
                    resolve_rope_thetas({key: value})


class RopeThetaPlumbingTests(unittest.TestCase):
    def test_attention_projects_with_the_configured_bases(self):
        """`Attention._project` must feed the per-axis bases into every rope call."""
        length = 4
        hidden = torch.zeros(1, length, NUM_HEADS * HEAD_DIM)
        indexes = torch.zeros(3, length, dtype=torch.long)

        class _Projection:
            def __init__(self, features):
                self.features = features

            def __call__(self, value):
                out = torch.zeros(value.shape[0], value.shape[1], self.features, dtype=value.dtype)
                return out.view(value.shape[0], value.shape[1], self.features // HEAD_DIM, HEAD_DIM)

        class _Identity:
            def __call__(self, value):
                return value

        class _FakeAttention:
            generation = False

            def __init__(self):
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    features = NUM_HEADS * HEAD_DIM if name.startswith("q_") else NUM_KV_HEADS * HEAD_DIM
                    setattr(self, name, _Projection(features))
                for name in ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen"):
                    features = NUM_HEADS * HEAD_DIM if name.startswith("q_") else NUM_KV_HEADS * HEAD_DIM
                    setattr(self, name, _Projection(features))
                for suffix in ("q_norm", "k_norm", "q_norm_hw", "k_norm_hw"):
                    for name in (suffix, f"{suffix}_mot_gen"):
                        setattr(self, name, _Identity())

        captured = []
        original = model_module._apply_llm_rope

        def record(query, key, positions, theta):
            captured.append(theta)
            return original(query, key, positions, theta)

        with mock.patch.object(model_module, "_apply_llm_rope", side_effect=record):
            Attention._project(_FakeAttention(), hidden, indexes, False, {"sensenova_rope_theta_t": 1234.0})
        self.assertEqual(captured, [1234.0, 10000.0, 10000.0])

        captured.clear()
        with mock.patch.object(model_module, "_apply_llm_rope", side_effect=record):
            Attention._project(
                _FakeAttention(),
                hidden,
                indexes,
                True,
                {"sensenova_rope_theta_t": 7.0, "sensenova_rope_theta_hw": 11.0},
            )
        self.assertEqual(captured, [7.0, 11.0, 11.0])

        captured.clear()
        with mock.patch.object(model_module, "_apply_llm_rope", side_effect=record):
            Attention._project(_FakeAttention(), hidden, indexes, False)
        self.assertEqual(captured, [5000000.0, 10000.0, 10000.0])

    def test_vision_embeddings_forward_takes_transformer_options(self):
        import inspect

        from sensenova_u15.model import VisionEmbeddings, VisionModel

        self.assertIn("transformer_options", inspect.signature(VisionEmbeddings.forward).parameters)
        self.assertIn("transformer_options", inspect.signature(VisionModel.forward).parameters)

    def test_rope_output_actually_depends_on_the_base(self):
        query = torch.randn(1, 2, 5, 64, generator=torch.Generator().manual_seed(0))
        key = torch.randn(1, 2, 5, 64, generator=torch.Generator().manual_seed(1))
        positions = torch.arange(5)
        base = _apply_llm_rope(query, key, positions, 5000000.0)[0]
        scaled = _apply_llm_rope(query, key, positions, 5000000.0 * 4.0)[0]
        self.assertFalse(torch.allclose(base, scaled))
        # an explicit default must be byte identical to the option-free default
        self.assertTrue(
            torch.equal(base, _apply_llm_rope(query, key, positions, resolve_rope_thetas()[0])[0])
        )

    def test_prefix_cache_key_includes_the_active_bases(self):
        source = (PACKAGE_ROOT / "sensenova_u15" / "model.py").read_text(encoding="utf-8")
        self.assertIn("rope_thetas = resolve_rope_thetas(transformer_options)", source)
        self.assertIn("rope_thetas,", source)

    def test_call_sites_pass_transformer_options(self):
        source = (PACKAGE_ROOT / "sensenova_u15" / "model.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("indexes, False, transformer_options"), 2)
        self.assertIn("indexes, True, transformer_options", source)
        self.assertIn('self.fm_modules["vision_model_mot_gen"](', source)
        self.assertIn("transformer_options=transformer_options", source)

    def test_no_kitchen_rope_kernel_is_used(self):
        source = (PACKAGE_ROOT / "sensenova_u15" / "model.py").read_text(encoding="utf-8")
        self.assertNotIn("quant_ops", source)
        self.assertNotIn("apply_rope_split_half", source)
        self.assertIn("torch.cat((-second, first), dim=-1)", source)


if __name__ == "__main__":
    unittest.main()
