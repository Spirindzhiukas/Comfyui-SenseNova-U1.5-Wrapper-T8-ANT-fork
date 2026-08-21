import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from sensenova_u15.model import _apply_interleaved_rope, _apply_llm_rope


def rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RopeTests(unittest.TestCase):
    def test_shared_split_half_rope_matches_upstream_math(self):
        generator = torch.Generator().manual_seed(0)
        query = torch.randn((2, 4, 7, 64), dtype=torch.bfloat16, generator=generator)
        key = torch.randn((2, 2, 7, 64), dtype=torch.bfloat16, generator=generator)
        positions = torch.stack((torch.arange(7), torch.arange(7) + 3))
        frequencies = 5000000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
        angles = positions.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cosine = embedding.cos().to(query.dtype)
        sine = embedding.sin().to(query.dtype)
        expected_query = query * cosine + rotate_half(query) * sine
        expected_key = key * cosine + rotate_half(key) * sine

        actual_query, actual_key = _apply_llm_rope(query, key, positions, 5000000.0)
        torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)

    def test_shared_interleaved_rope_matches_upstream_math(self):
        value = torch.randn((2, 7, 32), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1))
        positions = torch.arange(7)
        frequencies = 10000.0 ** (-torch.arange(0, 32, 2, dtype=torch.float32) / 32)
        angles = positions.float().unsqueeze(-1) * frequencies
        cosine = angles.cos().unsqueeze(0)
        sine = angles.sin().unsqueeze(0)
        expected = torch.empty_like(value, dtype=torch.float32)
        value_float = value.float()
        expected[..., 0::2] = value_float[..., 0::2] * cosine - value_float[..., 1::2] * sine
        expected[..., 1::2] = value_float[..., 0::2] * sine + value_float[..., 1::2] * cosine

        actual = _apply_interleaved_rope(value, positions, 10000.0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
