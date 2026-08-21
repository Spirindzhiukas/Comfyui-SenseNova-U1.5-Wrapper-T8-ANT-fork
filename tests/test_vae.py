import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from sensenova_u15.loader import load_pixel_vae


class PixelVaeTests(unittest.TestCase):
    def test_pixel_vae_round_trip_and_scale(self):
        pixels = torch.linspace(0.0, 1.0, 60).reshape(1, 4, 5, 3)
        vae = load_pixel_vae()
        latent = vae.encode(pixels)
        expected_latent = (pixels.movedim(-1, 1) * 2.0 - 1.0).to(vae.vae_dtype).to(latent.dtype)
        torch.testing.assert_close(latent, expected_latent, rtol=0, atol=0)
        expected_pixels = expected_latent.float().add(1.0).div(2.0).movedim(1, -1)
        torch.testing.assert_close(vae.decode(latent), expected_pixels, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
