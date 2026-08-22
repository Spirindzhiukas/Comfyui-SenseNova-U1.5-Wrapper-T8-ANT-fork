import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import comfy.model_sampling
import comfy.samplers

from sensenova_u15.sampling import (
    SenseNovaModelSampling,
    inverse_time_snr_shift,
    resolution_noise_scale,
    time_snr_shift,
    upstream_sigmas,
    upstream_timesteps,
)


class SamplingMathTests(unittest.TestCase):
    def test_upstream_schedule_matches_direct_formula(self):
        for steps in (1, 2, 8, 50):
            for shift in (1.0, 3.0):
                base = torch.linspace(0.0, 1.0, steps + 1)
                expected = 1.0 - shift * (1.0 - base) / (1.0 + (shift - 1.0) * (1.0 - base))
                torch.testing.assert_close(upstream_timesteps(steps, shift), expected, rtol=0, atol=0)
                torch.testing.assert_close(upstream_sigmas(steps, shift), 1.0 - expected, rtol=0, atol=0)

    def test_shift_inverse_round_trip(self):
        values = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
        for shift in (1.0, 3.0, 7.0):
            shifted = time_snr_shift(shift, values)
            torch.testing.assert_close(inverse_time_snr_shift(shift, shifted), values, rtol=1e-14, atol=1e-14)

    def test_normal_scheduler_is_exact_upstream_schedule(self):
        sampling = SenseNovaModelSampling(None)
        for shift in (1.0, 3.0):
            sampling.set_parameters(shift=shift)
            for steps in (1, 2, 8, 50):
                actual = comfy.samplers.normal_scheduler(sampling, steps)
                expected = upstream_sigmas(steps, shift)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)

    def test_resolution_noise_scale(self):
        self.assertEqual(resolution_noise_scale(2048, 2048), 8.0)
        self.assertEqual(resolution_noise_scale(4096, 4096), 16.0)
        self.assertEqual(resolution_noise_scale(8192, 8192), 16.0)
        self.assertEqual(resolution_noise_scale(2049, 2049), 8.125)

    def test_noise_scaling_uses_padded_token_grid(self):
        sampling = SenseNovaModelSampling(None)
        noise = torch.ones((1, 3, 33, 65))
        latent = torch.zeros_like(noise)
        actual = sampling.noise_scaling(torch.tensor(1.0), noise, latent)
        expected_scale = resolution_noise_scale(33, 65)
        torch.testing.assert_close(actual, torch.full_like(noise, expected_scale))

    def test_comfy_euler_requires_negative_velocity(self):
        image = torch.tensor([2.0, -1.0])
        velocity = torch.tensor([0.25, -0.5])
        t = 0.2
        t_next = 0.7
        sigma = torch.tensor(1.0 - t)
        sigma_next = torch.tensor(1.0 - t_next)
        denoised = comfy.model_sampling.CONST().calculate_denoised(sigma, -velocity, image)
        derivative = (image - denoised) / sigma
        comfy_euler = image + derivative * (sigma_next - sigma)
        upstream_euler = image + (t_next - t) * velocity
        torch.testing.assert_close(comfy_euler, upstream_euler)


if __name__ == "__main__":
    unittest.main()
