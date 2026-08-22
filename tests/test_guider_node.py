import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

spec = importlib.util.spec_from_file_location(
    "comfyui_sensenova_u15_t8_tests",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

from comfyui_sensenova_u15_t8_tests.nodes import EmptySenseNovaLatentImage, SenseNovaEditGuiderImpl, SenseNovaExtension, SenseNovaReferenceImage, SenseNovaU15EightStepLoRA, _prefix_cache_sample_wrapper


class EditGuiderNodeTests(unittest.TestCase):
    def setUp(self):
        self.positive = object()
        self.image = object()
        self.negative = object()
        self.outputs = {
            id(self.positive): torch.tensor(3.0),
            id(self.image): torch.tensor(2.0),
            id(self.negative): torch.tensor(-1.0),
        }
        self.guider = object.__new__(SenseNovaEditGuiderImpl)
        self.guider.inner_model = object()
        self.guider.conds = {
            "positive": self.positive,
            "image_condition": self.image,
            "negative": self.negative,
        }

    def run_case(self, cfg, img_cfg, expected, expected_branches):
        self.guider.set_cfg(cfg, img_cfg)

        def calc_cond_batch(_model, conds, *_args, **_kwargs):
            return [self.outputs[id(cond)] for cond in conds]

        with patch("comfy.samplers.calc_cond_batch", side_effect=calc_cond_batch) as mocked:
            actual = self.guider.predict_noise(torch.zeros(1), torch.zeros(1))
        torch.testing.assert_close(actual, torch.tensor(expected))
        self.assertEqual(mocked.call_args.args[1], expected_branches)

    def test_one_two_and_three_branch_shortcuts(self):
        self.run_case(1.0, 1.0, 3.0, [self.positive])
        self.run_case(4.0, 1.0, 6.0, [self.image, self.positive])
        self.run_case(2.0, 2.0, 7.0, [self.negative, self.positive])
        self.run_case(4.0, 2.0, 9.0, [self.negative, self.image, self.positive])

    def test_reference_node_rejects_image_batch_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "requires one IMAGE"):
            SenseNovaReferenceImage.execute(
                positive=[],
                negative=[],
                images={"image": torch.zeros((2, 16, 16, 3))},
            )

    def test_reference_node_accepts_multiple_named_images(self):
        image_1 = torch.zeros((1, 16, 16, 3))
        image_2 = torch.ones((1, 16, 16, 3))
        with patch("node_helpers.conditioning_set_values", side_effect=lambda conditioning, values, **_kwargs: values):
            result = SenseNovaReferenceImage.execute(
                positive=[],
                negative=[],
                images={"image": image_1, "image_2": image_2},
            )
        self.assertEqual(result[0]["sensenova_reference_images"], [image_1, image_2])
        self.assertEqual(result[1]["sensenova_reference_images"], [image_1, image_2])

    def test_reference_node_autogrow_inputs_have_readable_names(self):
        images = SenseNovaReferenceImage.define_schema().inputs[2]
        self.assertEqual(images.display_name, "reference images")
        self.assertEqual([value.id for value in images.get_all()[1:3]], ["image", "image_2"])
        self.assertEqual([value.display_name for value in images.get_all()[1:3]], ["reference image", "reference image"])

    def test_prefix_cache_is_execution_local_and_cleared(self):
        original_options = {"transformer_options": {}}
        guider = type("Guider", (), {"model_options": original_options})()
        seen_cache = None

        class Executor:
            class_obj = guider

            def __call__(self, *_args, **_kwargs):
                nonlocal seen_cache
                seen_cache = guider.model_options["transformer_options"]["sensenova_prefix_cache"]
                seen_cache["entry"] = torch.ones(1)
                return "done"

        self.assertEqual(_prefix_cache_sample_wrapper(Executor()), "done")
        self.assertEqual(seen_cache, {})
        self.assertIs(guider.model_options, original_options)

        class FailingExecutor(Executor):
            def __call__(self, *_args, **_kwargs):
                nonlocal seen_cache
                seen_cache = guider.model_options["transformer_options"]["sensenova_prefix_cache"]
                seen_cache["entry"] = torch.ones(1)
                raise RuntimeError("cancelled")

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            _prefix_cache_sample_wrapper(FailingExecutor())
        self.assertEqual(seen_cache, {})
        self.assertIs(guider.model_options, original_options)

    def test_empty_latent_rejects_unvalidated_batches(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            EmptySenseNovaLatentImage.execute(width=64, height=64, batch_size=2)

    def test_extension_has_fixed_node_outputs(self):
        nodes = asyncio.run(SenseNovaExtension().get_node_list())
        actual = {node.define_schema().node_id: len(node.define_schema().outputs) for node in nodes}
        self.assertEqual(
            actual,
            {
                "SenseNovaU15Loader": 3,
                "SenseNovaU15EightStepLoRA": 1,
                "EmptySenseNovaLatentImage": 1,
                "SenseNovaSamplingOptions": 1,
                "SenseNovaReferenceImage": 2,
                "SenseNovaEditGuider": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
