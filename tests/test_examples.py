import json
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ExampleWorkflowTests(unittest.TestCase):
    def load_example(self, name):
        return json.loads((PACKAGE_ROOT / "examples" / name).read_text(encoding="utf-8"))

    def test_t2i_uses_supported_sampler_contract(self):
        workflow = self.load_example("t2i_api.json")
        sampler = workflow["6"]["inputs"]
        self.assertEqual(sampler["model"], ["2", 0])
        self.assertEqual((sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]), ("euler", "normal", 1.0))

    def test_edit_scheduler_and_guider_share_patched_model(self):
        workflow = self.load_example("edit_api.json")
        guider_model = workflow["7"]["inputs"]["model"]
        scheduler_model = workflow["10"]["inputs"]["model"]
        self.assertEqual(guider_model, ["2", 0])
        self.assertEqual(scheduler_model, guider_model)
        self.assertEqual(workflow["9"]["inputs"]["sampler_name"], "euler")
        self.assertEqual((workflow["10"]["inputs"]["scheduler"], workflow["10"]["inputs"]["denoise"]), ("normal", 1.0))

    def test_two_way_edit_uses_image_only_negative(self):
        workflow = self.load_example("edit_two_way_api.json")
        sampler = workflow["8"]["inputs"]
        self.assertEqual(sampler["model"], ["2", 0])
        self.assertEqual(sampler["positive"], ["6", 0])
        self.assertEqual(sampler["negative"], ["6", 1])
        self.assertEqual((sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]), ("euler", "normal", 1.0))


if __name__ == "__main__":
    unittest.main()
