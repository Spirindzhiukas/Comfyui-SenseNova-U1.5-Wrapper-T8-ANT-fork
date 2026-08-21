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

    def test_frontend_workflows_have_resolved_links(self):
        for name in ("t2i_workflow.json", "edit_workflow.json", "multi_reference_edit_workflow.json"):
            workflow = self.load_example(name)
            nodes = {node["id"]: node for node in workflow["nodes"]}
            links = {link[0]: link for link in workflow["links"]}
            with self.subTest(name=name):
                self.assertEqual(workflow["last_node_id"], max(nodes))
                self.assertEqual(workflow["last_link_id"], max(links))
                for link_id, (_, origin_id, origin_slot, target_id, target_slot, link_type) in links.items():
                    self.assertEqual(nodes[origin_id]["outputs"][origin_slot]["type"], link_type, link_id)
                    self.assertEqual(nodes[target_id]["inputs"][target_slot]["type"], link_type, link_id)
                    self.assertEqual(nodes[target_id]["inputs"][target_slot]["link"], link_id, link_id)

    def test_frontend_t2i_uses_official_defaults(self):
        workflow = self.load_example("t2i_workflow.json")
        sampler = next(node for node in workflow["nodes"] if node["type"] == "KSampler")
        self.assertEqual(sampler["widgets_values"][2:], [50, 4, "euler", "normal", 1])

    def test_frontend_edit_and_multi_reference_contracts(self):
        edit = self.load_example("edit_workflow.json")
        edit_reference = next(node for node in edit["nodes"] if node["type"] == "SenseNovaReferenceImage")
        self.assertEqual([value["name"] for value in edit_reference["inputs"]], ["positive", "negative", "image"])

        multi = self.load_example("multi_reference_edit_workflow.json")
        multi_reference = next(node for node in multi["nodes"] if node["type"] == "SenseNovaReferenceImage")
        self.assertEqual([value["name"] for value in multi_reference["inputs"][-2:]], ["image", "image_2"])
        guider = next(node for node in multi["nodes"] if node["type"] == "SenseNovaEditGuider")
        scheduler = next(node for node in multi["nodes"] if node["type"] == "BasicScheduler")
        self.assertEqual(guider["widgets_values"], [4, 2])
        self.assertEqual(scheduler["widgets_values"], ["normal", 50, 1])
        links = {link[0]: link for link in multi["links"]}
        guider_model = links[guider["inputs"][0]["link"]][1:3]
        scheduler_model = links[scheduler["inputs"][0]["link"]][1:3]
        self.assertEqual(guider_model, scheduler_model)


if __name__ == "__main__":
    unittest.main()
