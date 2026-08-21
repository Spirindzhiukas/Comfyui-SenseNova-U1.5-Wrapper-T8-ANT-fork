import json
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ExampleWorkflowTests(unittest.TestCase):
    def load_example(self, name):
        return json.loads((PACKAGE_ROOT / "examples" / name).read_text(encoding="utf-8"))

    def test_examples_are_frontend_workflows(self):
        examples = sorted((PACKAGE_ROOT / "examples").glob("*.json"))
        self.assertEqual([path.name for path in examples], [
            "edit_workflow.json",
            "multi_reference_edit_workflow.json",
            "t2i_workflow.json",
        ])
        for path in examples:
            workflow = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(name=path.name):
                self.assertIsInstance(workflow.get("nodes"), list)
                self.assertIsInstance(workflow.get("links"), list)
                self.assertIn("version", workflow)

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
        self.assertEqual([value["name"] for value in edit_reference["inputs"]], ["positive", "negative", "images.image"])

        multi = self.load_example("multi_reference_edit_workflow.json")
        multi_reference = next(node for node in multi["nodes"] if node["type"] == "SenseNovaReferenceImage")
        self.assertEqual([value["name"] for value in multi_reference["inputs"][-2:]], ["images.image", "images.image_2"])
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
