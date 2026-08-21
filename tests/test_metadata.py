import json
import re
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def test_registry_metadata(self):
        metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["name"], "sensenova-u15-t8")
        self.assertEqual(metadata["tool"]["comfy"]["PublisherId"], "t8star")
        self.assertEqual(metadata["tool"]["comfy"]["DisplayName"], "SenseNova U1.5 (T8)")
        self.assertTrue(metadata["project"]["urls"]["Model Download"].startswith("https://huggingface.co/t8star/"))

    def test_all_examples_are_valid_json(self):
        examples = sorted((PACKAGE_ROOT / "examples").glob("*.json"))
        self.assertGreaterEqual(len(examples), 3)
        for path in examples:
            with self.subTest(path=path.name):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_registry_package_excludes_local_and_large_files(self):
        patterns = set((PACKAGE_ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines())
        self.assertTrue({"roadmap.md", "*.safetensors", "oracles/", "tools/"}.issubset(patterns))

    def test_readme_local_links_and_visuals_exist(self):
        readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        local_links = [value for value in re.findall(r"\]\(([^)]+)\)", readme) if "://" not in value]
        for value in local_links:
            with self.subTest(value=value):
                self.assertTrue((PACKAGE_ROOT / value).is_file())
        for name in (
            "t2i-workflow.jpg",
            "edit-workflow.jpg",
            "multi-reference-edit-workflow.jpg",
            "result-t2i-2048.png",
            "result-multi-reference-2048.png",
        ):
            with self.subTest(name=name):
                self.assertGreater((PACKAGE_ROOT / "docs" / "images" / name).stat().st_size, 1024)


if __name__ == "__main__":
    unittest.main()
