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
        self.assertEqual(metadata["project"]["version"], "1.3.3")
        self.assertEqual(metadata["tool"]["comfy"]["PublisherId"], "t8star")
        self.assertEqual(metadata["tool"]["comfy"]["DisplayName"], "SenseNova U1.5 (T8)")
        self.assertTrue(metadata["project"]["urls"]["Model Download"].startswith("https://huggingface.co/t8star/"))
        self.assertEqual(
            metadata["project"]["urls"]["Repository"],
            "https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8",
        )
        self.assertEqual(
            metadata["project"]["urls"]["Homepage"],
            "https://registry.comfy.org/nodes/sensenova-u15-t8",
        )

    def test_frontend_extension_is_packaged(self):
        extension = (PACKAGE_ROOT / "web" / "sensenova_reference_labels_v131e.js").read_text(encoding="utf-8")
        self.assertIn("SenseNovaReferenceImage", extension)
        self.assertIn("migrateLegacyReferenceInputs", extension)
        self.assertIn('WEB_DIRECTORY = "./web"', (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8"))
        self.assertGreater((PACKAGE_ROOT / "sensenova_u15" / "checkpoint_contract.json").stat().st_size, 100_000)

    def test_all_examples_are_valid_json(self):
        examples = sorted((PACKAGE_ROOT / "examples").glob("*.json"))
        self.assertGreaterEqual(len(examples), 3)
        for path in examples:
            with self.subTest(path=path.name):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_registry_package_excludes_local_and_large_files(self):
        patterns = set((PACKAGE_ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines())
        self.assertTrue({"roadmap.md", "*.safetensors", "oracles/", "tools/"}.issubset(patterns))
        self.assertFalse(any(pattern in {"*.json", "sensenova_u15/", "checkpoint_contract.json"} for pattern in patterns))

    def test_readme_local_links_and_visuals_exist(self):
        for readme_name in ("README.md", "README_EN.md"):
            readme = (PACKAGE_ROOT / readme_name).read_text(encoding="utf-8")
            local_links = [value for value in re.findall(r"\]\(([^)]+)\)", readme) if "://" not in value]
            for value in local_links:
                with self.subTest(readme=readme_name, value=value):
                    self.assertTrue((PACKAGE_ROOT / value).is_file())
        for name in (
            "t2i-workflow.jpg",
            "edit-workflow.jpg",
            "multi-reference-edit-workflow.jpg",
            "result-sft-t2i-2048.png",
            "result-t2i-8step-2048.png",
            "result-t2i-2048.png",
            "result-multi-reference-2048.png",
            "result-garment-edit-2048.png",
        ):
            with self.subTest(name=name):
                self.assertGreater((PACKAGE_ROOT / "docs" / "images" / name).stat().st_size, 1024)


if __name__ == "__main__":
    unittest.main()
