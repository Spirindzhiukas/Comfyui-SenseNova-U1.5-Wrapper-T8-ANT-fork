import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.convert_lora_to_comfy import COMFY_PREFIX, convert_lora


def write_lora(path, tensors, metadata=None):
    header = {"__metadata__": metadata or {"tensor_kind": "neo_hf_lora", "format": "pt"}}
    data = bytearray()
    for name, shape, value in tensors:
        start = len(data)
        data.extend(value)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, len(data)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(data)


def read_safetensors(path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        return header, handle.read()


class ConvertLoraTests(unittest.TestCase):
    def make_source(self, root, prefix=""):
        source = root / "official.safetensors"
        base = f"{prefix}language_model.model.layers.0.self_attn.q_proj_mot_gen"
        write_lora(
            source,
            [
                (f"{base}.alpha", [], b"aa"),
                (f"{base}.lora_down.weight", [1, 2], b"bbbb"),
                (f"{base}.lora_up.weight", [2, 1], b"cccc"),
            ],
        )
        return source

    def test_prefix_conversion_preserves_raw_tensor_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            output = root / "output" / "comfy.safetensors"
            manifest = convert_lora(source, output, "test/repo", "deadbeef", verify_official=False)

            source_header, source_data = read_safetensors(source)
            output_header, output_data = read_safetensors(output)
            source_keys = set(source_header) - {"__metadata__"}
            output_keys = set(output_header) - {"__metadata__"}
            self.assertEqual(output_keys, {f"{COMFY_PREFIX}{name}" for name in source_keys})
            self.assertEqual(output_data, source_data)
            self.assertEqual(output_header["__metadata__"]["source_revision"], "deadbeef")
            self.assertEqual(manifest["conversion"]["module_count"], 1)
            self.assertTrue(Path(f"{output}.manifest.json").is_file())

    def test_rejects_incomplete_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incomplete.safetensors"
            write_lora(source, [("layer.alpha", [], b"aa")])
            with self.assertRaisesRegex(ValueError, "incomplete LoRA module"):
                convert_lora(
                    source,
                    root / "out.safetensors",
                    "test/repo",
                    "deadbeef",
                    verify_official=False,
                )

    def test_rejects_already_converted_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, prefix=COMFY_PREFIX)
            with self.assertRaisesRegex(ValueError, "already use"):
                convert_lora(
                    source,
                    root / "out.safetensors",
                    "test/repo",
                    "deadbeef",
                    verify_official=False,
                )

    def test_default_rejects_unpinned_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            with self.assertRaisesRegex(ValueError, "pinned official"):
                convert_lora(source, root / "out.safetensors", "test/repo", "deadbeef")


if __name__ == "__main__":
    unittest.main()
