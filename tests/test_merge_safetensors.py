import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.merge_safetensors import SIDECAR_FILES, inspect_source, merge_model


def write_shard(path, tensors):
    header = {}
    data = bytearray()
    for name, dtype, shape, value in tensors:
        start = len(data)
        data.extend(value)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(data)


def read_header(path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size)), 8 + size


class MergeSafetensorsTests(unittest.TestCase):
    def make_source(self, root):
        source = root / "source"
        source.mkdir()
        write_shard(
            source / "model-00001-of-00002.safetensors",
            [("a", "BF16", [2], b"abcd"), ("b", "F32", [1], b"efgh")],
        )
        write_shard(
            source / "model-00002-of-00002.safetensors",
            [("c", "BF16", [1, 2], b"ijkl")],
        )
        index = {
            "metadata": {},
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00001-of-00002.safetensors",
                "c": "model-00002-of-00002.safetensors",
            },
        }
        (source / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        for name in SIDECAR_FILES:
            (source / name).write_text("{}", encoding="utf-8")
        return source

    def test_stream_merge_preserves_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            output = root / "output" / "model.safetensors"
            manifest = merge_model(source, output, "test/repo", "deadbeef")

            header, data_start = read_header(output)
            self.assertEqual(header["__metadata__"]["source_revision"], "deadbeef")
            self.assertEqual(set(header) - {"__metadata__"}, {"a", "b", "c"})
            with output.open("rb") as handle:
                handle.seek(data_start)
                self.assertEqual(handle.read(), b"abcdefghijkl")
            self.assertEqual(manifest["output"]["tensor_count"], 3)
            self.assertEqual(manifest["output"]["tensor_data_bytes"], 12)
            self.assertTrue(Path(f"{output}.manifest.json").is_file())
            self.assertFalse(Path(f"{output}.partial").exists())

    def test_rejects_index_header_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            index_path = source / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["missing"] = "model-00001-of-00002.safetensors"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "index/header key mismatch"):
                inspect_source(source)

    def test_rejects_residual_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            (source / "download.incomplete").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "incomplete or locked"):
                inspect_source(source)


if __name__ == "__main__":
    unittest.main()
