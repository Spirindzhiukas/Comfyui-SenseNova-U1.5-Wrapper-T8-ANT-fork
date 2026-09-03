"""Windows/CRLF tolerance for the bundled tokenizer assets.

The digest pin in ``sensenova_u15/loader.py`` originally refused to load a
node folder that git had rewritten to CRLF (``core.autocrlf=true``), which is a
checkout problem rather than a model problem. These tests keep the tolerant
behaviour and the ``.gitattributes`` pin in place.
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import sensenova_u15.loader as loader


class TokenizerAssetTests(unittest.TestCase):
    def test_bundled_assets_match_their_pinned_digests(self):
        loader._validate_tokenizer_assets()

    def test_digest_kind_recognises_crlf_checkouts(self):
        for name, expected in loader.TOKENIZER_ASSET_SHA256.items():
            raw = (loader.TOKENIZER_DIR / name).read_bytes()
            with self.subTest(name=name):
                self.assertEqual(loader._tokenizer_digest_kind(raw, expected), "raw")
                self.assertEqual(
                    loader._tokenizer_digest_kind(raw.replace(b"\n", b"\r\n"), expected),
                    "normalized",
                )
                self.assertIsNone(loader._tokenizer_digest_kind(raw + b" ", expected))

    def test_crlf_checkout_loads_with_a_note_but_no_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in loader.TOKENIZER_ASSET_SHA256:
                crlf = (loader.TOKENIZER_DIR / name).read_bytes().replace(b"\n", b"\r\n")
                self.assertEqual(crlf.count(b"\r\r"), 0)
                (root / name).write_bytes(crlf)
            with mock.patch.object(loader, "TOKENIZER_DIR", root), redirect_stdout(io.StringIO()) as out:
                loader._validate_tokenizer_assets()
            report = out.getvalue()
            self.assertIn("CRLF", report)
            self.assertNotIn("WARNING", report)

    def test_unexpected_digest_warns_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in loader.TOKENIZER_ASSET_SHA256:
                (root / name).write_bytes(b"tampered")
            with mock.patch.object(loader, "TOKENIZER_DIR", root), redirect_stdout(io.StringIO()) as out:
                loader._validate_tokenizer_assets()
            report = out.getvalue()
            self.assertIn("digest mismatch", report)
            self.assertIn("core.autocrlf", report)

    def test_missing_asset_is_still_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(loader, "TOKENIZER_DIR", Path(tmp)),
                self.assertRaisesRegex(ValueError, "tokenizer asset missing"),
            ):
                loader._validate_tokenizer_assets()

    def test_repository_declares_lf_for_pinned_assets(self):
        # The loader tolerates a CRLF checkout, but the repository must still
        # declare LF so a fresh clone never produces one in the first place.
        attributes = (PACKAGE_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("sensenova_u15/tokenizer/* text eol=lf", attributes)
        for pattern in ("*.py", "*.json", "*.js", "*.txt"):
            with self.subTest(pattern=pattern):
                self.assertIn(f"{pattern} text eol=lf", attributes)


if __name__ == "__main__":
    unittest.main()
