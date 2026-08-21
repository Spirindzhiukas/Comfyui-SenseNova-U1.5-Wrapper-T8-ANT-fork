import sys
import unittest
from pathlib import Path


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from transformers import Qwen2Tokenizer

from sensenova_u15.text_encoder import SenseNovaTokenizer, build_generation_prompt, build_unconditional_prompt


class TextEncoderTests(unittest.TestCase):
    def test_generation_tokens_match_official_tokenizer(self):
        prompt = "画一只蓝眼睛的狐狸"
        asset_dir = Path(__file__).resolve().parents[1] / "sensenova_u15" / "tokenizer"
        reference = Qwen2Tokenizer.from_pretrained(asset_dir, local_files_only=True)
        expected = reference(build_generation_prompt(prompt), add_special_tokens=True)["input_ids"]

        tokenizer = SenseNovaTokenizer()
        actual = tokenizer.tokenize_with_weights(prompt)["sensenova_u15"][0]
        actual = [int(value[0]) for value in actual]
        self.assertEqual(actual, expected)
        self.assertEqual(actual[-1], 151670)

    def test_empty_prompt_matches_official_unconditional_query(self):
        asset_dir = Path(__file__).resolve().parents[1] / "sensenova_u15" / "tokenizer"
        reference = Qwen2Tokenizer.from_pretrained(asset_dir, local_files_only=True)
        expected = reference(build_unconditional_prompt(), add_special_tokens=True)["input_ids"]
        actual = SenseNovaTokenizer().tokenize_with_weights("")["sensenova_u15"][0]
        self.assertEqual([int(value[0]) for value in actual], expected)


if __name__ == "__main__":
    unittest.main()
