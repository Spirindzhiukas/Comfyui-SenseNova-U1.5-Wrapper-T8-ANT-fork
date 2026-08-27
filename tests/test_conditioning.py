import sys
import unittest
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from sensenova_u15.conditioning import (
    IMAGE_CONTEXT_ID,
    IMAGE_END_ID,
    IMAGE_LABEL_IDS,
    IMAGE_START_ID,
    block_causal_mask,
    condition_input_ids,
    conditioned_input_length,
    preprocess_reference,
    smart_resize,
    thw_indexes,
)
from sensenova_u15.text_encoder import SenseNovaTokenizer


class ConditioningTests(unittest.TestCase):
    def test_reference_resize_matches_official_buckets(self):
        self.assertEqual(smart_resize(512, 512), (512, 512))
        self.assertEqual(smart_resize(256, 512), (384, 736))
        self.assertEqual(smart_resize(4096, 4096), (2048, 2048))
        image = torch.zeros((1, 256, 512, 3))
        self.assertEqual(tuple(preprocess_reference(image).shape), (1, 3, 384, 736))

    def test_reference_node_rejects_image_batches(self):
        with self.assertRaisesRegex(ValueError, "accepts one image"):
            preprocess_reference(torch.zeros((2, 512, 512, 3)))

    def test_condition_query_inserts_reference_block(self):
        tokenizer = SenseNovaTokenizer()
        pairs = tokenizer.tokenize_with_weights("change the sky")["sensenova_u15"][0]
        input_ids = torch.tensor([[int(value[0]) for value in pairs]])
        conditioned = condition_input_ids(input_ids, [(2, 3)])
        values = conditioned[0].tolist()
        start = values.index(IMAGE_START_ID, values.index(151644) + 1)
        self.assertEqual(values[start + 1 : start + 7], [IMAGE_CONTEXT_ID] * 6)
        self.assertEqual(values[start + 7], IMAGE_END_ID)

    def test_image_only_query_and_block_mask(self):
        input_ids = torch.tensor([[0]])
        conditioned = condition_input_ids(input_ids, [(2, 3)], image_only=True)
        indexes = thw_indexes(conditioned, [(2, 3)])
        selected = conditioned[0] == IMAGE_CONTEXT_ID
        self.assertEqual(indexes[0, 0, selected].unique().numel(), 1)
        self.assertEqual(indexes[0, 1, selected].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(indexes[0, 2, selected].tolist(), [0, 1, 2, 0, 1, 2])
        mask = block_causal_mask(indexes)
        context_positions = selected.nonzero(as_tuple=True)[0]
        block = mask[0, 0][context_positions[:, None], context_positions[None, :]]
        self.assertTrue(torch.equal(block, torch.zeros_like(block)))

    def test_conditioned_length_matches_real_queries(self):
        tokenizer = SenseNovaTokenizer()
        pairs = tokenizer.tokenize_with_weights("change the sky")["sensenova_u15"][0]
        input_ids = torch.tensor([[int(value[0]) for value in pairs]])
        for image_only in (False, True):
            conditioned = condition_input_ids(input_ids, [(2, 3)], image_only=image_only)
            expected = conditioned_input_length(input_ids.shape[1], [(2, 3)], image_only=image_only)
            self.assertEqual(conditioned.shape[1], expected)

    def test_multi_reference_query_matches_official_labels_and_grids(self):
        tokenizer = SenseNovaTokenizer()
        pairs = tokenizer.tokenize_with_weights("combine the subjects")['sensenova_u15'][0]
        input_ids = torch.tensor([[int(value[0]) for value in pairs]])
        grids = [(2, 3), (1, 4)]
        conditioned = condition_input_ids(input_ids, grids)
        values = conditioned[0].tolist()
        first = values.index(IMAGE_START_ID)
        self.assertEqual(values[first - len(IMAGE_LABEL_IDS[0]) : first], list(IMAGE_LABEL_IDS[0]))
        second = values.index(IMAGE_START_ID, first + 1)
        self.assertEqual(values[second - len(IMAGE_LABEL_IDS[1]) : second], list(IMAGE_LABEL_IDS[1]))

        indexes = thw_indexes(conditioned, grids)
        selected = conditioned[0] == IMAGE_CONTEXT_ID
        self.assertEqual(indexes[0, 1, selected].tolist(), [0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(indexes[0, 2, selected].tolist(), [0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
        self.assertEqual(
            conditioned.shape[1],
            conditioned_input_length(input_ids.shape[1], grids),
        )

        image_only = condition_input_ids(input_ids, grids, image_only=True)
        self.assertEqual(
            image_only.shape[1],
            conditioned_input_length(input_ids.shape[1], grids, image_only=True),
        )


if __name__ == "__main__":
    unittest.main()
