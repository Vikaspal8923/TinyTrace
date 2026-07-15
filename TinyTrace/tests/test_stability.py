import unittest

import torch

from tinytrace.config import TinyTraceConfig
from tinytrace.data import tinytrace_collate_fn
from tinytrace.model import TinyTraceModel
from tinytrace.tokenizers import NumericTokenizer

from test_vision import FakeMobileCLIPBackbone


class StabilityTests(unittest.TestCase):
    def test_numeric_tokenizer_round_trip(self) -> None:
        config = TinyTraceConfig()
        tokenizer = NumericTokenizer(config.time_vocab, width=6)

        encoded = tokenizer.encode([0.0, 12.5])

        self.assertEqual(tokenizer.decode_sequence(encoded), [0.0, 12.5])

    def test_variable_frame_collation_adds_padding_mask(self) -> None:
        def sample(frame_count: int, token_count: int) -> dict:
            return {
                "frames": torch.rand(frame_count, 3, 16, 16),
                "frame_times": torch.arange(frame_count, dtype=torch.float32),
                "token_ids": torch.arange(1, token_count + 1),
                "label_types": torch.zeros(token_count, dtype=torch.long),
                "events": [],
                "instruction": "test",
            }

        batch = tinytrace_collate_fn([sample(2, 4), sample(4, 6)])

        self.assertEqual(batch["frames"].shape, (2, 4, 3, 16, 16))
        self.assertEqual(batch["frame_times"].shape, (2, 4))
        self.assertEqual(
            batch["frame_mask"].tolist(),
            [[True, True, False, False], [True, True, True, True]],
        )

    def test_attention_padding_mask_covers_each_padded_frame_group(self) -> None:
        config = TinyTraceConfig(max_frames=2)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        token_ids = torch.tensor([[config.bos_token_id, config.video_token_id, config.pad_token_id]])
        frame_mask = torch.tensor([[True, False]])

        mask = model._build_key_padding_mask(token_ids, frame_mask, num_frames=2)

        tokens_per_frame = config.compressed_visual_tokens + config.time_tokens_per_frame
        self.assertFalse(mask[0, :tokens_per_frame].any())
        self.assertTrue(mask[0, tokens_per_frame : 2 * tokens_per_frame].all())
        self.assertTrue(mask[0, -1])

    def test_generation_rejects_unsupported_batches(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        with self.assertRaisesRegex(ValueError, "batch size 1"):
            model.generate(
                torch.rand(2, 1, 3, 16, 16),
                torch.zeros(2, 1),
                torch.tensor([[1, 3], [1, 3]]),
                max_new_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
