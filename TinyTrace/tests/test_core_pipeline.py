import tempfile
import types
import unittest
from pathlib import Path

import torch

from tinytrace import (
    EventParseError,
    TinyTraceConfig,
    TinyTraceModel,
    decode_event_sequence,
    serialize_example,
)
from tinytrace.model import TinyTraceOutput
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer

from test_vision import FakeMobileCLIPBackbone


def tokenizers(config: TinyTraceConfig):
    return (
        CharTokenizer(config.text_vocab_size),
        NumericTokenizer(config.time_vocab, width=6),
        NumericTokenizer(config.score_vocab, width=3),
    )


class CorePipelineTests(unittest.TestCase):
    def test_event_serialization_and_parser_round_trip(self) -> None:
        config = TinyTraceConfig()
        text, time, score = tokenizers(config)
        events = [{"timestamp": [1.2, 3.4], "score": [4.5], "caption": "person runs"}]

        token_ids, label_types, prompt_length = serialize_example(
            events,
            "localize the event",
            config,
            text,
            time,
            score,
        )
        decoded = decode_event_sequence(
            token_ids[prompt_length:],
            config,
            text,
            time,
            score,
            strict=True,
        )

        self.assertEqual(decoded, events)
        self.assertEqual(len(token_ids), len(label_types))
        self.assertTrue(all(label_type == -1 for label_type in label_types[:prompt_length]))

    def test_malformed_parser_is_safe_by_default_and_strict_on_request(self) -> None:
        config = TinyTraceConfig()
        text, time, score = tokenizers(config)
        malformed = [config.total_token_vocab + 10, config.sync_token_id, config.eos_token_id]

        self.assertEqual(
            decode_event_sequence(malformed, config, text, time, score),
            [],
        )
        with self.assertRaises(EventParseError):
            decode_event_sequence(malformed, config, text, time, score, strict=True)

    def test_lcem_forward_shape_and_loss(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        text, time, score = tokenizers(config)
        token_ids, label_types, _ = serialize_example(
            [{"timestamp": [0.0, 1.0], "score": [3.0], "caption": "action"}],
            "localize",
            config,
            text,
            time,
            score,
        )
        tokens = torch.tensor([token_ids])
        types_tensor = torch.tensor([label_types])

        output = model(
            torch.rand(1, 1, 3, 32, 32),
            torch.zeros(1, 1),
            tokens,
            labels=tokens,
            label_types=types_tensor,
        )

        expected_length = config.compressed_visual_tokens + config.time_tokens_per_frame + len(token_ids)
        self.assertEqual(output.logits.shape, (1, expected_length, config.total_token_vocab))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

    def test_generation_switches_time_score_caption(self) -> None:
        config = TinyTraceConfig(
            timestamp_value_count=0,
            score_value_count=0,
            min_caption_tokens=0,
            max_caption_tokens=0,
            max_events=1,
        )
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        def scripted_forward(self, frames, frame_times, token_ids, **kwargs):
            batch = token_ids.size(0)
            text_logits = torch.full((batch, 1, config.text_vocab_size + 1), float("-inf"))
            time_logits = torch.full((batch, 1, len(config.time_vocab)), float("-inf"))
            score_logits = torch.full((batch, 1, len(config.score_vocab)), float("-inf"))
            text_logits[:, :, config.text_vocab_size] = 0.0
            time_logits[:, :, 0] = 0.0
            score_logits[:, :, 0] = 0.0
            return TinyTraceOutput(
                loss=None,
                logits=torch.empty(batch, 1, config.total_token_vocab),
                text_logits=text_logits,
                time_logits=time_logits,
                score_logits=score_logits,
            )

        model.forward = types.MethodType(scripted_forward, model)
        prompt = torch.tensor([[config.bos_token_id, config.video_token_id]])
        generated = model.generate(
            torch.rand(1, 1, 3, 16, 16),
            torch.zeros(1, 1),
            prompt,
            max_new_tokens=4,
        )

        self.assertEqual(
            generated[0, -4:].tolist(),
            [
                config.sync_token_id,
                config.sync_token_id,
                config.sync_token_id,
                config.eos_token_id,
            ],
        )

    def test_checkpoint_config_and_state_round_trip(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "tinytrace.pt"
            torch.save(
                {"model_state": model.state_dict(), "config": config.to_dict()},
                checkpoint_path,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            restored_config = TinyTraceConfig.from_dict(checkpoint["config"])
            restored = TinyTraceModel(
                restored_config,
                mobileclip_backbone=FakeMobileCLIPBackbone(),
            )
            restored.load_state_dict(checkpoint["model_state"])

        self.assertEqual(restored_config, config)
        self.assertTrue(torch.equal(restored.text_head.weight, model.text_head.weight))


if __name__ == "__main__":
    unittest.main()
