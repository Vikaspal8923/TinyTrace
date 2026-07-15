from __future__ import annotations

from enum import IntEnum

from .config import TinyTraceConfig
from .tokenizers import CharTokenizer, NumericTokenizer


class LabelType(IntEnum):
    IGNORE = -1
    TEXT = 0
    TEXT_SYNC = 1
    TIME = 2
    SCORE = 3
    TIME_SYNC = 4
    SCORE_SYNC = 5


def serialize_example(
    events: list[dict],
    instruction: str,
    config: TinyTraceConfig,
    text_tokenizer: CharTokenizer,
    time_tokenizer: NumericTokenizer,
    score_tokenizer: NumericTokenizer,
) -> tuple[list[int], list[int], int]:
    """Build the canonical instruction + causal event target sequence."""
    instruction_ids = [config.bos_token_id]
    instruction_ids.extend(text_tokenizer.encode(instruction)[: config.max_text_len])
    instruction_ids.append(config.video_token_id)
    prompt_length = len(instruction_ids)

    token_ids = list(instruction_ids)
    label_types = [LabelType.IGNORE] * prompt_length

    for event_index, event in enumerate(events):
        timestamps = event.get("timestamp")
        scores = event.get("score")
        caption = event.get("caption")
        if not isinstance(timestamps, list) or len(timestamps) != config.timestamp_value_count:
            raise ValueError(
                f"Event {event_index} must contain {config.timestamp_value_count} timestamp values."
            )
        if not isinstance(scores, list) or len(scores) != config.score_value_count:
            raise ValueError(f"Event {event_index} must contain {config.score_value_count} score value.")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Event {event_index} must contain a non-empty caption.")

        time_ids = [
            config.sync_token_id if token_id == 0 else config.time_token_base + token_id
            for token_id in time_tokenizer.encode([float(value) for value in timestamps])
        ]
        score_ids = [
            config.sync_token_id if token_id == 0 else config.score_token_base + token_id
            for token_id in score_tokenizer.encode([float(value) for value in scores])
        ]
        caption_ids = text_tokenizer.encode(caption)[: config.max_caption_tokens]
        caption_ids.append(config.sync_token_id)

        token_ids.extend(time_ids)
        label_types.extend(
            LabelType.TIME_SYNC if token_id == config.sync_token_id else LabelType.TIME
            for token_id in time_ids
        )
        token_ids.extend(score_ids)
        label_types.extend(
            LabelType.SCORE_SYNC if token_id == config.sync_token_id else LabelType.SCORE
            for token_id in score_ids
        )
        token_ids.extend(caption_ids)
        label_types.extend(
            LabelType.TEXT_SYNC if token_id == config.sync_token_id else LabelType.TEXT
            for token_id in caption_ids
        )

    token_ids.append(config.eos_token_id)
    label_types.append(LabelType.TEXT)
    return token_ids, [int(label_type) for label_type in label_types], prompt_length
