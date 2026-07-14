from __future__ import annotations

from .config import TinyTraceConfig
from .tokenizers import CharTokenizer, NumericTokenizer


def decode_event_sequence(
    token_ids: list[int],
    config: TinyTraceConfig,
    text_tokenizer: CharTokenizer,
    time_tokenizer: NumericTokenizer,
    score_tokenizer: NumericTokenizer,
) -> list[dict]:
    events: list[dict] = []
    mode = "time"
    time_ids: list[int] = []
    score_ids: list[int] = []
    caption_ids: list[int] = []

    def flush_event() -> None:
        nonlocal time_ids, score_ids, caption_ids
        if not time_ids and not score_ids and not caption_ids:
            return
        events.append(
            {
                "timestamp": time_tokenizer.decode_values(time_ids),
                "score": score_tokenizer.decode_values(score_ids),
                "caption": text_tokenizer.decode(caption_ids).strip(),
            }
        )
        time_ids = []
        score_ids = []
        caption_ids = []

    for token_id in token_ids:
        if token_id == config.eos_token_id:
            flush_event()
            break
        if token_id == config.sync_token_id:
            if mode == "time":
                mode = "score"
            elif mode == "score":
                mode = "caption"
            else:
                flush_event()
                mode = "time"
            continue
        if token_id >= config.score_token_base:
            score_ids.append(token_id - config.score_token_base)
        elif token_id >= config.time_token_base:
            time_ids.append(token_id - config.time_token_base)
        else:
            caption_ids.append(token_id)

    return events
