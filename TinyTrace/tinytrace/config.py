import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TinyTraceConfig:
    image_size: int = 256
    max_frames: int = 12
    visual_hidden_dim: int = 1024
    compressed_visual_tokens: int = 4
    time_tokens_per_frame: int = 6

    mobileclip_model_name: str = "mobileclip_s0"
    mobileclip_checkpoint: str = "checkpoints/mobileclip_s0.pt"
    mobileclip_checkpoint_sha256: str = "809b408eff74f8058843e86a1f92967097d42ba782450e85b8f4867b7f0ca0b7"
    freeze_visual_encoder: bool = True
    mobileclip_image_mean: tuple[float, float, float] = (
        0.48145466,
        0.4578275,
        0.40821073,
    )
    mobileclip_image_std: tuple[float, float, float] = (
        0.26862954,
        0.26130258,
        0.27577711,
    )

    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.0

    text_vocab_size: int = 256
    max_text_len: int = 48
    max_caption_tokens: int = 20
    min_caption_tokens: int = 5
    max_events: int = 3
    max_generated_tokens: int = 128
    timestamp_value_count: int = 2
    score_value_count: int = 1

    time_vocab: tuple[str, ...] = field(
        default_factory=lambda: ("<sync>", "<sep>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".")
    )
    score_vocab: tuple[str, ...] = field(
        default_factory=lambda: ("<sync>", "<sep>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".")
    )

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    video_token_id: int = 3
    instruction_token_offset: int = 4

    @property
    def text_token_base(self) -> int:
        return 0

    @property
    def sync_token_id(self) -> int:
        return self.text_vocab_size

    @property
    def time_token_base(self) -> int:
        return self.text_vocab_size + 1

    @property
    def score_token_base(self) -> int:
        return self.time_token_base + len(self.time_vocab)

    @property
    def total_token_vocab(self) -> int:
        return self.score_token_base + len(self.score_vocab)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TinyTraceConfig":
        known_fields = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(known_fields))
        if unknown:
            raise ValueError(f"Unknown TinyTrace config fields: {', '.join(unknown)}")

        normalized = dict(values)
        for field_name in ("time_vocab", "score_vocab", "mobileclip_image_mean", "mobileclip_image_std"):
            if field_name in normalized:
                normalized[field_name] = tuple(normalized[field_name])
        return cls(**normalized)

    @classmethod
    def from_json(cls, path: str | Path) -> "TinyTraceConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
