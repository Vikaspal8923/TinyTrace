from dataclasses import dataclass, field


@dataclass
class TinyTraceConfig:
    image_size: int = 96
    patch_size: int = 16
    max_frames: int = 8
    visual_hidden_dim: int = 128
    compressed_visual_tokens: int = 4
    time_tokens_per_frame: int = 6

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
    max_generated_tokens: int = 96
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
