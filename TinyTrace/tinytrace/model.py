from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyTraceConfig
from .vision import MobileCLIPSpatialEncoder, SlotCompressor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LightweightVisualEncoder(nn.Module):
    def __init__(
        self,
        config: TinyTraceConfig,
        mobileclip_backbone: nn.Module | None = None,
        load_pretrained_visual: bool = True,
    ) -> None:
        super().__init__()
        self.mobileclip = MobileCLIPSpatialEncoder(
            config,
            backbone=mobileclip_backbone,
            load_pretrained=load_pretrained_visual,
        )
        self.compressor = SlotCompressor(
            input_dim=config.visual_hidden_dim,
            output_dim=config.d_model,
            num_slots=config.compressed_visual_tokens,
        )
        self.config = config

    def set_mobileclip_trainable(self, trainable: bool, strategy: str = "full") -> None:
        self.mobileclip.set_trainable(trainable, strategy=strategy)

    def extract_patch_features(self, frames: torch.Tensor) -> torch.Tensor:
        batch, num_frames, channels, height, width = frames.shape
        patches = self.mobileclip(frames.reshape(batch * num_frames, channels, height, width))
        return patches.view(batch, num_frames, patches.size(1), patches.size(2))

    def compress_patch_features(self, patch_features: torch.Tensor) -> torch.Tensor:
        if patch_features.ndim != 4:
            raise ValueError(
                "MobileCLIP patch features must have shape "
                "[batch, num_frames, num_patches, channels]."
            )
        batch, num_frames, num_patches, channels = patch_features.shape
        if channels != self.config.visual_hidden_dim:
            raise ValueError(
                f"Expected {self.config.visual_hidden_dim} MobileCLIP channels, received {channels}."
            )
        flattened = patch_features.reshape(batch * num_frames, num_patches, channels)
        compressed = self.compressor(flattened)
        return compressed.view(batch, num_frames, self.config.compressed_visual_tokens, -1)

    def forward(
        self,
        frames: torch.Tensor,
        patch_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patch_features is None:
            patch_features = self.extract_patch_features(frames)
        elif patch_features.shape[:2] != frames.shape[:2]:
            raise ValueError(
                "Cached MobileCLIP features do not match the frame batch/time dimensions: "
                f"{tuple(patch_features.shape[:2])} vs. {tuple(frames.shape[:2])}."
            )
        return self.compress_patch_features(patch_features)


class DecoderBlock(nn.Module):
    def __init__(self, config: TinyTraceConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.mlp_ratio),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.mlp_ratio, config.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class TinyTraceOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    text_logits: torch.Tensor
    time_logits: torch.Tensor
    score_logits: torch.Tensor


class TinyTraceModel(nn.Module):
    def __init__(
        self,
        config: TinyTraceConfig,
        mobileclip_backbone: nn.Module | None = None,
        load_pretrained_visual: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = LightweightVisualEncoder(
            config,
            mobileclip_backbone=mobileclip_backbone,
            load_pretrained_visual=load_pretrained_visual,
        )
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.d_model)
        self.sync_embedding = nn.Parameter(torch.randn(config.d_model))
        self.time_embeddings = nn.Embedding(len(config.time_vocab), config.d_model)
        self.score_embeddings = nn.Embedding(len(config.score_vocab), config.d_model)
        self.token_type_embeddings = nn.Embedding(4, config.d_model)
        self.position = PositionalEncoding(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

        self.text_head = nn.Linear(config.d_model, config.text_vocab_size + 1)
        self.time_head = nn.Linear(config.d_model, len(config.time_vocab))
        self.score_head = nn.Linear(config.d_model, len(config.score_vocab))

    def set_visual_encoder_trainable(self, trainable: bool, strategy: str = "full") -> None:
        self.visual_encoder.set_mobileclip_trainable(trainable, strategy=strategy)

    def _encode_frame_time_ids(self, frame_times: torch.Tensor) -> torch.Tensor:
        if frame_times.ndim != 2:
            raise ValueError("frame_times must have shape [batch, num_frames].")

        token_to_id = {token: index for index, token in enumerate(self.config.time_vocab)}
        rows: list[list[list[int]]] = []
        for sample_times in frame_times.detach().cpu().tolist():
            sample_rows = []
            for value in sample_times:
                formatted = format(float(value), "0>6.1f")
                if len(formatted) != self.config.time_tokens_per_frame:
                    raise ValueError(
                        f"Frame timestamp {value} cannot be represented as a "
                        f"{self.config.time_tokens_per_frame}-token TRACE timestamp."
                    )
                try:
                    sample_rows.append([token_to_id[character] for character in formatted])
                except KeyError as exc:
                    raise ValueError(f"Unsupported character in frame timestamp {formatted!r}.") from exc
            rows.append(sample_rows)
        return torch.tensor(rows, dtype=torch.long, device=frame_times.device)

    def build_visual_prefix(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        visual_patch_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_tokens = self.visual_encoder(frames, patch_features=visual_patch_features)
        if visual_tokens.shape[:2] != frame_times.shape:
            raise ValueError(
                "Frame/time shape mismatch: visual encoder produced "
                f"{tuple(visual_tokens.shape[:2])}, frame_times has {tuple(frame_times.shape)}."
            )

        time_ids = self._encode_frame_time_ids(frame_times)
        time_tokens = self.time_embeddings(time_ids)
        per_frame_tokens = torch.cat([visual_tokens, time_tokens], dim=2)
        return per_frame_tokens.flatten(1, 2)

    def _build_key_padding_mask(
        self,
        token_ids: torch.Tensor,
        frame_mask: torch.Tensor | None,
        num_frames: int,
    ) -> torch.Tensor | None:
        if frame_mask is None:
            frame_mask = torch.ones(
                token_ids.size(0),
                num_frames,
                dtype=torch.bool,
                device=token_ids.device,
            )
        if frame_mask.shape != (token_ids.size(0), num_frames):
            raise ValueError(
                f"frame_mask must have shape {(token_ids.size(0), num_frames)}, "
                f"received {tuple(frame_mask.shape)}."
            )

        tokens_per_frame = self.config.compressed_visual_tokens + self.config.time_tokens_per_frame
        visual_padding = (~frame_mask.bool()).repeat_interleave(tokens_per_frame, dim=1)
        text_padding = token_ids.eq(self.config.pad_token_id)
        combined = torch.cat([visual_padding, text_padding], dim=1)
        return combined if combined.any() else None

    def _expected_phase_lengths(self) -> tuple[int, int]:
        time_len = (
            self.config.timestamp_value_count * 6
            + max(0, self.config.timestamp_value_count - 1)
        )
        score_len = (
            self.config.score_value_count * 3
            + max(0, self.config.score_value_count - 1)
        )
        return time_len, score_len

    def _numeric_format_mask(self, device: torch.device, mode: str, position: int, vocab_size: int) -> torch.Tensor:
        allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        digit_slice = slice(2, 12)
        dot_idx = 12
        sep_idx = 1

        if mode == "time":
            if position in {0, 1, 2, 3, 5, 7, 8, 9, 10, 12}:
                allowed[digit_slice] = True
            elif position in {4, 11}:
                allowed[dot_idx] = True
            elif position == 6:
                allowed[sep_idx] = True
        else:
            if position in {0, 2}:
                allowed[digit_slice] = True
            elif position == 1:
                allowed[dot_idx] = True

        return allowed

    def _infer_token_type_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_types = torch.zeros_like(token_ids)
        token_types[token_ids == self.config.sync_token_id] = 3
        token_types[(token_ids >= self.config.time_token_base) & (token_ids < self.config.score_token_base)] = 1
        token_types[token_ids >= self.config.score_token_base] = 2
        return token_types

    def _decode_numeric_values(self, ids: list[int], vocab: tuple[str, ...]) -> list[float]:
        id_to_token = {index: token for index, token in enumerate(vocab)}
        values: list[float] = []
        current: list[str] = []
        for idx in ids:
            token = id_to_token[idx]
            if token == "<sep>":
                if current:
                    values.append(float("".join(current)))
                    current = []
            elif token != "<sync>":
                current.append(token)
        if current:
            values.append(float("".join(current)))
        return values

    def _encode_numeric_values(self, values: list[float], mode: str) -> list[int]:
        width = 6 if mode == "time" else 3
        vocab = self.config.time_vocab if mode == "time" else self.config.score_vocab
        token_to_id = {token: index for index, token in enumerate(vocab)}
        encoded: list[int] = []
        for index, value in enumerate(values):
            for char in format(float(value), f"0>{width}.1f"):
                encoded.append(token_to_id[char])
            if index < len(values) - 1:
                encoded.append(token_to_id["<sep>"])
        encoded.append(token_to_id["<sync>"])
        return encoded

    def _constrain_time_ids(self, time_ids: list[int], clip_end: float) -> list[int]:
        values = self._decode_numeric_values(time_ids, self.config.time_vocab)
        if not values:
            return time_ids
        clipped = [min(max(value, 0.0), clip_end) for value in values]
        if len(clipped) >= 2:
            clipped[1] = max(clipped[0], clipped[1])
        return self._encode_numeric_values(clipped[: self.config.timestamp_value_count], mode="time")

    def embed_mixed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_type_ids = self._infer_token_type_ids(token_ids)
        embeddings = []
        for row, type_row in zip(token_ids, token_type_ids):
            row_embeddings = []
            for token_id, token_type in zip(row.tolist(), type_row.tolist()):
                if token_id == self.config.sync_token_id:
                    base_embedding = self.sync_embedding
                elif token_id >= self.config.score_token_base:
                    base_embedding = self.score_embeddings.weight[token_id - self.config.score_token_base]
                elif token_id >= self.config.time_token_base:
                    base_embedding = self.time_embeddings.weight[token_id - self.config.time_token_base]
                else:
                    base_embedding = self.text_embeddings.weight[token_id]
                row_embeddings.append(base_embedding + self.token_type_embeddings.weight[token_type])
            embeddings.append(torch.stack(row_embeddings, dim=0))
        return torch.stack(embeddings, dim=0)

    def forward(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        token_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        label_types: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        visual_patch_features: torch.Tensor | None = None,
    ) -> TinyTraceOutput:
        visual_tokens = self.build_visual_prefix(
            frames,
            frame_times,
            visual_patch_features=visual_patch_features,
        )
        token_embeddings = self.embed_mixed_tokens(token_ids)
        x = torch.cat([visual_tokens, token_embeddings], dim=1)
        x = self.dropout(self.position(x))

        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = self._build_key_padding_mask(token_ids, frame_mask, frames.size(1))
        for block in self.blocks:
            x = block(x, attn_mask, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        text_logits = self.text_head(x)
        time_logits = self.time_head(x)
        score_logits = self.score_head(x)

        full_logits = torch.full(
            (x.size(0), x.size(1), self.config.total_token_vocab),
            float("-inf"),
            device=x.device,
        )
        full_logits[:, :, : self.config.text_vocab_size] = text_logits[:, :, : self.config.text_vocab_size]
        full_logits[:, :, self.config.sync_token_id] = text_logits[:, :, self.config.text_vocab_size]
        full_logits[:, :, self.config.time_token_base : self.config.score_token_base] = time_logits
        full_logits[:, :, self.config.score_token_base :] = score_logits

        loss = None
        if labels is not None and label_types is not None:
            prompt_len = visual_tokens.size(1)
            hidden_text = text_logits[:, prompt_len:-1]
            hidden_time = time_logits[:, prompt_len:-1]
            hidden_score = score_logits[:, prompt_len:-1]

            target_tokens = labels[:, 1:]
            target_types = label_types[:, 1:]
            valid_mask = target_types >= 0

            loss_terms = []

            text_mask = (target_types == 0) & valid_mask
            if text_mask.any():
                loss_terms.append(F.cross_entropy(hidden_text[text_mask], target_tokens[text_mask]))

            caption_sync_mask = (target_types == 1) & valid_mask
            if caption_sync_mask.any():
                sync_targets = torch.full_like(target_tokens[caption_sync_mask], self.config.text_vocab_size)
                loss_terms.append(F.cross_entropy(hidden_text[caption_sync_mask], sync_targets))

            time_mask = (target_types == 2) & valid_mask
            if time_mask.any():
                loss_terms.append(
                    F.cross_entropy(hidden_time[time_mask], target_tokens[time_mask] - self.config.time_token_base)
                )

            time_sync_mask = (target_types == 4) & valid_mask
            if time_sync_mask.any():
                time_sync_targets = torch.zeros_like(target_tokens[time_sync_mask])
                loss_terms.append(F.cross_entropy(hidden_time[time_sync_mask], time_sync_targets))

            score_mask = (target_types == 3) & valid_mask
            if score_mask.any():
                loss_terms.append(
                    F.cross_entropy(hidden_score[score_mask], target_tokens[score_mask] - self.config.score_token_base)
                )

            score_sync_mask = (target_types == 5) & valid_mask
            if score_sync_mask.any():
                score_sync_targets = torch.zeros_like(target_tokens[score_sync_mask])
                loss_terms.append(F.cross_entropy(hidden_score[score_sync_mask], score_sync_targets))

            # A caption <sync> is followed either by EOS (final event) or by
            # the first timestamp token (another event). The task heads are
            # otherwise trained independently, so this explicit boundary loss
            # calibrates the cross-head decision used by generation.
            previous_types = label_types[:, :-1]
            boundary_mask = (previous_types == 1) & valid_mask
            if boundary_mask.any():
                boundary_logits = torch.cat(
                    [
                        hidden_text[:, :, self.config.eos_token_id].unsqueeze(-1),
                        hidden_time,
                    ],
                    dim=-1,
                )
                boundary_tokens = target_tokens[boundary_mask]
                boundary_targets = torch.where(
                    boundary_tokens == self.config.eos_token_id,
                    torch.zeros_like(boundary_tokens),
                    boundary_tokens - self.config.time_token_base + 1,
                )
                loss_terms.append(F.cross_entropy(boundary_logits[boundary_mask], boundary_targets))

            loss = sum(loss_terms) if loss_terms else None

        return TinyTraceOutput(loss=loss, logits=full_logits, text_logits=text_logits, time_logits=time_logits, score_logits=score_logits)

    @torch.no_grad()
    def generate(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        frame_mask: torch.Tensor | None = None,
        visual_patch_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prompt_ids.size(0) != 1:
            raise ValueError(
                "TinyTrace generation currently supports batch size 1. "
                "Per-sequence adaptive head state is not implemented yet."
            )
        generated = prompt_ids.clone()
        mode = "time"
        phase_token_count = 0
        event_count = 0
        phase_start_index = generated.size(1)
        clip_end = float(frame_times.max().item())
        min_time_tokens, min_score_tokens = self._expected_phase_lengths()
        for _ in range(max_new_tokens):
            output = self.forward(
                frames,
                frame_times,
                generated,
                frame_mask=frame_mask,
                visual_patch_features=visual_patch_features,
            )
            if mode == "boundary":
                if event_count >= self.config.max_events:
                    next_token = torch.full(
                        (generated.size(0), 1),
                        self.config.eos_token_id,
                        dtype=torch.long,
                        device=generated.device,
                    )
                else:
                    time_logits = output.time_logits[:, -1, :].clone()
                    allowed = self._numeric_format_mask(
                        time_logits.device,
                        "time",
                        position=0,
                        vocab_size=time_logits.size(-1),
                    )
                    time_logits[:, ~allowed] = float("-inf")
                    best_time_logit, best_time_id = torch.max(time_logits, dim=-1, keepdim=True)
                    eos_logit = output.text_logits[:, -1, self.config.eos_token_id].unsqueeze(-1)
                    choose_eos = eos_logit >= best_time_logit
                    next_token = best_time_id + self.config.time_token_base
                    next_token = torch.where(
                        choose_eos,
                        torch.full_like(next_token, self.config.eos_token_id),
                        next_token,
                    )
                    if not bool(choose_eos.item()):
                        mode = "time"
                        phase_token_count = 1

                generated = torch.cat([generated, next_token], dim=1)
                if next_token[0, 0].item() == self.config.eos_token_id:
                    break
                phase_start_index = generated.size(1) - 1
                continue

            if mode == "time":
                next_logits = output.time_logits[:, -1, :].clone()
            elif mode == "score":
                next_logits = output.score_logits[:, -1, :].clone()
            else:
                next_logits = output.text_logits[:, -1, :].clone()

            if mode == "time":
                if phase_token_count < min_time_tokens:
                    next_logits[:, 0] = float("-inf")
                    allowed = self._numeric_format_mask(next_logits.device, "time", phase_token_count, next_logits.size(-1))
                    next_logits[:, ~allowed] = float("-inf")
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                next_token = torch.where(
                    next_token == 0,
                    torch.full_like(next_token, self.config.sync_token_id),
                    next_token + self.config.time_token_base,
                )
            elif mode == "score":
                if phase_token_count < min_score_tokens:
                    next_logits[:, 0] = float("-inf")
                    allowed = self._numeric_format_mask(next_logits.device, "score", phase_token_count, next_logits.size(-1))
                    next_logits[:, ~allowed] = float("-inf")
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                next_token = torch.where(
                    next_token == 0,
                    torch.full_like(next_token, self.config.sync_token_id),
                    next_token + self.config.score_token_base,
                )
            else:
                if phase_token_count == 0:
                    next_logits[:, self.config.eos_token_id] = float("-inf")
                if phase_token_count < self.config.min_caption_tokens:
                    next_logits[:, self.config.text_vocab_size] = float("-inf")
                if phase_token_count >= self.config.max_caption_tokens:
                    next_logits[:, : self.config.text_vocab_size] = float("-inf")
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                next_token = torch.where(
                    next_token == self.config.text_vocab_size,
                    torch.full_like(next_token, self.config.sync_token_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=1)

            token_value = next_token[0, 0].item()
            if token_value == self.config.sync_token_id:
                if mode == "time":
                    time_phase = generated[0, phase_start_index:].tolist()
                    constrained = self._constrain_time_ids(
                        [
                            0 if token == self.config.sync_token_id else token - self.config.time_token_base
                            for token in time_phase
                        ],
                        clip_end=clip_end,
                    )
                    constrained_tokens = [
                        self.config.sync_token_id if token == 0 else token + self.config.time_token_base
                        for token in constrained
                    ]
                    generated = torch.cat(
                        [
                            generated[:, :phase_start_index],
                            torch.tensor(constrained_tokens, dtype=generated.dtype, device=generated.device).unsqueeze(0),
                        ],
                        dim=1,
                    )
                    mode = "score"
                elif mode == "score":
                    mode = "caption"
                else:
                    event_count += 1
                    mode = "boundary"
                phase_token_count = 0
                phase_start_index = generated.size(1)
            elif token_value == self.config.eos_token_id:
                break
            else:
                phase_token_count += 1
        return generated
