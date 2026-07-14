from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyTraceConfig


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
    def __init__(self, config: TinyTraceConfig) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, config.visual_hidden_dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.frame_projector = nn.Sequential(
            nn.Linear(config.visual_hidden_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.compression_queries = nn.Parameter(torch.randn(config.compressed_visual_tokens, config.visual_hidden_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.config = config

    def forward(self, frames: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        batch, num_frames, channels, height, width = frames.shape
        patches = self.patch_embed(frames.view(batch * num_frames, channels, height, width))
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches.view(batch, num_frames, patches.size(1), patches.size(2))

        queries = self.compression_queries.unsqueeze(0).unsqueeze(0).expand(batch, num_frames, -1, -1)
        scores = torch.matmul(queries, patches.transpose(-1, -2)) / math.sqrt(patches.size(-1))
        weights = torch.softmax(scores, dim=-1)
        compressed = torch.matmul(weights, patches)
        compressed = self.frame_projector(compressed)

        frame_times = frame_times.unsqueeze(-1)
        time_features = self.time_mlp(frame_times).unsqueeze(2)

        visual_tokens = torch.cat([compressed, time_features], dim=2)
        return visual_tokens.view(batch, -1, visual_tokens.size(-1))


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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
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
    def __init__(self, config: TinyTraceConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = LightweightVisualEncoder(config)
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.d_model)
        self.sync_embedding = nn.Parameter(torch.randn(config.d_model))
        self.time_embeddings = nn.Embedding(len(config.time_vocab), config.d_model)
        self.score_embeddings = nn.Embedding(len(config.score_vocab), config.d_model)
        self.position = PositionalEncoding(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

        self.text_head = nn.Linear(config.d_model, config.text_vocab_size + 1)
        self.time_head = nn.Linear(config.d_model, len(config.time_vocab))
        self.score_head = nn.Linear(config.d_model, len(config.score_vocab))

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

    def embed_mixed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for row in token_ids:
            row_embeddings = []
            for token_id in row.tolist():
                if token_id == self.config.sync_token_id:
                    row_embeddings.append(self.sync_embedding)
                elif token_id >= self.config.score_token_base:
                    row_embeddings.append(self.score_embeddings.weight[token_id - self.config.score_token_base])
                elif token_id >= self.config.time_token_base:
                    row_embeddings.append(self.time_embeddings.weight[token_id - self.config.time_token_base])
                else:
                    row_embeddings.append(self.text_embeddings.weight[token_id])
            embeddings.append(torch.stack(row_embeddings, dim=0))
        return torch.stack(embeddings, dim=0)

    def forward(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        token_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        label_types: torch.Tensor | None = None,
    ) -> TinyTraceOutput:
        visual_tokens = self.visual_encoder(frames, frame_times)
        token_embeddings = self.embed_mixed_tokens(token_ids)
        x = torch.cat([visual_tokens, token_embeddings], dim=1)
        x = self.dropout(self.position(x))

        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask)
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

            loss = sum(loss_terms) if loss_terms else None

        return TinyTraceOutput(loss=loss, logits=full_logits, text_logits=text_logits, time_logits=time_logits, score_logits=score_logits)

    @torch.no_grad()
    def generate(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        generated = prompt_ids.clone()
        mode = "time"
        phase_token_count = 0
        event_count = 0
        min_time_tokens, min_score_tokens = self._expected_phase_lengths()
        for _ in range(max_new_tokens):
            output = self.forward(frames, frame_times, generated)
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
                    mode = "score"
                elif mode == "score":
                    mode = "caption"
                else:
                    event_count += 1
                    if event_count >= self.config.max_events:
                        eos = torch.full_like(next_token, self.config.eos_token_id)
                        generated = torch.cat([generated[:, :-1], eos], dim=1)
                        break
                    mode = "time"
                phase_token_count = 0
            elif token_value == self.config.eos_token_id:
                break
            else:
                phase_token_count += 1
        return generated
