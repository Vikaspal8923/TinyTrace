from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyTraceConfig


class MobileCLIPSpatialEncoder(nn.Module):
    """Frozen MobileCLIP-S0 image tower exposing pre-pooling spatial tokens.

    Apple's public ``encode_image`` API returns one pooled vector per image.
    TinyTRACE needs a token sequence for slot compression, so this adapter uses
    the S0 tower's embedding, backbone, and expansion layers before its global
    pooling head. For a 256x256 image this is expected to produce an 8x8 map
    with 1024 channels, represented as 64 spatial tokens.
    """

    def __init__(
        self,
        config: TinyTraceConfig,
        backbone: nn.Module | None = None,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = (
            backbone if backbone is not None else self._load_official_backbone(load_pretrained=load_pretrained)
        )

        if config.freeze_visual_encoder:
            self.backbone.requires_grad_(False)
        self.backbone.eval()

    def _load_official_backbone(self, load_pretrained: bool) -> nn.Module:
        checkpoint = Path(self.config.mobileclip_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[1] / checkpoint
        if load_pretrained and not checkpoint.is_file():
            raise FileNotFoundError(
                "MobileCLIP-S0 checkpoint not found at "
                f"{checkpoint}. Download the official checkpoint and set "
                "mobileclip_checkpoint in the TinyTrace config."
            )

        try:
            import mobileclip
        except ImportError as exc:
            raise ImportError(
                "The official Apple MobileCLIP package is required. Install "
                "apple/ml-mobileclip before constructing TinyTraceModel."
            ) from exc

        clip_model, _, _ = mobileclip.create_model_and_transforms(
            self.config.mobileclip_model_name,
            pretrained=str(checkpoint) if load_pretrained else None,
            reparameterize=False,
        )
        return clip_model.image_encoder.model

    def train(self, mode: bool = True) -> "MobileCLIPSpatialEncoder":
        super().train(mode)
        # MobileCLIP-S0 contains BatchNorm layers and must stay in evaluation
        # mode while frozen, even when the surrounding TinyTrace model trains.
        if self.config.freeze_visual_encoder:
            self.backbone.eval()
        return self

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.size(1) != 3:
            raise ValueError("MobileCLIP frames must have shape [batch, 3, height, width].")
        frames = frames.float().clamp(0.0, 1.0)
        return F.interpolate(
            frames,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self.preprocess(frames)
        context = torch.no_grad() if self.config.freeze_visual_encoder else torch.enable_grad()
        with context:
            features = self.backbone.forward_embeddings(frames)
            features = self.backbone.forward_tokens(features)
            features = self.backbone.conv_exp(features)

        if features.ndim != 4:
            raise RuntimeError(
                "MobileCLIP spatial extraction expected [batch, channels, height, width], "
                f"but received {tuple(features.shape)}."
            )
        if features.size(1) != self.config.visual_hidden_dim:
            raise RuntimeError(
                f"Configured visual_hidden_dim={self.config.visual_hidden_dim}, but MobileCLIP "
                f"produced {features.size(1)} channels."
            )
        return features.flatten(2).transpose(1, 2)


class SlotCompressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_slots: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_slots, input_dim))
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(patch_tokens.size(0), -1, -1)
        scores = torch.matmul(queries, patch_tokens.transpose(-1, -2)) / patch_tokens.size(-1) ** 0.5
        weights = torch.softmax(scores, dim=-1)
        return self.projector(torch.matmul(weights, patch_tokens))
