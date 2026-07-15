import unittest

import torch
import torch.nn as nn

from tinytrace.config import TinyTraceConfig
from tinytrace.model import TinyTraceModel
from tinytrace.vision import MobileCLIPSpatialEncoder, SlotCompressor


class FakeMobileCLIPBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 64, kernel_size=1)
        self.expansion = nn.Conv2d(64, 1024, kernel_size=1)

    def forward_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stem(frames)

    def forward_tokens(self, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(features, (8, 8))

    def conv_exp(self, features: torch.Tensor) -> torch.Tensor:
        return self.expansion(features)


class MobileCLIPSpatialEncoderTests(unittest.TestCase):
    def test_spatial_tokens_have_expected_shape_and_backbone_is_frozen(self) -> None:
        config = TinyTraceConfig()
        backbone = FakeMobileCLIPBackbone()
        encoder = MobileCLIPSpatialEncoder(config, backbone=backbone)

        tokens = encoder(torch.rand(2, 3, 96, 96))

        self.assertEqual(tokens.shape, (2, 64, 1024))
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))

    def test_frozen_backbone_stays_in_eval_mode(self) -> None:
        encoder = MobileCLIPSpatialEncoder(TinyTraceConfig(), backbone=FakeMobileCLIPBackbone())

        encoder.train()

        self.assertFalse(encoder.backbone.training)

    def test_slot_compressor_shape(self) -> None:
        compressor = SlotCompressor(input_dim=1024, output_dim=192, num_slots=4)

        compressed = compressor(torch.rand(2, 64, 1024))

        self.assertEqual(compressed.shape, (2, 4, 192))

    def test_model_builds_interleaved_visual_and_discrete_time_prefix(self) -> None:
        config = TinyTraceConfig(max_frames=2)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        frames = torch.rand(1, 2, 3, 96, 96)
        frame_times = torch.tensor([[0.0, 12.5]])

        prefix = model.build_visual_prefix(frames, frame_times)
        time_ids = model._encode_frame_time_ids(frame_times)

        self.assertEqual(prefix.shape, (1, 20, config.d_model))
        self.assertEqual(time_ids.shape, (1, 2, 6))
        decoded = "".join(config.time_vocab[index] for index in time_ids[0, 1].tolist())
        self.assertEqual(decoded, "0012.5")


if __name__ == "__main__":
    unittest.main()
