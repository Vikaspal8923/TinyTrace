from __future__ import annotations

import json
import hashlib
import random
import re
import subprocess
import uuid
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import TinyTraceConfig
from .serialization import serialize_example
from .tokenizers import CharTokenizer, NumericTokenizer


class SyntheticTinyTraceDataset(Dataset):
    def __init__(self, config: TinyTraceConfig, size: int = 256, seed: int = 7) -> None:
        self.config = config
        self.size = size
        self.rng = random.Random(seed)
        self.text_tokenizer = CharTokenizer(config.text_vocab_size)
        self.time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
        self.score_tokenizer = NumericTokenizer(config.score_vocab, width=3)

        self.samples = [self._make_sample(index) for index in range(size)]

    def _make_sample(self, index: int) -> dict:
        num_frames = self.config.max_frames
        frame_times = torch.linspace(0.0, float(num_frames - 1), steps=num_frames)
        frames = self._make_structured_frames(index, num_frames)

        num_events = self.rng.randint(1, self.config.max_events)
        events = []
        for event_idx in range(num_events):
            start = round(event_idx * 1.3 + self.rng.uniform(0.0, 0.4), 1)
            end = round(start + self.rng.uniform(0.6, 1.4), 1)
            score = round(min(9.9, 3.0 + event_idx * 1.5 + self.rng.uniform(0.0, 1.0)), 1)
            caption = f"event {event_idx + 1} action {index % 5}"
            events.append({"timestamp": [start, end], "score": [score], "caption": caption})

        instruction = "localize events and describe them"
        token_ids, label_types, prompt_length = self._serialize_example(events, instruction)
        return {
            "frames": frames,
            "frame_times": frame_times,
            "events": events,
            "instruction": instruction,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label_types": torch.tensor(label_types, dtype=torch.long),
            "prompt_length": prompt_length,
        }

    def _make_structured_frames(self, index: int, num_frames: int) -> torch.Tensor:
        """Create deterministic, MobileCLIP-visible patterns for overfit tests.

        Pure random noise is out of distribution for a frozen image encoder and
        can collapse to nearly identical representations. These samples retain
        synthetic simplicity while making identity available only through the
        visual prefix, not through the shared instruction.
        """
        size = self.config.image_size
        action_id = index % 5
        frames = torch.zeros(num_frames, 3, size, size)
        x = torch.linspace(0.0, 1.0, steps=size).view(1, size).expand(size, size)
        y = torch.linspace(0.0, 1.0, steps=size).view(size, 1).expand(size, size)
        stripe_width = max(2, size // 12)

        for frame_index in range(num_frames):
            frames[frame_index, 0] = (action_id + 1) / 5.0
            frames[frame_index, 1] = x if action_id % 2 == 0 else y
            frames[frame_index, 2] = (frame_index + 1) / num_frames
            stripe_start = (action_id * stripe_width * 2 + frame_index * stripe_width) % size
            stripe_end = min(size, stripe_start + stripe_width)
            if action_id % 2 == 0:
                frames[frame_index, :, :, stripe_start:stripe_end] = 1.0
            else:
                frames[frame_index, :, stripe_start:stripe_end, :] = 1.0
        return frames

    def _serialize_example(self, events: list[dict], instruction: str) -> tuple[list[int], list[int], int]:
        return serialize_example(
            events,
            instruction,
            self.config,
            self.text_tokenizer,
            self.time_tokenizer,
            self.score_tokenizer,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class JsonTinyTraceDataset(Dataset):
    def __init__(
        self,
        annotation_path: str | Path,
        config: TinyTraceConfig,
        frame_cache_dir: str | Path | None = None,
        allow_random_frames: bool = True,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.config = config
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self.allow_random_frames = allow_random_frames
        self.text_tokenizer = CharTokenizer(config.text_vocab_size)
        self.time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
        self.score_tokenizer = NumericTokenizer(config.score_vocab, width=3)

        payload = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("TinyTrace JSON dataset must be a list of samples.")
        self.items = payload

    def _convert_sample(self, item: dict) -> dict:
        num_frames = int(item.get("num_frames", self.config.max_frames))
        height = int(item.get("image_size", self.config.image_size))
        width = int(item.get("image_size", self.config.image_size))

        if "video_path" in item:
            frames, frame_times_tensor = self._load_video_frames_cached(item["video_path"], num_frames)
        elif "frames_path" in item:
            frames = torch.load(item["frames_path"], map_location="cpu", weights_only=True).float()
            frame_times_tensor = torch.linspace(0.0, float(num_frames - 1), steps=num_frames)
        elif self.allow_random_frames:
            frames = torch.rand(num_frames, 3, height, width)
            frame_times_tensor = torch.linspace(0.0, float(num_frames - 1), steps=num_frames)
        else:
            raise ValueError(
                "Real-data sample must define video_path or frames_path; "
                "random fallback frames are disabled."
            )

        instruction = item.get("instruction", "localize events and describe them")
        events = item.get("events", [])
        token_ids, label_types, prompt_length = self._serialize_example(events, instruction)

        return {
            "source_id": item.get("source_id"),
            "video_path": item.get("video_path"),
            "frames": frames,
            "frame_times": frame_times_tensor,
            "events": events,
            "instruction": instruction,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label_types": torch.tensor(label_types, dtype=torch.long),
            "prompt_length": prompt_length,
        }

    def _load_video_frames_cached(
        self,
        video_path: str,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.frame_cache_dir is None:
            return self._load_video_frames(video_path, num_frames)

        source = Path(video_path)
        stat = source.stat() if source.exists() else None
        identity = "|".join(
            [
                str(source.resolve()),
                str(stat.st_size if stat else "missing"),
                str(stat.st_mtime_ns if stat else "missing"),
                str(num_frames),
                str(self.config.image_size),
            ]
        )
        cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        cache_path = self.frame_cache_dir / f"{cache_key}.pt"
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            return cached["frames"], cached["frame_times"]

        frames, frame_times = self._load_video_frames(video_path, num_frames)
        self.frame_cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save({"frames": frames, "frame_times": frame_times}, temporary_path)
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return frames, frame_times

    def _load_video_frames(self, video_path: str, num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        duration = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                text=True,
            ).strip()
        )
        if duration <= 0:
            raise ValueError(f"Invalid video duration: {video_path}")

        safe_end = max(duration - 0.25, 0.0)
        frame_times = torch.linspace(0.0, safe_end, steps=num_frames)
        frames = []
        for timestamp in frame_times.tolist():
            frame = self._read_single_frame(video_path, timestamp)
            frames.append(frame.unsqueeze(0))
        return torch.cat(frames, dim=0), frame_times

    def _read_single_frame(self, video_path: str, timestamp: float) -> torch.Tensor:
        size = self.config.image_size
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-vf",
            f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
            "-",
        ]
        expected = size * size * 3
        for candidate in (timestamp, max(timestamp - 0.1, 0.0), max(timestamp - 0.3, 0.0)):
            command[4] = f"{candidate:.3f}"
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0 and len(result.stdout) == expected:
                frame_bytes = bytearray(result.stdout)
                frame = torch.frombuffer(frame_bytes, dtype=torch.uint8).view(size, size, 3).permute(2, 0, 1).float() / 255.0
                return frame
        raise ValueError(f"Could not decode frame near {timestamp:.3f}s from {video_path}")

    def _serialize_example(self, events: list[dict], instruction: str) -> tuple[list[int], list[int], int]:
        return serialize_example(
            events,
            instruction,
            self.config,
            self.text_tokenizer,
            self.time_tokenizer,
            self.score_tokenizer,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self._convert_sample(self.items[index])


def tinytrace_collate_fn(batch: list[dict]) -> dict:
    max_seq = max(sample["token_ids"].size(0) for sample in batch)
    max_frames = max(sample["frames"].size(0) for sample in batch)
    padded_tokens = []
    padded_types = []
    padded_frames = []
    padded_frame_times = []
    frame_masks = []

    for sample in batch:
        pad_len = max_seq - sample["token_ids"].size(0)
        padded_tokens.append(
            torch.cat(
                [sample["token_ids"], torch.full((pad_len,), 0, dtype=torch.long)],
                dim=0,
            )
        )
        padded_types.append(
            torch.cat(
                [sample["label_types"], torch.full((pad_len,), -1, dtype=torch.long)],
                dim=0,
            )
        )

        frame_count = sample["frames"].size(0)
        frame_pad = max_frames - frame_count
        padded_frames.append(
            torch.cat(
                [
                    sample["frames"],
                    torch.zeros(
                        frame_pad,
                        *sample["frames"].shape[1:],
                        dtype=sample["frames"].dtype,
                    ),
                ],
                dim=0,
            )
        )
        padded_frame_times.append(
            torch.cat(
                [sample["frame_times"], torch.zeros(frame_pad, dtype=sample["frame_times"].dtype)],
                dim=0,
            )
        )
        frame_masks.append(
            torch.cat(
                [torch.ones(frame_count, dtype=torch.bool), torch.zeros(frame_pad, dtype=torch.bool)],
                dim=0,
            )
        )

    return {
        "frames": torch.stack(padded_frames, dim=0),
        "frame_times": torch.stack(padded_frame_times, dim=0),
        "frame_mask": torch.stack(frame_masks, dim=0),
        "token_ids": torch.stack(padded_tokens, dim=0),
        "label_types": torch.stack(padded_types, dim=0),
        "events": [sample["events"] for sample in batch],
        "instruction": [sample["instruction"] for sample in batch],
    }
