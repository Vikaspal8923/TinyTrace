from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import TinyTraceConfig
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
        frames = torch.rand(num_frames, 3, self.config.image_size, self.config.image_size)

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

    def _serialize_example(self, events: list[dict], instruction: str) -> tuple[list[int], list[int], int]:
        instruction_ids = [self.config.bos_token_id]
        instruction_ids.extend(self.text_tokenizer.encode(instruction)[: self.config.max_text_len])
        instruction_ids.append(self.config.video_token_id)
        prompt_length = len(instruction_ids)

        token_ids = instruction_ids
        label_types = [-1] * len(instruction_ids)

        for event in events:
            time_ids = [
                self.config.sync_token_id if idx == 0 else self.config.time_token_base + idx
                for idx in self.time_tokenizer.encode(event["timestamp"])
            ]
            score_ids = [
                self.config.sync_token_id if idx == 0 else self.config.score_token_base + idx
                for idx in self.score_tokenizer.encode(event["score"])
            ]
            caption_ids = self.text_tokenizer.encode(event["caption"])[: self.config.max_caption_tokens]
            caption_ids.append(self.config.sync_token_id)

            token_ids.extend(time_ids)
            label_types.extend([2 if token != self.config.sync_token_id else 4 for token in time_ids])

            token_ids.extend(score_ids)
            label_types.extend([3 if token != self.config.sync_token_id else 5 for token in score_ids])

            token_ids.extend(caption_ids)
            label_types.extend([0 if token != self.config.sync_token_id else 1 for token in caption_ids])

        token_ids.append(self.config.eos_token_id)
        label_types.append(0)
        return token_ids, label_types, prompt_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class JsonTinyTraceDataset(Dataset):
    def __init__(self, annotation_path: str | Path, config: TinyTraceConfig) -> None:
        self.annotation_path = Path(annotation_path)
        self.config = config
        self.text_tokenizer = CharTokenizer(config.text_vocab_size)
        self.time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
        self.score_tokenizer = NumericTokenizer(config.score_vocab, width=3)

        payload = json.loads(self.annotation_path.read_text())
        if not isinstance(payload, list):
            raise ValueError("TinyTrace JSON dataset must be a list of samples.")
        self.samples = [self._convert_sample(item) for item in payload]

    def _convert_sample(self, item: dict) -> dict:
        num_frames = int(item.get("num_frames", self.config.max_frames))
        height = int(item.get("image_size", self.config.image_size))
        width = int(item.get("image_size", self.config.image_size))

        if "video_path" in item:
            frames, frame_times_tensor = self._load_video_frames(item["video_path"], num_frames)
        elif "frames_path" in item:
            frames = torch.load(item["frames_path"]).float()
            frame_times_tensor = torch.linspace(0.0, float(num_frames - 1), steps=num_frames)
        else:
            frames = torch.rand(num_frames, 3, height, width)
            frame_times_tensor = torch.linspace(0.0, float(num_frames - 1), steps=num_frames)

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
        instruction_ids = [self.config.bos_token_id]
        instruction_ids.extend(self.text_tokenizer.encode(instruction)[: self.config.max_text_len])
        instruction_ids.append(self.config.video_token_id)
        prompt_length = len(instruction_ids)

        token_ids = instruction_ids
        label_types = [-1] * len(instruction_ids)

        for event in events:
            time_ids = [
                self.config.sync_token_id if idx == 0 else self.config.time_token_base + idx
                for idx in self.time_tokenizer.encode(event["timestamp"])
            ]
            score_ids = [
                self.config.sync_token_id if idx == 0 else self.config.score_token_base + idx
                for idx in self.score_tokenizer.encode(event["score"])
            ]
            caption_ids = self.text_tokenizer.encode(event["caption"])[: self.config.max_caption_tokens]
            caption_ids.append(self.config.sync_token_id)

            token_ids.extend(time_ids)
            label_types.extend([2 if token != self.config.sync_token_id else 4 for token in time_ids])

            token_ids.extend(score_ids)
            label_types.extend([3 if token != self.config.sync_token_id else 5 for token in score_ids])

            token_ids.extend(caption_ids)
            label_types.extend([0 if token != self.config.sync_token_id else 1 for token in caption_ids])

        token_ids.append(self.config.eos_token_id)
        label_types.append(0)
        return token_ids, label_types, prompt_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


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
