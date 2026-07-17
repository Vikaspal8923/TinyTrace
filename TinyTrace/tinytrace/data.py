from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
import uuid
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import TinyTraceConfig
from .serialization import caption_budget_metadata, serialize_example
from .tokenizers import CharTokenizer, NumericTokenizer


FRAME_CACHE_FORMAT_VERSION = 2


def sample_uniform_frame_times(
    duration: float,
    requested_frames: int,
    safety_margin: float = 0.25,
) -> torch.Tensor:
    """Return deterministic, monotonic timestamps inside a safe decode range.

    Videos whose safe decode range collapses to zero return one valid timestamp
    instead of duplicating the first frame to imitate additional evidence.
    Variable-frame collation supplies the corresponding padding mask.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Video duration must be finite and positive, received {duration!r}.")
    if not isinstance(requested_frames, int) or isinstance(requested_frames, bool) or requested_frames < 1:
        raise ValueError("requested_frames must be a positive integer.")
    if not math.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError("safety_margin must be finite and non-negative.")

    safe_end = max(float(duration) - float(safety_margin), 0.0)
    if requested_frames == 1 or safe_end == 0.0:
        return torch.zeros(1, dtype=torch.float32)
    return torch.linspace(0.0, safe_end, steps=requested_frames, dtype=torch.float32)


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
        task_mode = "caption"
        token_ids, label_types, prompt_length = self._serialize_example(events, instruction, task_mode)
        return {
            "frames": frames,
            "frame_times": frame_times,
            "events": events,
            "instruction": instruction,
            "task_mode": task_mode,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label_types": torch.tensor(label_types, dtype=torch.long),
            "prompt_length": prompt_length,
            "caption_budget": caption_budget_metadata(events, self.config, self.text_tokenizer),
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

    def _serialize_example(
        self, events: list[dict], instruction: str, task_mode: str = "caption"
    ) -> tuple[list[int], list[int], int]:
        return serialize_example(
            events,
            instruction,
            self.config,
            self.text_tokenizer,
            self.time_tokenizer,
            self.score_tokenizer,
            task_mode=task_mode,
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
        validate_videos_on_init: bool = False,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.config = config
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self.allow_random_frames = allow_random_frames
        self.max_retries_per_sample = 8
        self.text_tokenizer = CharTokenizer(config.text_vocab_size)
        self.time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
        self.score_tokenizer = NumericTokenizer(config.score_vocab, width=3)

        payload = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("TinyTrace JSON dataset must be a list of samples.")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("Each TinyTrace JSON sample must be an object.")
        self.items = (
            self._filter_valid_items(payload)
            if validate_videos_on_init
            else [self._resolve_item_video_path(item) for item in payload]
        )

    def _resolve_item_video_path(self, item: dict) -> dict:
        if not isinstance(item, dict):
            raise ValueError("Each TinyTrace JSON sample must be an object.")
        resolved = dict(item)
        if item.get("video_path"):
            source = self._resolve_media_path(str(item["video_path"]))
            resolved["video_path"] = str(source)
        if item.get("frames_path"):
            source = self._resolve_media_path(str(item["frames_path"]))
            resolved["frames_path"] = str(source)
        return resolved

    def _resolve_media_path(self, media_path: str) -> Path:
        source = Path(media_path)
        if source.is_absolute():
            return source
        if source.is_file():
            return source.resolve()

        annotation_root = self.annotation_path.parent
        candidates = [
            annotation_root / source,
            annotation_root.parent / source,
            annotation_root.parent.parent / source,
        ]

        source_parts = source.parts
        if source_parts and source_parts[0] == "final_qvhighlights_tinytrace":
            trimmed = Path(*source_parts[1:])
            candidates.extend(
                [
                    annotation_root.parent / trimmed,
                    annotation_root.parent.parent / trimmed,
                ]
            )

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return source

    def _filter_valid_items(self, payload: list[dict]) -> list[dict]:
        valid_items = []
        skipped_missing = 0
        skipped_invalid = 0
        for item in payload:
            video_path = item.get("video_path")
            if not video_path:
                valid_items.append(item)
                continue

            source = self._resolve_media_path(video_path)
            if not source.is_file():
                skipped_missing += 1
                continue
            item = dict(item)
            item["video_path"] = str(source)

            try:
                duration = self._probe_video_duration(str(source))
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
                skipped_invalid += 1
                continue
            if duration <= 0:
                skipped_invalid += 1
                continue

            try:
                max_event_time = 0.0
                for event in item.get("events", []):
                    timestamp = event.get("timestamp", [])
                    if isinstance(timestamp, list) and timestamp:
                        max_event_time = max(max_event_time, max(float(value) for value in timestamp))
            except (AttributeError, TypeError, ValueError):
                skipped_invalid += 1
                continue

            if max_event_time > duration + 0.5:
                skipped_invalid += 1
                continue
            valid_items.append(item)

        if not valid_items:
            raise ValueError(
                "No valid TinyTrace samples remain after video validation. "
                "Check downloaded clips and annotation paths."
            )

        if skipped_missing or skipped_invalid:
            print(
                "JsonTinyTraceDataset filtered invalid samples: "
                f"kept={len(valid_items)} skipped_missing={skipped_missing} skipped_invalid={skipped_invalid}"
            )
        return valid_items

    @staticmethod
    def _probe_video_duration(video_path: str) -> float:
        probe = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=duration:format=duration",
                "-of",
                "json",
                video_path,
            ],
            text=True,
        )
        payload = json.loads(probe)
        candidates = []
        for stream in payload.get("streams", []):
            value = stream.get("duration")
            if value not in (None, "N/A", ""):
                try:
                    candidates.append(float(value))
                except ValueError:
                    pass
        format_value = payload.get("format", {}).get("duration")
        if format_value not in (None, "N/A", ""):
            try:
                candidates.append(float(format_value))
            except ValueError:
                pass
        positive = [value for value in candidates if value > 0]
        return min(positive) if positive else 0.0

    def _convert_sample(self, item: dict) -> dict:
        num_frames = item.get("num_frames", self.config.max_frames)
        if not isinstance(num_frames, int) or isinstance(num_frames, bool):
            raise ValueError("num_frames must be an integer.")
        if not 1 <= num_frames <= self.config.max_frames:
            raise ValueError(
                f"num_frames must be between 1 and configured max_frames={self.config.max_frames}."
            )
        image_size = item.get("image_size", self.config.image_size)
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 1:
            raise ValueError("image_size must be a positive integer.")
        height = image_size
        width = image_size

        if "video_path" in item:
            frames, frame_times_tensor = self._load_video_frames_cached(item["video_path"], num_frames)
        elif "frames_path" in item:
            loaded_frames = torch.load(item["frames_path"], map_location="cpu", weights_only=True)
            if not isinstance(loaded_frames, torch.Tensor):
                raise ValueError("frames_path must contain a frame tensor.")
            frames = loaded_frames.float()
            frame_times_tensor = torch.linspace(
                0.0,
                float(max(frames.size(0) - 1, 0)),
                steps=frames.size(0),
            )
        elif self.allow_random_frames:
            frames = torch.rand(num_frames, 3, height, width)
            frame_times_tensor = torch.linspace(0.0, float(num_frames - 1), steps=num_frames)
        else:
            raise ValueError(
                "Real-data sample must define video_path or frames_path; "
                "random fallback frames are disabled."
            )

        if (
            frames.ndim != 4
            or not 1 <= frames.size(0) <= num_frames
            or frames.size(1) != 3
        ):
            raise ValueError(
                "Decoded frames must contain between 1 and "
                f"{num_frames} RGB frames, received {tuple(frames.shape)}."
            )
        if not frames.is_floating_point() or not torch.isfinite(frames).all():
            raise ValueError("Decoded frames must be finite floating-point tensors.")
        if frames.numel() and (frames.min() < 0 or frames.max() > 1):
            raise ValueError("Decoded frames must be in the [0, 1] range.")
        if frame_times_tensor.shape != (frames.size(0),):
            raise ValueError("frame_times must contain one timestamp per decoded frame.")

        instruction = item.get("instruction", "localize events and describe them")
        events = item.get("events", [])
        task_mode = item.get("task_mode", "caption")
        token_ids, label_types, prompt_length = self._serialize_example(
            events, instruction, task_mode
        )

        return {
            "source_id": item.get("source_id"),
            "video_path": item.get("video_path"),
            "frames": frames,
            "frame_times": frame_times_tensor,
            "events": events,
            "instruction": instruction,
            "task_mode": task_mode,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label_types": torch.tensor(label_types, dtype=torch.long),
            "prompt_length": prompt_length,
            "caption_budget": caption_budget_metadata(events, self.config, self.text_tokenizer),
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
                str(FRAME_CACHE_FORMAT_VERSION),
            ]
        )
        cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        cache_path = self.frame_cache_dir / f"{cache_key}.pt"
        if cache_path.is_file():
            cached = self._read_frame_cache(cache_path, num_frames)
            if cached is not None:
                return cached

        frames, frame_times = self._load_video_frames(video_path, num_frames)
        self.frame_cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(
                {
                    "format_version": FRAME_CACHE_FORMAT_VERSION,
                    "frames": frames,
                    "frame_times": frame_times,
                },
                temporary_path,
            )
            try:
                temporary_path.replace(cache_path)
            except PermissionError:
                # On Windows another worker may have published and immediately
                # opened the same cache entry. Its complete atomic result wins;
                # only fail when no destination was actually published.
                if not cache_path.is_file():
                    raise
        finally:
            temporary_path.unlink(missing_ok=True)
        return frames, frame_times

    def _read_frame_cache(
        self,
        cache_path: Path,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if not isinstance(cached, dict):
                raise ValueError("cache payload is not an object")
            if cached.get("format_version") != FRAME_CACHE_FORMAT_VERSION:
                raise ValueError("cache format version does not match")
            frames = cached["frames"]
            frame_times = cached["frame_times"]
            expected_spatial_shape = (3, self.config.image_size, self.config.image_size)
            if (
                not isinstance(frames, torch.Tensor)
                or frames.ndim != 4
                or not 1 <= frames.size(0) <= num_frames
                or tuple(frames.shape[1:]) != expected_spatial_shape
            ):
                raise ValueError(
                    "cached frames must contain between 1 and "
                    f"{num_frames} frames with shape {expected_spatial_shape}, "
                    f"received {getattr(frames, 'shape', None)}"
                )
            if frames.dtype != torch.float32 or not torch.isfinite(frames).all():
                raise ValueError("cached frames must be finite float32 tensors")
            if frames.numel() and (frames.min() < 0 or frames.max() > 1):
                raise ValueError("cached frames must be in the [0, 1] range")
            if not isinstance(frame_times, torch.Tensor) or tuple(frame_times.shape) != (frames.size(0),):
                raise ValueError("cached frame_times have an invalid shape")
            if frame_times.dtype != torch.float32 or not torch.isfinite(frame_times).all():
                raise ValueError("cached frame_times must be finite float32 tensors")
            if frame_times.numel() and frame_times[0] < 0:
                raise ValueError("cached frame_times cannot be negative")
            if frame_times.numel() > 1 and (frame_times[1:] < frame_times[:-1]).any():
                raise ValueError("cached frame_times must be monotonically nondecreasing")
            return frames, frame_times
        except PermissionError as exc:
            # A concurrent Windows reader/writer can hold the file briefly.
            # Treat that as a cache miss without deleting another worker's entry.
            print(f"TinyTrace frame cache is temporarily unavailable {cache_path}: {exc}")
            return None
        except Exception as exc:
            cache_path.unlink(missing_ok=True)
            print(f"Ignoring invalid TinyTrace frame cache {cache_path}: {exc}")
            return None

    def _load_video_frames(self, video_path: str, num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        duration = self._probe_video_duration(video_path)
        if duration <= 0:
            raise ValueError(f"Invalid video duration: {video_path}")

        frame_times = sample_uniform_frame_times(duration, num_frames)
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

    def _serialize_example(
        self, events: list[dict], instruction: str, task_mode: str = "caption"
    ) -> tuple[list[int], list[int], int]:
        return serialize_example(
            events,
            instruction,
            self.config,
            self.text_tokenizer,
            self.time_tokenizer,
            self.score_tokenizer,
            task_mode=task_mode,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        if not self.items:
            raise IndexError("TinyTrace dataset is empty.")
        errors: list[str] = []
        first_exception: Exception | None = None
        for offset in range(min(self.max_retries_per_sample, len(self.items))):
            candidate_index = (index + offset) % len(self.items)
            item = self.items[candidate_index]
            try:
                sample = self._convert_sample(item)
            except Exception as exc:
                if first_exception is None:
                    first_exception = exc
                video_path = item.get("video_path") or item.get("frames_path") or f"item[{candidate_index}]"
                errors.append(f"{video_path}: {exc}")
                continue
            if offset > 0:
                print(
                    f"JsonTinyTraceDataset skipped invalid sample at index={index} "
                    f"and used fallback index={candidate_index}."
                )
            return sample
        if len(self.items) == 1 and first_exception is not None:
            raise first_exception
        joined = "; ".join(errors[:3])
        raise RuntimeError(
            "Unable to load a valid TinyTrace sample after retries. "
            f"Recent errors: {joined}"
        )


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
        "task_mode": [sample.get("task_mode", "caption") for sample in batch],
        "caption_budget": [
            sample.get(
                "caption_budget",
                {
                    "available": False,
                    "max_caption_tokens": None,
                    "event_count": 0,
                    "truncated_event_count": 0,
                    "original_caption_tokens": 0,
                    "retained_caption_tokens": 0,
                    "events": [],
                },
            )
            for sample in batch
        ],
    }
