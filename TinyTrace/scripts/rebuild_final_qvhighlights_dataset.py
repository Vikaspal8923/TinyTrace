from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the final TinyTrace-ready QVHighlights dataset folder from downloaded clips."
    )
    parser.add_argument("--source-json", type=str, default="dataset/qvhighlights/mt_fmt-2000-valid.json")
    parser.add_argument("--video-dir", type=str, default="dataset/qvhighlights/videos/train")
    parser.add_argument("--output-dir", type=str, default="final_qvhighlights_tinytrace")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=3.5)
    parser.add_argument("--link-mode", type=str, default="hardlink", choices=("hardlink", "copy"))
    return parser.parse_args()


def extract_query(prompt: str) -> str:
    match = re.search(r"sentence query: '(.*?)'\. Please return", prompt, flags=re.DOTALL)
    if match:
        return match.group(1)
    return prompt.replace("<video>\n", "").strip()


def build_events(
    times: list[list[float]],
    scores: list[list[float]],
    max_events: int,
    score_threshold: float,
    caption: str,
) -> list[dict]:
    points = [(float(t[0]), float(s[0])) for t, s in zip(times, scores) if t and s]
    segments = []
    current = None

    for time_value, score_value in points:
        if score_value < score_threshold:
            if current is not None:
                segments.append(current)
                current = None
            continue

        if current is None:
            current = {"start": time_value, "end": time_value, "scores": [score_value]}
        elif abs(time_value - current["end"]) <= 0.51:
            current["end"] = time_value
            current["scores"].append(score_value)
        else:
            segments.append(current)
            current = {"start": time_value, "end": time_value, "scores": [score_value]}

    if current is not None:
        segments.append(current)

    if not segments:
        top_points = sorted(points, key=lambda pair: pair[1], reverse=True)[:max_events]
        segments = [{"start": t, "end": t, "scores": [s]} for t, s in top_points]

    segments = sorted(segments, key=lambda seg: (sum(seg["scores"]) / len(seg["scores"])), reverse=True)[:max_events]
    segments = sorted(segments, key=lambda seg: seg["start"])

    return [
        {
            "timestamp": [round(seg["start"], 1), round(seg["end"], 1)],
            "score": [round(sum(seg["scores"]) / len(seg["scores"]), 1)],
            "caption": caption,
        }
        for seg in segments
    ]


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def write_readme(
    path: Path,
    total_count: int,
    train_count: int,
    val_count: int,
    source_json_name: str,
    seed: int,
) -> None:
    content = f"""# Final QVHighlights TinyTrace Dataset

This folder contains the cleaned TinyTrace-ready QVHighlights subset prepared from successfully downloaded valid clips.

## Structure

```text
final_qvhighlights_tinytrace/
├── videos/
│   ├── train/
│   └── val/
├── annotations/
│   ├── qvh_raw_valid.json
│   ├── tinytrace_train.json
│   ├── tinytrace_val.json
│   ├── train_ids.txt
│   ├── val_ids.txt
│   └── download_manifest.json
└── README.md
```

## Contents

- `videos/train/`: {train_count} valid QVHighlights clips
- `videos/val/`: {val_count} valid QVHighlights clips
- `annotations/qvh_raw_valid.json`: original-format filtered annotations for the {total_count} valid clips
- `annotations/tinytrace_train.json`: TinyTrace-ready training JSON
- `annotations/tinytrace_val.json`: TinyTrace-ready validation JSON
- `annotations/train_ids.txt`: train clip filenames
- `annotations/val_ids.txt`: val clip filenames
- `annotations/download_manifest.json`: creation metadata and split settings

## Split Rule

- filenames collected from the valid downloaded clips in `{source_json_name}`
- sorted, then shuffled with seed `{seed}`
- 90/10 split
- final counts:
  - train: `{train_count}`
  - val: `{val_count}`
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    source_json_path = root / args.source_json
    video_dir = root / args.video_dir
    output_dir = root / args.output_dir

    if not source_json_path.is_file():
        raise FileNotFoundError(f"Source JSON not found: {source_json_path}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    items = json.loads(source_json_path.read_text(encoding="utf-8"))
    existing_by_name = {path.name: path.resolve() for path in video_dir.glob("*.mp4")}

    valid_items = []
    seen_names = set()
    for item in items:
        video_name = Path(item["video"]).name
        if video_name in seen_names:
            continue
        local_path = existing_by_name.get(video_name)
        if local_path is None:
            continue
        cloned = dict(item)
        cloned["local_video_path"] = str(local_path)
        valid_items.append(cloned)
        seen_names.add(video_name)

    if not valid_items:
        raise ValueError("No valid downloaded items were found to rebuild the final dataset.")

    ordered_names = sorted(Path(item["local_video_path"]).name for item in valid_items)
    rng = random.Random(args.seed)
    rng.shuffle(ordered_names)
    train_count = int(len(ordered_names) * args.train_ratio)
    train_names = set(ordered_names[:train_count])
    val_names = set(ordered_names[train_count:])

    output_tmp = output_dir.with_name(output_dir.name + "_tmp")
    if output_tmp.exists():
        shutil.rmtree(output_tmp)
    (output_tmp / "videos" / "train").mkdir(parents=True, exist_ok=True)
    (output_tmp / "videos" / "val").mkdir(parents=True, exist_ok=True)
    (output_tmp / "annotations").mkdir(parents=True, exist_ok=True)

    raw_valid = []
    tinytrace_train = []
    tinytrace_val = []

    for item in valid_items:
        source_path = Path(item["local_video_path"])
        video_name = source_path.name
        split = "train" if video_name in train_names else "val"
        target_path = output_tmp / "videos" / split / video_name
        link_or_copy(source_path, target_path, args.link_mode)

        query = extract_query(item["conversations"][0]["value"])
        events = build_events(item["times"], item["scores"], args.max_events, args.score_threshold, query)
        tinytrace_item = {
            "source_id": item["id"],
            "video_path": str(target_path.relative_to(output_tmp)),
            "instruction": "localize highlight events and describe them",
            "events": events,
        }
        raw_item = dict(item)
        raw_item.pop("local_video_path", None)
        raw_valid.append(raw_item)
        if split == "train":
            tinytrace_train.append(tinytrace_item)
        else:
            tinytrace_val.append(tinytrace_item)

    annotations_dir = output_tmp / "annotations"
    (annotations_dir / "qvh_raw_valid.json").write_text(json.dumps(raw_valid, indent=2), encoding="utf-8")
    (annotations_dir / "tinytrace_train.json").write_text(json.dumps(tinytrace_train, indent=2), encoding="utf-8")
    (annotations_dir / "tinytrace_val.json").write_text(json.dumps(tinytrace_val, indent=2), encoding="utf-8")
    (annotations_dir / "train_ids.txt").write_text("\n".join(sorted(train_names)) + "\n", encoding="utf-8")
    (annotations_dir / "val_ids.txt").write_text("\n".join(sorted(val_names)) + "\n", encoding="utf-8")
    manifest = {
        "source_json": args.source_json,
        "video_dir": args.video_dir,
        "total_valid_videos": len(valid_items),
        "train_count": len(tinytrace_train),
        "val_count": len(tinytrace_val),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "max_events": args.max_events,
        "score_threshold": args.score_threshold,
        "link_mode": args.link_mode,
    }
    (annotations_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(
        output_tmp / "README.md",
        total_count=len(valid_items),
        train_count=len(tinytrace_train),
        val_count=len(tinytrace_val),
        source_json_name=args.source_json,
        seed=args.seed,
    )

    backup_dir = output_dir.with_name(output_dir.name + "_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        output_dir.replace(backup_dir)
    output_tmp.replace(output_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    print(f"Rebuilt {output_dir} with {len(valid_items)} valid clips.")
    print(f"train={len(tinytrace_train)} val={len(tinytrace_val)}")


if __name__ == "__main__":
    main()
