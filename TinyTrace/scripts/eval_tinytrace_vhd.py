from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from tinytrace import JsonTinyTraceDataset, TinyTraceConfig, TinyTraceModel, decode_event_sequence
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="TinyTrace/outputs-qvh-smoke/tinytrace.pt")
    parser.add_argument("--dataset-json", type=str, default="TinyTrace/data/qvh_tinytrace_subset.json")
    parser.add_argument("--source-json", type=str, default="dataset/mt_fmt-8k.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default="TinyTrace/outputs-qvh-smoke/qvh_metrics.json")
    return parser.parse_args()


def infer_duration_from_video_name(video_path: str) -> float:
    stem = Path(video_path).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer duration from video name: {video_path}")
    start = float(parts[-2])
    end = float(parts[-1])
    return max(0.0, end - start)


def event_to_clip_scores(events: list[dict], duration: float, clip_length: float = 2.0) -> list[float]:
    clip_count = max(1, int(math.ceil(duration / clip_length)))
    clip_scores = np.zeros(clip_count, dtype=np.float32)
    clip_hits = np.zeros(clip_count, dtype=np.float32)

    for event in events:
        timestamps = event.get("timestamp", [])
        scores = event.get("score", [])
        if len(timestamps) < 2 or not scores:
            continue
        start = max(0.0, min(float(timestamps[0]), duration))
        end = max(start, min(float(timestamps[1]), duration))
        score = float(scores[0])
        start_idx = max(0, min(clip_count - 1, int(start / clip_length)))
        end_idx = max(0, min(clip_count - 1, int(end / clip_length)))
        for clip_idx in range(start_idx, end_idx + 1):
            clip_scores[clip_idx] += score
            clip_hits[clip_idx] += 1.0

    nonzero = clip_hits > 0
    clip_scores[nonzero] = clip_scores[nonzero] / clip_hits[nonzero]
    return clip_scores.tolist()


def dense_gt_clip_scores(item: dict, duration: float, clip_length: float = 2.0) -> np.ndarray:
    clip_count = max(1, int(math.ceil(duration / clip_length)))
    clip_scores = np.zeros(clip_count, dtype=np.float32)
    clip_hits = np.zeros(clip_count, dtype=np.float32)

    for time_row, score_row in zip(item.get("times", []), item.get("scores", [])):
        if not time_row or not score_row:
            continue
        timestamp = float(time_row[0])
        score = float(score_row[0])
        clip_idx = max(0, min(clip_count - 1, int(timestamp / clip_length)))
        clip_scores[clip_idx] += score
        clip_hits[clip_idx] += 1.0

    nonzero = clip_hits > 0
    clip_scores[nonzero] = clip_scores[nonzero] / clip_hits[nonzero]
    return clip_scores


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    ranked = labels[order]
    tp = np.cumsum(ranked)
    precision = tp / (np.arange(len(ranked)) + 1.0)
    return float((precision * ranked).sum() / positives)


def evaluate_highlight_metrics(predictions: list[dict], ground_truth: list[dict]) -> OrderedDict:
    thresholds = [(2.0, "Fair"), (3.0, "Good"), (4.0, "VeryGood")]
    qid2pred = {int(item["qid"]): np.array(item["pred_saliency_scores"], dtype=np.float32) for item in predictions}
    qid2gt = {int(item["qid"]): np.array(item["gt_saliency_scores"], dtype=np.float32) for item in ground_truth}

    results: OrderedDict[str, float] = OrderedDict()
    for threshold, label in thresholds:
        aps = []
        hits = []
        for qid, gt_scores in qid2gt.items():
            pred_scores = qid2pred[qid]
            size = min(len(pred_scores), len(gt_scores))
            pred_scores = pred_scores[:size]
            gt_scores = gt_scores[:size]
            gt_binary = (gt_scores >= threshold).astype(np.float32)
            aps.append(average_precision(pred_scores, gt_binary))
            top_idx = int(np.argmax(pred_scores)) if len(pred_scores) else 0
            hits.append(float(gt_binary[top_idx]) if len(gt_binary) else 0.0)
        results[f"HL-min-{label}-mAP"] = float(np.mean(aps) * 100.0)
        results[f"HL-min-{label}-Hit1"] = float(np.mean(hits) * 100.0)
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = TinyTraceConfig()
    model = TinyTraceModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = JsonTinyTraceDataset(args.dataset_json, config=config)
    source_items = json.loads(Path(args.source_json).read_text())
    source_by_id = {int(item["id"]): item for item in source_items}

    text_tokenizer = CharTokenizer(config.text_vocab_size)
    time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
    score_tokenizer = NumericTokenizer(config.score_vocab, width=3)

    predictions = []
    ground_truth = []

    for sample in dataset:
        source_id = int(sample["source_id"])
        duration = infer_duration_from_video_name(sample["video_path"])
        prompt_ids = sample["token_ids"][: sample["prompt_length"]].unsqueeze(0).to(device)
        generated = model.generate(
            sample["frames"].unsqueeze(0).to(device),
            sample["frame_times"].unsqueeze(0).to(device),
            prompt_ids,
            max_new_tokens=config.max_generated_tokens,
        )
        predicted_events = decode_event_sequence(
            generated[0].tolist()[prompt_ids.size(1):],
            config,
            text_tokenizer,
            time_tokenizer,
            score_tokenizer,
        )

        pred_clip_scores = event_to_clip_scores(predicted_events, duration)
        gt_clip_scores = dense_gt_clip_scores(source_by_id[source_id], duration).tolist()
        qid = source_id

        predictions.append(
            {
                "qid": qid,
                "pred_saliency_scores": pred_clip_scores,
                "predicted_events": predicted_events,
                "video_path": sample["video_path"],
            }
        )
        ground_truth.append(
            {
                "qid": qid,
                "gt_saliency_scores": gt_clip_scores,
                "events": sample["events"],
                "video_path": sample["video_path"],
            }
        )

    metrics = evaluate_highlight_metrics(predictions, ground_truth)
    output = {
        "metrics": metrics,
        "predictions": predictions,
    }

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(output, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
