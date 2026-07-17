from __future__ import annotations

from typing import Iterable
import math
from pathlib import Path


def temporal_iou(first: list[float], second: list[float]) -> float:
    if len(first) < 2 or len(second) < 2:
        return 0.0
    start = max(float(first[0]), float(second[0]))
    end = min(float(first[1]), float(second[1]))
    intersection = max(0.0, end - start)
    union = max(float(first[1]), float(second[1])) - min(float(first[0]), float(second[0]))
    if union <= 0:
        return 0.0
    return intersection / union


def _best_ious(ground_truth: list[dict], predicted: list[dict]) -> list[float]:
    if not ground_truth:
        return []
    values = []
    for gt_event in ground_truth:
        gt_ts = gt_event.get("timestamp", [])
        if not predicted:
            values.append(0.0)
            continue
        values.append(
            max(
                temporal_iou(gt_ts, pred_event.get("timestamp", []))
                for pred_event in predicted
            )
        )
    return values


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _video_duration_from_path(video_path: str) -> float | None:
    parts = Path(video_path).stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return max(0.0, float(parts[-1]) - float(parts[-2]))
    except ValueError:
        return None


def _events_to_clip_scores(events: list[dict], duration: float, clip_length: float = 2.0) -> list[float]:
    clip_count = max(1, int(math.ceil(duration / clip_length)))
    totals = [0.0] * clip_count
    counts = [0] * clip_count
    for event in events:
        timestamps = event.get("timestamp", [])
        scores = event.get("score", [])
        if len(timestamps) != 2 or not scores:
            continue
        start = min(max(float(timestamps[0]), 0.0), duration)
        end = min(max(float(timestamps[1]), start), duration)
        first = min(int(start / clip_length), clip_count - 1)
        last = min(int(end / clip_length), clip_count - 1)
        for index in range(first, last + 1):
            totals[index] += float(scores[0])
            counts[index] += 1
    return [total / count if count else 0.0 for total, count in zip(totals, counts)]


def _average_precision(scores: list[float], labels: list[bool]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(ranked, start=1):
        if labels[index]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def evaluate_qvhighlights(samples: list[dict], positive_score: float = 3.0) -> dict[str, float]:
    average_precisions: list[float] = []
    hit_at_one: list[float] = []
    for sample in samples:
        if sample.get("task_mode") != "highlight":
            continue
        duration = _video_duration_from_path(str(sample.get("video_path", "")))
        if not duration:
            continue
        ground_truth = _events_to_clip_scores(list(sample.get("ground_truth", [])), duration)
        predicted = _events_to_clip_scores(list(sample.get("predicted", [])), duration)
        labels = [score >= positive_score for score in ground_truth]
        average_precisions.append(_average_precision(predicted, labels))
        best_index = max(range(len(predicted)), key=predicted.__getitem__)
        hit_at_one.append(float(labels[best_index]))
    return {
        "qvh_mAP": _mean(average_precisions),
        "qvh_HIT_at_1": _mean(hit_at_one),
    }


def evaluate_event_predictions(samples: list[dict]) -> dict[str, float]:
    best_ious: list[float] = []
    top1_ious: list[float] = []
    score_errors: list[float] = []
    caption_exact_hits: list[float] = []
    event_count_errors: list[float] = []

    for sample in samples:
        ground_truth = list(sample.get("ground_truth", []))
        predicted = list(sample.get("predicted", []))
        event_count_errors.append(abs(len(predicted) - len(ground_truth)))
        best_ious.extend(_best_ious(ground_truth, predicted))

        if ground_truth:
            top1_ious.append(
                temporal_iou(
                    ground_truth[0].get("timestamp", []),
                    predicted[0].get("timestamp", []) if predicted else [],
                )
            )
            if predicted:
                gt_score = float(ground_truth[0].get("score", [0.0])[0])
                pred_score = float(predicted[0].get("score", [0.0])[0])
                score_errors.append(abs(pred_score - gt_score))
                gt_caption = str(ground_truth[0].get("caption", "")).strip().lower()
                pred_caption = str(predicted[0].get("caption", "")).strip().lower()
                caption_exact_hits.append(float(gt_caption == pred_caption and bool(gt_caption)))
            else:
                score_errors.append(abs(float(ground_truth[0].get("score", [0.0])[0])))
                caption_exact_hits.append(0.0)

    metrics = {
        "temporal_mean_iou": _mean(best_ious),
        "r1_iou_0.3": _mean(1.0 if value >= 0.3 else 0.0 for value in top1_ious),
        "r1_iou_0.5": _mean(1.0 if value >= 0.5 else 0.0 for value in top1_ious),
        "score_mae": _mean(score_errors),
        "caption_exact_match": _mean(caption_exact_hits),
        "event_count_mae": _mean(event_count_errors),
    }
    metrics.update(evaluate_qvhighlights(samples))
    return metrics
