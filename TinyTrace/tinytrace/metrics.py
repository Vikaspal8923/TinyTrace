from __future__ import annotations

from typing import Iterable


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

    return {
        "temporal_mean_iou": _mean(best_ious),
        "r1_iou_0.3": _mean(1.0 if value >= 0.3 else 0.0 for value in top1_ious),
        "r1_iou_0.5": _mean(1.0 if value >= 0.5 else 0.0 for value in top1_ious),
        "score_mae": _mean(score_errors),
        "caption_exact_match": _mean(caption_exact_hits),
        "event_count_mae": _mean(event_count_errors),
    }
