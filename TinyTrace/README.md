# TinyTrace

TinyTrace is a lightweight reimplementation of the TRACE causal event modeling idea for constrained environments. It keeps the TRACE event structure and generation order:

- `time -> score -> caption`
- structured event output
- causal autoregressive decoding

while replacing the heavy backbone with a smaller visual encoder and a compact decoder-only transformer.

## What This Repo Contains

- lightweight TinyTrace model code
- training and evaluation scripts
- a synthetic sample dataset for smoke testing
- a converter for small QVHighlights subsets
- TRACE-style highlight evaluation for the QVHighlights setting

## Current Status

TinyTrace is currently a working prototype:

- synthetic-data training works
- real-video loading works
- small real-data QVHighlights subset training works
- TRACE-style highlight metrics run

It is not yet a final strong model. To get better predictions, you need a larger clean video subset and longer training.

## Project Structure

Main code paths:

- `TinyTrace/tinytrace/` : model, data loader, tokenizers, parser
- `TinyTrace/scripts/train_tinytrace.py` : training
- `TinyTrace/scripts/eval_tinytrace.py` : inspect one sample prediction
- `TinyTrace/scripts/eval_tinytrace_vhd.py` : TRACE-style QVHighlights metrics
- `TinyTrace/scripts/prepare_qvhighlights_subset.py` : convert downloaded QVHighlights clips into TinyTrace JSON
- `TinyTrace/configs/tinytrace_baseline.json` : baseline config
- `TinyTrace/data/sample_dataset.json` : minimal sample dataset
- `TinyTrace/trace_lightwieght.md` : project design specification

## Environment Setup

```bash
python3 -m venv TinyTrace/.venv
TinyTrace/.venv/bin/pip install -r TinyTrace/requirements.txt
```

## Smoke Test With Synthetic Data

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --epochs 3 \
  --batch-size 8 \
  --dataset-size 128

PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py
```

## Train With A Small JSON Dataset

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/sample_dataset.json

PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py \
  --dataset-json TinyTrace/data/sample_dataset.json
```

Each TinyTrace sample looks like:

```json
{
  "instruction": "localize events and describe them",
  "num_frames": 8,
  "events": [
    {
      "timestamp": [0.2, 1.4],
      "score": [3.6],
      "caption": "person starts activity"
    }
  ]
}
```

## QVHighlights: What You Need

For TinyTrace, QVHighlights is a good first real dataset because it matches TRACE's highlight-detection setting.

But the dataset setup has two different parts:

1. annotations
2. videos

You need both.

Files you already downloaded:

- `dataset/mt_fmt-8k.json`
- `dataset/val.caption_coco_format.json`

What they are for:

- `mt_fmt-8k.json` : training-style annotation source
- `val.caption_coco_format.json` : useful for evaluation/reference, not your first training file

Important:

- `mt_fmt-8k.json` alone is not enough
- videos alone are not enough
- TinyTrace training needs matching annotation rows and matching `.mp4` files

## Is Your Current QVHighlights Download Enough?

For first prototype work:

- yes, it is enough to start
- yes, it is enough to test the full TinyTrace pipeline

For proper training:

- no, it is not enough yet

Right now you have a few downloaded clips in:

- `qvhighlights/videos/train/`

TinyTrace currently filters bad/corrupted clips and builds a small clean subset from the usable ones.

## How To Prepare TinyTrace Training Data From QVHighlights

Put matching QVHighlights train clips here:

```bash
qvhighlights/videos/train/
```

Then convert the downloaded clips into TinyTrace JSON:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/prepare_qvhighlights_subset.py \
  --source-json dataset/mt_fmt-8k.json \
  --video-dir qvhighlights/videos/train \
  --output-json TinyTrace/data/qvh_tinytrace_subset.json \
  --max-samples 8
```

This script:

- finds matching videos by filename
- skips unreadable/corrupted clips
- extracts the query text
- converts the annotation into TinyTrace event format

## Train On Real QVHighlights Clips

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --epochs 10 \
  --batch-size 2 \
  --output-dir TinyTrace/outputs-qvh
```

For a quick smoke run:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --epochs 1 \
  --batch-size 1 \
  --output-dir TinyTrace/outputs-qvh-smoke
```

## Check One Video's Prediction

To inspect what TinyTrace currently predicts for one video:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py \
  --checkpoint TinyTrace/outputs-qvh-smoke/tinytrace.pt \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --sample-index 0
```

It prints:

- `ground_truth`
- `predicted`

Both are event lists with:

- `timestamp`
- `score`
- `caption`

## TRACE-Style Metrics For QVHighlights

TinyTrace now supports TRACE-style highlight metrics for the QVHighlights setting:

- `HL-mAP`
- `HL-Hit1`

Run:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace_vhd.py \
  --checkpoint TinyTrace/outputs-qvh-smoke/tinytrace.pt \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --source-json dataset/mt_fmt-8k.json \
  --save-path TinyTrace/outputs-qvh-smoke/qvh_metrics.json
```

This produces metrics similar in style to TRACE's QVHighlights evaluation:

- `HL-min-Fair-mAP`
- `HL-min-Fair-Hit1`
- `HL-min-Good-mAP`
- `HL-min-Good-Hit1`
- `HL-min-VeryGood-mAP`
- `HL-min-VeryGood-Hit1`

## Recommended Next Step

If you want better TinyTrace results now, the best next step is:

1. download more valid matching QVHighlights clips
2. regenerate `TinyTrace/data/qvh_tinytrace_subset.json`
3. train for more epochs
4. rerun `eval_tinytrace_vhd.py`

That will help much more than changing the metric.
