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

TinyTrace is currently an architecture-aligned prototype:

- MobileCLIP-S0 is the visual encoder and remains frozen during training
- pre-pooling MobileCLIP spatial features are compressed with learned slots
- each frame contributes TRACE-style fixed-width discrete time embeddings
- the LCEM prefix is ordered as per-frame visual slots + time tokens, followed by instruction/event tokens
- variable-length frame batches are padded and attention-masked
- real-video frames are decoded lazily and can be cached across epochs
- checkpoints include optimizer/config/history state and support resume
- training supports validation, best/latest checkpoints, and prediction snapshots
- focused architecture tests pass

It is not yet a final trained model. Tiny-subset overfitting and staged training
must be validated before scaling the real-video dataset.

The MobileCLIP checkpoint is intentionally not committed. The setup helper
downloads Apple's official `mobileclip_s0.pt` and verifies its SHA-256 digest.

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

On Windows PowerShell:

```powershell
python -m venv TinyTrace/.venv
TinyTrace/.venv/Scripts/python.exe -m pip install -r TinyTrace/requirements.txt
TinyTrace/.venv/Scripts/python.exe TinyTrace/scripts/setup_mobileclip.py
```

Run the focused tests before training:

```powershell
cd TinyTrace
.venv/Scripts/python.exe -m unittest discover -s tests -v
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
  --dataset-json TinyTrace/data/sample_dataset.json \
  --frame-cache-dir TinyTrace/.cache/frames

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

To auto-download valid QVHighlights clips, run the downloader from the project root folder:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/download_qvhighlights_subset.py --target-count 50
```

This will:

- create `dataset/qvhighlights/videos/train/` if it does not exist
- download valid clips automatically
- save the matched annotation subset to `dataset/qvhighlights/mt_fmt-50-valid.json`

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
  --val-dataset-json TinyTrace/data/qvh_tinytrace_val.json \
  --epochs 10 \
  --batch-size 2 \
  --output-dir TinyTrace/outputs-qvh
```

Real-data training does not silently substitute random frames. Use
`--allow-random-frames` only for deliberate debugging. Resume a stopped run by
setting the total target epoch and loading `latest.pt`:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --val-dataset-json TinyTrace/data/qvh_tinytrace_val.json \
  --epochs 20 \
  --resume TinyTrace/outputs-qvh/checkpoints/latest.pt \
  --output-dir TinyTrace/outputs-qvh
```

Each run writes `config.json`, `history.json`, `checkpoints/latest.pt`,
`checkpoints/best.pt`, periodic checkpoints, and epoch prediction JSON files.

For a quick smoke run:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --epochs 1 \
  --batch-size 1 \
  --output-dir TinyTrace/outputs-qvh-smoke
```

## Use The Final 500-Video Dataset Folder

The final cleaned dataset prepared for TinyTrace is stored here:

```bash
final_qvhighlights_tinytrace/
```

It contains:

- `videos/train/` : 450 valid training clips
- `videos/val/` : 50 valid validation clips
- `annotations/tinytrace_train.json` : TinyTrace-ready training annotations
- `annotations/tinytrace_val.json` : TinyTrace-ready validation annotations
- `annotations/qvh_raw_valid_500.json` : filtered original-format annotations for the same 500 valid clips

You do not need to move this folder anywhere else. Keep it exactly under:

```bash
final_qvhighlights_tinytrace/
```

and run training from the project root:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json final_qvhighlights_tinytrace/annotations/tinytrace_train.json \
  --epochs 10 \
  --batch-size 2 \
  --output-dir TinyTrace/outputs-qvh-final
```

To inspect one validation sample after training:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py \
  --checkpoint TinyTrace/outputs-qvh-final/tinytrace.pt \
  --dataset-json final_qvhighlights_tinytrace/annotations/tinytrace_val.json \
  --sample-index 0
```

To run TRACE-style highlight metrics on the validation split:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace_vhd.py \
  --checkpoint TinyTrace/outputs-qvh-final/tinytrace.pt \
  --dataset-json final_qvhighlights_tinytrace/annotations/tinytrace_val.json \
  --source-json final_qvhighlights_tinytrace/annotations/qvh_raw_valid_500.json \
  --save-path TinyTrace/outputs-qvh-final/qvh_val_metrics.json
```

## Final Training Setup

Before final training, TinyTrace needs:

1. the Python dependencies from `TinyTrace/requirements.txt`
2. the pretrained `MobileCLIP-S0` checkpoint

Install dependencies:

```bash
cd /home/vikaspal/Desktop/Traceall
TinyTrace/.venv/bin/pip install -r TinyTrace/requirements.txt
```

Download and verify the MobileCLIP checkpoint:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/setup_mobileclip.py
```

That will place the pretrained checkpoint at:

```bash
TinyTrace/checkpoints/mobileclip_s0.pt
```

This checkpoint is the pretrained weight file for the MobileCLIP visual backbone.
TinyTrace does not learn those visual features from scratch. It starts from this
already-trained MobileCLIP model, freezes it, and trains the lightweight
TinyTrace parts on top of it.

## One-File Final Training Run

Use this training profile file:

```bash
TinyTrace/configs/final_train_qvh500.json
```

Edit that file if you want to change:

- epochs
- batch size
- learning rate
- device
- output folder
- dataset paths

Then run final training with one command:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/run_training_profile.py \
  --profile TinyTrace/configs/final_train_qvh500.json
```

Even simpler, you can use the dedicated final-training launcher:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_final_qvh500.py
```

If the MobileCLIP checkpoint is missing, this launcher will first try to
download it automatically and then start training.

This profile currently trains on:

- `final_qvhighlights_tinytrace/annotations/tinytrace_train.json`
- validates on `final_qvhighlights_tinytrace/annotations/tinytrace_val.json`

## Check One Video's Prediction

To inspect what TinyTrace currently predicts for one video:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py \
  --checkpoint TinyTrace/outputs-qvh-smoke/checkpoints/best.pt \
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
  --checkpoint TinyTrace/outputs-qvh-smoke/checkpoints/best.pt \
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
