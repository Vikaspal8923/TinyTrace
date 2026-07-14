# TinyTrace

TinyTrace is a lightweight reimplementation of the TRACE causal event modeling idea for constrained environments. This version keeps the event math and the `time -> score -> caption` generation order, while replacing the heavy backbone with a compact visual encoder and a small decoder-only transformer.

## What This Baseline Includes

- Lightweight visual encoder with patch projection and token compression
- Separate time, score, and text token spaces
- Decoder-only LCEM-style event generator
- Adaptive generation flow using `<sync>` transitions
- Synthetic training data so the pipeline can be validated before real dataset training

## Run

```bash
python3 -m venv TinyTrace/.venv
TinyTrace/.venv/bin/pip install -r TinyTrace/requirements.txt
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py --epochs 3 --batch-size 8 --dataset-size 128
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py
```

## Train With A JSON Dataset

The baseline also accepts a simple JSON annotation file:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py --dataset-json TinyTrace/data/sample_dataset.json
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py --dataset-json TinyTrace/data/sample_dataset.json
```

Each sample should look like:

```json
{
  "instruction": "localize events and describe them",
  "num_frames": 8,
  "frame_times": [0.0, 1.0, 2.0, 3.0],
  "events": [
    {
      "timestamp": [0.2, 1.4],
      "score": [3.6],
      "caption": "person starts activity"
    }
  ]
}
```

## Prepare From Downloaded QVHighlights Clips

If you have downloaded a few matching clips referenced by `dataset/mt_fmt-8k.json`, you can convert them into TinyTrace training format:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/prepare_qvhighlights_subset.py \
  --source-json dataset/mt_fmt-8k.json \
  --video-dir qvhighlights/videos/train \
  --output-json TinyTrace/data/qvh_tinytrace_subset.json
```

Then train on the converted subset:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --epochs 10 \
  --batch-size 2 \
  --output-dir TinyTrace/outputs-qvh
```

## Current Scope

This is the first runnable baseline. It is designed to verify the architecture and training loop locally before wiring in real video datasets and a stronger mobile vision backbone.
