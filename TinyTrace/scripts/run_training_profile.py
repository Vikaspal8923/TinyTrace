from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyTrace training from a single JSON profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="TinyTrace/configs/final_train_qvh500.json",
        help="Path to the training profile JSON.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(model_config_path: Path) -> Path:
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    checkpoint = Path(model_config["mobileclip_checkpoint"])
    if checkpoint.is_absolute():
        return checkpoint
    return PROJECT_ROOT / checkpoint


def resolve_profile_path(profile_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    workspace_root = profile_path.parents[2]
    workspace_candidate = workspace_root / candidate
    if workspace_candidate.exists() or (candidate.parts and candidate.parts[0] in {"TinyTrace", "final_qvhighlights_tinytrace", "dataset"}):
        return workspace_candidate
    return profile_path.parent / candidate


def ensure_mobileclip_checkpoint(profile_path: Path, profile: dict) -> None:
    model_config_path = resolve_profile_path(profile_path, profile["model_config"]).resolve()
    checkpoint_path = resolve_checkpoint_path(model_config_path)
    if checkpoint_path.is_file():
        return

    setup_script = PROJECT_ROOT / "scripts" / "setup_mobileclip.py"
    command = [sys.executable, str(setup_script), "--destination", str(checkpoint_path)]
    print("MobileCLIP checkpoint missing. Downloading it now:")
    print(" ".join(command))
    print()
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    device = str(profile.get("device", "cuda"))
    if not device.startswith("cuda"):
        raise ValueError(
            f"Training profile device must be a CUDA GPU device, received {device!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training, but no CUDA device is available.")

    ensure_mobileclip_checkpoint(profile_path, profile)

    train_script = str(resolve_profile_path(profile_path, profile["train_script"]))
    command = [
        sys.executable,
        train_script,
        "--config",
        str(resolve_profile_path(profile_path, profile["model_config"])),
        "--dataset-json",
        str(resolve_profile_path(profile_path, profile["train_dataset_json"])),
        "--val-dataset-json",
        str(resolve_profile_path(profile_path, profile["val_dataset_json"])),
        "--output-dir",
        str(resolve_profile_path(profile_path, profile["output_dir"])),
        "--frame-cache-dir",
        str(resolve_profile_path(profile_path, profile["frame_cache_dir"])),
        "--device",
        device,
        "--epochs",
        str(profile["epochs"]),
        "--batch-size",
        str(profile["batch_size"]),
        "--lr",
        str(profile["lr"]),
        "--weight-decay",
        str(profile["weight_decay"]),
        "--gradient-clip",
        str(profile["gradient_clip"]),
        "--save-every",
        str(profile["save_every"]),
        "--prediction-every",
        str(profile["prediction_every"]),
        "--prediction-samples",
        str(profile["prediction_samples"]),
        "--metrics-every",
        str(profile.get("metrics_every", 1)),
        "--num-workers",
        str(profile["num_workers"]),
        "--log-every",
        str(profile.get("log_every", 25)),
        "--stage2-start-epoch",
        str(profile.get("stage2_start_epoch", 0)),
        "--stage2-visual-lr-scale",
        str(profile.get("stage2_visual_lr_scale", 0.1)),
        "--stage2-unfreeze-strategy",
        str(profile.get("stage2_unfreeze_strategy", "conv_exp")),
        "--seed",
        str(profile["seed"]),
    ]

    print("Running TinyTrace training profile:")
    print(json.dumps(profile, indent=2))
    print()
    print("Command:")
    print(" ".join(command))
    print()

    result = subprocess.run(command)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
