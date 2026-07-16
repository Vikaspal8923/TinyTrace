from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace import (
    JsonTinyTraceDataset,
    SyntheticTinyTraceDataset,
    TinyTraceConfig,
    TinyTraceModel,
    decode_event_sequence,
    evaluate_event_predictions,
    tinytrace_collate_fn,
)
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyTrace with reproducible artifacts.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--dataset-json", type=str, default="")
    parser.add_argument("--val-dataset-json", type=str, default="")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/tinytrace_baseline.json"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--frame-cache-dir", type=str, default=str(PROJECT_ROOT / ".cache/frames"))
    parser.add_argument("--allow-random-frames", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--prediction-every", type=int, default=1)
    parser.add_argument("--prediction-samples", type=int, default=2)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--stage2-start-epoch", type=int, default=0)
    parser.add_argument("--stage2-visual-lr-scale", type=float, default=0.1)
    parser.add_argument("--stage2-unfreeze-strategy", type=str, default="conv_exp")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def build_dataset(
    annotation_path: str,
    config: TinyTraceConfig,
    frame_cache_dir: str,
    allow_random_frames: bool,
    synthetic_size: int | None = None,
    seed: int = 7,
) -> Dataset:
    if annotation_path:
        return JsonTinyTraceDataset(
            annotation_path,
            config=config,
            frame_cache_dir=frame_cache_dir,
            allow_random_frames=allow_random_frames,
        )
    if synthetic_size is None:
        raise ValueError("A validation dataset path is required for real-data validation.")
    return SyntheticTinyTraceDataset(config=config, size=synthetic_size, seed=seed)


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tinytrace_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in ("frames", "frame_times", "frame_mask", "token_ids", "label_types")
    }


def run_epoch(
    model: TinyTraceModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    epoch: int,
    log_every: int,
    split: str,
    max_steps_per_epoch: int = 0,
) -> float:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    steps = 0
    total_steps = len(loader)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step_index, raw_batch in enumerate(loader, start=1):
            if max_steps_per_epoch > 0 and step_index > max_steps_per_epoch:
                break
            batch = move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["frames"],
                batch["frame_times"],
                batch["token_ids"],
                labels=batch["token_ids"],
                label_types=batch["label_types"],
                frame_mask=batch["frame_mask"],
            )
            if output.loss is None:
                continue
            if training:
                output.loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        gradient_clip,
                    )
                optimizer.step()
            running_loss += float(output.loss.detach().cpu())
            steps += 1
            if log_every > 0 and (step_index % log_every == 0 or step_index == total_steps):
                average_loss = running_loss / steps
                print(
                    f"{split}_step epoch={epoch} step={step_index}/{total_steps} "
                    f"loss={float(output.loss.detach().cpu()):.6f} avg_loss={average_loss:.6f}"
                )
    if steps == 0:
        raise RuntimeError("No loss-producing batches were available.")
    return running_loss / steps


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_optimizer(
    model: TinyTraceModel,
    lr: float,
    weight_decay: float,
    visual_lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    visual_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("visual_encoder.mobileclip."):
            visual_params.append(parameter)
        else:
            other_params.append(parameter)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if visual_params:
        param_groups.append({"params": visual_params, "lr": lr * visual_lr_scale})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def save_checkpoint(
    path: Path,
    model: TinyTraceModel,
    optimizer: torch.optim.Optimizer,
    config: TinyTraceConfig,
    epoch: int,
    best_loss: float,
    history: list[dict],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "history": history,
        },
        temporary,
    )
    temporary.replace(path)


@torch.no_grad()
def collect_predictions(
    model: TinyTraceModel,
    dataset: Dataset,
    config: TinyTraceConfig,
    device: torch.device,
    sample_limit: int | None = None,
) -> list[dict]:
    model.eval()
    text_tokenizer = CharTokenizer(config.text_vocab_size)
    time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
    score_tokenizer = NumericTokenizer(config.score_vocab, width=3)
    predictions = []
    total = len(dataset) if sample_limit is None else min(sample_limit, len(dataset))
    for index in range(total):
        sample = dataset[index]
        frames = sample["frames"].unsqueeze(0).to(device)
        frame_times = sample["frame_times"].unsqueeze(0).to(device)
        frame_mask = sample.get("frame_mask")
        frame_mask = frame_mask.unsqueeze(0).to(device) if frame_mask is not None else None
        prompt_ids = sample["token_ids"][: sample["prompt_length"]].unsqueeze(0).to(device)
        patch_features = model.visual_encoder.extract_patch_features(frames)
        generated = model.generate(
            frames,
            frame_times,
            prompt_ids,
            max_new_tokens=config.max_generated_tokens,
            frame_mask=frame_mask,
            visual_patch_features=patch_features,
        )
        predicted = decode_event_sequence(
            generated[0, prompt_ids.size(1) :].tolist(),
            config,
            text_tokenizer,
            time_tokenizer,
            score_tokenizer,
        )
        predictions.append(
            {
                "sample_index": index,
                "source_id": sample.get("source_id"),
                "video_path": sample.get("video_path"),
                "ground_truth": sample["events"],
                "predicted": predicted,
            }
        )
    return predictions


@torch.no_grad()
def dump_predictions(
    model: TinyTraceModel,
    dataset: Dataset,
    config: TinyTraceConfig,
    device: torch.device,
    path: Path,
    sample_count: int,
) -> None:
    predictions = collect_predictions(model, dataset, config, device, sample_limit=sample_count)
    atomic_json(path, predictions)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if not args.dataset_json and not args.allow_random_frames:
        # Synthetic mode is always allowed when no dataset json is provided.
        args.allow_random_frames = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        config = TinyTraceConfig.from_dict(resume_payload["config"])
    else:
        config = TinyTraceConfig.from_json(args.config)

    if args.dataset_json and not Path(args.dataset_json).is_file():
        raise FileNotFoundError(f"Training dataset JSON not found: {args.dataset_json}")
    if args.val_dataset_json and not Path(args.val_dataset_json).is_file():
        raise FileNotFoundError(f"Validation dataset JSON not found: {args.val_dataset_json}")

    train_dataset = build_dataset(
        args.dataset_json,
        config,
        args.frame_cache_dir,
        args.allow_random_frames,
        synthetic_size=args.dataset_size,
        seed=args.seed,
    )
    val_dataset = (
        build_dataset(
            args.val_dataset_json,
            config,
            args.frame_cache_dir,
            args.allow_random_frames,
        )
        if args.val_dataset_json
        else None
    )
    pin_memory = device.type == "cuda"
    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    val_loader = (
        build_loader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        )
        if val_dataset is not None
        else None
    )

    model = TinyTraceModel(config, load_pretrained_visual=resume_payload is None).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    start_epoch = 1
    best_loss = float("inf")
    history: list[dict] = []
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_loss = float(resume_payload.get("best_loss", float("inf")))
        history = list(resume_payload.get("history", []))
        print(f"resumed={args.resume} start_epoch={start_epoch}")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    Path(args.frame_cache_dir).mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", config.to_dict())
    stage2_activated = False

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}; target epochs is {args.epochs}."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if (
            args.stage2_start_epoch > 0
            and epoch >= args.stage2_start_epoch
            and not stage2_activated
        ):
            model.set_visual_encoder_trainable(True, strategy=args.stage2_unfreeze_strategy)
            optimizer = build_optimizer(
                model,
                args.lr,
                args.weight_decay,
                visual_lr_scale=args.stage2_visual_lr_scale,
            )
            stage2_activated = True
            print(
                f"stage2_enabled epoch={epoch} strategy={args.stage2_unfreeze_strategy} "
                f"visual_lr_scale={args.stage2_visual_lr_scale}"
            )
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.gradient_clip,
            epoch=epoch,
            log_every=args.log_every,
            split="train",
            max_steps_per_epoch=args.max_steps_per_epoch,
        )
        val_loss = (
            run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                gradient_clip=0.0,
                epoch=epoch,
                log_every=args.log_every,
                split="val",
                max_steps_per_epoch=args.max_steps_per_epoch,
            )
            if val_loader is not None
            else None
        )
        monitored_loss = val_loss if val_loss is not None else train_loss
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        atomic_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            + (f"val_loss={val_loss:.6f}" if val_loss is not None else "val_loss=n/a")
        )

        improved = monitored_loss < best_loss
        if improved:
            best_loss = monitored_loss
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            config,
            epoch,
            best_loss,
            history,
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                config,
                epoch,
                best_loss,
                history,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "history": history,
                },
                output_dir / "tinytrace.pt",
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch-{epoch:04d}.pt",
                model,
                optimizer,
                config,
                epoch,
                best_loss,
                history,
            )
        if args.prediction_every > 0 and epoch % args.prediction_every == 0:
            prediction_dataset = val_dataset if val_dataset is not None else train_dataset
            dump_predictions(
                model,
                prediction_dataset,
                config,
                device,
                prediction_dir / f"epoch-{epoch:04d}.json",
                args.prediction_samples,
            )
        if args.metrics_every > 0 and val_dataset is not None and epoch % args.metrics_every == 0:
            metrics_predictions = collect_predictions(model, val_dataset, config, device, sample_limit=None)
            metrics = evaluate_event_predictions(metrics_predictions)
            atomic_json(output_dir / "metrics.json", metrics)
            atomic_json(output_dir / f"metrics-epoch-{epoch:04d}.json", metrics)
            print("metrics " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
