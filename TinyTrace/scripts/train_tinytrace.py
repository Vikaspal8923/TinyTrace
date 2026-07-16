from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
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
from tinytrace.training import (
    OPTIMIZER_FORMAT_VERSION,
    AmpSettings,
    EarlyStopping,
    JsonlLogger,
    TrainingConfig,
    build_named_optimizer,
    build_warmup_cosine_scheduler,
    current_learning_rates,
    load_optimizer_state_compat,
    optimizer_group_summary,
    resolve_amp_settings,
    set_scheduler_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyTrace with reproducible artifacts.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--amp", choices=("off", "auto", "fp16", "bf16"), default="off")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=1)
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_settings: AmpSettings | None = None,
    event_logger: JsonlLogger | None = None,
    global_step: int = 0,
) -> tuple[dict, int]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    component_totals: dict[str, float] = {}
    weighted_component_totals: dict[str, float] = {}
    target_counts: dict[str, int] = {}
    gradient_norm_total = 0.0
    gradient_norm_steps = 0
    clipped_steps = 0
    optimizer_steps = 0
    examples = 0
    steps = 0
    total_steps = len(loader)
    epoch_started = time.perf_counter()
    amp_settings = amp_settings or AmpSettings(False, None, False)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step_index, raw_batch in enumerate(loader, start=1):
            if max_steps_per_epoch > 0 and step_index > max_steps_per_epoch:
                break
            batch = move_batch(raw_batch, device)
            examples += int(batch["frames"].size(0))
            if training:
                optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_settings.dtype)
                if amp_settings.enabled
                else nullcontext()
            )
            with autocast_context:
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
                if scaler is not None and scaler.is_enabled():
                    scale_before = scaler.get_scale()
                    scaler.scale(output.loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    scale_before = None
                    output.loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    gradient_clip if gradient_clip > 0 else float("inf"),
                    error_if_nonfinite=True,
                )
                gradient_norm_value = float(gradient_norm.detach().cpu())
                gradient_norm_total += gradient_norm_value
                gradient_norm_steps += 1
                if gradient_clip > 0 and gradient_norm_value > gradient_clip:
                    clipped_steps += 1

                optimizer_step_performed = True
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_performed = scaler.get_scale() >= scale_before
                else:
                    optimizer.step()
                if optimizer_step_performed:
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1
                    optimizer_steps += 1

            loss_value = float(output.loss.detach().float().cpu())
            running_loss += loss_value
            for name, component in output.loss_components.items():
                count = int(output.target_counts[name].cpu())
                component_totals[name] = component_totals.get(name, 0.0) + float(
                    component.detach().float().cpu()
                ) * count
                weighted_component_totals[name] = weighted_component_totals.get(name, 0.0) + float(
                    output.weighted_loss_components[name].detach().float().cpu()
                ) * count
                target_counts[name] = target_counts.get(name, 0) + count
            steps += 1
            step_record = {
                "record_type": "step",
                "split": split,
                "epoch": epoch,
                "step": step_index,
                "global_step": global_step,
                "loss": loss_value,
                "loss_components": {
                    name: float(value.detach().float().cpu())
                    for name, value in output.loss_components.items()
                },
                "weighted_loss_components": {
                    name: float(value.detach().float().cpu())
                    for name, value in output.weighted_loss_components.items()
                },
                "target_counts": {
                    name: int(value.cpu()) for name, value in output.target_counts.items()
                },
                "learning_rates": current_learning_rates(optimizer) if optimizer is not None else {},
            }
            if training and gradient_norm_steps:
                step_record["gradient_norm"] = gradient_norm_value
                step_record["gradient_clipped"] = bool(
                    gradient_clip > 0 and gradient_norm_value > gradient_clip
                )
            if event_logger is not None:
                event_logger.log(step_record)
            if log_every > 0 and (step_index % log_every == 0 or step_index == total_steps):
                average_loss = running_loss / steps
                print(
                    f"{split}_step epoch={epoch} step={step_index}/{total_steps} "
                    f"loss={float(output.loss.detach().cpu()):.6f} avg_loss={average_loss:.6f}"
                )
    if steps == 0:
        raise RuntimeError("No loss-producing batches were available.")
    elapsed = time.perf_counter() - epoch_started
    metrics = {
        "loss": running_loss / steps,
        "loss_components": {
            name: component_totals[name] / target_counts[name] for name in component_totals
        },
        "weighted_loss_components": {
            name: weighted_component_totals[name] / target_counts[name]
            for name in weighted_component_totals
        },
        "target_counts": target_counts,
        "steps": steps,
        "optimizer_steps": optimizer_steps,
        "examples": examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed if elapsed > 0 else 0.0,
        "mean_gradient_norm": (
            gradient_norm_total / gradient_norm_steps if gradient_norm_steps else None
        ),
        "clipped_steps": clipped_steps,
        "learning_rates": current_learning_rates(optimizer) if optimizer is not None else {},
    }
    if event_logger is not None:
        event_logger.log(
            {
                "record_type": "epoch",
                "split": split,
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
            }
        )
    return metrics, global_step


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: TinyTraceModel,
    optimizer: torch.optim.Optimizer,
    config: TinyTraceConfig,
    epoch: int,
    best_loss: float,
    history: list[dict],
    training_state: dict | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    training_config: TrainingConfig | None = None,
    early_stopping: EarlyStopping | None = None,
    global_step: int = 0,
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
            "training_state": dict(training_state or {}),
            "optimizer_format_version": OPTIMIZER_FORMAT_VERSION,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "training_config": training_config.to_dict() if training_config is not None else None,
            "early_stopping_state": early_stopping.state_dict() if early_stopping is not None else None,
            "global_step": global_step,
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

    cli_training_config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        visual_lr_scale=args.stage2_visual_lr_scale,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        amp_mode=args.amp,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
    )
    saved_training_config = resume_payload.get("training_config") if resume_payload else None
    training_config = (
        TrainingConfig(**saved_training_config) if saved_training_config else cli_training_config
    )
    if training_config.early_stopping_patience > 0 and val_loader is None:
        raise ValueError("Validation data is required when early stopping is enabled.")

    steps_per_epoch = len(train_loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_optimizer_steps = args.epochs * steps_per_epoch
    if total_optimizer_steps < 1:
        raise ValueError("Training must contain at least one optimizer step.")
    warmup_steps = min(
        int(round(total_optimizer_steps * training_config.warmup_ratio)),
        max(total_optimizer_steps - 1, 0),
    )

    model = TinyTraceModel(config, load_pretrained_visual=resume_payload is None).to(device)
    start_epoch = 1
    best_loss = float("inf")
    history: list[dict] = []
    global_step = 0
    stage2_activated = False
    active_stage2_strategy = args.stage2_unfreeze_strategy
    active_stage2_start_epoch = args.stage2_start_epoch
    optimizer_format_version = OPTIMIZER_FORMAT_VERSION
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        saved_training_state = resume_payload.get("training_state", {})
        optimizer_format_version = int(resume_payload.get("optimizer_format_version", 1))
        legacy_stage2 = (
            optimizer_format_version < OPTIMIZER_FORMAT_VERSION
            and len(resume_payload["optimizer_state"].get("param_groups", [])) > 1
        )
        stage2_activated = bool(saved_training_state.get("stage2_activated", False)) or legacy_stage2
        active_stage2_strategy = str(
            saved_training_state.get("stage2_unfreeze_strategy", args.stage2_unfreeze_strategy)
        )
        active_stage2_start_epoch = int(
            saved_training_state.get("stage2_start_epoch", args.stage2_start_epoch)
        )
        if stage2_activated:
            model.set_visual_encoder_trainable(True, strategy=active_stage2_strategy)

    optimizer = build_named_optimizer(model, training_config)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=training_config.min_lr_ratio,
    )
    amp_settings = resolve_amp_settings(training_config.amp_mode, device)
    scaler = (
        torch.amp.GradScaler(device.type, enabled=True)
        if amp_settings.use_grad_scaler
        else None
    )
    early_stopping = EarlyStopping(
        patience=training_config.early_stopping_patience,
        min_delta=training_config.early_stopping_min_delta,
        min_epochs=training_config.early_stopping_min_epochs,
    )
    if resume_payload is not None:
        load_optimizer_state_compat(
            optimizer,
            model,
            resume_payload["optimizer_state"],
            training_config,
            optimizer_format_version=optimizer_format_version,
        )
        start_epoch = int(resume_payload["epoch"]) + 1
        best_loss = float(resume_payload.get("best_loss", float("inf")))
        history = list(resume_payload.get("history", []))
        global_step = int(resume_payload.get("global_step", (start_epoch - 1) * steps_per_epoch))
        saved_total_steps = resume_payload.get("training_state", {}).get("total_optimizer_steps")
        if saved_total_steps is not None and int(saved_total_steps) != total_optimizer_steps:
            raise ValueError(
                "Cannot change the planned optimizer-step count when resuming a scheduled run: "
                f"checkpoint={saved_total_steps}, requested={total_optimizer_steps}."
            )
        if resume_payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        else:
            set_scheduler_step(scheduler, global_step)
        if scaler is not None and resume_payload.get("scaler_state") is not None:
            scaler.load_state_dict(resume_payload["scaler_state"])
        if resume_payload.get("early_stopping_state") is not None:
            early_stopping.load_state_dict(resume_payload["early_stopping_state"])
        else:
            early_stopping.best = best_loss
        print(f"resumed={args.resume} start_epoch={start_epoch}")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    Path(args.frame_cache_dir).mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", config.to_dict())
    atomic_json(output_dir / "training_config.json", training_config.to_dict())
    atomic_json(output_dir / "optimizer_groups.json", optimizer_group_summary(optimizer))
    event_logger = JsonlLogger(output_dir / "training_log.jsonl")
    event_logger.log(
        {
            "record_type": "run_start",
            "start_epoch": start_epoch,
            "total_epochs": args.epochs,
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "amp_enabled": amp_settings.enabled,
            "amp_dtype": str(amp_settings.dtype),
            "optimizer_groups": optimizer_group_summary(optimizer),
        }
    )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}; target epochs is {args.epochs}."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if (
            active_stage2_start_epoch > 0
            and epoch >= active_stage2_start_epoch
            and not stage2_activated
        ):
            model.set_visual_encoder_trainable(True, strategy=active_stage2_strategy)
            stage2_activated = True
            current_group_summary = optimizer_group_summary(optimizer)
            atomic_json(output_dir / "optimizer_groups.json", current_group_summary)
            event_logger.log(
                {
                    "record_type": "stage_transition",
                    "epoch": epoch,
                    "stage": 2,
                    "visual_unfreeze_strategy": active_stage2_strategy,
                    "visual_lr_scale": training_config.visual_lr_scale,
                    "optimizer_groups": current_group_summary,
                }
            )
            print(
                f"stage2_enabled epoch={epoch} strategy={active_stage2_strategy} "
                f"visual_lr_scale={training_config.visual_lr_scale}"
            )
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            training_config.gradient_clip,
            epoch=epoch,
            log_every=args.log_every,
            split="train",
            max_steps_per_epoch=args.max_steps_per_epoch,
            scheduler=scheduler,
            scaler=scaler,
            amp_settings=amp_settings,
            event_logger=event_logger,
            global_step=global_step,
        )
        validation_result = (
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
                amp_settings=amp_settings,
                event_logger=event_logger,
                global_step=global_step,
            )
            if val_loader is not None
            else None
        )
        val_metrics = validation_result[0] if validation_result is not None else None
        monitored_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        selection_split = "validation" if val_metrics is not None else "training_fallback"
        improved, should_stop = early_stopping.update(
            monitored_loss,
            epoch,
            active=global_step >= warmup_steps,
        )
        best_loss = early_stopping.best
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"] if val_metrics is not None else None,
            "train": train_metrics,
            "validation": val_metrics,
            "selection_split": selection_split,
            "improved": improved,
            "global_step": global_step,
        }
        history.append(record)
        atomic_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            + (
                f"val_loss={val_metrics['loss']:.6f}"
                if val_metrics is not None
                else "val_loss=n/a"
            )
        )

        checkpoint_training_state = {
            "stage2_activated": stage2_activated,
            "stage2_unfreeze_strategy": active_stage2_strategy,
            "stage2_start_epoch": active_stage2_start_epoch,
            "stage2_visual_lr_scale": training_config.visual_lr_scale,
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "selection_split": selection_split,
        }
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            config,
            epoch,
            best_loss,
            history,
            training_state=checkpoint_training_state,
            scheduler=scheduler,
            scaler=scaler,
            training_config=training_config,
            early_stopping=early_stopping,
            global_step=global_step,
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
                training_state=checkpoint_training_state,
                scheduler=scheduler,
                scaler=scaler,
                training_config=training_config,
                early_stopping=early_stopping,
                global_step=global_step,
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
                training_state=checkpoint_training_state,
                scheduler=scheduler,
                scaler=scaler,
                training_config=training_config,
                early_stopping=early_stopping,
                global_step=global_step,
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
        if should_stop:
            event_logger.log(
                {
                    "record_type": "early_stop",
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "bad_epochs": early_stopping.bad_epochs,
                }
            )
            print(
                f"early_stopping epoch={epoch} best_loss={best_loss:.6f} "
                f"bad_epochs={early_stopping.bad_epochs}"
            )
            break


if __name__ == "__main__":
    main()
