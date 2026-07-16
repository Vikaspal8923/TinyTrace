from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .model import TinyTraceModel


OPTIMIZER_FORMAT_VERSION = 2
PARAMETER_GROUP_ORDER = (
    "compression",
    "embeddings",
    "lcem",
    "task_heads",
    "mobileclip",
)


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    visual_lr_scale: float = 0.1
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    amp_mode: str = "off"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("learning_rate", "weight_decay", "gradient_clip", "visual_lr_scale"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number.")
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        for name in ("warmup_ratio", "min_lr_ratio"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number in [0, 1].")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.amp_mode not in {"off", "auto", "fp16", "bf16"}:
            raise ValueError("amp_mode must be one of: off, auto, fp16, bf16.")
        if (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience < 0
        ):
            raise ValueError("early_stopping_patience must be a non-negative integer.")
        if (
            not isinstance(self.early_stopping_min_epochs, int)
            or isinstance(self.early_stopping_min_epochs, bool)
            or self.early_stopping_min_epochs < 1
        ):
            raise ValueError("early_stopping_min_epochs must be a positive integer.")
        if (
            not isinstance(self.early_stopping_min_delta, (int, float))
            or isinstance(self.early_stopping_min_delta, bool)
            or not math.isfinite(float(self.early_stopping_min_delta))
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta must be finite and non-negative.")

    def to_dict(self) -> dict:
        return asdict(self)


def _parameter_category(name: str) -> str:
    if name.startswith("visual_encoder.mobileclip."):
        return "mobileclip"
    if name.startswith("visual_encoder.compressor."):
        return "compression"
    if name.startswith(
        (
            "text_embeddings.",
            "time_embeddings.",
            "score_embeddings.",
            "token_type_embeddings.",
        )
    ) or name == "sync_embedding":
        return "embeddings"
    if name.startswith("blocks.") or name.startswith("final_norm."):
        return "lcem"
    if name.startswith(("text_head.", "time_head.", "score_head.")):
        return "task_heads"
    raise ValueError(f"Trainable parameter is not assigned to a known optimizer group: {name}")


def build_named_optimizer(
    model: TinyTraceModel,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """Build deterministic named groups, including currently frozen MobileCLIP.

    Including the visual parameters from the beginning lets staged unfreezing
    preserve Adam and scheduler state without replacing the optimizer.
    """
    grouped: dict[tuple[str, str], list[torch.nn.Parameter]] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        identifier = id(parameter)
        if identifier in seen:
            raise ValueError(f"Parameter appears more than once in model.named_parameters(): {name}")
        seen.add(identifier)
        category = _parameter_category(name)
        decay_kind = "decay" if parameter.ndim >= 2 else "no_decay"
        grouped.setdefault((category, decay_kind), []).append(parameter)

    parameter_groups = []
    for category in PARAMETER_GROUP_ORDER:
        for decay_kind in ("decay", "no_decay"):
            parameters = grouped.get((category, decay_kind), [])
            if not parameters:
                continue
            learning_rate = config.learning_rate
            if category == "mobileclip":
                learning_rate *= config.visual_lr_scale
            parameter_groups.append(
                {
                    "params": parameters,
                    "group_name": f"{category}.{decay_kind}",
                    "lr": learning_rate,
                    "initial_lr": learning_rate,
                    "weight_decay": config.weight_decay if decay_kind == "decay" else 0.0,
                }
            )

    assigned = {id(parameter) for group in parameter_groups for parameter in group["params"]}
    if assigned != seen:
        raise ValueError("Optimizer parameter assignment is incomplete or contains duplicates.")
    return torch.optim.AdamW(parameter_groups)


def optimizer_group_summary(optimizer: torch.optim.Optimizer) -> list[dict]:
    summary = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = group["params"]
        summary.append(
            {
                "name": group.get("group_name", f"group_{index}"),
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in parameters if parameter.requires_grad
                ),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
        )
    return summary


def _build_legacy_optimizer(
    model: TinyTraceModel,
    learning_rate: float,
    weight_decay: float,
    visual_lr_scale: float,
) -> torch.optim.AdamW:
    visual_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("visual_encoder.mobileclip."):
            visual_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    groups = []
    if other_parameters:
        groups.append({"params": other_parameters, "lr": learning_rate})
    if visual_parameters:
        groups.append({"params": visual_parameters, "lr": learning_rate * visual_lr_scale})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def load_optimizer_state_compat(
    optimizer: torch.optim.Optimizer,
    model: TinyTraceModel,
    state_dict: dict,
    training_config: TrainingConfig,
    optimizer_format_version: int,
) -> None:
    if optimizer_format_version >= OPTIMIZER_FORMAT_VERSION:
        optimizer.load_state_dict(state_dict)
        return

    legacy = _build_legacy_optimizer(
        model,
        training_config.learning_rate,
        training_config.weight_decay,
        training_config.visual_lr_scale,
    )
    legacy.load_state_dict(state_dict)
    for parameter, state in legacy.state.items():
        optimizer.state[parameter] = state


def warmup_cosine_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps < 1:
        raise ValueError("total_steps must be positive.")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps).")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1].")
    clamped_step = min(max(int(step), 0), total_steps)
    if warmup_steps > 0 and clamped_step < warmup_steps:
        return float(clamped_step + 1) / float(warmup_steps)
    decay_steps = total_steps - warmup_steps
    decay_position = (clamped_step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_position))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    multiplier = lambda step: warmup_cosine_multiplier(  # noqa: E731
        step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def set_scheduler_step(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    completed_steps: int,
) -> None:
    """Position a scheduler when migrating a checkpoint without scheduler state."""
    if completed_steps < 0:
        raise ValueError("completed_steps cannot be negative.")
    scheduler.last_epoch = completed_steps
    learning_rates = [
        base_lr * schedule(completed_steps)
        for base_lr, schedule in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    for group, learning_rate in zip(scheduler.optimizer.param_groups, learning_rates):
        group["lr"] = learning_rate
    scheduler._last_lr = learning_rates
    scheduler._step_count = completed_steps + 1


@dataclass(frozen=True)
class AmpSettings:
    enabled: bool
    dtype: torch.dtype | None
    use_grad_scaler: bool


def resolve_amp_settings(mode: str, device: torch.device) -> AmpSettings:
    if mode == "off":
        return AmpSettings(False, None, False)
    if mode == "auto":
        if device.type != "cuda":
            return AmpSettings(False, None, False)
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if mode == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 AMP is supported only on CUDA devices.")
        return AmpSettings(True, torch.float16, True)
    if mode == "bf16":
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("bf16 AMP requires a CPU or CUDA device.")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bf16 AMP.")
        return AmpSettings(True, torch.bfloat16, False)
    raise ValueError(f"Unsupported AMP mode: {mode}")


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    min_epochs: int = 1
    best: float = float("inf")
    bad_epochs: int = 0

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def update(self, value: float, epoch: int, *, active: bool = True) -> tuple[bool, bool]:
        if not math.isfinite(value):
            raise ValueError("Early-stopping metric must be finite.")
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        elif active and epoch >= self.min_epochs:
            self.bad_epochs += 1
        should_stop = (
            self.enabled
            and active
            and epoch >= self.min_epochs
            and self.bad_epochs >= self.patience
        )
        return improved, should_stop

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: dict) -> None:
        self.best = float(state.get("best", self.best))
        self.bad_epochs = int(state.get("bad_epochs", self.bad_epochs))


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def current_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
