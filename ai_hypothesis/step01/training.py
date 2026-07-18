"""Training and evaluation pipeline for the Step 1 neural-unit experiments."""

from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .baselines import predict_baselines
from .model import (
    REFERENCE_10M_CONFIG,
    Step01Output,
    Step01Unit,
    UnitConfig,
    decode_predictions,
)
from .schema import BENCHMARK_VERSION, VALID_LABELS
from .torch_data import Step01TorchDataset, collate_samples, make_loader


@dataclass(frozen=True, slots=True)
class TrainConfig:
    experiment_name: str = "step01_reference_10m"
    architecture_version: str = "step01-unit-v0"
    benchmark_version: str = BENCHMARK_VERSION
    seed: int = 1
    train_count: int = 100_000
    validation_count: int = 20_000
    test_count: int = 20_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_training_steps: int = 5_000
    eval_interval: int = 250
    early_stopping_patience: int = 8
    gradient_clip_norm: float = 1.0
    uncertainty_threshold: float = 0.5
    num_workers: int = 0
    device: str = "auto"
    output_dir: str = "results/step01/reference_10m/seed_1"

    def validate(self) -> None:
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError(
                f"config benchmark {self.benchmark_version!r} does not match "
                f"implementation {BENCHMARK_VERSION!r}"
            )
        for name in ("train_count", "validation_count", "test_count", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_training_steps <= 0:
            raise ValueError("max_training_steps must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in [0, 1]")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


def load_experiment_config(path: str | Path) -> tuple[UnitConfig, TrainConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unit_config = UnitConfig(**payload["architecture"])
    train_config = TrainConfig(**payload["training"])
    unit_config.validate()
    train_config.validate()
    return unit_config, train_config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    output: Step01Output,
    label_targets: torch.Tensor,
    uncertainty_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    answerable = label_targets != -100
    if answerable.any():
        label_loss = F.cross_entropy(
            output.label_logits[answerable],
            label_targets[answerable],
        )
    else:
        label_loss = output.label_logits.sum() * 0.0

    uncertainty_loss = F.binary_cross_entropy_with_logits(
        output.uncertainty_logits,
        uncertainty_targets,
    )
    total = label_loss + uncertainty_loss
    return total, {
        "label_loss": float(label_loss.detach().cpu()),
        "uncertainty_loss": float(uncertainty_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        **batch,
        "features": batch["features"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
        "label_targets": batch["label_targets"].to(device, non_blocking=True),
        "uncertainty_targets": batch["uncertainty_targets"].to(
            device, non_blocking=True
        ),
    }


def evaluate_model(
    model: Step01Unit,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    uncertainty_threshold: float,
) -> dict[str, Any]:
    model.eval()

    total = 0
    correct = 0
    invalid = 0
    total_loss = 0.0
    batch_count = 0

    true_uncertain = 0
    predicted_uncertain = 0
    correct_uncertain = 0

    by_task: dict[str, list[int]] = {}
    by_difficulty: dict[str, list[int]] = {}
    by_task_difficulty: dict[str, list[int]] = {}

    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            output = model(batch["features"], batch["mask"])
            loss, _ = compute_loss(
                output,
                batch["label_targets"],
                batch["uncertainty_targets"],
            )
            predictions = decode_predictions(
                output,
                uncertainty_threshold=uncertainty_threshold,
            )

            total_loss += float(loss.detach().cpu())
            batch_count += 1

            for sample, prediction in zip(
                batch["samples"], predictions, strict=True
            ):
                total += 1
                is_correct = prediction == sample.label
                correct += int(is_correct)
                invalid += int(prediction not in VALID_LABELS[sample.task])

                actual_uncertain = sample.label == "UNCERTAIN"
                predicted_is_uncertain = prediction == "UNCERTAIN"
                true_uncertain += int(actual_uncertain)
                predicted_uncertain += int(predicted_is_uncertain)
                correct_uncertain += int(actual_uncertain and predicted_is_uncertain)

                task_key = sample.task.value
                difficulty_key = sample.difficulty.value
                combined_key = f"{task_key}/{difficulty_key}"
                for mapping, key in (
                    (by_task, task_key),
                    (by_difficulty, difficulty_key),
                    (by_task_difficulty, combined_key),
                ):
                    bucket = mapping.setdefault(key, [0, 0])
                    bucket[0] += int(is_correct)
                    bucket[1] += 1

    if total == 0:
        raise ValueError("evaluation loader produced no samples")

    task_accuracy = {
        key: values[0] / values[1] for key, values in sorted(by_task.items())
    }
    difficulty_accuracy = {
        key: values[0] / values[1]
        for key, values in sorted(by_difficulty.items())
    }
    task_difficulty_accuracy = {
        key: values[0] / values[1]
        for key, values in sorted(by_task_difficulty.items())
    }

    uncertainty_precision = (
        correct_uncertain / predicted_uncertain if predicted_uncertain else 0.0
    )
    uncertainty_recall = (
        correct_uncertain / true_uncertain if true_uncertain else 0.0
    )

    return {
        "count": total,
        "loss": total_loss / max(batch_count, 1),
        "accuracy": correct / total,
        "macro_task_accuracy": sum(task_accuracy.values()) / len(task_accuracy),
        "invalid_output_rate": invalid / total,
        "uncertainty_precision": uncertainty_precision,
        "uncertainty_recall": uncertainty_recall,
        "by_task": task_accuracy,
        "by_difficulty": difficulty_accuracy,
        "by_task_difficulty": task_difficulty_accuracy,
    }


def evaluate_deterministic_baselines(*, split: str, count: int) -> dict[str, Any]:
    dataset = Step01TorchDataset(split, count)
    stats: dict[str, dict[str, list[int]]] = {}

    for sample in dataset:
        task_stats = stats.setdefault(sample.task.value, {})
        for name, prediction in predict_baselines(sample).items():
            bucket = task_stats.setdefault(name, [0, 0])
            bucket[0] += int(prediction == sample.label)
            bucket[1] += 1

    return {
        task: {
            baseline: {
                "accuracy": values[0] / values[1],
                "count": values[1],
            }
            for baseline, values in sorted(baselines.items())
        }
        for task, baselines in sorted(stats.items())
    }


def benchmark_inference(
    model: Step01Unit,
    *,
    device: torch.device,
    batch_widths: tuple[int, ...] = (1, 4, 16, 64, 256),
    warmup_runs: int = 5,
    timed_runs: int = 30,
) -> dict[str, Any]:
    model.eval()
    dataset = Step01TorchDataset("test", max(batch_widths))
    results: dict[str, Any] = {}

    with torch.inference_mode():
        for width in batch_widths:
            batch = collate_samples([dataset[index] for index in range(width)])
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)

            for _ in range(warmup_runs):
                model(features, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            for _ in range(timed_runs):
                model(features, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            results[str(width)] = {
                "batch_width": width,
                "timed_runs": timed_runs,
                "batch_latency_ms": elapsed / timed_runs * 1000.0,
                "unit_evaluations_per_second": width * timed_runs / elapsed,
            }

    return results


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_training(
    *,
    unit_config: UnitConfig = REFERENCE_10M_CONFIG,
    train_config: TrainConfig = TrainConfig(),
    verbose: bool = True,
) -> dict[str, Any]:
    unit_config.validate()
    train_config.validate()
    seed_everything(train_config.seed)

    device = resolve_device(train_config.device)
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"

    model = Step01Unit(unit_config).to(device)
    parameter_count = model.trainable_parameter_count()
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    train_loader = make_loader(
        split="train",
        count=train_config.train_count,
        batch_size=train_config.batch_size,
        shuffle=True,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )
    validation_loader = make_loader(
        split="validation",
        count=train_config.validation_count,
        batch_size=train_config.batch_size,
        shuffle=False,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )
    test_loader = make_loader(
        split="test",
        count=train_config.test_count,
        batch_size=train_config.batch_size,
        shuffle=False,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )

    if verbose:
        print(
            json.dumps(
                {
                    "event": "training_start",
                    "device": str(device),
                    "parameter_count": parameter_count,
                    "max_training_steps": train_config.max_training_steps,
                }
            )
        )

    best_score = float("-inf")
    best_step = 0
    evaluations_without_improvement = 0
    validation_history: list[dict[str, Any]] = []
    train_iterator = iter(train_loader)
    training_start = time.perf_counter()

    for step in range(1, train_config.max_training_steps + 1):
        try:
            raw_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            raw_batch = next(train_iterator)

        batch = _move_batch(raw_batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["features"], batch["mask"])
        loss, loss_parts = compute_loss(
            output,
            batch["label_targets"],
            batch["uncertainty_targets"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip_norm
        )
        optimizer.step()

        should_evaluate = (
            step % train_config.eval_interval == 0
            or step == train_config.max_training_steps
        )
        if not should_evaluate:
            continue

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device=device,
            uncertainty_threshold=train_config.uncertainty_threshold,
        )
        record = {
            "step": step,
            "train_loss": loss_parts,
            "validation": validation_metrics,
        }
        validation_history.append(record)

        score = float(validation_metrics["macro_task_accuracy"])
        if score > best_score:
            best_score = score
            best_step = step
            evaluations_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "unit_config": asdict(unit_config),
                    "train_config": asdict(train_config),
                    "step": step,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            evaluations_without_improvement += 1

        if verbose:
            print(
                json.dumps(
                    {
                        "event": "validation",
                        "step": step,
                        "train_total_loss": loss_parts["total_loss"],
                        "accuracy": validation_metrics["accuracy"],
                        "macro_task_accuracy": validation_metrics[
                            "macro_task_accuracy"
                        ],
                        "invalid_output_rate": validation_metrics[
                            "invalid_output_rate"
                        ],
                    }
                )
            )

        if evaluations_without_improvement >= train_config.early_stopping_patience:
            if verbose:
                print(
                    json.dumps(
                        {
                            "event": "early_stop",
                            "step": step,
                            "best_step": best_step,
                        }
                    )
                )
            break

    training_duration_seconds = time.perf_counter() - training_start

    if not checkpoint_path.exists():
        raise RuntimeError("training finished without producing a checkpoint")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        uncertainty_threshold=train_config.uncertainty_threshold,
    )
    baseline_metrics = evaluate_deterministic_baselines(
        split="test",
        count=train_config.test_count,
    )
    inference_metrics = benchmark_inference(model, device=device)

    result = {
        "experiment_name": train_config.experiment_name,
        "architecture_version": train_config.architecture_version,
        "benchmark_version": train_config.benchmark_version,
        "git_revision": _git_revision(),
        "device": str(device),
        "parameter_count": parameter_count,
        "unit_config": asdict(unit_config),
        "train_config": asdict(train_config),
        "best_step": best_step,
        "best_validation_score": best_score,
        "training_duration_seconds": training_duration_seconds,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "validation_history": validation_history,
        "test": test_metrics,
        "deterministic_baselines": baseline_metrics,
        "inference": inference_metrics,
    }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if verbose:
        print(
            json.dumps(
                {
                    "event": "training_complete",
                    "best_step": best_step,
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_task_accuracy": test_metrics[
                        "macro_task_accuracy"
                    ],
                    "result_path": str(result_path),
                }
            )
        )

    return result
