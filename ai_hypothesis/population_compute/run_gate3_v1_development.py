"""Run the frozen Gate-3 v1 development seed-0 recipe.

The CLI intentionally exposes no scientific tuning knobs. Confirmation is unreachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .gate3_v1_development import (
    GATE3_V1_DEVELOPMENT_BOOTSTRAP_SAMPLES,
    GATE3_V1_DEVELOPMENT_EVAL_BATCH_SIZE,
    GATE3_V1_DEVELOPMENT_EXPERIMENT_VERSION,
    GATE3_V1_DEVELOPMENT_WORLD_COUNT,
    Gate3V1DevelopmentResult,
    Gate3V1TrainingConfig,
    Gate3V1TrainingSummary,
    _prefix_targets,
    _training_candidate,
    _training_world_seed,
    build_gate3_v1_paired_summaries,
    evaluate_gate3_v1_condition,
)
from .gate3_v1_model import Gate3V1Scorer, encode_gate3_v1_child_input
from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_DEPTHS,
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    GATE3_V1_RESERVE_CAPACITIES,
    Gate3V1ControlMode,
    generate_gate3_v1_world,
)


GATE3_V1_FIRST_DEVELOPMENT_TRAINING_SEED = 0
Progress = Callable[[int, int, int, float], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def train_gate3_v1_with_progress(
    *,
    training_seed: int,
    config: Gate3V1TrainingConfig,
    device: torch.device | str,
    progress: Progress | None = None,
) -> tuple[Gate3V1Scorer, Gate3V1TrainingSummary]:
    """Progress-enabled execution of the frozen reference training recipe."""

    config.validate()
    if training_seed != GATE3_V1_FIRST_DEVELOPMENT_TRAINING_SEED:
        raise ValueError("admitted Gate-3 v1 development runner is bound to training seed 0")
    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = Gate3V1Scorer(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    model.train()

    for step in range(config.steps):
        depth = GATE3_V1_DEPTHS[step % len(GATE3_V1_DEPTHS)]
        worlds = tuple(
            generate_gate3_v1_world(
                seed=_training_world_seed(
                    training_seed=training_seed,
                    step=step,
                    sample_index=sample_index,
                    depth=depth,
                ),
                depth=depth,
            )
            for sample_index in range(config.batch_size)
        )
        candidates = tuple(
            _training_candidate(
                training_seed=training_seed,
                step=step,
                sample_index=sample_index,
                depth=depth,
            )
            for sample_index in range(config.batch_size)
        )
        targets = tuple(
            _prefix_targets(
                noisy_hints=world.public.noisy_hints,
                candidate=candidate,
                depth=depth,
            )
            for world, candidate in zip(worlds, candidates, strict=True)
        )
        states = model.initial_state(config.batch_size, device=target_device)
        phase_losses: list[torch.Tensor] = []
        for prefix_index in range(depth):
            inputs = torch.stack(
                [
                    encode_gate3_v1_child_input(
                        world=world.public,
                        child_depth=prefix_index + 1,
                        observed_hint=world.public.noisy_hints[prefix_index],
                        branch_action=candidate[prefix_index],
                        sink=False,
                        device=target_device,
                    )
                    for world, candidate in zip(worlds, candidates, strict=True)
                ],
                dim=0,
            )
            states = model.advance(states, inputs, repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD)
            predictions = model.score(states)
            target_tensor = torch.tensor(
                [row[prefix_index] for row in targets],
                dtype=predictions.dtype,
                device=target_device,
            )
            phase_losses.append(loss_fn(predictions, target_tensor))
        loss = torch.stack(phase_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Gate-3 v1 training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        scalar_loss = float(loss.detach().item())
        losses.append(scalar_loss)
        if progress is not None:
            progress(step + 1, config.steps, depth, scalar_loss)

    return model, Gate3V1TrainingSummary(
        training_seed=training_seed,
        steps=config.steps,
        examples_seen=config.steps * config.batch_size,
        initial_loss=losses[0],
        final_loss=losses[-1],
        mean_last_50_loss=sum(losses[-50:]) / len(losses[-50:]),
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _checkpoint_payload(
    model: Gate3V1Scorer,
    *,
    summary: Gate3V1TrainingSummary,
    config: Gate3V1TrainingConfig,
) -> dict[str, object]:
    return {
        "experiment_version": GATE3_V1_DEVELOPMENT_EXPERIMENT_VERSION,
        "scientific_status": "FROZEN_GATE3_V1_DEVELOPMENT_SEED0",
        "evaluation_split": "development",
        "confirmation_opened": False,
        "training_seed": summary.training_seed,
        "learned_parameter_count": summary.learned_parameter_count,
        "parameter_fingerprint": summary.parameter_fingerprint,
        "training_config": {
            "steps": config.steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "model": asdict(config.model),
        },
        "state_dict": model.state_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"Gate-3 v1 output already exists: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-3 v1 development runner requires CUDA")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True)
    started = time.monotonic()
    config = Gate3V1TrainingConfig()
    last_reported = 0

    print("Gate-3 v1 development-only seed 0", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Learned parameters: {Gate3V1Scorer().trainable_parameter_count():,}", flush=True)

    def progress(done: int, total: int, depth: int, loss: float) -> None:
        nonlocal last_reported
        if done != total and done - last_reported < 10:
            return
        last_reported = done
        elapsed = (time.monotonic() - started) / 60.0
        print(
            f"TRAIN {_bar(done, total)} {100.0 * done / total:6.2f}% "
            f"{done:4d}/{total} S{depth} loss={loss:.6f} elapsed={elapsed:.1f}m",
            flush=True,
        )

    model, training = train_gate3_v1_with_progress(
        training_seed=GATE3_V1_FIRST_DEVELOPMENT_TRAINING_SEED,
        config=config,
        device="cuda",
        progress=progress,
    )

    checkpoint_path = output_root / "gate3-v1-development-checkpoint.pt"
    torch.save(_checkpoint_payload(model, summary=training, config=config), checkpoint_path)

    conditions = []
    total_cells = sum(len(GATE3_V1_RESERVE_CAPACITIES[depth]) for depth in GATE3_V1_DEPTHS) * len(Gate3V1ControlMode)
    completed = 0
    eval_started = time.monotonic()
    for depth in GATE3_V1_DEPTHS:
        for capacity in GATE3_V1_RESERVE_CAPACITIES[depth]:
            for mode in Gate3V1ControlMode:
                condition = evaluate_gate3_v1_condition(
                    model,
                    depth=depth,
                    reserve_capacity=capacity,
                    mode=mode,
                    world_count=GATE3_V1_DEVELOPMENT_WORLD_COUNT,
                    evaluation_batch_size=GATE3_V1_DEVELOPMENT_EVAL_BATCH_SIZE,
                    device="cuda",
                )
                conditions.append(condition)
                completed += 1
                productive = sum(condition.productive_work_fraction_by_world) / condition.world_count
                elapsed = (time.monotonic() - eval_started) / 60.0
                print(
                    f"EVAL  {_bar(completed, total_cells)} {100.0 * completed / total_cells:6.2f}% "
                    f"{completed:2d}/{total_cells} S{depth} L={capacity:<3d} {mode.value:<23s} "
                    f"coverage={condition.coverage_rate:.4f} productive={productive:.3f} elapsed={elapsed:.1f}m",
                    flush=True,
                )

    paired = build_gate3_v1_paired_summaries(
        conditions,
        bootstrap_samples=GATE3_V1_DEVELOPMENT_BOOTSTRAP_SAMPLES,
    )
    result = Gate3V1DevelopmentResult(
        experiment_version=GATE3_V1_DEVELOPMENT_EXPERIMENT_VERSION,
        evaluation_split="development",
        confirmation_opened=False,
        training=training,
        training_config=config,
        evaluation_world_count=GATE3_V1_DEVELOPMENT_WORLD_COUNT,
        evaluation_batch_size=GATE3_V1_DEVELOPMENT_EVAL_BATCH_SIZE,
        bootstrap_samples=GATE3_V1_DEVELOPMENT_BOOTSTRAP_SAMPLES,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    result_path = output_root / "gate3-v1-development.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime = {
        "scientific_status": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
        "confirmation_opened": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - started,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "result_sha256": _sha256(result_path),
        "training_seed": 0,
        "learned_parameter_count": training.learned_parameter_count,
        "parameter_fingerprint": training.parameter_fingerprint,
    }
    runtime_path = output_root / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "GATE3_V1_DEVELOPMENT_ONLY_COMPLETE",
                "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
                "confirmation_opened": False,
                "training_seed": 0,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": runtime["checkpoint_sha256"],
                "result": str(result_path),
                "result_sha256": runtime["result_sha256"],
                "learned_parameter_count": training.learned_parameter_count,
                "parameter_fingerprint": training.parameter_fingerprint,
                "primary_comparisons": result.to_dict()["primary_comparisons"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
