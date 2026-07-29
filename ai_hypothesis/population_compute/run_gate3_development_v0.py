"""Execute the frozen Gate-3 v0 development seed-0 recipe with progress output.

This CLI intentionally exposes no scientific tuning flags. It requires CUDA and produces
DEVELOPMENT-ONLY evidence; the confirmation world domain remains unreachable.
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

from .gate3_development import (
    GATE3_DEVELOPMENT_BOOTSTRAP_SAMPLES,
    GATE3_DEVELOPMENT_EVAL_BATCH_SIZE,
    GATE3_DEVELOPMENT_EVAL_WORLD_COUNT,
    Gate3DevelopmentResult,
    Gate3TrainingConfig,
    Gate3TrainingSummary,
    _candidate_path,
    _target_score,
    _training_world_seed,
    build_gate3_paired_summaries,
    evaluate_gate3_condition,
    gate3_stable_training_conditions,
)
from .gate3_hypothesis_model import Gate3HypothesisScorer, encode_gate3_phase_input
from .gate3_hypothesis_population import (
    GATE3_DEPTHS,
    GATE3_WIDTHS_BY_DEPTH,
    Gate3ControlMode,
    build_gate3_condition_plan,
    generate_gate3_world,
)


GATE3_FIRST_DEVELOPMENT_TRAINING_SEED = 0
Progress = Callable[[int, int, int, int, float], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def train_gate3_development_with_progress(
    *,
    training_seed: int,
    config: Gate3TrainingConfig,
    device: torch.device | str,
    progress: Progress | None = None,
) -> tuple[Gate3HypothesisScorer, Gate3TrainingSummary]:
    """Progress-enabled copy of the frozen scorer training loop.

    Qualification compares its checkpoint fingerprint with the silent reference training function
    on the same deterministic smoke recipe before this path is admitted locally.
    """

    config.validate()
    if training_seed != GATE3_FIRST_DEVELOPMENT_TRAINING_SEED:
        raise ValueError("Gate-3 v0 admitted development CLI is bound to training seed 0")

    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = Gate3HypothesisScorer(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()
    conditions = gate3_stable_training_conditions()
    losses: list[float] = []
    examples_seen = 0
    model.train()

    for step in range(config.steps):
        depth, width = conditions[step % len(conditions)]
        worlds = tuple(
            generate_gate3_world(
                seed=_training_world_seed(
                    training_seed=training_seed,
                    step=step,
                    sample_index=sample_index,
                    depth=depth,
                    width=width,
                ),
                depth=depth,
            )
            for sample_index in range(config.batch_size)
        )
        candidates = tuple(
            _candidate_path(
                training_seed=training_seed,
                step=step,
                sample_index=sample_index,
                depth=depth,
                width=width,
            )
            for sample_index in range(config.batch_size)
        )
        plan = build_gate3_condition_plan(
            worlds[0],
            width=width,
            mode=Gate3ControlMode.STABLE_DIVERSE,
        )
        states = model.initial_state(config.batch_size, device=target_device)
        phase_losses: list[torch.Tensor] = []

        for phase in plan.phases:
            phase_inputs = torch.stack(
                [
                    encode_gate3_phase_input(
                        depth=depth,
                        observation=world.observations[phase.phase_index],
                        branch_action=(candidate[phase.phase_index] if phase.phase_index < depth else None),
                        device=target_device,
                    )
                    for world, candidate in zip(worlds, candidates, strict=True)
                ],
                dim=0,
            )
            states = model.advance(
                states,
                phase_inputs,
                repeats=phase.recurrent_updates_per_evaluated_state,
            )
            predictions = model.score(states)
            targets = torch.tensor(
                [
                    _target_score(world, candidate, phase_index=phase.phase_index)
                    for world, candidate in zip(worlds, candidates, strict=True)
                ],
                dtype=predictions.dtype,
                device=target_device,
            )
            phase_losses.append(loss_fn(predictions, targets))

        loss = torch.stack(phase_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Gate-3 development training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        scalar_loss = float(loss.detach().item())
        losses.append(scalar_loss)
        examples_seen += config.batch_size
        if progress is not None:
            progress(step + 1, config.steps, depth, width, scalar_loss)

    summary = Gate3TrainingSummary(
        training_seed=training_seed,
        steps=config.steps,
        examples_seen=examples_seen,
        initial_loss=losses[0],
        final_loss=losses[-1],
        mean_last_50_loss=sum(losses[-50:]) / len(losses[-50:]),
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
        stable_training_condition_count=len(conditions),
    )
    return model, summary


def _checkpoint_payload(
    model: Gate3HypothesisScorer,
    *,
    training_summary: Gate3TrainingSummary,
    training_config: Gate3TrainingConfig,
) -> dict[str, object]:
    return {
        "experiment_version": "gate3-hypothesis-population-development-v0",
        "scientific_status": "FROZEN_GATE3_DEVELOPMENT_SEED0",
        "evaluation_split": "development",
        "confirmation_opened": False,
        "training_seed": training_summary.training_seed,
        "learned_parameter_count": training_summary.learned_parameter_count,
        "parameter_fingerprint": training_summary.parameter_fingerprint,
        "training_config": {
            "steps": training_config.steps,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "gradient_clip_norm": training_config.gradient_clip_norm,
            "model": asdict(training_config.model),
        },
        "state_dict": model.state_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"Gate-3 development output already exists: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("The admitted Gate-3 development seed-0 runner requires CUDA")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True)
    started = time.monotonic()
    config = Gate3TrainingConfig()
    last_reported = 0

    print("Gate-3 v0 development-only seed 0", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Learned parameters: {Gate3HypothesisScorer(config.model).trainable_parameter_count():,}", flush=True)

    def progress(done: int, total: int, depth: int, width: int, loss: float) -> None:
        nonlocal last_reported
        if done != total and done - last_reported < 10:
            return
        last_reported = done
        elapsed = (time.monotonic() - started) / 60.0
        print(
            f"TRAIN {_bar(done, total)} {100.0 * done / total:6.2f}% "
            f"{done:4d}/{total} H{depth} W={width:<3d} loss={loss:.6f} elapsed={elapsed:.1f}m",
            flush=True,
        )

    model, training_summary = train_gate3_development_with_progress(
        training_seed=GATE3_FIRST_DEVELOPMENT_TRAINING_SEED,
        config=config,
        device="cuda",
        progress=progress,
    )

    checkpoint_path = output_root / "gate3-development-checkpoint.pt"
    torch.save(
        _checkpoint_payload(
            model,
            training_summary=training_summary,
            training_config=config,
        ),
        checkpoint_path,
    )

    conditions = []
    total_cells = sum(len(GATE3_WIDTHS_BY_DEPTH[depth]) for depth in GATE3_DEPTHS) * len(Gate3ControlMode)
    completed_cells = 0
    eval_started = time.monotonic()
    for depth in GATE3_DEPTHS:
        for width in GATE3_WIDTHS_BY_DEPTH[depth]:
            for mode in Gate3ControlMode:
                condition = evaluate_gate3_condition(
                    model,
                    depth=depth,
                    width=width,
                    mode=mode,
                    world_count=GATE3_DEVELOPMENT_EVAL_WORLD_COUNT,
                    evaluation_batch_size=GATE3_DEVELOPMENT_EVAL_BATCH_SIZE,
                    device="cuda",
                )
                conditions.append(condition)
                completed_cells += 1
                elapsed = (time.monotonic() - eval_started) / 60.0
                print(
                    f"EVAL  {_bar(completed_cells, total_cells)} "
                    f"{100.0 * completed_cells / total_cells:6.2f}% "
                    f"{completed_cells:2d}/{total_cells} H{depth} W={width:<3d} "
                    f"{mode.value:<23s} exact={condition.exact_solve_rate:.4f} elapsed={elapsed:.1f}m",
                    flush=True,
                )

    paired = build_gate3_paired_summaries(
        conditions,
        bootstrap_samples=GATE3_DEVELOPMENT_BOOTSTRAP_SAMPLES,
    )
    result = Gate3DevelopmentResult(
        experiment_version="gate3-hypothesis-population-development-v0",
        evaluation_split="development",
        confirmation_opened=False,
        training=training_summary,
        training_config=config,
        evaluation_world_count=GATE3_DEVELOPMENT_EVAL_WORLD_COUNT,
        evaluation_batch_size=GATE3_DEVELOPMENT_EVAL_BATCH_SIZE,
        bootstrap_samples=GATE3_DEVELOPMENT_BOOTSTRAP_SAMPLES,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    result_path = output_root / "gate3-development.json"
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
        "training_seed": GATE3_FIRST_DEVELOPMENT_TRAINING_SEED,
        "learned_parameter_count": training_summary.learned_parameter_count,
        "parameter_fingerprint": training_summary.parameter_fingerprint,
    }
    runtime_path = output_root / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    primary = result.to_dict()["primary_comparisons"]
    print(
        json.dumps(
            {
                "status": "GATE3_DEVELOPMENT_ONLY_COMPLETE",
                "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
                "confirmation_opened": False,
                "training_seed": GATE3_FIRST_DEVELOPMENT_TRAINING_SEED,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": runtime["checkpoint_sha256"],
                "learned_parameter_count": training_summary.learned_parameter_count,
                "parameter_fingerprint": training_summary.parameter_fingerprint,
                "result": str(result_path),
                "primary_comparisons": primary,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
