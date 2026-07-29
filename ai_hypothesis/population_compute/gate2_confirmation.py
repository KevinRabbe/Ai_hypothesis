"""Frozen Gate-2 confirmation execution with observational progress callbacks.

This module is separate from the measured development implementation.  It deliberately reuses
its exact model/world/update functions while exposing progress for long local runs.  Regression
tests must prove the progress-enabled path produces the same deterministic checkpoint/evaluation
as the existing silent development helpers for matched inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import nn

from .gate2_development import (
    Gate2ConditionEvaluation,
    Gate2PairedSummary,
    Gate2TrainingConfig,
    Gate2TrainingSummary,
    _payload_targets,
    _training_world_seed,
    build_gate2_paired_summaries,
    evaluate_gate2_condition,
    gate2_stable_training_conditions,
    generate_gate2_split_worlds,
)
from .gate2_persistent_model import (
    Gate2PersistentStateModel,
    build_gate2_tensor_batch,
    parallel_persistent_forward,
)
from .gate2_persistent_state_capacity import (
    GATE2_ENTITY_COUNTS,
    Gate2ControlMode,
    gate2_population_widths,
    generate_gate2_world,
)

GATE2_CONFIRMATION_EXPERIMENT_VERSION = "gate2-persistent-state-confirmation-v0"
GATE2_CONFIRMATION_TRAINING_SEEDS = (3, 4, 5)
GATE2_CONFIRMATION_WORLD_COUNT = 512
GATE2_CONFIRMATION_EVALUATION_BATCH_SIZE = 64
GATE2_CONFIRMATION_BOOTSTRAP_SAMPLES = 2_000
GATE2_CONFIRMATION_CONDITION_COUNT = 36

TrainingProgress = Callable[[int, int, int, int, float], None]
EvaluationProgress = Callable[[int, int, int, int, Gate2ControlMode], None]


@dataclass(frozen=True, slots=True)
class Gate2ConfirmationPrimaryComparison:
    comparison: str
    entity_count: int
    treatment_width: int
    exact_solve_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate2ConfirmationSeedResult:
    experiment_version: str
    evaluation_split: str
    confirmation_opened: bool
    training: Gate2TrainingSummary
    training_config: Gate2TrainingConfig
    evaluation_world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    conditions: tuple[Gate2ConditionEvaluation, ...]
    paired_summaries: tuple[Gate2PairedSummary, ...]
    primary_comparisons: tuple[Gate2ConfirmationPrimaryComparison, ...]
    width1_identity_passed: bool
    seed_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_version": self.experiment_version,
            "evaluation_split": self.evaluation_split,
            "confirmation_opened": self.confirmation_opened,
            "training": asdict(self.training),
            "training_config": {
                "steps": self.training_config.steps,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "weight_decay": self.training_config.weight_decay,
                "gradient_clip_norm": self.training_config.gradient_clip_norm,
                "model": asdict(self.training_config.model),
            },
            "evaluation_world_count": self.evaluation_world_count,
            "evaluation_batch_size": self.evaluation_batch_size,
            "bootstrap_samples": self.bootstrap_samples,
            "conditions": [row.to_dict() for row in self.conditions],
            "paired_summaries": [row.to_dict() for row in self.paired_summaries],
            "primary_comparisons": [row.to_dict() for row in self.primary_comparisons],
            "width1_identity_passed": self.width1_identity_passed,
            "seed_passed": self.seed_passed,
            "scientific_status": "CONFIRMATION_SEED_RESULT",
            "gate2_verdict": "NOT_ASSIGNED_UNTIL_ALL_SEEDS_AND_RESOURCE_PROTOCOL_COMPLETE",
        }


def frozen_confirmation_training_config() -> Gate2TrainingConfig:
    """Return the exact recipe frozen after development seeds 0/1/2."""

    return Gate2TrainingConfig(
        steps=1_000,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
    )


def train_gate2_confirmation_model(
    *,
    training_seed: int,
    config: Gate2TrainingConfig | None = None,
    device: torch.device | str = "cpu",
    progress: TrainingProgress | None = None,
) -> tuple[Gate2PersistentStateModel, Gate2TrainingSummary]:
    """Execute the frozen training loop with an observational-only progress callback."""

    if training_seed not in GATE2_CONFIRMATION_TRAINING_SEEDS:
        raise ValueError("Gate-2 confirmation training seed must be one of 3, 4, 5")
    config = frozen_confirmation_training_config() if config is None else config
    config.validate()

    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = Gate2PersistentStateModel(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    conditions = gate2_stable_training_conditions()
    examples_seen = 0
    model.train()

    for step in range(config.steps):
        entity_count, width = conditions[step % len(conditions)]
        worlds = tuple(
            generate_gate2_world(
                seed=_training_world_seed(
                    training_seed=training_seed,
                    step=step,
                    sample_index=sample_index,
                    entity_count=entity_count,
                    width=width,
                ),
                entity_count=entity_count,
            )
            for sample_index in range(config.batch_size)
        )
        batch = build_gate2_tensor_batch(
            worlds,
            width=width,
            mode=Gate2ControlMode.STABLE_PERSISTENT,
            device=target_device,
        )
        output = parallel_persistent_forward(model, batch)
        targets = _payload_targets(batch.answer_payloads, dtype=output.logits.dtype)
        loss = loss_fn(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Gate-2 confirmation training produced non-finite loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        loss_value = float(loss.detach().item())
        losses.append(loss_value)
        examples_seen += config.batch_size
        if progress is not None:
            progress(step + 1, config.steps, entity_count, width, loss_value)

    summary = Gate2TrainingSummary(
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


def evaluate_gate2_confirmation(
    model: Gate2PersistentStateModel,
    *,
    world_count: int = GATE2_CONFIRMATION_WORLD_COUNT,
    batch_size: int = GATE2_CONFIRMATION_EVALUATION_BATCH_SIZE,
    bootstrap_samples: int = GATE2_CONFIRMATION_BOOTSTRAP_SAMPLES,
    device: torch.device | str = "cpu",
    progress: EvaluationProgress | None = None,
) -> tuple[tuple[Gate2ConditionEvaluation, ...], tuple[Gate2PairedSummary, ...]]:
    """Evaluate the frozen confirmation split while reporting completed matrix cells."""

    if world_count != GATE2_CONFIRMATION_WORLD_COUNT:
        raise ValueError("Gate-2 confirmation world_count is frozen at 512")
    if batch_size != GATE2_CONFIRMATION_EVALUATION_BATCH_SIZE:
        raise ValueError("Gate-2 confirmation evaluation batch size is frozen at 64")
    if bootstrap_samples != GATE2_CONFIRMATION_BOOTSTRAP_SAMPLES:
        raise ValueError("Gate-2 confirmation bootstrap sample count is frozen at 2000")

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    expected_count = model.trainable_parameter_count()
    expected_fingerprint = model.parameter_fingerprint()
    rows: list[Gate2ConditionEvaluation] = []

    for entity_count in GATE2_ENTITY_COUNTS:
        worlds = generate_gate2_split_worlds(
            split="confirmation",
            entity_count=entity_count,
            world_count=world_count,
            allow_confirmation=True,
        )
        for width in gate2_population_widths(entity_count):
            for mode in Gate2ControlMode:
                row = evaluate_gate2_condition(
                    model,
                    worlds,
                    width=width,
                    mode=mode,
                    batch_size=batch_size,
                    device=target_device,
                )
                if row.learned_parameter_count != expected_count:
                    raise RuntimeError("Gate-2 confirmation changed learned parameter count")
                if row.parameter_fingerprint != expected_fingerprint:
                    raise RuntimeError("Gate-2 confirmation changed checkpoint fingerprint")
                rows.append(row)
                if progress is not None:
                    progress(
                        len(rows),
                        GATE2_CONFIRMATION_CONDITION_COUNT,
                        entity_count,
                        width,
                        mode,
                    )

    if len(rows) != GATE2_CONFIRMATION_CONDITION_COUNT:
        raise RuntimeError("Gate-2 confirmation matrix is incomplete")
    if model.parameter_fingerprint() != expected_fingerprint:
        raise RuntimeError("Gate-2 confirmation evaluation mutated the checkpoint")

    paired = build_gate2_paired_summaries(rows, bootstrap_samples=bootstrap_samples)
    return tuple(rows), paired


def run_gate2_confirmation_seed(
    *,
    training_seed: int,
    device: torch.device | str = "cpu",
    training_progress: TrainingProgress | None = None,
    evaluation_progress: EvaluationProgress | None = None,
) -> tuple[Gate2PersistentStateModel, Gate2ConfirmationSeedResult]:
    """Run one frozen confirmation training seed and evaluate its exact acceptance rule."""

    config = frozen_confirmation_training_config()
    model, training = train_gate2_confirmation_model(
        training_seed=training_seed,
        config=config,
        device=device,
        progress=training_progress,
    )
    conditions, paired = evaluate_gate2_confirmation(
        model,
        device=device,
        progress=evaluation_progress,
    )

    primary = _primary_confirmation_comparisons(paired)
    width1_identity_passed = _width1_identity_passed(paired)
    seed_passed = width1_identity_passed and len(primary) == 4 and all(row.passed for row in primary)

    result = Gate2ConfirmationSeedResult(
        experiment_version=GATE2_CONFIRMATION_EXPERIMENT_VERSION,
        evaluation_split="confirmation",
        confirmation_opened=True,
        training=training,
        training_config=config,
        evaluation_world_count=GATE2_CONFIRMATION_WORLD_COUNT,
        evaluation_batch_size=GATE2_CONFIRMATION_EVALUATION_BATCH_SIZE,
        bootstrap_samples=GATE2_CONFIRMATION_BOOTSTRAP_SAMPLES,
        conditions=conditions,
        paired_summaries=paired,
        primary_comparisons=primary,
        width1_identity_passed=width1_identity_passed,
        seed_passed=seed_passed,
    )
    return model, result


def _primary_confirmation_comparisons(
    paired: tuple[Gate2PairedSummary, ...],
) -> tuple[Gate2ConfirmationPrimaryComparison, ...]:
    selected: list[Gate2ConfirmationPrimaryComparison] = []
    for row in paired:
        wanted = (
            row.comparison == "stable_width_vs_width1"
            and ((row.entity_count, row.treatment_width) in {(64, 64), (256, 256)})
        ) or (
            row.comparison in {"stable_vs_reshuffled", "stable_vs_reset"}
            and row.entity_count == 256
            and row.treatment_width == 256
        )
        if not wanted:
            continue
        selected.append(
            Gate2ConfirmationPrimaryComparison(
                comparison=row.comparison,
                entity_count=row.entity_count,
                treatment_width=row.treatment_width,
                exact_solve_delta=row.exact_solve_delta,
                bootstrap_ci_low=row.bootstrap_ci_low,
                bootstrap_ci_high=row.bootstrap_ci_high,
                passed=row.bootstrap_ci_low > 0.0,
            )
        )

    order = {"stable_width_vs_width1": 0, "stable_vs_reshuffled": 1, "stable_vs_reset": 2}
    selected.sort(key=lambda row: (order[row.comparison], row.entity_count))
    if len(selected) != 4:
        raise RuntimeError("Gate-2 confirmation primary comparison set is incomplete")
    return tuple(selected)


def _width1_identity_passed(paired: tuple[Gate2PairedSummary, ...]) -> bool:
    identities = [
        row
        for row in paired
        if row.comparison == "stable_vs_reshuffled" and row.treatment_width == 1
    ]
    if len(identities) != len(GATE2_ENTITY_COUNTS):
        return False
    return all(
        row.exact_solve_delta == 0.0
        and row.treatment_only == 0
        and row.reference_only == 0
        and row.bootstrap_ci_low == 0.0
        and row.bootstrap_ci_high == 0.0
        for row in identities
    )
