"""Training and evaluation for the fixed-parameter collective-relay gate.

The experiment deliberately trains one shared checkpoint and then reuses that exact
parameter state across every population point and communication ablation. Training
examples are information-complete at their active population size so the loss never
rewards guessing a target that is absent from the visible scope.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from .collective_relay import (
    RELAY_DIFFICULTIES,
    RelayDifficulty,
    RelayWorld,
    generate_relay_dataset,
    generate_relay_world,
    relay_scope_thresholds,
)
from .contract import (
    DEVELOPMENT_POPULATION_SIZES,
    CommunicationMode,
    CurveAssessment,
    PopulationCondition,
    PopulationRunMetrics,
    assess_scaling_curve,
)
from .relay_model import (
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
)


RELAY_EXPERIMENT_VERSION = "population-compute-relay-training-v0"
TRAINING_SEED_LIMIT = 1_000_000_000
TRAINING_SEED_STRIDE = 50_000_000
TRAINING_STEP_STRIDE = 8_192
DEVELOPMENT_SEED_START = 1_000_000_000
CONFIRMATION_SEED_START = 1_500_000_000
SPLIT_SEED_SPAN = 500_000_000


@dataclass(frozen=True, slots=True)
class RelayTrainingConfig:
    steps: int = 2_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    model: RelayPopulationConfig = RelayPopulationConfig()

    def validate(self) -> None:
        self.model.validate()
        if self.steps <= 0:
            raise ValueError("training steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("training batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if self.steps * TRAINING_STEP_STRIDE >= TRAINING_SEED_STRIDE:
            raise ValueError("training steps exceed the reserved per-seed world range")
        if self.batch_size * 4 >= TRAINING_STEP_STRIDE:
            raise ValueError("training batch_size exceeds the deterministic step seed budget")


@dataclass(frozen=True, slots=True)
class RelayTrainingSummary:
    training_seed: int
    steps: int
    examples_seen: int
    initial_loss: float
    final_loss: float
    mean_last_50_loss: float
    learned_parameter_count: int
    parameter_fingerprint: str


@dataclass(frozen=True, slots=True)
class RelayScopeCohortResult:
    """Exact solve behavior for worlds sharing one first-complete population threshold."""

    scope_threshold: int
    task_count: int
    solved_count: int

    def validate(self) -> None:
        if self.scope_threshold <= 0:
            raise ValueError("scope_threshold must be positive")
        if self.task_count <= 0:
            raise ValueError("scope cohort task_count must be positive")
        if not 0 <= self.solved_count <= self.task_count:
            raise ValueError("scope cohort solved_count must be within task_count")

    @property
    def solve_rate(self) -> float:
        return self.solved_count / self.task_count

    def to_dict(self) -> dict[str, object]:
        return {
            "scope_threshold": self.scope_threshold,
            "task_count": self.task_count,
            "solved_count": self.solved_count,
            "solve_rate": self.solve_rate,
        }


@dataclass(frozen=True, slots=True)
class RelayEvaluationResult:
    metrics: PopulationRunMetrics
    bit_accuracy: float
    scope_cohorts: tuple[RelayScopeCohortResult, ...]

    def validate(self) -> None:
        self.metrics.validate()
        if not math.isfinite(self.bit_accuracy) or not 0.0 <= self.bit_accuracy <= 1.0:
            raise ValueError("bit_accuracy must be finite and within [0, 1]")
        if not self.scope_cohorts:
            raise ValueError("relay evaluation must contain scope cohorts")
        for cohort in self.scope_cohorts:
            cohort.validate()
        thresholds = tuple(cohort.scope_threshold for cohort in self.scope_cohorts)
        if tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError("scope cohort thresholds must be unique and increasing")
        if sum(cohort.task_count for cohort in self.scope_cohorts) != self.metrics.task_count:
            raise ValueError("scope cohort task counts do not cover the evaluation")
        expected_complete = sum(
            cohort.task_count
            for cohort in self.scope_cohorts
            if cohort.scope_threshold
            <= self.metrics.condition.nominal_population_size
        )
        if expected_complete != self.metrics.information_complete_count:
            raise ValueError("scope cohorts disagree with information-complete accounting")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        run = self.metrics
        return {
            "training_seed": run.training_seed,
            "benchmark_seed": run.benchmark_seed,
            "difficulty": run.difficulty,
            "learned_parameter_count": run.learned_parameter_count,
            "parameter_fingerprint": run.parameter_fingerprint,
            "condition": {
                "nominal_population_size": run.condition.nominal_population_size,
                "active_state_count": run.condition.active_state_count,
                "recurrent_rounds": run.condition.recurrent_rounds,
                "communication_mode": run.condition.communication_mode.value,
                "worker_updates": run.condition.worker_updates,
            },
            "task_count": run.task_count,
            "solved_count": run.solved_count,
            "solve_rate": run.solve_rate,
            "information_complete_count": run.information_complete_count,
            "information_complete_rate": run.information_complete_rate,
            "solved_information_complete_count": run.solved_information_complete_count,
            "solve_rate_given_information_complete": run.solve_rate_given_information_complete,
            "solved_information_incomplete_count": run.solved_information_incomplete_count,
            "solve_rate_given_information_incomplete": run.solve_rate_given_information_incomplete,
            "scope_cohorts": [cohort.to_dict() for cohort in self.scope_cohorts],
            "bit_accuracy": self.bit_accuracy,
            "messages_emitted": run.messages_emitted,
            "communicated_scalar_count": run.communicated_scalar_count,
            "peak_worker_state_bytes": run.peak_worker_state_bytes,
            "elapsed_seconds": run.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class RelayDevelopmentResult:
    experiment_version: str
    evaluation_split: str
    training: RelayTrainingSummary
    training_config: RelayTrainingConfig
    evaluation_world_count: int
    evaluation_batch_size: int
    evaluations: tuple[RelayEvaluationResult, ...]
    assessments: tuple[tuple[str, CurveAssessment], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_version": self.experiment_version,
            "evaluation_split": self.evaluation_split,
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
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "assessments": [
                {
                    "difficulty": difficulty,
                    "population_sizes": list(assessment.population_sizes),
                    "solve_rates": list(assessment.solve_rates),
                    "information_complete_rates": list(assessment.information_complete_rates),
                    "solve_rates_given_information_complete": list(
                        assessment.solve_rates_given_information_complete
                    ),
                    "endpoint_gain": assessment.endpoint_gain,
                    "nondecreasing_steps": assessment.nondecreasing_steps,
                    "communication_endpoint_advantage": assessment.communication_endpoint_advantage,
                    "passes_per_curve_scaling_signal": assessment.passes_scaling_signal,
                    "reasons": list(assessment.reasons),
                }
                for difficulty, assessment in self.assessments
            ],
            "interpretation_note": (
                "Development assessments are diagnostic only. Gate-v0 acceptance requires "
                "frozen confirmation worlds across at least three independent training seeds."
            ),
        }


def train_relay_model(
    *,
    training_seed: int,
    config: RelayTrainingConfig = RelayTrainingConfig(),
    device: torch.device | str = "cpu",
) -> tuple[RelayPopulationModel, RelayTrainingSummary]:
    """Train one checkpoint on complete-information relay worlds across population sizes."""

    config.validate()
    _validate_training_seed(training_seed)
    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = RelayPopulationModel(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    examples_seen = 0
    model.train()

    for step in range(config.steps):
        difficulty = RELAY_DIFFICULTIES[step % len(RELAY_DIFFICULTIES)]
        thresholds = relay_scope_thresholds(difficulty)
        threshold_cycle = step // len(RELAY_DIFFICULTIES)
        active_workers = thresholds[threshold_cycle % len(thresholds)]
        worlds = training_world_batch(
            training_seed=training_seed,
            step=step,
            difficulty=difficulty,
            active_workers=active_workers,
            batch_size=config.batch_size,
        )
        batch = build_relay_tensor_batch(
            worlds,
            active_workers=active_workers,
            device=target_device,
        )
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError("training batch contains information-incomplete relay worlds")

        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
        )
        targets = (batch.target_bits > 0).to(dtype=output.logits.dtype)
        loss = loss_fn(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("relay training produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        losses.append(float(loss.detach().item()))
        examples_seen += config.batch_size

    fingerprint = model.parameter_fingerprint()
    summary = RelayTrainingSummary(
        training_seed=training_seed,
        steps=config.steps,
        examples_seen=examples_seen,
        initial_loss=losses[0],
        final_loss=losses[-1],
        mean_last_50_loss=sum(losses[-50:]) / len(losses[-50:]),
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=fingerprint,
    )
    return model, summary


def training_world_batch(
    *,
    training_seed: int,
    step: int,
    difficulty: RelayDifficulty,
    active_workers: int,
    batch_size: int,
) -> tuple[RelayWorld, ...]:
    """Return deterministic worlds whose first-complete threshold is active_workers."""

    _validate_training_seed(training_seed)
    if step < 0:
        raise ValueError("training step must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    thresholds = relay_scope_thresholds(difficulty)
    try:
        threshold_index = thresholds.index(active_workers)
    except ValueError as exc:
        raise ValueError("active_workers is not an admissible complete-information threshold") from exc

    modulus = len(thresholds)
    base = training_seed * TRAINING_SEED_STRIDE + step * TRAINING_STEP_STRIDE
    first_seed = base + ((threshold_index - (base % modulus)) % modulus)
    seeds = tuple(first_seed + offset * modulus for offset in range(batch_size))
    if seeds[-1] >= (training_seed + 1) * TRAINING_SEED_STRIDE:
        raise ValueError("training batch escaped its reserved per-seed world range")
    worlds = tuple(generate_relay_world(seed, difficulty) for seed in seeds)
    if any(world.scope_threshold != active_workers for world in worlds):
        raise AssertionError("training seed alignment did not preserve scope threshold")
    return worlds


def evaluate_relay_split(
    model: RelayPopulationModel,
    *,
    training_seed: int,
    split: str = "development",
    world_count: int = 1_000,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    allow_confirmation: bool = False,
) -> tuple[RelayEvaluationResult, ...]:
    """Evaluate one immutable checkpoint on matched worlds across the full population curve."""

    _validate_training_seed(training_seed)
    if split == "confirmation" and not allow_confirmation:
        raise ValueError("confirmation split is locked unless allow_confirmation=True")
    benchmark_seed = split_seed_start(split)
    if world_count <= 0:
        raise ValueError("evaluation world_count must be positive")
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    if benchmark_seed + world_count > split_seed_limit(split):
        raise ValueError("evaluation worlds exceed the reserved split seed range")

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    expected_parameter_count = model.trainable_parameter_count()
    expected_fingerprint = model.parameter_fingerprint()
    results: list[RelayEvaluationResult] = []

    for difficulty in RELAY_DIFFICULTIES:
        worlds = generate_relay_dataset(
            start_seed=benchmark_seed,
            world_count=world_count,
            difficulty=difficulty,
        )
        for communication_mode in (
            CommunicationMode.SPARSE_SHARED_V0,
            CommunicationMode.NO_COMMUNICATION,
        ):
            for active_workers in DEVELOPMENT_POPULATION_SIZES:
                result = evaluate_relay_condition(
                    model,
                    worlds,
                    training_seed=training_seed,
                    benchmark_seed=benchmark_seed,
                    active_workers=active_workers,
                    communication_mode=communication_mode,
                    batch_size=batch_size,
                    device=target_device,
                )
                if result.metrics.learned_parameter_count != expected_parameter_count:
                    raise RuntimeError("relay evaluation changed learned parameter count")
                if result.metrics.parameter_fingerprint != expected_fingerprint:
                    raise RuntimeError("relay evaluation changed parameter fingerprint")
                results.append(result)

    if model.parameter_fingerprint() != expected_fingerprint:
        raise RuntimeError("relay evaluation mutated the trained checkpoint")
    return tuple(results)


def evaluate_relay_condition(
    model: RelayPopulationModel,
    worlds: Sequence[RelayWorld],
    *,
    training_seed: int,
    benchmark_seed: int,
    active_workers: int,
    communication_mode: CommunicationMode,
    batch_size: int,
    device: torch.device | str,
) -> RelayEvaluationResult:
    """Evaluate one difficulty/population/communication condition."""

    if not worlds:
        raise ValueError("relay evaluation requires at least one world")
    if batch_size <= 0:
        raise ValueError("relay evaluation batch_size must be positive")
    difficulty = worlds[0].difficulty
    if any(world.difficulty != difficulty for world in worlds):
        raise ValueError("one relay evaluation condition must use one difficulty")
    target_device = torch.device(device)
    parameter_count = model.trainable_parameter_count()
    fingerprint = model.parameter_fingerprint()

    thresholds = relay_scope_thresholds(difficulty)
    cohort_counts: dict[int, list[int]] = {
        threshold: [0, 0] for threshold in thresholds
    }
    solved_count = 0
    information_complete_count = 0
    solved_complete_count = 0
    correct_bits = 0
    total_bits = 0
    messages_emitted = 0
    communicated_scalars = 0
    peak_state_bytes = 0

    _synchronize(target_device)
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(worlds), batch_size):
            world_batch = worlds[offset : offset + batch_size]
            batch = build_relay_tensor_batch(
                world_batch,
                active_workers=active_workers,
                device=target_device,
            )
            output = model(
                batch,
                communication_mode=communication_mode,
            )
            predicted = decode_node_logits(output.logits)
            solved = predicted.eq(batch.answer_keys)
            complete = batch.information_complete
            solved_count += int(solved.sum().item())
            information_complete_count += int(complete.sum().item())
            solved_complete_count += int((solved & complete).sum().item())

            solved_rows = tuple(bool(value) for value in solved.detach().cpu().tolist())
            for world, row_solved in zip(world_batch, solved_rows, strict=True):
                cohort = cohort_counts.get(world.scope_threshold)
                if cohort is None:
                    raise RuntimeError("relay world used an unexpected scope threshold")
                cohort[0] += 1
                cohort[1] += int(row_solved)

            target_bits = batch.target_bits > 0
            predicted_bits = output.logits >= 0
            correct_bits += int(predicted_bits.eq(target_bits).sum().item())
            total_bits += int(target_bits.numel())
            messages_emitted += output.telemetry.messages_emitted
            communicated_scalars += output.telemetry.communicated_scalar_count
            logical_state_bytes = (
                len(world_batch)
                * active_workers
                * model.config.state_width
                * output.final_states.element_size()
            )
            peak_state_bytes = max(peak_state_bytes, logical_state_bytes)
    _synchronize(target_device)
    elapsed = time.perf_counter() - started

    metrics = PopulationRunMetrics(
        training_seed=training_seed,
        benchmark_seed=benchmark_seed,
        difficulty=difficulty.name,
        learned_parameter_count=parameter_count,
        parameter_fingerprint=fingerprint,
        condition=PopulationCondition(
            nominal_population_size=active_workers,
            active_state_count=active_workers,
            recurrent_rounds=difficulty.hop_count,
            communication_mode=communication_mode,
        ),
        task_count=len(worlds),
        solved_count=solved_count,
        information_complete_count=information_complete_count,
        solved_information_complete_count=solved_complete_count,
        messages_emitted=messages_emitted,
        communicated_scalar_count=communicated_scalars,
        peak_worker_state_bytes=peak_state_bytes,
        elapsed_seconds=elapsed,
    )
    metrics.validate()
    scope_cohorts = tuple(
        RelayScopeCohortResult(
            scope_threshold=threshold,
            task_count=cohort_counts[threshold][0],
            solved_count=cohort_counts[threshold][1],
        )
        for threshold in thresholds
    )
    result = RelayEvaluationResult(
        metrics=metrics,
        bit_accuracy=correct_bits / total_bits,
        scope_cohorts=scope_cohorts,
    )
    result.validate()
    return result


def assess_relay_results(
    evaluations: Iterable[RelayEvaluationResult],
) -> tuple[tuple[str, CurveAssessment], ...]:
    """Apply the frozen per-curve diagnostic assessment to matched development results."""

    rows = tuple(evaluations)
    assessments: list[tuple[str, CurveAssessment]] = []
    for difficulty in (difficulty.name for difficulty in RELAY_DIFFICULTIES):
        communicating = tuple(
            row.metrics
            for row in rows
            if row.metrics.difficulty == difficulty
            and row.metrics.condition.communication_mode
            is CommunicationMode.SPARSE_SHARED_V0
        )
        no_communication = tuple(
            row.metrics
            for row in rows
            if row.metrics.difficulty == difficulty
            and row.metrics.condition.communication_mode
            is CommunicationMode.NO_COMMUNICATION
        )
        assessments.append(
            (difficulty, assess_scaling_curve(communicating, no_communication))
        )
    return tuple(assessments)


def run_development_experiment(
    *,
    training_seed: int,
    training_config: RelayTrainingConfig = RelayTrainingConfig(),
    evaluation_world_count: int = 1_000,
    evaluation_batch_size: int = 64,
    device: torch.device | str = "cpu",
    evaluation_split: str = "development",
    allow_confirmation: bool = False,
) -> RelayDevelopmentResult:
    model, training = train_relay_model(
        training_seed=training_seed,
        config=training_config,
        device=device,
    )
    evaluations = evaluate_relay_split(
        model,
        training_seed=training_seed,
        split=evaluation_split,
        world_count=evaluation_world_count,
        batch_size=evaluation_batch_size,
        device=device,
        allow_confirmation=allow_confirmation,
    )
    assessments = assess_relay_results(evaluations)
    return RelayDevelopmentResult(
        experiment_version=RELAY_EXPERIMENT_VERSION,
        evaluation_split=evaluation_split,
        training=training,
        training_config=training_config,
        evaluation_world_count=evaluation_world_count,
        evaluation_batch_size=evaluation_batch_size,
        evaluations=evaluations,
        assessments=assessments,
    )


def save_relay_checkpoint(
    model: RelayPopulationModel,
    training: RelayTrainingSummary,
    training_config: RelayTrainingConfig,
    path: Path | str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_version": RELAY_EXPERIMENT_VERSION,
        "training_seed": training.training_seed,
        "training_summary": asdict(training),
        "training_config": {
            "steps": training_config.steps,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "gradient_clip_norm": training_config.gradient_clip_norm,
            "model": asdict(training_config.model),
        },
        "state_dict": model.state_dict(),
        "parameter_fingerprint": model.parameter_fingerprint(),
    }
    torch.save(payload, output)
    return output


def load_relay_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[RelayPopulationModel, dict[str, object]]:
    payload = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
    if payload.get("experiment_version") != RELAY_EXPERIMENT_VERSION:
        raise ValueError("unexpected relay checkpoint experiment version")
    config_data = payload.get("training_config", {}).get("model")
    if not isinstance(config_data, dict):
        raise ValueError("relay checkpoint is missing model configuration")
    model = RelayPopulationModel(RelayPopulationConfig(**config_data)).to(device)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("relay checkpoint is missing state_dict")
    model.load_state_dict(state_dict)
    expected = payload.get("parameter_fingerprint")
    actual = model.parameter_fingerprint()
    if expected != actual:
        raise ValueError("relay checkpoint parameter fingerprint does not match payload")
    return model, payload


def write_relay_result(result: RelayDevelopmentResult, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def split_seed_start(split: str) -> int:
    if split == "development":
        return DEVELOPMENT_SEED_START
    if split == "confirmation":
        return CONFIRMATION_SEED_START
    raise ValueError(f"unknown relay evaluation split {split!r}")


def split_seed_limit(split: str) -> int:
    return split_seed_start(split) + SPLIT_SEED_SPAN


def _validate_training_seed(training_seed: int) -> None:
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    start = training_seed * TRAINING_SEED_STRIDE
    if start >= TRAINING_SEED_LIMIT:
        raise ValueError("training_seed exceeds the reserved training seed range")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
