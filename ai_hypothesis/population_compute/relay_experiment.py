"""Train one shared relay model and evaluate fixed-weight population-compute curves."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

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
    GateCriteria,
    PopulationCondition,
    PopulationRunMetrics,
    assess_scaling_curve,
    validate_fixed_parameter_identity,
)
from .relay_model import (
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
)


RELAY_CHECKPOINT_VERSION = "relay-population-checkpoint-v0"
RELAY_TRAINING_PROTOCOL = "balanced-complete-populations-v0"
TRAINING_POPULATION_SIZES: tuple[int, ...] = (4, 16, 64, 256)
TRAINING_WORLD_SEED_BASE = 10_000_000
DEVELOPMENT_WORLD_SEED_BASE = 1_000_000_000
CONFIRMATION_WORLD_SEED_BASE = 2_000_000_000
SEED_BLOCK_SIZE = 1_000_000
_MAX_SCOPE_CYCLE = max(
    len(relay_scope_thresholds(difficulty)) for difficulty in RELAY_DIFFICULTIES
)


@dataclass(frozen=True, slots=True)
class RelayTrainingConfig:
    training_seed: int = 0
    steps: int = 2_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reserved_world_slots = self.steps * self.batch_size * _MAX_SCOPE_CYCLE
        if reserved_world_slots > SEED_BLOCK_SIZE:
            raise ValueError("training world schedule exceeds reserved seed block")
        _seed_block_start(
            TRAINING_WORLD_SEED_BASE,
            self.training_seed,
            upper_bound=DEVELOPMENT_WORLD_SEED_BASE,
            label="training",
        )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")


@dataclass(frozen=True, slots=True)
class RelayTrainingBatchPlan:
    step: int
    difficulty: RelayDifficulty
    active_workers: int
    scope_threshold: int

    def validate(self) -> None:
        if self.step < 0:
            raise ValueError("training step must be non-negative")
        self.difficulty.validate()
        if self.active_workers not in TRAINING_POPULATION_SIZES:
            raise ValueError("training active_workers is outside the frozen training ladder")
        if self.active_workers < self.difficulty.hop_count:
            raise ValueError("training population cannot contain this relay difficulty")
        admissible = tuple(
            threshold
            for threshold in relay_scope_thresholds(self.difficulty)
            if threshold <= self.active_workers
        )
        if self.scope_threshold not in admissible:
            raise ValueError("training scope_threshold is not complete at active_workers")


@dataclass(frozen=True, slots=True)
class RelayTrainingSummary:
    training_seed: int
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    first_loss: float
    final_loss: float
    best_loss: float
    elapsed_seconds: float
    learned_parameter_count: int
    parameter_fingerprint: str
    training_protocol: str = RELAY_TRAINING_PROTOCOL
    population_batch_counts: tuple[tuple[int, int], ...] = ()
    difficulty_batch_counts: tuple[tuple[str, int], ...] = ()
    scope_threshold_batch_counts: tuple[tuple[int, int], ...] = ()

    def validate(self) -> None:
        if self.training_seed < 0 or self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("training summary contains invalid scope")
        for value in (self.first_loss, self.final_loss, self.best_loss, self.elapsed_seconds):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("training summary values must be finite and non-negative")
        if self.learned_parameter_count <= 0:
            raise ValueError("learned_parameter_count must be positive")
        if not self.parameter_fingerprint:
            raise ValueError("parameter_fingerprint must be non-empty")
        if self.training_protocol != RELAY_TRAINING_PROTOCOL:
            raise ValueError("unexpected relay training protocol")
        self._validate_counts(
            self.population_batch_counts,
            expected_keys=TRAINING_POPULATION_SIZES,
            label="population",
        )
        self._validate_counts(
            self.difficulty_batch_counts,
            expected_keys=tuple(difficulty.name for difficulty in RELAY_DIFFICULTIES),
            label="difficulty",
        )
        all_thresholds = tuple(
            sorted(
                {
                    threshold
                    for difficulty in RELAY_DIFFICULTIES
                    for threshold in relay_scope_thresholds(difficulty)
                }
            )
        )
        self._validate_counts(
            self.scope_threshold_batch_counts,
            expected_keys=all_thresholds,
            label="scope threshold",
        )

    def _validate_counts(
        self,
        values: tuple[tuple[object, int], ...],
        *,
        expected_keys: tuple[object, ...],
        label: str,
    ) -> None:
        if tuple(key for key, _ in values) != expected_keys:
            raise ValueError(f"training {label} counts do not use the frozen key order")
        counts = tuple(count for _, count in values)
        if any(count < 0 for count in counts) or sum(counts) != self.steps:
            raise ValueError(f"training {label} counts do not reconcile with steps")


@dataclass(frozen=True, slots=True)
class RelayDevelopmentResult:
    training: RelayTrainingSummary
    model_config: RelayPopulationConfig
    benchmark_seed: int
    world_count_per_difficulty: int
    runs: tuple[PopulationRunMetrics, ...]
    assessments: tuple[tuple[str, CurveAssessment], ...]

    def validate(self) -> None:
        self.training.validate()
        self.model_config.validate()
        if self.benchmark_seed < 0:
            raise ValueError("benchmark_seed must be non-negative")
        if self.world_count_per_difficulty <= 0:
            raise ValueError("world_count_per_difficulty must be positive")

        expected_keys = {
            (difficulty.name, mode, population_size)
            for difficulty in RELAY_DIFFICULTIES
            for mode in (
                CommunicationMode.NO_COMMUNICATION,
                CommunicationMode.SPARSE_SHARED_V0,
            )
            for population_size in DEVELOPMENT_POPULATION_SIZES
        }
        actual_keys: set[tuple[str, CommunicationMode, int]] = set()
        for run in self.runs:
            run.validate()
            key = (
                run.difficulty,
                run.condition.communication_mode,
                run.condition.nominal_population_size,
            )
            if key in actual_keys:
                raise ValueError("development result contains a duplicate condition")
            actual_keys.add(key)
        if actual_keys != expected_keys:
            raise ValueError("development result does not contain the frozen 30 conditions")

        validate_fixed_parameter_identity(self.runs)
        fingerprints = {run.parameter_fingerprint for run in self.runs}
        counts = {run.learned_parameter_count for run in self.runs}
        if fingerprints != {self.training.parameter_fingerprint}:
            raise ValueError("development curves do not use the trained checkpoint fingerprint")
        if counts != {self.training.learned_parameter_count}:
            raise ValueError("development curves changed the trained parameter count")

        assessment_names = tuple(name for name, _ in self.assessments)
        if assessment_names != tuple(difficulty.name for difficulty in RELAY_DIFFICULTIES):
            raise ValueError("development result assessments do not match relay difficulties")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "experiment": "population-compute-scaling-v0",
            "benchmark": "collective-relay-v0",
            "split": "development",
            "training": asdict(self.training),
            "model_config": asdict(self.model_config),
            "benchmark_seed": self.benchmark_seed,
            "world_count_per_difficulty": self.world_count_per_difficulty,
            "population_sizes": list(DEVELOPMENT_POPULATION_SIZES),
            "communication_modes": [
                CommunicationMode.NO_COMMUNICATION.value,
                CommunicationMode.SPARSE_SHARED_V0.value,
            ],
            "runs": [_run_to_dict(run) for run in self.runs],
            "assessments": {
                difficulty: {
                    "population_sizes": list(assessment.population_sizes),
                    "solve_rates": list(assessment.solve_rates),
                    "information_complete_rates": list(
                        assessment.information_complete_rates
                    ),
                    "solve_rates_given_information_complete": list(
                        assessment.solve_rates_given_information_complete
                    ),
                    "endpoint_gain": assessment.endpoint_gain,
                    "nondecreasing_steps": assessment.nondecreasing_steps,
                    "communication_endpoint_advantage": (
                        assessment.communication_endpoint_advantage
                    ),
                    "passes_scaling_signal": assessment.passes_scaling_signal,
                    "reasons": list(assessment.reasons),
                }
                for difficulty, assessment in self.assessments
            },
        }


def relay_training_schedule(steps: int) -> tuple[RelayTrainingBatchPlan, ...]:
    """Build deterministic population-balanced complete-information training batches.

    Populations 4/16/64/256 receive equal batch exposure up to a one-batch remainder.
    Population 1 is omitted because a multi-hop relay is never information-complete there.
    Within each population, admissible difficulties and controlled scope thresholds cycle
    deterministically so training does not privilege only the 256-state condition.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    plans: list[RelayTrainingBatchPlan] = []
    for step in range(steps):
        active_workers = TRAINING_POPULATION_SIZES[step % len(TRAINING_POPULATION_SIZES)]
        population_round = step // len(TRAINING_POPULATION_SIZES)
        difficulties = tuple(
            difficulty
            for difficulty in RELAY_DIFFICULTIES
            if difficulty.hop_count <= active_workers
        )
        difficulty = difficulties[population_round % len(difficulties)]
        difficulty_round = population_round // len(difficulties)
        thresholds = tuple(
            threshold
            for threshold in relay_scope_thresholds(difficulty)
            if threshold <= active_workers
        )
        scope_threshold = thresholds[difficulty_round % len(thresholds)]
        plan = RelayTrainingBatchPlan(
            step=step,
            difficulty=difficulty,
            active_workers=active_workers,
            scope_threshold=scope_threshold,
        )
        plan.validate()
        plans.append(plan)
    return tuple(plans)


def train_relay_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_config: RelayPopulationConfig = RelayPopulationConfig(),
    training_config: RelayTrainingConfig = RelayTrainingConfig(),
    device: torch.device | str = "cpu",
) -> RelayTrainingSummary:
    """Train one shared model across balanced information-complete populations."""

    model_config.validate()
    training_config.validate()
    schedule = relay_training_schedule(training_config.steps)
    target_device = torch.device(device)
    torch.manual_seed(training_config.training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.training_seed)

    model = RelayPopulationModel(model_config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    losses: list[float] = []
    _synchronize(target_device)
    started = time.perf_counter()
    model.train()
    for plan in schedule:
        worlds = _training_worlds_for_plan(
            training_config.training_seed,
            plan,
            training_config.batch_size,
        )
        batch = build_relay_tensor_batch(
            worlds,
            active_workers=plan.active_workers,
            device=target_device,
        )
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError("training curriculum emitted an information-incomplete world")
        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=plan.difficulty.hop_count,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("relay training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            training_config.gradient_clip_norm,
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
    _synchronize(target_device)
    elapsed = time.perf_counter() - started

    population_counts = Counter(plan.active_workers for plan in schedule)
    difficulty_counts = Counter(plan.difficulty.name for plan in schedule)
    threshold_counts = Counter(plan.scope_threshold for plan in schedule)
    all_thresholds = tuple(
        sorted(
            {
                threshold
                for difficulty in RELAY_DIFFICULTIES
                for threshold in relay_scope_thresholds(difficulty)
            }
        )
    )
    summary = RelayTrainingSummary(
        training_seed=training_config.training_seed,
        steps=training_config.steps,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        first_loss=losses[0],
        final_loss=losses[-1],
        best_loss=min(losses),
        elapsed_seconds=elapsed,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
        training_protocol=RELAY_TRAINING_PROTOCOL,
        population_batch_counts=tuple(
            (population, population_counts.get(population, 0))
            for population in TRAINING_POPULATION_SIZES
        ),
        difficulty_batch_counts=tuple(
            (difficulty.name, difficulty_counts.get(difficulty.name, 0))
            for difficulty in RELAY_DIFFICULTIES
        ),
        scope_threshold_batch_counts=tuple(
            (threshold, threshold_counts.get(threshold, 0))
            for threshold in all_thresholds
        ),
    )
    summary.validate()

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": RELAY_CHECKPOINT_VERSION,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "training_summary": asdict(summary),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return summary


def load_relay_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[RelayPopulationModel, RelayTrainingSummary]:
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if payload.get("checkpoint_version") != RELAY_CHECKPOINT_VERSION:
        raise ValueError("unexpected relay checkpoint version")
    model_config = RelayPopulationConfig(**payload["model_config"])
    raw_summary = dict(payload["training_summary"])
    raw_summary["population_batch_counts"] = tuple(
        tuple(item) for item in raw_summary.get("population_batch_counts", ())
    )
    raw_summary["difficulty_batch_counts"] = tuple(
        tuple(item) for item in raw_summary.get("difficulty_batch_counts", ())
    )
    raw_summary["scope_threshold_batch_counts"] = tuple(
        tuple(item) for item in raw_summary.get("scope_threshold_batch_counts", ())
    )
    training_summary = RelayTrainingSummary(**raw_summary)
    model = RelayPopulationModel(model_config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.trainable_parameter_count() != training_summary.learned_parameter_count:
        raise RuntimeError("relay checkpoint learned parameter count changed on reload")
    if model.parameter_fingerprint() != training_summary.parameter_fingerprint:
        raise RuntimeError("relay checkpoint fingerprint changed on reload")
    training_summary.validate()
    return model, training_summary


def evaluate_relay_development(
    model: RelayPopulationModel,
    training_summary: RelayTrainingSummary,
    *,
    benchmark_seed: int = 0,
    world_count_per_difficulty: int = 1_000,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    criteria: GateCriteria = GateCriteria(),
) -> RelayDevelopmentResult:
    """Evaluate the frozen development curve without changing learned weights."""

    training_summary.validate()
    criteria.validate()
    if benchmark_seed < 0:
        raise ValueError("benchmark_seed must be non-negative")
    if not 0 < world_count_per_difficulty <= SEED_BLOCK_SIZE:
        raise ValueError("world_count_per_difficulty is outside reserved seed block")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    benchmark_start = _development_world_seed(benchmark_seed, 0)

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    parameter_count = model.trainable_parameter_count()
    fingerprint = model.parameter_fingerprint()
    if (
        parameter_count != training_summary.learned_parameter_count
        or fingerprint != training_summary.parameter_fingerprint
    ):
        raise ValueError("evaluation model does not match trained checkpoint identity")

    runs: list[PopulationRunMetrics] = []
    for difficulty in RELAY_DIFFICULTIES:
        worlds = generate_relay_dataset(
            start_seed=benchmark_start,
            world_count=world_count_per_difficulty,
            difficulty=difficulty,
        )
        for mode in (
            CommunicationMode.NO_COMMUNICATION,
            CommunicationMode.SPARSE_SHARED_V0,
        ):
            for population_size in DEVELOPMENT_POPULATION_SIZES:
                runs.append(
                    evaluate_relay_condition(
                        model,
                        worlds,
                        training_seed=training_summary.training_seed,
                        benchmark_seed=benchmark_seed,
                        difficulty=difficulty,
                        population_size=population_size,
                        communication_mode=mode,
                        batch_size=batch_size,
                        expected_parameter_count=parameter_count,
                        expected_fingerprint=fingerprint,
                        device=target_device,
                    )
                )

    if model.parameter_fingerprint() != fingerprint:
        raise RuntimeError("evaluation mutated the frozen relay checkpoint")

    assessments: list[tuple[str, CurveAssessment]] = []
    for difficulty in RELAY_DIFFICULTIES:
        difficulty_runs = [run for run in runs if run.difficulty == difficulty.name]
        assessments.append(
            (
                difficulty.name,
                assess_scaling_curve(
                    (
                        run
                        for run in difficulty_runs
                        if run.condition.communication_mode
                        is CommunicationMode.SPARSE_SHARED_V0
                    ),
                    (
                        run
                        for run in difficulty_runs
                        if run.condition.communication_mode
                        is CommunicationMode.NO_COMMUNICATION
                    ),
                    criteria=criteria,
                ),
            )
        )

    result = RelayDevelopmentResult(
        training=training_summary,
        model_config=model.config,
        benchmark_seed=benchmark_seed,
        world_count_per_difficulty=world_count_per_difficulty,
        runs=tuple(runs),
        assessments=tuple(assessments),
    )
    result.validate()
    return result


def evaluate_relay_condition(
    model: RelayPopulationModel,
    worlds: Sequence[RelayWorld],
    *,
    training_seed: int,
    benchmark_seed: int,
    difficulty: RelayDifficulty,
    population_size: int,
    communication_mode: CommunicationMode,
    batch_size: int,
    expected_parameter_count: int,
    expected_fingerprint: str,
    device: torch.device | str,
) -> PopulationRunMetrics:
    """Evaluate one population point while preserving the canonical scope decomposition."""

    if communication_mode not in {
        CommunicationMode.NO_COMMUNICATION,
        CommunicationMode.SPARSE_SHARED_V0,
    }:
        raise ValueError("relay development supports only frozen v0 communication modes")
    if population_size not in DEVELOPMENT_POPULATION_SIZES:
        raise ValueError("population_size is outside the frozen development curve")
    if not worlds:
        raise ValueError("at least one relay world is required")
    if any(world.difficulty != difficulty for world in worlds):
        raise ValueError("condition worlds do not match difficulty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if model.trainable_parameter_count() != expected_parameter_count:
        raise RuntimeError("learned parameter count changed before condition evaluation")
    if model.parameter_fingerprint() != expected_fingerprint:
        raise RuntimeError("parameter fingerprint changed before condition evaluation")

    target_device = torch.device(device)
    solved = 0
    information_complete = 0
    solved_information_complete = 0
    messages = 0
    communicated_scalars = 0
    _synchronize(target_device)
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(worlds), batch_size):
            batch_worlds = worlds[offset : offset + batch_size]
            batch = build_relay_tensor_batch(
                batch_worlds,
                active_workers=population_size,
                device=target_device,
            )
            output = model(
                batch,
                communication_mode=communication_mode,
                recurrent_rounds=difficulty.hop_count,
            )
            correct = decode_node_logits(output.logits).eq(batch.answer_keys)
            solved += int(correct.sum().item())
            information_complete += int(batch.information_complete.sum().item())
            solved_information_complete += int(
                (correct & batch.information_complete).sum().item()
            )
            messages += output.telemetry.messages_emitted
            communicated_scalars += output.telemetry.communicated_scalar_count
    _synchronize(target_device)
    elapsed = time.perf_counter() - started

    parameter = next(model.parameters())
    condition = PopulationCondition(
        nominal_population_size=population_size,
        active_state_count=population_size,
        recurrent_rounds=difficulty.hop_count,
        communication_mode=communication_mode,
    )
    metrics = PopulationRunMetrics(
        training_seed=training_seed,
        benchmark_seed=benchmark_seed,
        difficulty=difficulty.name,
        learned_parameter_count=expected_parameter_count,
        parameter_fingerprint=expected_fingerprint,
        condition=condition,
        task_count=len(worlds),
        solved_count=solved,
        information_complete_count=information_complete,
        solved_information_complete_count=solved_information_complete,
        messages_emitted=messages,
        communicated_scalar_count=communicated_scalars,
        peak_worker_state_bytes=(
            population_size * model.config.state_width * parameter.element_size()
        ),
        elapsed_seconds=elapsed,
    )
    metrics.validate()
    return metrics


def _training_worlds_for_plan(
    training_seed: int,
    plan: RelayTrainingBatchPlan,
    batch_size: int,
) -> tuple[RelayWorld, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    plan.validate()
    block_start = _seed_block_start(
        TRAINING_WORLD_SEED_BASE,
        training_seed,
        upper_bound=DEVELOPMENT_WORLD_SEED_BASE,
        label="training",
    )
    segment_size = batch_size * _MAX_SCOPE_CYCLE
    segment_start = block_start + plan.step * segment_size
    segment_limit = segment_start + segment_size
    block_limit = block_start + SEED_BLOCK_SIZE
    if segment_limit > block_limit:
        raise ValueError("training plan exceeds reserved seed block")

    thresholds = relay_scope_thresholds(plan.difficulty)
    selected_seeds = tuple(
        seed
        for seed in range(segment_start, segment_limit)
        if thresholds[seed % len(thresholds)] == plan.scope_threshold
    )[:batch_size]
    if len(selected_seeds) != batch_size:
        raise RuntimeError("training schedule could not fill its threshold-matched batch")
    worlds = tuple(generate_relay_world(seed, plan.difficulty) for seed in selected_seeds)
    if any(world.scope_threshold != plan.scope_threshold for world in worlds):
        raise RuntimeError("training world scope threshold does not match batch plan")
    return worlds


def _run_to_dict(run: PopulationRunMetrics) -> dict[str, object]:
    run.validate()
    condition = run.condition
    return {
        "training_seed": run.training_seed,
        "benchmark_seed": run.benchmark_seed,
        "difficulty": run.difficulty,
        "learned_parameter_count": run.learned_parameter_count,
        "parameter_fingerprint": run.parameter_fingerprint,
        "condition": {
            "nominal_population_size": condition.nominal_population_size,
            "active_state_count": condition.active_state_count,
            "recurrent_rounds": condition.recurrent_rounds,
            "communication_mode": condition.communication_mode.value,
            "worker_updates_per_task": condition.worker_updates,
        },
        "task_count": run.task_count,
        "solved_count": run.solved_count,
        "solve_rate": run.solve_rate,
        "information_complete_count": run.information_complete_count,
        "information_complete_rate": run.information_complete_rate,
        "solved_information_complete_count": run.solved_information_complete_count,
        "solve_rate_given_information_complete": (
            run.solve_rate_given_information_complete
        ),
        "solved_information_incomplete_count": run.solved_information_incomplete_count,
        "solve_rate_given_information_incomplete": (
            run.solve_rate_given_information_incomplete
        ),
        "total_worker_updates": condition.worker_updates * run.task_count,
        "messages_emitted": run.messages_emitted,
        "communicated_scalar_count": run.communicated_scalar_count,
        "peak_worker_state_bytes": run.peak_worker_state_bytes,
        "elapsed_seconds": run.elapsed_seconds,
    }


def _seed_block_start(base: int, index: int, *, upper_bound: int, label: str) -> int:
    if index < 0:
        raise ValueError(f"{label} seed must be non-negative")
    start = base + index * SEED_BLOCK_SIZE
    if start < base or start + SEED_BLOCK_SIZE > upper_bound:
        raise ValueError(f"{label} seed exceeds its reserved world-seed range")
    return start


def _development_world_seed(benchmark_seed: int, ordinal: int) -> int:
    start = _seed_block_start(
        DEVELOPMENT_WORLD_SEED_BASE,
        benchmark_seed,
        upper_bound=CONFIRMATION_WORLD_SEED_BASE,
        label="development",
    )
    if not 0 <= ordinal < SEED_BLOCK_SIZE:
        raise ValueError("development ordinal is outside reserved seed block")
    return start + ordinal


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
