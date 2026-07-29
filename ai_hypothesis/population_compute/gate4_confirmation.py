"""Frozen Gate-4 adaptive-activation confirmation mechanics.

Confirmation reuses the exact Gate-4 scheduler/runtime and frozen checkpoints, but evaluates an
untouched deterministic 512-world namespace with 4,000 paired bootstrap samples.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from .gate4_adaptive_activation import (
    GATE4_CHECKPOINT_INDICES,
    GATE4_CONDITIONS,
    GATE4_DEPTH,
    GATE4_EVAL_BATCH_SIZE,
    GATE4_HINT_RELIABILITY,
    GATE4_RESERVE_CAPACITY,
    GATE4_SCHEDULED_SLOTS,
    GATE4_TOTAL_LEARNED_UPDATES,
    Gate4CheckpointIdentity,
    Gate4ConditionEvaluation,
    Gate4PairedSummary,
    Gate4SchedulerMode,
    run_gate4_world_batch,
)

GATE4_CONFIRMATION_VERSION = "gate4-adaptive-activation-confirmation-v0"
GATE4_CONFIRMATION_WORLD_COUNT = 512
GATE4_CONFIRMATION_EVAL_BATCH_SIZE = GATE4_EVAL_BATCH_SIZE
GATE4_CONFIRMATION_BOOTSTRAP_SAMPLES = 4_000
GATE4_CONFIRMATION_CHECKPOINT_INDICES = GATE4_CHECKPOINT_INDICES
GATE4_CONFIRMATION_CONDITIONS = GATE4_CONDITIONS


@dataclass(frozen=True, slots=True)
class Gate4ConfirmationWorld:
    world_index: int
    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.world_index < GATE4_CONFIRMATION_WORLD_COUNT:
            raise ValueError("Gate-4 confirmation world index is outside 0..511")
        self.public.validate()
        if self.public.depth != GATE4_DEPTH:
            raise ValueError("Gate-4 confirmation must remain at depth 8")
        if len(self.hidden_path) != GATE4_DEPTH or any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("Gate-4 confirmation hidden path must contain eight binary decisions")


@dataclass(frozen=True, slots=True)
class Gate4ConfirmationResult:
    experiment_version: str
    scientific_status: str
    confirmation_opened: bool
    training_performed: bool
    checkpoints: tuple[Gate4CheckpointIdentity, ...]
    world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    depth: int
    hint_reliability: float
    reserve_capacity: int
    scheduled_slots: int
    total_learned_updates_per_world: int
    conditions: tuple[Gate4ConditionEvaluation, ...]
    paired_summaries: tuple[Gate4PairedSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_version": self.experiment_version,
            "scientific_status": self.scientific_status,
            "confirmation_opened": self.confirmation_opened,
            "training_performed": self.training_performed,
            "checkpoints": [asdict(row) for row in self.checkpoints],
            "world_count": self.world_count,
            "evaluation_batch_size": self.evaluation_batch_size,
            "bootstrap_samples": self.bootstrap_samples,
            "depth": self.depth,
            "hint_reliability": self.hint_reliability,
            "reserve_capacity": self.reserve_capacity,
            "scheduled_slots": self.scheduled_slots,
            "active_child_lanes": 2,
            "recurrent_updates_per_child": 8,
            "total_learned_updates_per_world": self.total_learned_updates_per_world,
            "conditions": [row.to_dict() for row in self.conditions],
            "paired_summaries": [row.to_dict() for row in self.paired_summaries],
            "scientific_decision": "PENDING_INDEPENDENT_CONFIRMATION_AUDIT",
        }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def gate4_confirmation_runtime_seed(*, world_index: int) -> int:
    return _seed_from_parts(
        "gate4-adaptive-activation-confirmation-runtime", world_index, GATE4_DEPTH
    )


def generate_gate4_confirmation_world(*, world_index: int) -> Gate4ConfirmationWorld:
    if not 0 <= world_index < GATE4_CONFIRMATION_WORLD_COUNT:
        raise ValueError("Gate-4 confirmation world index is outside 0..511")
    hidden_rng = random.Random(
        _seed_from_parts(
            "gate4-adaptive-activation-confirmation-hidden", world_index, GATE4_DEPTH
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE4_DEPTH))
    hint_rng = random.Random(
        _seed_from_parts(
            "gate4-adaptive-activation-confirmation-hints", world_index, GATE4_DEPTH
        )
    )
    noisy_hints = tuple(
        hidden_bit if hint_rng.random() < GATE4_HINT_RELIABILITY else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    world = Gate4ConfirmationWorld(
        world_index=world_index,
        public=Gate3V1PublicWorld(
            seed=gate4_confirmation_runtime_seed(world_index=world_index),
            depth=GATE4_DEPTH,
            noisy_hints=noisy_hints,
        ),
        hidden_path=hidden_path,
    )
    world.validate()
    return world


def evaluate_gate4_confirmation_condition(
    model: torch.nn.Module,
    *,
    checkpoint_index: int,
    mode: Gate4SchedulerMode,
    device: torch.device | str,
    world_count: int = GATE4_CONFIRMATION_WORLD_COUNT,
    evaluation_batch_size: int = GATE4_CONFIRMATION_EVAL_BATCH_SIZE,
) -> Gate4ConditionEvaluation:
    if checkpoint_index not in GATE4_CONFIRMATION_CHECKPOINT_INDICES:
        raise ValueError("Gate-4 confirmation checkpoint index must be 0, 1 or 2")
    if mode not in GATE4_CONFIRMATION_CONDITIONS:
        raise ValueError("Gate-4 confirmation condition is outside the frozen matrix")
    if world_count != GATE4_CONFIRMATION_WORLD_COUNT:
        raise ValueError("Gate-4 confirmation must use exactly 512 worlds")
    if evaluation_batch_size != GATE4_CONFIRMATION_EVAL_BATCH_SIZE:
        raise ValueError("Gate-4 confirmation batch size is frozen at 64")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    productive: list[int] = []
    sink: list[int] = []
    live_population: list[tuple[int, ...]] = []
    max_population: list[int] = []
    mean_population: list[float] = []
    distinct_depths: list[int] = []
    productive_by_depth: list[tuple[int, ...]] = []
    activated_depths: list[tuple[int, ...]] = []
    terminal_slots: list[tuple[int, ...]] = []
    terminal_count: list[int] = []
    unique_terminal_count: list[int] = []

    for start in range(0, world_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, world_count)
        worlds = tuple(
            generate_gate4_confirmation_world(world_index=index)
            for index in range(start, stop)
        )
        batch = run_gate4_world_batch(model, worlds, mode=mode, device=device)
        for world, result in zip(worlds, batch, strict=True):
            telemetry = result.telemetry
            covered.append(world.hidden_path in set(result.generated_terminal_paths))
            runtime_seeds.append(world.public.seed)
            productive.append(telemetry.productive_slots)
            sink.append(telemetry.sink_slots)
            live_population.append(telemetry.live_nonterminal_population_by_slot)
            max_population.append(telemetry.max_live_nonterminal_population)
            mean_population.append(telemetry.mean_live_nonterminal_population)
            distinct_depths.append(telemetry.distinct_parent_depths_activated)
            productive_by_depth.append(telemetry.productive_activations_by_parent_depth)
            activated_depths.append(telemetry.activated_parent_depth_by_slot)
            terminal_slots.append(telemetry.terminal_generation_slot_indices)
            terminal_count.append(telemetry.generated_terminal_count)
            unique_terminal_count.append(telemetry.unique_generated_terminal_count)

    vector = tuple(covered)
    return Gate4ConditionEvaluation(
        checkpoint_index=checkpoint_index,
        mode=mode,
        world_count=world_count,
        world_indices=tuple(range(world_count)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / world_count,
        productive_slots_by_world=tuple(productive),
        sink_slots_by_world=tuple(sink),
        live_nonterminal_population_by_slot_by_world=tuple(live_population),
        max_live_nonterminal_population_by_world=tuple(max_population),
        mean_live_nonterminal_population_by_world=tuple(mean_population),
        distinct_parent_depths_activated_by_world=tuple(distinct_depths),
        productive_activations_by_parent_depth_by_world=tuple(productive_by_depth),
        activated_parent_depth_by_slot_by_world=tuple(activated_depths),
        terminal_generation_slot_indices_by_world=tuple(terminal_slots),
        generated_terminal_count_by_world=tuple(terminal_count),
        unique_generated_terminal_count_by_world=tuple(unique_terminal_count),
        total_learned_updates_per_world=GATE4_TOTAL_LEARNED_UPDATES,
        reserve_capacity=GATE4_RESERVE_CAPACITY,
        learned_parameter_count=int(model.trainable_parameter_count()),
        parameter_fingerprint=str(model.parameter_fingerprint()),
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint_index: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts(
            "gate4-adaptive-activation-confirmation-bootstrap",
            checkpoint_index,
            comparison,
        )
    )
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE4_CONFIRMATION_BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[
            int(math.floor(0.025 * (GATE4_CONFIRMATION_BOOTSTRAP_SAMPLES - 1)))
        ],
        estimates[
            int(math.ceil(0.975 * (GATE4_CONFIRMATION_BOOTSTRAP_SAMPLES - 1)))
        ],
    )


def _paired_summary(
    *,
    comparison: str,
    checkpoint_index: int,
    treatment: Gate4ConditionEvaluation,
    reference: Gate4ConditionEvaluation,
) -> Gate4PairedSummary:
    if treatment.world_indices != reference.world_indices:
        raise ValueError("Gate-4 confirmation paired conditions use different world indices")
    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = GATE4_CONFIRMATION_WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=checkpoint_index,
        comparison=comparison,
    )
    return Gate4PairedSummary(
        comparison=comparison,
        checkpoint_index=checkpoint_index,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        world_count=GATE4_CONFIRMATION_WORLD_COUNT,
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=sum(differences) / GATE4_CONFIRMATION_WORLD_COUNT,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate4_confirmation_paired_summaries(
    conditions: Iterable[Gate4ConditionEvaluation],
) -> tuple[Gate4PairedSummary, ...]:
    rows = tuple(conditions)
    index = {(row.checkpoint_index, row.mode): row for row in rows}
    specs = (
        (
            "adaptive_score_vs_static_generation",
            Gate4SchedulerMode.ADAPTIVE_SCORE,
            Gate4SchedulerMode.STATIC_GENERATION,
        ),
        (
            "adaptive_score_vs_adaptive_hash",
            Gate4SchedulerMode.ADAPTIVE_SCORE,
            Gate4SchedulerMode.ADAPTIVE_HASH,
        ),
        (
            "static_generation_vs_adaptive_hash",
            Gate4SchedulerMode.STATIC_GENERATION,
            Gate4SchedulerMode.ADAPTIVE_HASH,
        ),
    )
    summaries: list[Gate4PairedSummary] = []
    for checkpoint_index in GATE4_CONFIRMATION_CHECKPOINT_INDICES:
        for comparison, treatment_mode, reference_mode in specs:
            summaries.append(
                _paired_summary(
                    comparison=comparison,
                    checkpoint_index=checkpoint_index,
                    treatment=index[(checkpoint_index, treatment_mode)],
                    reference=index[(checkpoint_index, reference_mode)],
                )
            )
    return tuple(summaries)
