"""Frozen Gate-3 v3 generation-pressure confirmation evaluation.

No training occurs. This module reuses the exact Gate-3 v3 scheduler/runtime and three frozen
Gate-3 v1 checkpoints on a distinct untouched confirmation namespace.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .gate3_v1_model import Gate3V1Scorer
from .gate3_v1_sparse_active_reserve import Gate3V1ControlMode, Gate3V1PublicWorld
from .gate3_v3_generation_pressure import (
    GATE3_V3_CHECKPOINT_INDICES,
    GATE3_V3_CONDITIONS,
    GATE3_V3_DEPTH,
    GATE3_V3_HINT_RELIABILITY,
    GATE3_V3_SCHEDULED_SLOTS,
    GATE3_V3_TOTAL_LEARNED_UPDATES,
    Gate3V3CheckpointIdentity,
    run_gate3_v3_world_batch,
)

GATE3_V3_CONFIRMATION_VERSION = "gate3-v3-generation-pressure-confirmation-v0"
GATE3_V3_CONFIRMATION_WORLD_COUNT = 512
GATE3_V3_CONFIRMATION_EVAL_BATCH_SIZE = 64
GATE3_V3_CONFIRMATION_BOOTSTRAP_SAMPLES = 4_000
GATE3_V3_CONFIRMATION_CONDITIONS = GATE3_V3_CONDITIONS
GATE3_V3_CONFIRMATION_CHECKPOINT_INDICES = GATE3_V3_CHECKPOINT_INDICES


@dataclass(frozen=True, slots=True)
class Gate3V3ConfirmationWorld:
    world_index: int
    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.world_index < GATE3_V3_CONFIRMATION_WORLD_COUNT:
            raise ValueError("Gate-3 v3 confirmation world index must be in 0..511")
        self.public.validate()
        if self.public.depth != GATE3_V3_DEPTH:
            raise ValueError("Gate-3 v3 confirmation must remain at depth 8")
        if len(self.hidden_path) != GATE3_V3_DEPTH or any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("Gate-3 v3 confirmation hidden path must contain eight binary decisions")


@dataclass(frozen=True, slots=True)
class Gate3V3ConfirmationCondition:
    checkpoint_index: int
    reserve_capacity: int
    mode: Gate3V1ControlMode
    world_count: int
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    coverage_rate: float
    productive_slots_by_world: tuple[int, ...]
    sink_slots_by_world: tuple[int, ...]
    preprune_widths_by_world: tuple[tuple[int, ...], ...]
    retained_widths_by_world: tuple[tuple[int, ...], ...]
    unique_retained_widths_by_world: tuple[tuple[int, ...], ...]
    binding_by_generation_by_world: tuple[tuple[bool, ...], ...]
    depth7_preprune_width_by_world: tuple[int, ...]
    depth7_retained_width_by_world: tuple[int, ...]
    depth7_expanded_parents_by_world: tuple[int, ...]
    generated_terminal_count_by_world: tuple[int, ...]
    unique_generated_terminal_count_by_world: tuple[int, ...]
    total_learned_updates_per_world: int
    learned_parameter_count: int
    parameter_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate3V3ConfirmationPair:
    comparison: str
    checkpoint_index: int
    treatment_capacity: int
    treatment_mode: Gate3V1ControlMode
    reference_capacity: int
    reference_mode: Gate3V1ControlMode
    world_count: int
    treatment_only: int
    reference_only: int
    both_covered: int
    neither_covered: int
    coverage_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["treatment_mode"] = self.treatment_mode.value
        payload["reference_mode"] = self.reference_mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate3V3ConfirmationResult:
    experiment_version: str
    scientific_status: str
    confirmation_opened: bool
    training_performed: bool
    checkpoints: tuple[Gate3V3CheckpointIdentity, ...]
    world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    depth: int
    hint_reliability: float
    scheduled_slots: int
    total_learned_updates_per_world: int
    conditions: tuple[Gate3V3ConfirmationCondition, ...]
    paired_summaries: tuple[Gate3V3ConfirmationPair, ...]

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
            "scheduled_slots": self.scheduled_slots,
            "active_child_lanes": 2,
            "recurrent_updates_per_child": 8,
            "total_learned_updates_per_world": self.total_learned_updates_per_world,
            "conditions": [row.to_dict() for row in self.conditions],
            "paired_summaries": [row.to_dict() for row in self.paired_summaries],
            "scientific_decision": "NOT_ASSIGNED_UNTIL_INDEPENDENT_CONFIRMATION_AUDIT",
        }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def gate3_v3_confirmation_runtime_seed(*, world_index: int) -> int:
    return _seed_from_parts(
        "gate3-v3-generation-pressure-confirmation-runtime",
        world_index,
        GATE3_V3_DEPTH,
    )


def generate_gate3_v3_confirmation_world(*, world_index: int) -> Gate3V3ConfirmationWorld:
    if not 0 <= world_index < GATE3_V3_CONFIRMATION_WORLD_COUNT:
        raise ValueError("Gate-3 v3 confirmation world index must be in 0..511")
    hidden_rng = random.Random(
        _seed_from_parts(
            "gate3-v3-generation-pressure-confirmation-hidden",
            world_index,
            GATE3_V3_DEPTH,
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE3_V3_DEPTH))
    hint_rng = random.Random(
        _seed_from_parts(
            "gate3-v3-generation-pressure-confirmation-hints",
            world_index,
            GATE3_V3_DEPTH,
        )
    )
    noisy_hints = tuple(
        hidden_bit if hint_rng.random() < GATE3_V3_HINT_RELIABILITY else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    world = Gate3V3ConfirmationWorld(
        world_index=world_index,
        public=Gate3V1PublicWorld(
            seed=gate3_v3_confirmation_runtime_seed(world_index=world_index),
            depth=GATE3_V3_DEPTH,
            noisy_hints=noisy_hints,
        ),
        hidden_path=hidden_path,
    )
    world.validate()
    return world


def evaluate_gate3_v3_confirmation_condition(
    model: Gate3V1Scorer,
    *,
    checkpoint_index: int,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    device: torch.device | str,
    world_count: int = GATE3_V3_CONFIRMATION_WORLD_COUNT,
    evaluation_batch_size: int = GATE3_V3_CONFIRMATION_EVAL_BATCH_SIZE,
) -> Gate3V3ConfirmationCondition:
    if checkpoint_index not in GATE3_V3_CONFIRMATION_CHECKPOINT_INDICES:
        raise ValueError("checkpoint index must be 0, 1 or 2")
    if world_count != GATE3_V3_CONFIRMATION_WORLD_COUNT:
        raise ValueError("Gate-3 v3 confirmation must use exactly 512 worlds")
    if evaluation_batch_size != GATE3_V3_CONFIRMATION_EVAL_BATCH_SIZE:
        raise ValueError("Gate-3 v3 confirmation batch size is frozen at 64")
    if (reserve_capacity, mode) not in GATE3_V3_CONFIRMATION_CONDITIONS:
        raise ValueError("condition is outside the frozen Gate-3 v3 confirmation matrix")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    productive: list[int] = []
    sink: list[int] = []
    preprune: list[tuple[int, ...]] = []
    retained: list[tuple[int, ...]] = []
    unique: list[tuple[int, ...]] = []
    binding: list[tuple[bool, ...]] = []
    depth7_pre: list[int] = []
    depth7_retained: list[int] = []
    depth7_expanded: list[int] = []
    terminal_count: list[int] = []
    unique_terminal_count: list[int] = []

    for start in range(0, world_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, world_count)
        worlds = tuple(
            generate_gate3_v3_confirmation_world(world_index=index)
            for index in range(start, stop)
        )
        batch = run_gate3_v3_world_batch(
            model,
            worlds,
            reserve_capacity=reserve_capacity,
            mode=mode,
            device=device,
        )
        for world, runtime_result in zip(worlds, batch, strict=True):
            telemetry = runtime_result.telemetry
            covered.append(world.hidden_path in set(runtime_result.generated_terminal_paths))
            runtime_seeds.append(world.public.seed)
            productive.append(telemetry.productive_slots)
            sink.append(telemetry.sink_slots)
            preprune.append(telemetry.preprune_widths)
            retained.append(telemetry.retained_widths)
            unique.append(telemetry.unique_retained_widths)
            binding.append(telemetry.binding_by_generation)
            depth7_pre.append(telemetry.depth7_preprune_width)
            depth7_retained.append(telemetry.depth7_retained_width)
            depth7_expanded.append(telemetry.depth7_expanded_parents)
            terminal_count.append(telemetry.generated_terminal_count)
            unique_terminal_count.append(telemetry.unique_generated_terminal_count)

    vector = tuple(covered)
    return Gate3V3ConfirmationCondition(
        checkpoint_index=checkpoint_index,
        reserve_capacity=reserve_capacity,
        mode=mode,
        world_count=world_count,
        world_indices=tuple(range(world_count)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / world_count,
        productive_slots_by_world=tuple(productive),
        sink_slots_by_world=tuple(sink),
        preprune_widths_by_world=tuple(preprune),
        retained_widths_by_world=tuple(retained),
        unique_retained_widths_by_world=tuple(unique),
        binding_by_generation_by_world=tuple(binding),
        depth7_preprune_width_by_world=tuple(depth7_pre),
        depth7_retained_width_by_world=tuple(depth7_retained),
        depth7_expanded_parents_by_world=tuple(depth7_expanded),
        generated_terminal_count_by_world=tuple(terminal_count),
        unique_generated_terminal_count_by_world=tuple(unique_terminal_count),
        total_learned_updates_per_world=GATE3_V3_TOTAL_LEARNED_UPDATES,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint_index: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts(
            "gate3-v3-generation-pressure-confirmation-bootstrap",
            checkpoint_index,
            comparison,
        )
    )
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE3_V3_CONFIRMATION_BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (GATE3_V3_CONFIRMATION_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE3_V3_CONFIRMATION_BOOTSTRAP_SAMPLES - 1)))],
    )


def _paired_summary(
    *,
    comparison: str,
    checkpoint_index: int,
    treatment: Gate3V3ConfirmationCondition,
    reference: Gate3V3ConfirmationCondition,
) -> Gate3V3ConfirmationPair:
    if treatment.world_indices != reference.world_indices:
        raise ValueError("Gate-3 v3 confirmation paired conditions must share world indices")
    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = GATE3_V3_CONFIRMATION_WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=checkpoint_index,
        comparison=comparison,
    )
    return Gate3V3ConfirmationPair(
        comparison=comparison,
        checkpoint_index=checkpoint_index,
        treatment_capacity=treatment.reserve_capacity,
        treatment_mode=treatment.mode,
        reference_capacity=reference.reserve_capacity,
        reference_mode=reference.mode,
        world_count=GATE3_V3_CONFIRMATION_WORLD_COUNT,
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=sum(differences) / GATE3_V3_CONFIRMATION_WORLD_COUNT,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate3_v3_confirmation_pairs(
    conditions: Iterable[Gate3V3ConfirmationCondition],
) -> tuple[Gate3V3ConfirmationPair, ...]:
    rows = tuple(conditions)
    index = {
        (row.checkpoint_index, row.reserve_capacity, row.mode): row
        for row in rows
    }
    specs = (
        ("stable_l256_vs_l64", 256, Gate3V1ControlMode.STABLE_RESERVE, 64, Gate3V1ControlMode.STABLE_RESERVE),
        ("stable_l64_vs_l16", 64, Gate3V1ControlMode.STABLE_RESERVE, 16, Gate3V1ControlMode.STABLE_RESERVE),
        ("stable_l256_vs_collapsed", 256, Gate3V1ControlMode.STABLE_RESERVE, 256, Gate3V1ControlMode.COLLAPSED_DIVERSITY),
        ("stable_l256_vs_reshuffled", 256, Gate3V1ControlMode.STABLE_RESERVE, 256, Gate3V1ControlMode.RESHUFFLED_CONTINUITY),
    )
    summaries: list[Gate3V3ConfirmationPair] = []
    for checkpoint_index in GATE3_V3_CONFIRMATION_CHECKPOINT_INDICES:
        for comparison, t_cap, t_mode, r_cap, r_mode in specs:
            summaries.append(
                _paired_summary(
                    comparison=comparison,
                    checkpoint_index=checkpoint_index,
                    treatment=index[(checkpoint_index, t_cap, t_mode)],
                    reference=index[(checkpoint_index, r_cap, r_mode)],
                )
            )
    return tuple(summaries)
