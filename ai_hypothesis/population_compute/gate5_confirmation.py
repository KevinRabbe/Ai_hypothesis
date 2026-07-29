"""Frozen Gate-5 v0 bounded score-visibility confirmation evaluator.

No training occurs here.  Confirmation reuses the exact Gate-5 development scheduler kernel and
three frozen scorer checkpoints, but evaluates a disjoint 512-world namespace and uses 4,000
paired bootstrap samples.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from .gate5_bounded_score_activation import (
    GATE5_CHECKPOINT_INDICES,
    GATE5_CONDITIONS,
    GATE5_DEPTH,
    GATE5_HINT_RELIABILITY,
    GATE5_NONINFERIORITY_MARGIN,
    GATE5_RESERVE_CAPACITY,
    GATE5_SCHEDULED_SLOTS,
    GATE5_STAGE_A_SLOTS,
    GATE5_STAGE_B_SLOTS,
    GATE5_TOTAL_LEARNED_UPDATES,
    Gate5CheckpointIdentity,
    Gate5ConditionEvaluation,
    Gate5PairedSummary,
    Gate5SchedulerMode,
)
from .gate5_bounded_score_batch import run_gate5_strict_world_batch

GATE5_CONFIRMATION_VERSION = "gate5-bounded-score-activation-confirmation-v0"
GATE5_CONFIRMATION_WORLD_COUNT = 512
GATE5_CONFIRMATION_EVAL_BATCH_SIZE = 64
GATE5_CONFIRMATION_BOOTSTRAP_SAMPLES = 4_000
GATE5_CONFIRMATION_CHECKPOINT_INDICES = GATE5_CHECKPOINT_INDICES
GATE5_CONFIRMATION_CONDITIONS = GATE5_CONDITIONS


@dataclass(frozen=True, slots=True)
class Gate5ConfirmationWorld:
    world_index: int
    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.world_index < GATE5_CONFIRMATION_WORLD_COUNT:
            raise ValueError("Gate-5 confirmation world index is outside 0..511")
        self.public.validate()
        if self.public.depth != GATE5_DEPTH:
            raise ValueError("Gate-5 confirmation must remain at depth 8")
        if len(self.hidden_path) != GATE5_DEPTH or any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("Gate-5 confirmation hidden path must contain eight binary decisions")


@dataclass(frozen=True, slots=True)
class Gate5ConfirmationResult:
    experiment_version: str
    scientific_status: str
    confirmation_opened: bool
    training_performed: bool
    checkpoints: tuple[Gate5CheckpointIdentity, ...]
    world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    depth: int
    hint_reliability: float
    reserve_capacity: int
    stage_a_slots: int
    stage_b_slots: int
    scheduled_slots: int
    total_learned_updates_per_world: int
    noninferiority_margin: float
    conditions: tuple[Gate5ConditionEvaluation, ...]
    paired_summaries: tuple[Gate5PairedSummary, ...]

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
            "stage_a_slots": self.stage_a_slots,
            "stage_b_slots": self.stage_b_slots,
            "scheduled_slots": self.scheduled_slots,
            "active_child_lanes": 2,
            "recurrent_updates_per_child": 8,
            "total_learned_updates_per_world": self.total_learned_updates_per_world,
            "noninferiority_margin": self.noninferiority_margin,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "paired_summaries": [summary.to_dict() for summary in self.paired_summaries],
            "scientific_decision": "PENDING_INDEPENDENT_CONFIRMATION_AUDIT",
        }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def gate5_confirmation_runtime_seed(*, world_index: int) -> int:
    return _seed_from_parts(
        "gate5-bounded-score-activation-confirmation-runtime",
        world_index,
        GATE5_DEPTH,
    )


def generate_gate5_confirmation_world(*, world_index: int) -> Gate5ConfirmationWorld:
    if not 0 <= world_index < GATE5_CONFIRMATION_WORLD_COUNT:
        raise ValueError("Gate-5 confirmation world index is outside 0..511")
    hidden_rng = random.Random(
        _seed_from_parts(
            "gate5-bounded-score-activation-confirmation-hidden",
            world_index,
            GATE5_DEPTH,
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE5_DEPTH))
    hint_rng = random.Random(
        _seed_from_parts(
            "gate5-bounded-score-activation-confirmation-hints",
            world_index,
            GATE5_DEPTH,
        )
    )
    noisy_hints = tuple(
        hidden_bit if hint_rng.random() < GATE5_HINT_RELIABILITY else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    world = Gate5ConfirmationWorld(
        world_index=world_index,
        public=Gate3V1PublicWorld(
            seed=gate5_confirmation_runtime_seed(world_index=world_index),
            depth=GATE5_DEPTH,
            noisy_hints=noisy_hints,
        ),
        hidden_path=hidden_path,
    )
    world.validate()
    return world


def evaluate_gate5_confirmation_condition(
    model: object,
    *,
    checkpoint_index: int,
    mode: Gate5SchedulerMode,
    device: torch.device | str,
) -> Gate5ConditionEvaluation:
    if checkpoint_index not in GATE5_CONFIRMATION_CHECKPOINT_INDICES:
        raise ValueError("Gate-5 confirmation checkpoint index must be 0, 1 or 2")
    if mode not in GATE5_CONFIRMATION_CONDITIONS:
        raise ValueError("Gate-5 confirmation condition is outside the frozen matrix")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    productive: list[int] = []
    sink: list[int] = []
    frontier: list[int] = []
    live_rows: list[tuple[int, ...]] = []
    depth_rows: list[tuple[int, ...]] = []
    visible_rows: list[tuple[int, ...]] = []
    score_obs_rows: list[tuple[int, ...]] = []
    total_score_obs: list[int] = []
    max_score_obs: list[int] = []
    visible_rank_rows: list[tuple[int, ...]] = []
    global_rank_rows: list[tuple[int, ...]] = []
    selected_path_rows: list[tuple[tuple[int, ...], ...]] = []
    terminal_count: list[int] = []
    unique_terminal_count: list[int] = []

    for start in range(0, GATE5_CONFIRMATION_WORLD_COUNT, GATE5_CONFIRMATION_EVAL_BATCH_SIZE):
        stop = min(start + GATE5_CONFIRMATION_EVAL_BATCH_SIZE, GATE5_CONFIRMATION_WORLD_COUNT)
        worlds = tuple(generate_gate5_confirmation_world(world_index=index) for index in range(start, stop))
        # The admitted strict kernel is unchanged from Gate-5 development.  It is intentionally
        # world-generator agnostic and only consumes validated public/hidden wrapper objects.
        batch = run_gate5_strict_world_batch(model, worlds, mode=mode, device=device)  # type: ignore[arg-type]
        for world, result in zip(worlds, batch, strict=True):
            telemetry = result.telemetry
            covered.append(world.hidden_path in set(result.generated_terminal_paths))
            runtime_seeds.append(world.public.seed)
            productive.append(telemetry.productive_slots)
            sink.append(telemetry.sink_slots)
            frontier.append(telemetry.stage_a_frontier_width)
            live_rows.append(telemetry.stage_b_live_population_by_slot)
            depth_rows.append(telemetry.stage_b_activated_parent_depth_by_slot)
            visible_rows.append(telemetry.stage_b_visible_candidate_count_by_slot)
            score_obs_rows.append(telemetry.stage_b_score_observation_count_by_slot)
            total_score_obs.append(telemetry.total_stage_b_score_observations)
            max_score_obs.append(telemetry.max_stage_b_score_observations)
            visible_rank_rows.append(telemetry.selected_visible_score_rank_by_slot)
            global_rank_rows.append(telemetry.selected_global_score_rank_by_slot)
            selected_path_rows.append(telemetry.selected_parent_paths_by_slot)
            terminal_count.append(telemetry.generated_terminal_count)
            unique_terminal_count.append(telemetry.unique_generated_terminal_count)

    vector = tuple(covered)
    return Gate5ConditionEvaluation(
        checkpoint_index=checkpoint_index,
        mode=mode,
        world_count=GATE5_CONFIRMATION_WORLD_COUNT,
        world_indices=tuple(range(GATE5_CONFIRMATION_WORLD_COUNT)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / GATE5_CONFIRMATION_WORLD_COUNT,
        productive_slots_by_world=tuple(productive),
        sink_slots_by_world=tuple(sink),
        stage_a_frontier_width_by_world=tuple(frontier),
        stage_b_live_population_by_slot_by_world=tuple(live_rows),
        stage_b_activated_parent_depth_by_slot_by_world=tuple(depth_rows),
        stage_b_visible_candidate_count_by_slot_by_world=tuple(visible_rows),
        stage_b_score_observation_count_by_slot_by_world=tuple(score_obs_rows),
        total_stage_b_score_observations_by_world=tuple(total_score_obs),
        max_stage_b_score_observations_by_world=tuple(max_score_obs),
        selected_visible_score_rank_by_slot_by_world=tuple(visible_rank_rows),
        selected_global_score_rank_by_slot_by_world=tuple(global_rank_rows),
        selected_parent_paths_by_slot_by_world=tuple(selected_path_rows),
        generated_terminal_count_by_world=tuple(terminal_count),
        unique_generated_terminal_count_by_world=tuple(unique_terminal_count),
        total_learned_updates_per_world=GATE5_TOTAL_LEARNED_UPDATES,
        reserve_capacity=GATE5_RESERVE_CAPACITY,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint_index: int, comparison: str
) -> tuple[float, float]:
    if len(differences) != GATE5_CONFIRMATION_WORLD_COUNT:
        raise ValueError("Gate-5 confirmation bootstrap requires 512 paired worlds")
    rng = random.Random(
        _seed_from_parts(
            "gate5-bounded-score-activation-confirmation-bootstrap",
            checkpoint_index,
            comparison,
        )
    )
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE5_CONFIRMATION_BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (GATE5_CONFIRMATION_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE5_CONFIRMATION_BOOTSTRAP_SAMPLES - 1)))],
    )


def _paired_summary(
    *,
    comparison: str,
    checkpoint_index: int,
    treatment: Gate5ConditionEvaluation,
    reference: Gate5ConditionEvaluation,
) -> Gate5PairedSummary:
    if treatment.world_indices != reference.world_indices:
        raise ValueError("Gate-5 confirmation paired conditions use different world indices")
    if treatment.world_count != GATE5_CONFIRMATION_WORLD_COUNT or reference.world_count != GATE5_CONFIRMATION_WORLD_COUNT:
        raise ValueError("Gate-5 confirmation pair does not contain 512 worlds")
    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = GATE5_CONFIRMATION_WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=checkpoint_index,
        comparison=comparison,
    )
    return Gate5PairedSummary(
        comparison=comparison,
        checkpoint_index=checkpoint_index,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        world_count=GATE5_CONFIRMATION_WORLD_COUNT,
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=sum(differences) / GATE5_CONFIRMATION_WORLD_COUNT,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate5_confirmation_paired_summaries(
    conditions: Iterable[Gate5ConditionEvaluation],
) -> tuple[Gate5PairedSummary, ...]:
    rows = tuple(conditions)
    index = {(row.checkpoint_index, row.mode): row for row in rows}
    specs = (
        ("bounded_score_k4_vs_global_score", Gate5SchedulerMode.BOUNDED_SCORE_K4, Gate5SchedulerMode.GLOBAL_SCORE),
        ("bounded_score_k8_vs_global_score", Gate5SchedulerMode.BOUNDED_SCORE_K8, Gate5SchedulerMode.GLOBAL_SCORE),
        ("bounded_score_k16_vs_global_score", Gate5SchedulerMode.BOUNDED_SCORE_K16, Gate5SchedulerMode.GLOBAL_SCORE),
        ("bounded_score_k32_vs_global_score", Gate5SchedulerMode.BOUNDED_SCORE_K32, Gate5SchedulerMode.GLOBAL_SCORE),
        (
            "bounded_score_k16_vs_bounded_hash_k16",
            Gate5SchedulerMode.BOUNDED_SCORE_K16,
            Gate5SchedulerMode.BOUNDED_HASH_K16,
        ),
    )
    summaries: list[Gate5PairedSummary] = []
    for checkpoint_index in GATE5_CONFIRMATION_CHECKPOINT_INDICES:
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
