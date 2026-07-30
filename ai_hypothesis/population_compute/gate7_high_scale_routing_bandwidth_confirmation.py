"""Bound execution core for Gate-7 routing-bandwidth frontier confirmation.

The module binds the exact screening result and transition checkpoint family, defines the one untouched
confirmation namespace, evaluates the fixed N4096/N8192 matrix in physical B64 batches, aggregates exactly
512 paired worlds per condition, and computes the preregistered confirmation intervals. It performs no
training, checkpoint selection, protocol mutation, adaptive K exposure, or second confirmation.
"""

from __future__ import annotations

import gc
import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .gate7_high_scale_frontier_prep import build_gate7_high_scale_immutable_frontier
from .gate7_high_scale_routing_bandwidth import load_verified_gate7_high_scale_checkpoint
from .gate7_high_scale_routing_bandwidth_confirmation_protocol import (
    GATE7_CONFIRMATION_ACTIVE_CHILD_LANES,
    GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES,
    GATE7_CONFIRMATION_CHECKPOINT_INDICES,
    GATE7_CONFIRMATION_CHECKPOINTS,
    GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE,
    GATE7_CONFIRMATION_GLOBAL_HASH,
    GATE7_CONFIRMATION_GLOBAL_SCORE,
    GATE7_CONFIRMATION_HINT_RELIABILITY,
    GATE7_CONFIRMATION_K_LADDER,
    GATE7_CONFIRMATION_LEARNED_PARAMETER_COUNT,
    GATE7_CONFIRMATION_POPULATIONS,
    GATE7_CONFIRMATION_RECURRENT_UPDATES_PER_CHILD,
    GATE7_CONFIRMATION_SCREENING_AUDIT_SHA256,
    GATE7_CONFIRMATION_SCREENING_RESULT_HEAD,
    GATE7_CONFIRMATION_SCREENING_RESULT_SHA256,
    GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS,
    GATE7_CONFIRMATION_VERSION,
    GATE7_CONFIRMATION_WORLD_COUNT,
    bounded_hash_condition,
    bounded_score_condition,
    build_confirmation_tier_plan,
)
from .gate7_high_scale_terminal_stage_b_prep import (
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
    run_gate7_high_scale_terminal_stage_b_preparation,
)
from .gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer
from .gate7_scale_neutral_transition_bridge import Gate7TransitionCheckpointIdentity

GATE7_CONFIRMATION_EXECUTION_ADMITTED = True
GATE7_CONFIRMATION_SCIENTIFIC_STATUS = (
    "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_CONFIRMATION_EVIDENCE"
)
GATE7_CONFIRMATION_PROTOCOL_HEAD = "b0f0cfca736186b9400f82a7539a54f888dc59e5"


@dataclass(frozen=True, slots=True)
class Gate7ConfirmationWorld:
    population: int
    world_index: int
    task_depth: int
    runtime_seed: int
    noisy_hints: tuple[int, ...]
    hidden_path: tuple[int, ...]
    hidden_terminal_path_id: int

    def validate(self) -> None:
        if self.population not in GATE7_CONFIRMATION_POPULATIONS:
            raise ValueError("confirmation population is outside the frozen matrix")
        if not 0 <= self.world_index < GATE7_CONFIRMATION_WORLD_COUNT:
            raise ValueError("confirmation world index is outside 0..511")
        expected_depth = self.population.bit_length()
        if self.task_depth != expected_depth:
            raise ValueError("confirmation task depth changed")
        if len(self.noisy_hints) != expected_depth or len(self.hidden_path) != expected_depth:
            raise ValueError("confirmation path/hint length changed")
        if any(bit not in (0, 1) for bit in self.noisy_hints + self.hidden_path):
            raise ValueError("confirmation paths and hints must remain binary")
        expected_id = 0
        for bit in self.hidden_path:
            expected_id = expected_id * 2 + bit
        if self.hidden_terminal_path_id != expected_id:
            raise ValueError("confirmation hidden terminal path ID changed")
        if not 0 <= self.hidden_terminal_path_id < 2 * self.population:
            raise ValueError("confirmation hidden terminal path is outside the task tree")


@dataclass(frozen=True, slots=True)
class Gate7ConfirmationBatchCondition:
    checkpoint_index: int
    population: int
    condition: str
    k: int | None
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    score_observations_per_world: tuple[int, ...]
    logical_stage_a_parent_slots: int
    logical_stage_b_parent_slots: int
    logical_learned_updates_per_world: int
    learned_parameter_count: int
    parameter_fingerprint: str
    wall_seconds: float
    peak_allocated_bytes: int
    selected_frontier_index_checksum: int
    terminal_score_checksum: float


@dataclass(frozen=True, slots=True)
class Gate7ConfirmationCondition:
    checkpoint_index: int
    population: int
    condition: str
    k: int | None
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    coverage_rate: float
    score_observations_per_world: tuple[int, ...]
    logical_stage_a_parent_slots: int
    logical_stage_b_parent_slots: int
    logical_learned_updates_per_world: int
    learned_parameter_count: int
    parameter_fingerprint: str
    batch_count: int
    wall_seconds: float
    peak_allocated_bytes: int
    selected_frontier_index_checksum: int
    terminal_score_checksum: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7ConfirmationPairedSummary:
    comparison: str
    checkpoint_index: int
    population: int
    k: int | None
    treatment_condition: str
    reference_condition: str
    coverage_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7ConfirmationStratifiedSummary:
    comparison: str
    population: int
    checkpoint_point_deltas: dict[int, float]
    pooled_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def gate7_confirmation_runtime_seed(*, population: int, world_index: int) -> int:
    if population not in GATE7_CONFIRMATION_POPULATIONS:
        raise ValueError("confirmation population is outside the frozen matrix")
    if not 0 <= world_index < GATE7_CONFIRMATION_WORLD_COUNT:
        raise ValueError("confirmation world index is outside 0..511")
    return _seed_from_parts(
        "gate7-high-scale-routing-bandwidth-confirmation-runtime-v0",
        population,
        world_index,
        population.bit_length(),
    )


def generate_gate7_confirmation_world(
    *, population: int, world_index: int
) -> Gate7ConfirmationWorld:
    """Generate one untouched confirmation world from the frozen namespace."""

    if population not in GATE7_CONFIRMATION_POPULATIONS:
        raise ValueError("confirmation population is outside the frozen matrix")
    if not 0 <= world_index < GATE7_CONFIRMATION_WORLD_COUNT:
        raise ValueError("confirmation world index is outside 0..511")
    task_depth = population.bit_length()
    hidden_rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-confirmation-hidden-v0",
            population,
            world_index,
            task_depth,
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(task_depth))
    hint_rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-confirmation-hints-v0",
            population,
            world_index,
            task_depth,
        )
    )
    noisy_hints = tuple(
        hidden_bit
        if hint_rng.random() < GATE7_CONFIRMATION_HINT_RELIABILITY
        else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    terminal_id = 0
    for bit in hidden_path:
        terminal_id = terminal_id * 2 + bit
    world = Gate7ConfirmationWorld(
        population=population,
        world_index=world_index,
        task_depth=task_depth,
        runtime_seed=gate7_confirmation_runtime_seed(
            population=population,
            world_index=world_index,
        ),
        noisy_hints=noisy_hints,
        hidden_path=hidden_path,
        hidden_terminal_path_id=terminal_id,
    )
    world.validate()
    return world


def load_verified_gate7_confirmation_checkpoint(
    *, checkpoint_index: int, checkpoint_path: Path, device: torch.device | str
) -> tuple[Gate7ScaleNeutralScorer, Gate7TransitionCheckpointIdentity]:
    """Load one exact transition checkpoint through the qualified screening loader."""

    if checkpoint_index not in GATE7_CONFIRMATION_CHECKPOINT_INDICES:
        raise ValueError("confirmation checkpoint index must be 0, 1 or 2")
    model, identity = load_verified_gate7_high_scale_checkpoint(
        checkpoint_index=checkpoint_index,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    expected = GATE7_CONFIRMATION_CHECKPOINTS[checkpoint_index]
    if identity.checkpoint_sha256 != expected["sha256"]:
        raise RuntimeError("confirmation checkpoint SHA differs from the frozen protocol")
    if identity.parameter_fingerprint != expected["fingerprint"]:
        raise RuntimeError("confirmation checkpoint fingerprint differs from the frozen protocol")
    if model.trainable_parameter_count() != GATE7_CONFIRMATION_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("confirmation scorer parameter count changed")
    return model, identity


def confirmation_world_batch(
    *, population: int, batch_start: int
) -> tuple[Gate7ConfirmationWorld, ...]:
    if batch_start % GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE:
        raise ValueError("confirmation batch start must align to 64 worlds")
    batch_stop = batch_start + GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE
    if not 0 <= batch_start < batch_stop <= GATE7_CONFIRMATION_WORLD_COUNT:
        raise ValueError("confirmation batch range is outside 0..511")
    worlds = tuple(
        generate_gate7_confirmation_world(population=population, world_index=index)
        for index in range(batch_start, batch_stop)
    )
    _validate_confirmation_world_batch(worlds)
    return worlds


def _validate_confirmation_world_batch(worlds: tuple[Gate7ConfirmationWorld, ...]) -> int:
    if len(worlds) != GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE:
        raise ValueError("confirmation execution requires exactly 64 physical worlds")
    population = worlds[0].population
    if population not in GATE7_CONFIRMATION_POPULATIONS:
        raise ValueError("confirmation batch population is outside the frozen matrix")
    indices = tuple(world.world_index for world in worlds)
    if indices != tuple(range(indices[0], indices[0] + GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE)):
        raise ValueError("confirmation batch world indices must be contiguous")
    if indices[0] % GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE:
        raise ValueError("confirmation batch world indices must align to 64")
    for world in worlds:
        world.validate()
        if world.population != population:
            raise ValueError("confirmation batch mixes populations")
    return population


def _condition_mode(condition: str) -> tuple[str, int | None]:
    if condition == GATE7_CONFIRMATION_GLOBAL_SCORE:
        return GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE, None
    if condition == GATE7_CONFIRMATION_GLOBAL_HASH:
        return GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH, None
    for k in GATE7_CONFIRMATION_K_LADDER:
        if condition == bounded_score_condition(k):
            return GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE, k
        if condition == bounded_hash_condition(k):
            return GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH, k
    raise ValueError("condition is outside the frozen confirmation matrix")


def build_gate7_confirmation_frontier(
    model: Gate7ScaleNeutralScorer,
    *,
    worlds: tuple[Gate7ConfirmationWorld, ...],
    device: torch.device | str,
) -> tuple[object, dict[str, float | int]]:
    population = _validate_confirmation_world_batch(worlds)
    gc.collect()
    target = torch.device(device)
    if torch.cuda.is_available() and target.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frontier = build_gate7_high_scale_immutable_frontier(
        model,
        population=population,
        noisy_hints_by_world=tuple(world.noisy_hints for world in worlds),
        device=target,
    )
    if torch.cuda.is_available() and target.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    else:
        peak = 0
    return frontier, {
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(peak),
        "frontier_storage_bytes": int(
            frontier.states.numel() * frontier.states.element_size()
            + frontier.scores.numel() * frontier.scores.element_size()
        ),
    }


def evaluate_gate7_confirmation_batch_condition(
    model: Gate7ScaleNeutralScorer,
    frontier: object,
    *,
    checkpoint_index: int,
    worlds: tuple[Gate7ConfirmationWorld, ...],
    condition: str,
) -> Gate7ConfirmationBatchCondition:
    population = _validate_confirmation_world_batch(worlds)
    plan = build_confirmation_tier_plan(population)
    if condition not in plan.conditions:
        raise ValueError("condition is not admitted for this confirmation population")
    if checkpoint_index not in GATE7_CONFIRMATION_CHECKPOINT_INDICES:
        raise ValueError("confirmation checkpoint index must be 0, 1 or 2")
    if getattr(frontier, "population", None) != population:
        raise ValueError("confirmation frontier population differs from the world batch")
    mode, k = _condition_mode(condition)
    target = frontier.states.device
    public_seeds = torch.tensor(
        [world.runtime_seed for world in worlds],
        dtype=torch.int64,
        device=target,
    )
    terminal_hints = tuple(world.noisy_hints[-1] for world in worlds)

    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    transcript = run_gate7_high_scale_terminal_stage_b_preparation(
        model,
        frontier,
        terminal_hints_by_world=terminal_hints,
        public_seeds=public_seeds,
        mode=mode,
        k=k,
        stage_b_slots=GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS,
    )
    if target.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    else:
        peak = 0
    wall = time.perf_counter() - started

    hidden_ids = torch.tensor(
        [world.hidden_terminal_path_id for world in worlds],
        dtype=torch.int64,
        device=target,
    )
    covered_tensor = (transcript.terminal_path_ids == hidden_ids[:, None, None]).any(dim=2).any(dim=1)
    covered = tuple(bool(value) for value in covered_tensor.detach().cpu().tolist())
    observations = tuple(
        int(value)
        for value in transcript.total_neural_score_observations_per_world().detach().cpu().tolist()
    )
    row = Gate7ConfirmationBatchCondition(
        checkpoint_index=checkpoint_index,
        population=population,
        condition=condition,
        k=k,
        world_indices=tuple(world.world_index for world in worlds),
        runtime_seeds=tuple(world.runtime_seed for world in worlds),
        covered_by_world=covered,
        score_observations_per_world=observations,
        logical_stage_a_parent_slots=plan.stage_a_parent_slots,
        logical_stage_b_parent_slots=plan.stage_b_parent_slots,
        logical_learned_updates_per_world=plan.logical_learned_updates_per_world,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
        wall_seconds=wall,
        peak_allocated_bytes=int(peak),
        selected_frontier_index_checksum=int(
            transcript.selected_frontier_indices.sum().detach().cpu()
        ),
        terminal_score_checksum=float(transcript.terminal_child_scores.sum().detach().cpu()),
    )
    del transcript, public_seeds, hidden_ids, covered_tensor
    return row


def aggregate_gate7_confirmation_condition(
    rows: tuple[Gate7ConfirmationBatchCondition, ...],
) -> Gate7ConfirmationCondition:
    expected_batches = GATE7_CONFIRMATION_WORLD_COUNT // GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE
    if len(rows) != expected_batches:
        raise ValueError("confirmation aggregation requires exactly eight B64 rows")
    first = rows[0]
    for row in rows:
        if (
            row.checkpoint_index,
            row.population,
            row.condition,
            row.k,
            row.logical_stage_a_parent_slots,
            row.logical_stage_b_parent_slots,
            row.logical_learned_updates_per_world,
            row.learned_parameter_count,
            row.parameter_fingerprint,
        ) != (
            first.checkpoint_index,
            first.population,
            first.condition,
            first.k,
            first.logical_stage_a_parent_slots,
            first.logical_stage_b_parent_slots,
            first.logical_learned_updates_per_world,
            first.learned_parameter_count,
            first.parameter_fingerprint,
        ):
            raise ValueError("confirmation batch-condition identity changed during aggregation")
    world_indices = tuple(index for row in rows for index in row.world_indices)
    if world_indices != tuple(range(GATE7_CONFIRMATION_WORLD_COUNT)):
        raise ValueError("confirmation aggregate must cover exact world indices 0..511")
    runtime_seeds = tuple(seed for row in rows for seed in row.runtime_seeds)
    covered = tuple(value for row in rows for value in row.covered_by_world)
    observations = tuple(value for row in rows for value in row.score_observations_per_world)
    if len(runtime_seeds) != GATE7_CONFIRMATION_WORLD_COUNT:
        raise ValueError("confirmation runtime-seed aggregation changed")
    if len(covered) != GATE7_CONFIRMATION_WORLD_COUNT:
        raise ValueError("confirmation coverage aggregation changed")
    if len(observations) != GATE7_CONFIRMATION_WORLD_COUNT:
        raise ValueError("confirmation observation aggregation changed")
    return Gate7ConfirmationCondition(
        checkpoint_index=first.checkpoint_index,
        population=first.population,
        condition=first.condition,
        k=first.k,
        world_indices=world_indices,
        runtime_seeds=runtime_seeds,
        covered_by_world=covered,
        coverage_rate=sum(int(value) for value in covered) / GATE7_CONFIRMATION_WORLD_COUNT,
        score_observations_per_world=observations,
        logical_stage_a_parent_slots=first.logical_stage_a_parent_slots,
        logical_stage_b_parent_slots=first.logical_stage_b_parent_slots,
        logical_learned_updates_per_world=first.logical_learned_updates_per_world,
        learned_parameter_count=first.learned_parameter_count,
        parameter_fingerprint=first.parameter_fingerprint,
        batch_count=expected_batches,
        wall_seconds=sum(row.wall_seconds for row in rows),
        peak_allocated_bytes=max(row.peak_allocated_bytes for row in rows),
        selected_frontier_index_checksum=sum(
            row.selected_frontier_index_checksum for row in rows
        ),
        terminal_score_checksum=sum(row.terminal_score_checksum for row in rows),
    )


def _bootstrap_quantiles(estimates: list[float]) -> tuple[float, float]:
    estimates.sort()
    return (
        estimates[int(math.floor(0.025 * (GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES - 1)))],
    )


def paired_gate7_confirmation_summary(
    *,
    comparison: str,
    treatment: Gate7ConfirmationCondition,
    reference: Gate7ConfirmationCondition,
) -> Gate7ConfirmationPairedSummary:
    if treatment.checkpoint_index != reference.checkpoint_index:
        raise ValueError("confirmation pair checkpoint mismatch")
    if treatment.population != reference.population:
        raise ValueError("confirmation pair population mismatch")
    if treatment.world_indices != reference.world_indices or treatment.runtime_seeds != reference.runtime_seeds:
        raise ValueError("confirmation pair does not share the exact same worlds")
    differences = tuple(
        int(left) - int(right)
        for left, right in zip(treatment.covered_by_world, reference.covered_by_world, strict=True)
    )
    rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-confirmation-paired-bootstrap-v0",
            treatment.population,
            treatment.checkpoint_index,
            comparison,
        )
    )
    count = GATE7_CONFIRMATION_WORLD_COUNT
    estimates = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES)
    ]
    low, high = _bootstrap_quantiles(estimates)
    return Gate7ConfirmationPairedSummary(
        comparison=comparison,
        checkpoint_index=treatment.checkpoint_index,
        population=treatment.population,
        k=treatment.k,
        treatment_condition=treatment.condition,
        reference_condition=reference.condition,
        coverage_delta=sum(differences) / count,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def stratified_gate7_confirmation_global_summary(
    *,
    population: int,
    treatment_by_checkpoint: dict[int, Gate7ConfirmationCondition],
    reference_by_checkpoint: dict[int, Gate7ConfirmationCondition],
) -> Gate7ConfirmationStratifiedSummary:
    if set(treatment_by_checkpoint) != set(GATE7_CONFIRMATION_CHECKPOINT_INDICES):
        raise ValueError("confirmation stratified treatment requires all three checkpoints")
    if set(reference_by_checkpoint) != set(GATE7_CONFIRMATION_CHECKPOINT_INDICES):
        raise ValueError("confirmation stratified reference requires all three checkpoints")
    differences: dict[int, tuple[int, ...]] = {}
    points: dict[int, float] = {}
    for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES:
        treatment = treatment_by_checkpoint[checkpoint]
        reference = reference_by_checkpoint[checkpoint]
        if treatment.population != population or reference.population != population:
            raise ValueError("confirmation stratified population mismatch")
        if treatment.world_indices != reference.world_indices or treatment.runtime_seeds != reference.runtime_seeds:
            raise ValueError("confirmation stratified pair does not share exact worlds")
        vector = tuple(
            int(left) - int(right)
            for left, right in zip(treatment.covered_by_world, reference.covered_by_world, strict=True)
        )
        differences[checkpoint] = vector
        points[checkpoint] = sum(vector) / GATE7_CONFIRMATION_WORLD_COUNT

    rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-confirmation-stratified-bootstrap-v0",
            population,
            "global_score_vs_global_hash",
        )
    )
    count = GATE7_CONFIRMATION_WORLD_COUNT
    estimates: list[float] = []
    for _ in range(GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES):
        stratum_means = []
        for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES:
            vector = differences[checkpoint]
            stratum_means.append(
                sum(vector[rng.randrange(count)] for _ in range(count)) / count
            )
        estimates.append(sum(stratum_means) / len(stratum_means))
    low, high = _bootstrap_quantiles(estimates)
    return Gate7ConfirmationStratifiedSummary(
        comparison="global_score_vs_global_hash_stratified",
        population=population,
        checkpoint_point_deltas=points,
        pooled_delta=sum(points.values()) / len(points),
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def confirmation_provenance() -> dict[str, Any]:
    return {
        "confirmation_version": GATE7_CONFIRMATION_VERSION,
        "confirmation_protocol_head": GATE7_CONFIRMATION_PROTOCOL_HEAD,
        "screening_result_head": GATE7_CONFIRMATION_SCREENING_RESULT_HEAD,
        "screening_result_sha256": GATE7_CONFIRMATION_SCREENING_RESULT_SHA256,
        "screening_audit_sha256": GATE7_CONFIRMATION_SCREENING_AUDIT_SHA256,
        "world_count": GATE7_CONFIRMATION_WORLD_COUNT,
        "evaluation_batch_size": GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE,
        "bootstrap_samples": GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES,
        "stage_b_parent_slots": GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS,
        "active_child_lanes": GATE7_CONFIRMATION_ACTIVE_CHILD_LANES,
        "recurrent_updates_per_child": GATE7_CONFIRMATION_RECURRENT_UPDATES_PER_CHILD,
    }
