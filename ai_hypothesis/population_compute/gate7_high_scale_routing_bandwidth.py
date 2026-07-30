"""Bound scientific execution core for Gate-7 high-scale routing-bandwidth screening.

This module binds the exact qualified transition checkpoints, defines the one fresh high-scale world
namespace, evaluates the frozen immutable-frontier / terminal-routing conditions, and computes the
preregistered paired intervals.  It performs no training, checkpoint selection, protocol mutation, or
confirmation execution.
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
from .gate7_high_scale_routing_bandwidth_protocol import (
    GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES,
    GATE7_HIGH_SCALE_CHECKPOINT_INDICES,
    GATE7_HIGH_SCALE_CHECKPOINTS,
    GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE,
    GATE7_HIGH_SCALE_GLOBAL_HASH,
    GATE7_HIGH_SCALE_GLOBAL_SCORE,
    GATE7_HIGH_SCALE_HINT_RELIABILITY,
    GATE7_HIGH_SCALE_K_LADDER,
    GATE7_HIGH_SCALE_LEARNED_PARAMETER_COUNT,
    GATE7_HIGH_SCALE_POPULATIONS,
    GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
    GATE7_HIGH_SCALE_VERSION,
    GATE7_HIGH_SCALE_WORLD_COUNT,
    bounded_hash_condition,
    bounded_score_condition,
    build_gate7_high_scale_tier_plan,
)
from .gate7_high_scale_terminal_stage_b_prep import (
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
    run_gate7_high_scale_terminal_stage_b_preparation,
)
from .gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer
from .gate7_scale_neutral_transition_bridge import (
    Gate7TransitionCheckpointIdentity,
    load_verified_gate7_transition_checkpoint,
)

GATE7_HIGH_SCALE_EXECUTION_ADMITTED = True
GATE7_HIGH_SCALE_SCIENTIFIC_STATUS = "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_SCREENING_EVIDENCE"
GATE7_HIGH_SCALE_ENGINEERING_RESULT_HEAD = "5305475ea1e295c84fadbce3533f13489b10d60d"
GATE7_HIGH_SCALE_ENGINEERING_SUMMARY_SHA256 = (
    "e40823e3e2787151f2a63607aa3d396f18e03428b715b8864af4f549631e2953"
)
GATE7_HIGH_SCALE_ENGINEERING_MANIFEST_SHA256 = (
    "8393f9b4f11aa90aa333c3443669306675d1e9cc746e1f1dc3aa5acd1523afe4"
)


@dataclass(frozen=True, slots=True)
class Gate7HighScaleWorld:
    population: int
    world_index: int
    task_depth: int
    runtime_seed: int
    noisy_hints: tuple[int, ...]
    hidden_path: tuple[int, ...]
    hidden_terminal_path_id: int

    def validate(self) -> None:
        plan = build_gate7_high_scale_tier_plan(self.population)
        if not 0 <= self.world_index < GATE7_HIGH_SCALE_WORLD_COUNT:
            raise ValueError("Gate-7 high-scale world index is outside 0..63")
        if self.task_depth != plan.world_depth:
            raise ValueError("Gate-7 high-scale task depth changed")
        if len(self.noisy_hints) != self.task_depth or len(self.hidden_path) != self.task_depth:
            raise ValueError("Gate-7 high-scale path/hint length changed")
        if any(bit not in (0, 1) for bit in self.noisy_hints + self.hidden_path):
            raise ValueError("Gate-7 high-scale paths and hints must remain binary")
        expected_id = 0
        for bit in self.hidden_path:
            expected_id = expected_id * 2 + bit
        if self.hidden_terminal_path_id != expected_id:
            raise ValueError("Gate-7 high-scale hidden terminal path ID changed")
        if not 0 <= self.hidden_terminal_path_id < 2 * self.population:
            raise ValueError("Gate-7 high-scale hidden terminal path is outside the task tree")


@dataclass(frozen=True, slots=True)
class Gate7HighScaleCondition:
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
    wall_seconds: float
    peak_allocated_bytes: int
    selected_frontier_index_checksum: int
    terminal_score_checksum: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7HighScalePairedSummary:
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
class Gate7HighScaleStratifiedSummary:
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


def gate7_high_scale_runtime_seed(*, population: int, world_index: int) -> int:
    plan = build_gate7_high_scale_tier_plan(population)
    if not 0 <= world_index < GATE7_HIGH_SCALE_WORLD_COUNT:
        raise ValueError("Gate-7 high-scale world index is outside 0..63")
    return _seed_from_parts(
        "gate7-high-scale-routing-bandwidth-runtime-v0",
        population,
        world_index,
        plan.world_depth,
    )


def generate_gate7_high_scale_world(*, population: int, world_index: int) -> Gate7HighScaleWorld:
    """Generate one untouched high-scale hidden/hint/runtime world from the frozen namespace."""

    plan = build_gate7_high_scale_tier_plan(population)
    if not 0 <= world_index < GATE7_HIGH_SCALE_WORLD_COUNT:
        raise ValueError("Gate-7 high-scale world index is outside 0..63")
    hidden_rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-hidden-v0",
            population,
            world_index,
            plan.world_depth,
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(plan.world_depth))
    hint_rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-hints-v0",
            population,
            world_index,
            plan.world_depth,
        )
    )
    noisy_hints = tuple(
        hidden_bit
        if hint_rng.random() < GATE7_HIGH_SCALE_HINT_RELIABILITY
        else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    terminal_id = 0
    for bit in hidden_path:
        terminal_id = terminal_id * 2 + bit
    world = Gate7HighScaleWorld(
        population=population,
        world_index=world_index,
        task_depth=plan.world_depth,
        runtime_seed=gate7_high_scale_runtime_seed(
            population=population,
            world_index=world_index,
        ),
        noisy_hints=noisy_hints,
        hidden_path=hidden_path,
        hidden_terminal_path_id=terminal_id,
    )
    world.validate()
    return world


def load_verified_gate7_high_scale_checkpoint(
    *, checkpoint_index: int, checkpoint_path: Path, device: torch.device | str
) -> tuple[Gate7ScaleNeutralScorer, Gate7TransitionCheckpointIdentity]:
    """Load one exact transition checkpoint and expose its verified scale-neutral scorer."""

    if checkpoint_index not in GATE7_HIGH_SCALE_CHECKPOINT_INDICES:
        raise ValueError("Gate-7 high-scale checkpoint index must be 0, 1 or 2")
    expected = GATE7_HIGH_SCALE_CHECKPOINTS[checkpoint_index]
    adapter, identity = load_verified_gate7_transition_checkpoint(
        checkpoint_index=checkpoint_index,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    if identity.checkpoint_sha256 != expected["sha256"]:
        raise RuntimeError("Gate-7 high-scale checkpoint SHA binding differs from the frozen protocol")
    if identity.parameter_fingerprint != expected["fingerprint"]:
        raise RuntimeError("Gate-7 high-scale checkpoint fingerprint differs from the frozen protocol")
    scorer = adapter.scorer
    if scorer.trainable_parameter_count() != GATE7_HIGH_SCALE_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("Gate-7 high-scale scorer parameter count changed")
    scorer.eval()
    scorer.to(device)
    return scorer, identity


def _validate_world_batch(worlds: tuple[Gate7HighScaleWorld, ...]) -> int:
    if len(worlds) != GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE:
        raise ValueError("Gate-7 high-scale execution requires exactly 64 paired worlds")
    population = worlds[0].population
    if population not in GATE7_HIGH_SCALE_POPULATIONS:
        raise ValueError("Gate-7 high-scale population is outside the frozen ladder")
    if tuple(world.world_index for world in worlds) != tuple(range(GATE7_HIGH_SCALE_WORLD_COUNT)):
        raise ValueError("Gate-7 high-scale world indices must remain the exact 0..63 namespace")
    for world in worlds:
        world.validate()
        if world.population != population:
            raise ValueError("Gate-7 high-scale world batch mixes populations")
    return population


def _condition_mode(condition: str) -> tuple[str, int | None]:
    if condition == GATE7_HIGH_SCALE_GLOBAL_SCORE:
        return GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE, None
    if condition == GATE7_HIGH_SCALE_GLOBAL_HASH:
        return GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH, None
    for k in GATE7_HIGH_SCALE_K_LADDER:
        if condition == bounded_score_condition(k):
            return GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE, k
        if condition == bounded_hash_condition(k):
            return GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH, k
    raise ValueError("condition is outside the frozen Gate-7 high-scale matrix")


def build_gate7_high_scale_scientific_frontier(
    model: Gate7ScaleNeutralScorer,
    *,
    worlds: tuple[Gate7HighScaleWorld, ...],
    device: torch.device | str,
) -> tuple[object, dict[str, float | int]]:
    """Build one common Stage-A frontier from public hints only and return engineering telemetry."""

    population = _validate_world_batch(worlds)
    gc.collect()
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frontier = build_gate7_high_scale_immutable_frontier(
        model,
        population=population,
        noisy_hints_by_world=tuple(world.noisy_hints for world in worlds),
        device=device,
    )
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
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


def evaluate_gate7_high_scale_condition(
    model: Gate7ScaleNeutralScorer,
    frontier: object,
    *,
    checkpoint_index: int,
    worlds: tuple[Gate7HighScaleWorld, ...],
    condition: str,
) -> Gate7HighScaleCondition:
    """Evaluate one frozen routing condition; hidden answers are used only after terminal execution."""

    population = _validate_world_batch(worlds)
    if checkpoint_index not in GATE7_HIGH_SCALE_CHECKPOINT_INDICES:
        raise ValueError("Gate-7 high-scale checkpoint index must be 0, 1 or 2")
    if getattr(frontier, "population", None) != population:
        raise ValueError("Gate-7 high-scale frontier population differs from the world batch")
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
        stage_b_slots=GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
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
    plan = build_gate7_high_scale_tier_plan(population)
    row = Gate7HighScaleCondition(
        checkpoint_index=checkpoint_index,
        population=population,
        condition=condition,
        k=k,
        world_indices=tuple(world.world_index for world in worlds),
        runtime_seeds=tuple(world.runtime_seed for world in worlds),
        covered_by_world=covered,
        coverage_rate=sum(int(value) for value in covered) / GATE7_HIGH_SCALE_WORLD_COUNT,
        score_observations_per_world=observations,
        logical_stage_a_parent_slots=plan.stage_a_parent_slots,
        logical_stage_b_parent_slots=plan.stage_b_parent_slots,
        logical_learned_updates_per_world=plan.total_logical_learned_updates,
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


def _bootstrap_quantiles(estimates: list[float]) -> tuple[float, float]:
    estimates.sort()
    return (
        estimates[int(math.floor(0.025 * (GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES - 1)))],
    )


def paired_gate7_high_scale_summary(
    *,
    comparison: str,
    treatment: Gate7HighScaleCondition,
    reference: Gate7HighScaleCondition,
) -> Gate7HighScalePairedSummary:
    if treatment.checkpoint_index != reference.checkpoint_index:
        raise ValueError("Gate-7 high-scale pair checkpoint mismatch")
    if treatment.population != reference.population:
        raise ValueError("Gate-7 high-scale pair population mismatch")
    if treatment.world_indices != reference.world_indices or treatment.runtime_seeds != reference.runtime_seeds:
        raise ValueError("Gate-7 high-scale pair does not share the exact same worlds")
    differences = tuple(
        int(left) - int(right)
        for left, right in zip(treatment.covered_by_world, reference.covered_by_world, strict=True)
    )
    rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-paired-bootstrap-v0",
            treatment.population,
            treatment.checkpoint_index,
            comparison,
        )
    )
    count = GATE7_HIGH_SCALE_WORLD_COUNT
    estimates = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES)
    ]
    low, high = _bootstrap_quantiles(estimates)
    return Gate7HighScalePairedSummary(
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


def stratified_gate7_high_scale_global_summary(
    *,
    population: int,
    treatment_by_checkpoint: dict[int, Gate7HighScaleCondition],
    reference_by_checkpoint: dict[int, Gate7HighScaleCondition],
) -> Gate7HighScaleStratifiedSummary:
    if set(treatment_by_checkpoint) != set(GATE7_HIGH_SCALE_CHECKPOINT_INDICES):
        raise ValueError("stratified Gate-7 treatment requires all three checkpoints")
    if set(reference_by_checkpoint) != set(GATE7_HIGH_SCALE_CHECKPOINT_INDICES):
        raise ValueError("stratified Gate-7 reference requires all three checkpoints")
    differences: dict[int, tuple[int, ...]] = {}
    points: dict[int, float] = {}
    for checkpoint in GATE7_HIGH_SCALE_CHECKPOINT_INDICES:
        treatment = treatment_by_checkpoint[checkpoint]
        reference = reference_by_checkpoint[checkpoint]
        if treatment.population != population or reference.population != population:
            raise ValueError("stratified Gate-7 population mismatch")
        if treatment.world_indices != reference.world_indices or treatment.runtime_seeds != reference.runtime_seeds:
            raise ValueError("stratified Gate-7 pair does not share the exact worlds")
        vector = tuple(
            int(left) - int(right)
            for left, right in zip(treatment.covered_by_world, reference.covered_by_world, strict=True)
        )
        differences[checkpoint] = vector
        points[checkpoint] = sum(vector) / GATE7_HIGH_SCALE_WORLD_COUNT

    rng = random.Random(
        _seed_from_parts(
            "gate7-high-scale-routing-bandwidth-stratified-bootstrap-v0",
            population,
            "global_score_vs_global_hash",
        )
    )
    count = GATE7_HIGH_SCALE_WORLD_COUNT
    estimates: list[float] = []
    for _ in range(GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES):
        stratum_means = []
        for checkpoint in GATE7_HIGH_SCALE_CHECKPOINT_INDICES:
            vector = differences[checkpoint]
            stratum_means.append(
                sum(vector[rng.randrange(count)] for _ in range(count)) / count
            )
        estimates.append(sum(stratum_means) / len(stratum_means))
    low, high = _bootstrap_quantiles(estimates)
    return Gate7HighScaleStratifiedSummary(
        comparison="global_score_vs_global_hash_stratified",
        population=population,
        checkpoint_point_deltas=points,
        pooled_delta=sum(points.values()) / len(points),
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )
