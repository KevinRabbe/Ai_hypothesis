"""B64 frontier and fixed-condition execution for Gate-7 continuation."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .gate7_high_scale_frontier_prep import build_gate7_high_scale_immutable_frontier
from .gate7_high_scale_routing_bandwidth_continuation_protocol import (
    GATE7_CONTINUATION_CHECKPOINT_INDICES,
    GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
    GATE7_CONTINUATION_GLOBAL_HASH,
    GATE7_CONTINUATION_GLOBAL_SCORE,
    GATE7_CONTINUATION_K_LADDER,
    GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS,
    GATE7_CONTINUATION_WORLD_COUNT,
    bounded_hash_condition,
    bounded_score_condition,
    build_continuation_tier_plan,
)
from .gate7_high_scale_routing_bandwidth_continuation_worlds import (
    Gate7ContinuationWorld,
    validate_continuation_world_batch,
)
from .gate7_high_scale_terminal_stage_b_prep import (
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
    run_gate7_high_scale_terminal_stage_b_preparation,
)
from .gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer


@dataclass(frozen=True, slots=True)
class Gate7ContinuationBatchCondition:
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
class Gate7ContinuationCondition:
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


def _condition_mode(condition: str) -> tuple[str, int | None]:
    if condition == GATE7_CONTINUATION_GLOBAL_SCORE:
        return GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE, None
    if condition == GATE7_CONTINUATION_GLOBAL_HASH:
        return GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH, None
    for k in GATE7_CONTINUATION_K_LADDER:
        if condition == bounded_score_condition(k):
            return GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE, k
        if condition == bounded_hash_condition(k):
            return GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH, k
    raise ValueError("condition is outside the frozen continuation matrix")


def build_gate7_continuation_frontier(
    model: Gate7ScaleNeutralScorer,
    *,
    worlds: tuple[Gate7ContinuationWorld, ...],
    device: torch.device | str,
) -> tuple[object, dict[str, float | int]]:
    population = validate_continuation_world_batch(worlds)
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


def evaluate_gate7_continuation_batch_condition(
    model: Gate7ScaleNeutralScorer,
    frontier: object,
    *,
    checkpoint_index: int,
    worlds: tuple[Gate7ContinuationWorld, ...],
    condition: str,
) -> Gate7ContinuationBatchCondition:
    population = validate_continuation_world_batch(worlds)
    plan = build_continuation_tier_plan(population)
    if condition not in plan.conditions:
        raise ValueError("condition is not admitted for this continuation population")
    if checkpoint_index not in GATE7_CONTINUATION_CHECKPOINT_INDICES:
        raise ValueError("continuation checkpoint index must be 0, 1 or 2")
    if getattr(frontier, "population", None) != population:
        raise ValueError("continuation frontier population differs from the world batch")
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
        stage_b_slots=GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS,
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
    row = Gate7ContinuationBatchCondition(
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


def aggregate_gate7_continuation_condition(
    rows: tuple[Gate7ContinuationBatchCondition, ...],
) -> Gate7ContinuationCondition:
    expected_batches = GATE7_CONTINUATION_WORLD_COUNT // GATE7_CONTINUATION_EVALUATION_BATCH_SIZE
    if len(rows) != expected_batches:
        raise ValueError("continuation aggregation requires exactly eight B64 rows")
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
            raise ValueError("continuation batch-condition identity changed during aggregation")
    world_indices = tuple(index for row in rows for index in row.world_indices)
    if world_indices != tuple(range(GATE7_CONTINUATION_WORLD_COUNT)):
        raise ValueError("continuation aggregate must cover exact world indices 0..511")
    runtime_seeds = tuple(seed for row in rows for seed in row.runtime_seeds)
    covered = tuple(value for row in rows for value in row.covered_by_world)
    observations = tuple(value for row in rows for value in row.score_observations_per_world)
    if len(runtime_seeds) != GATE7_CONTINUATION_WORLD_COUNT:
        raise ValueError("continuation runtime-seed aggregation changed")
    if len(covered) != GATE7_CONTINUATION_WORLD_COUNT:
        raise ValueError("continuation coverage aggregation changed")
    if len(observations) != GATE7_CONTINUATION_WORLD_COUNT:
        raise ValueError("continuation observation aggregation changed")
    return Gate7ContinuationCondition(
        checkpoint_index=first.checkpoint_index,
        population=first.population,
        condition=first.condition,
        k=first.k,
        world_indices=world_indices,
        runtime_seeds=runtime_seeds,
        covered_by_world=covered,
        coverage_rate=sum(int(value) for value in covered) / GATE7_CONTINUATION_WORLD_COUNT,
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
