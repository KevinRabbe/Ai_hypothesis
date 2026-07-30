"""Data-blind complete scale-neutral Stage-A frontier construction for Gate-7 preparation.

The builder consumes public noisy hints only and produces one immutable lexicographically ordered
recurrent-state/score frontier. Frontier index i is the binary depth-d path value i, so no candidate
objects or additional path tensor are required. No hidden answer, checkpoint loading, scientific world
namespace, result classification, or admitted runner lives here.
"""

from __future__ import annotations

import torch

from .gate3_v1_sparse_active_reserve import GATE3_V1_RECURRENT_UPDATES_PER_CHILD
from .gate7_high_scale_index_bank_prep import Gate7HighScaleImmutableFrontier
from .gate7_high_scale_routing_bandwidth_protocol import (
    GATE7_HIGH_SCALE_LEARNED_PARAMETER_COUNT,
    build_gate7_high_scale_tier_plan,
)
from .gate7_scale_neutral_memory_bounded_prep import (
    advance_gate7_scale_neutral_memory_bounded,
)
from .gate7_scale_neutral_model_prep import (
    Gate7ScaleNeutralScorer,
    encode_gate7_scale_neutral_child_inputs_batch,
)

GATE7_HIGH_SCALE_FRONTIER_PREPARATION_ONLY = True
GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS = 1_048_576


def validate_gate7_high_scale_public_hints(
    *,
    population: int,
    noisy_hints_by_world: tuple[tuple[int, ...], ...],
) -> int:
    plan = build_gate7_high_scale_tier_plan(population)
    if not noisy_hints_by_world:
        raise ValueError("at least one public Gate-7 world is required")
    for row in noisy_hints_by_world:
        if len(row) != plan.world_depth:
            raise ValueError("public hint row length differs from the frozen tier world depth")
        if any(value not in (0, 1) for value in row):
            raise ValueError("public Gate-7 hints must remain binary")
    return plan.world_depth


def gate7_high_scale_frontier_chunk_ranges(total_rows: int) -> tuple[tuple[int, int], ...]:
    """Return the fixed contiguous recurrent-row partition used by every complete layer."""

    if total_rows <= 0:
        raise ValueError("complete-frontier recurrent row count must be positive")
    if GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS <= 0:
        raise RuntimeError("Gate-7 frontier recurrent-row chunk must remain positive")
    return tuple(
        (
            start,
            min(start + GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS, total_rows),
        )
        for start in range(0, total_rows, GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS)
    )


def _productive_inputs_for_action_chunk(
    *,
    hint_by_world: torch.Tensor,
    world_depth: int,
    child_depth: int,
    parent_count: int,
    branch_action: int,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    if branch_action not in (0, 1):
        raise ValueError("complete-frontier action must remain binary")
    if hint_by_world.ndim != 1 or hint_by_world.dtype != torch.int64:
        raise ValueError("public layer hints must use int64 [batch]")
    if not 0 <= row_start < row_end <= hint_by_world.shape[0] * parent_count:
        raise ValueError("recurrent-row chunk is outside the complete layer")

    device = hint_by_world.device
    row_ids = torch.arange(row_start, row_end, dtype=torch.int64, device=device)
    world_indices = torch.div(row_ids, parent_count, rounding_mode="floor")
    observed_hints = hint_by_world[world_indices]
    count = row_end - row_start
    return encode_gate7_scale_neutral_child_inputs_batch(
        world_depths=torch.full((count,), world_depth, dtype=torch.int64, device=device),
        child_depths=torch.full((count,), child_depth, dtype=torch.int64, device=device),
        observed_hints=observed_hints,
        branch_actions=torch.full(
            (count,), branch_action, dtype=torch.int64, device=device
        ),
        sink=torch.zeros(count, dtype=torch.bool, device=device),
    )


def build_gate7_high_scale_immutable_frontier(
    model: Gate7ScaleNeutralScorer,
    *,
    population: int,
    noisy_hints_by_world: tuple[tuple[int, ...], ...],
    device: torch.device | str,
) -> Gate7HighScaleImmutableFrontier:
    """Build the exact complete Stage-A frontier under the frozen scale-neutral scorer.

    Output order is world-major and lexicographic by binary action path. The function performs only
    productive Stage-A transitions; the final public hint at world_depth is reserved for Stage B.

    Action lanes are executed separately, recurrent updates are applied one step at a time, and each
    action lane is partitioned into fixed contiguous recurrent-row chunks. The same B64 output frontier
    is preallocated in full; chunking changes only transient execution and does not change world count,
    logical work, output order, learned parameters, or model arithmetic.
    """

    plan = build_gate7_high_scale_tier_plan(population)
    world_depth = validate_gate7_high_scale_public_hints(
        population=population,
        noisy_hints_by_world=noisy_hints_by_world,
    )
    if model.trainable_parameter_count() != GATE7_HIGH_SCALE_LEARNED_PARAMETER_COUNT:
        raise ValueError("Gate-7 frontier model must contain exactly 19,649 learned parameters")

    target = torch.device(device)
    model = model.to(target)
    model.eval()
    batch = len(noisy_hints_by_world)
    states = model.initial_state(batch, device=target).reshape(batch, 1, 64)
    scores = torch.zeros((batch, 1), dtype=torch.float32, device=target)

    with torch.inference_mode():
        for child_depth in range(1, plan.frontier_depth + 1):
            parent_count = states.shape[1]
            total_parent_rows = batch * parent_count
            parent_states = states.reshape(total_parent_rows, 64)
            hint_by_world = torch.tensor(
                [row[child_depth - 1] for row in noisy_hints_by_world],
                dtype=torch.int64,
                device=target,
            )
            next_states = torch.empty(
                (batch, parent_count * 2, 64),
                dtype=torch.float32,
                device=target,
            )
            next_scores = torch.empty(
                (batch, parent_count * 2),
                dtype=torch.float32,
                device=target,
            )
            next_states_flat = next_states.reshape(total_parent_rows * 2, 64)
            next_scores_flat = next_scores.reshape(total_parent_rows * 2)
            chunk_ranges = gate7_high_scale_frontier_chunk_ranges(total_parent_rows)

            for branch_action in (0, 1):
                for row_start, row_end in chunk_ranges:
                    child_inputs = _productive_inputs_for_action_chunk(
                        hint_by_world=hint_by_world,
                        world_depth=world_depth,
                        child_depth=child_depth,
                        parent_count=parent_count,
                        branch_action=branch_action,
                        row_start=row_start,
                        row_end=row_end,
                    )
                    child_states = advance_gate7_scale_neutral_memory_bounded(
                        model,
                        parent_states[row_start:row_end],
                        child_inputs,
                        repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
                    )
                    child_scores = model.score(child_states)
                    output_positions = (
                        torch.arange(row_start, row_end, dtype=torch.int64, device=target) * 2
                        + branch_action
                    )
                    next_states_flat.index_copy_(0, output_positions, child_states)
                    next_scores_flat.index_copy_(0, output_positions, child_scores)
                    del child_inputs, child_states, child_scores, output_positions

            del parent_states, hint_by_world, next_states_flat, next_scores_flat
            states = next_states
            scores = next_scores

    frontier = Gate7HighScaleImmutableFrontier(
        states=states,
        scores=scores,
        population=population,
    )
    frontier.validate()
    return frontier
