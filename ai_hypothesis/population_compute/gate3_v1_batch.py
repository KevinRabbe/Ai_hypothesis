"""Batched eager runtime for Gate-3 v1 sparse-active search.

Every world contributes exactly two neural lanes in every frozen search round. Productive worlds
use those lanes for branch children; exhausted worlds use matched-work sink lanes. Reserve/search
decisions remain per-world and answer-blind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .gate3_v1_model import (
    Gate3V1NeuralCandidate,
    Gate3V1RuntimeResult,
    Gate3V1RuntimeTelemetry,
    Gate3V1Scorer,
    _apply_neural_control,
    _rank_neural_candidates,
    encode_gate3_v1_child_input,
)
from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    Gate3V1ControlMode,
    Gate3V1PublicWorld,
    build_gate3_v1_condition_plan,
    make_gate3_v1_accounting,
)


@dataclass(frozen=True, slots=True)
class Gate3V1BatchResult:
    depth: int
    reserve_capacity: int
    mode: Gate3V1ControlMode
    world_seeds: tuple[int, ...]
    world_results: tuple[Gate3V1RuntimeResult, ...]


def run_gate3_v1_public_world_batch(
    model: Gate3V1Scorer,
    worlds: Iterable[Gate3V1PublicWorld],
    *,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    device: torch.device | str = "cpu",
) -> Gate3V1BatchResult:
    materialized = tuple(worlds)
    if not materialized:
        raise ValueError("at least one Gate-3 v1 public world is required")
    for world in materialized:
        world.validate()
    depth = materialized[0].depth
    if any(world.depth != depth for world in materialized):
        raise ValueError("one Gate-3 v1 batch must use one depth tier")

    plan = build_gate3_v1_condition_plan(
        depth=depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
    )
    target_device = torch.device(device)
    model = model.to(target_device)
    batch_size = len(materialized)

    initial_states = model.initial_state(batch_size, device=target_device)
    reserves: list[tuple[Gate3V1NeuralCandidate, ...]] = [
        (
            Gate3V1NeuralCandidate(
                path=(),
                state=initial_states[index].clone(),
                score=0.0,
            ),
        )
        for index in range(batch_size)
    ]
    terminals: list[list[tuple[int, ...]]] = [[] for _ in materialized]
    reserve_counts: list[list[int]] = [[] for _ in materialized]
    unique_counts: list[list[int]] = [[] for _ in materialized]
    productive_rounds = [0 for _ in materialized]

    with torch.inference_mode():
        for expansion_index in range(plan.search_rounds):
            lane_states: list[torch.Tensor] = []
            lane_inputs: list[torch.Tensor] = []
            productive: list[bool] = []
            parent_by_world: list[Gate3V1NeuralCandidate | None] = []
            remaining_by_world: list[list[Gate3V1NeuralCandidate]] = []

            for world_index, world in enumerate(materialized):
                reserve = reserves[world_index]
                if not reserve:
                    sink_states = model.initial_state(2, device=target_device)
                    sink_input = encode_gate3_v1_child_input(
                        world=world,
                        child_depth=world.depth,
                        observed_hint=None,
                        branch_action=None,
                        sink=True,
                        device=target_device,
                    )
                    lane_states.extend((sink_states[0], sink_states[1]))
                    lane_inputs.extend((sink_input, sink_input.clone()))
                    productive.append(False)
                    parent_by_world.append(None)
                    remaining_by_world.append([])
                    continue

                ranked = _rank_neural_candidates(
                    reserve,
                    world_seed=world.seed,
                    expansion_index=expansion_index,
                )
                parent = ranked[0]
                remaining = [candidate for candidate in reserve if candidate.path != parent.path]
                next_depth = parent.depth + 1
                hint = world.noisy_hints[next_depth - 1]
                lane_states.extend((parent.state.clone(), parent.state.clone()))
                lane_inputs.extend(
                    (
                        encode_gate3_v1_child_input(
                            world=world,
                            child_depth=next_depth,
                            observed_hint=hint,
                            branch_action=0,
                            sink=False,
                            device=target_device,
                        ),
                        encode_gate3_v1_child_input(
                            world=world,
                            child_depth=next_depth,
                            observed_hint=hint,
                            branch_action=1,
                            sink=False,
                            device=target_device,
                        ),
                    )
                )
                productive.append(True)
                parent_by_world.append(parent)
                remaining_by_world.append(remaining)

            flat_states = torch.stack(lane_states, dim=0)
            flat_inputs = torch.stack(lane_inputs, dim=0)
            advanced = model.advance(
                flat_states,
                flat_inputs,
                repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            )
            scores = model.score(advanced)

            for world_index, world in enumerate(materialized):
                if not productive[world_index]:
                    reserve_counts[world_index].append(0)
                    unique_counts[world_index].append(0)
                    continue

                parent = parent_by_world[world_index]
                assert parent is not None
                remaining = remaining_by_world[world_index]
                next_depth = parent.depth + 1
                lane_offset = 2 * world_index
                for action in (0, 1):
                    path = parent.path + (action,)
                    if next_depth == world.depth:
                        terminals[world_index].append(path)
                    else:
                        remaining.append(
                            Gate3V1NeuralCandidate(
                                path=path,
                                state=advanced[lane_offset + action].clone(),
                                score=float(scores[lane_offset + action].item()),
                            )
                        )

                reserves[world_index] = _apply_neural_control(
                    tuple(remaining),
                    reserve_capacity=reserve_capacity,
                    mode=mode,
                    world_seed=world.seed,
                    expansion_index=expansion_index,
                )
                productive_rounds[world_index] += 1
                reserve_counts[world_index].append(len(reserves[world_index]))
                unique_counts[world_index].append(
                    len({candidate.path for candidate in reserves[world_index]})
                )

    results: list[Gate3V1RuntimeResult] = []
    for world_index, world in enumerate(materialized):
        accounting = make_gate3_v1_accounting(
            depth=depth,
            reserve_capacity=reserve_capacity,
            mode=mode,
            productive_rounds=productive_rounds[world_index],
        )
        results.append(
            Gate3V1RuntimeResult(
                generated_terminal_paths=tuple(terminals[world_index]),
                telemetry=Gate3V1RuntimeTelemetry(
                    depth=depth,
                    reserve_capacity=reserve_capacity,
                    mode=mode,
                    productive_rounds=accounting.productive_rounds,
                    sink_rounds=accounting.sink_rounds,
                    productive_learned_updates=accounting.productive_learned_updates,
                    sink_learned_updates=accounting.sink_learned_updates,
                    total_learned_updates=accounting.total_learned_updates,
                    reserve_population_by_round=tuple(reserve_counts[world_index]),
                    unique_reserve_population_by_round=tuple(unique_counts[world_index]),
                    generated_terminal_count=len(terminals[world_index]),
                    unique_generated_terminal_count=len(set(terminals[world_index])),
                ),
            )
        )

    return Gate3V1BatchResult(
        depth=depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
        world_seeds=tuple(world.seed for world in materialized),
        world_results=tuple(results),
    )
