"""Strict admitted Gate-5 batched runtime.

Bounded conditions do not rank or otherwise inspect non-sampled candidate scores until after the
parent for the current slot has been irrevocably selected.  Any full-reserve score ranking below
that point is evaluation-only telemetry and cannot feed back into search state or the selection.
"""

from __future__ import annotations

from typing import Iterable

import torch

from .gate3_v1_model import Gate3V1NeuralCandidate, Gate3V1Scorer
from .gate3_v1_sparse_active_reserve import GATE3_V1_UPDATES_PER_ROUND
from .gate5_bounded_score_activation import (
    GATE5_CONDITIONS,
    GATE5_DEPTH,
    GATE5_EVAL_BATCH_SIZE,
    GATE5_RESERVE_CAPACITY,
    GATE5_SCHEDULED_SLOTS,
    GATE5_STAGE_A_FRONTIER,
    GATE5_STAGE_A_SLOTS,
    GATE5_STAGE_B_SLOTS,
    GATE5_TOTAL_LEARNED_UPDATES,
    GATE5_WORLD_COUNT,
    Gate5ConditionEvaluation,
    Gate5EvaluationWorld,
    Gate5SchedulerMode,
    Gate5WorldResult,
    Gate5WorldTelemetry,
    _BOUNDED_K,
    _advance_parent_batch,
    _bounded_visible_candidates,
    _hash_select,
    _score_rank,
    generate_gate5_development_world,
)


def run_gate5_strict_world_batch(
    model: Gate3V1Scorer,
    worlds: Iterable[Gate5EvaluationWorld],
    *,
    mode: Gate5SchedulerMode,
    device: torch.device | str,
) -> tuple[Gate5WorldResult, ...]:
    world_tuple = tuple(worlds)
    if not world_tuple:
        return ()
    if mode not in GATE5_CONDITIONS:
        raise ValueError("Gate-5 scheduler mode is outside the frozen matrix")
    for world in world_tuple:
        world.validate()

    target_device = torch.device(device)
    model = model.to(target_device)
    populations: list[tuple[Gate3V1NeuralCandidate, ...]] = [
        (
            Gate3V1NeuralCandidate(
                path=(),
                state=model.initial_state(1, device=target_device)[0],
                score=0.0,
            ),
        )
        for _ in world_tuple
    ]
    generated_terminals: list[list[tuple[int, ...]]] = [[] for _ in world_tuple]
    productive = [0 for _ in world_tuple]

    stage_b_live: list[list[int]] = [[] for _ in world_tuple]
    stage_b_depths: list[list[int]] = [[] for _ in world_tuple]
    stage_b_visible: list[list[int]] = [[] for _ in world_tuple]
    stage_b_score_obs: list[list[int]] = [[] for _ in world_tuple]
    selected_visible_rank: list[list[int]] = [[] for _ in world_tuple]
    selected_global_rank: list[list[int]] = [[] for _ in world_tuple]
    selected_paths: list[list[tuple[int, ...]]] = [[] for _ in world_tuple]

    with torch.inference_mode():
        # Stage A: all conditions build the exact same complete depth-6 frontier.
        for _parent_depth in range(6):
            selected = tuple(populations)
            children_by_world = _advance_parent_batch(
                model,
                world_tuple,
                selected,
                device=target_device,
            )
            next_populations: list[tuple[Gate3V1NeuralCandidate, ...]] = []
            for world_offset, (parents, children) in enumerate(
                zip(selected, children_by_world, strict=True)
            ):
                productive[world_offset] += len(parents)
                next_populations.append(tuple(children))
            populations = next_populations

        if any(len(population) != GATE5_STAGE_A_FRONTIER for population in populations):
            raise RuntimeError("Gate-5 Stage A did not create exactly 64 depth-6 hypotheses")
        if any(count != GATE5_STAGE_A_SLOTS for count in productive):
            raise RuntimeError("Gate-5 Stage A did not consume exactly 63 parent slots")

        # Stage B: strict selection-visible information boundary.
        for local_slot in range(GATE5_STAGE_B_SLOTS):
            absolute_slot = GATE5_STAGE_A_SLOTS + local_slot
            selected_rows: list[tuple[Gate3V1NeuralCandidate, ...]] = []
            for world_offset, (world, population) in enumerate(zip(world_tuple, populations, strict=True)):
                if not population:
                    raise RuntimeError("Gate-5 Stage B unexpectedly exhausted the live reserve")

                if mode is Gate5SchedulerMode.GLOBAL_SCORE:
                    # Full score visibility is the explicit reference condition.
                    global_ranked = _score_rank(
                        population,
                        world_seed=world.public.seed,
                        expansion_index=absolute_slot,
                    )
                    visible = global_ranked
                    parent = global_ranked[0]
                    score_observations = len(visible)
                    visible_rank = 1
                    global_rank = 1
                else:
                    # IMPORTANT: only the sampled subset is constructed before parent choice.
                    visible = _bounded_visible_candidates(
                        population,
                        mode=mode,
                        world_seed=world.public.seed,
                        slot_index=absolute_slot,
                    )
                    if mode is Gate5SchedulerMode.BOUNDED_HASH_K16:
                        # No neural score comparison before selection in the hash control.
                        parent = _hash_select(
                            visible,
                            world_seed=world.public.seed,
                            slot_index=absolute_slot,
                        )
                        score_observations = 0
                        # Post-decision diagnostic only.
                        visible_score_order = _score_rank(
                            visible,
                            world_seed=world.public.seed,
                            expansion_index=absolute_slot,
                        )
                        visible_rank = 1 + next(
                            index
                            for index, candidate in enumerate(visible_score_order)
                            if candidate.path == parent.path
                        )
                    else:
                        # Learned bounded selector sees only K sampled scores before choice.
                        visible_score_order = _score_rank(
                            visible,
                            world_seed=world.public.seed,
                            expansion_index=absolute_slot,
                        )
                        parent = visible_score_order[0]
                        score_observations = len(visible)
                        visible_rank = 1

                    # Full-reserve ranking occurs only after the bounded parent is fixed.
                    # It is evaluation-only telemetry and never influences search state.
                    global_ranked_after_decision = _score_rank(
                        population,
                        world_seed=world.public.seed,
                        expansion_index=absolute_slot,
                    )
                    global_rank = 1 + next(
                        index
                        for index, candidate in enumerate(global_ranked_after_decision)
                        if candidate.path == parent.path
                    )

                stage_b_live[world_offset].append(len(population))
                stage_b_depths[world_offset].append(parent.depth)
                stage_b_visible[world_offset].append(len(visible))
                stage_b_score_obs[world_offset].append(score_observations)
                selected_visible_rank[world_offset].append(visible_rank)
                selected_global_rank[world_offset].append(global_rank)
                selected_paths[world_offset].append(parent.path)

                selected_rows.append((parent,))
                populations[world_offset] = tuple(
                    candidate for candidate in population if candidate.path != parent.path
                )

            children_by_world = _advance_parent_batch(
                model,
                world_tuple,
                tuple(selected_rows),
                device=target_device,
            )
            next_populations = []
            for world_offset, (population, parent_tuple, children) in enumerate(
                zip(populations, selected_rows, children_by_world, strict=True)
            ):
                parent = parent_tuple[0]
                productive[world_offset] += 1
                if parent.depth + 1 == GATE5_DEPTH:
                    generated_terminals[world_offset].extend(child.path for child in children)
                    updated = population
                else:
                    updated = population + tuple(children)
                if len(updated) > GATE5_RESERVE_CAPACITY:
                    raise RuntimeError("Gate-5 live reserve exceeded frozen L256 capacity")
                next_populations.append(updated)
            populations = next_populations

    results: list[Gate5WorldResult] = []
    for world_offset in range(len(world_tuple)):
        if productive[world_offset] != GATE5_SCHEDULED_SLOTS:
            raise RuntimeError("Gate-5 admitted topology did not use exactly 159 productive slots")
        total_updates = productive[world_offset] * GATE3_V1_UPDATES_PER_ROUND
        if total_updates != GATE5_TOTAL_LEARNED_UPDATES:
            raise RuntimeError("Gate-5 learned-work identity was violated")
        if len(stage_b_live[world_offset]) != GATE5_STAGE_B_SLOTS:
            raise RuntimeError("Gate-5 Stage-B telemetry is incomplete")

        if mode in _BOUNDED_K:
            k = _BOUNDED_K[mode]
            for live_count, visible_count in zip(
                stage_b_live[world_offset], stage_b_visible[world_offset], strict=True
            ):
                if visible_count != min(k, live_count):
                    raise RuntimeError("Gate-5 bounded visibility violated frozen K")
        if mode is Gate5SchedulerMode.GLOBAL_SCORE:
            if stage_b_visible[world_offset] != stage_b_live[world_offset]:
                raise RuntimeError("Gate-5 global scheduler did not observe complete live reserve")
        if mode is Gate5SchedulerMode.BOUNDED_HASH_K16:
            if any(value != 0 for value in stage_b_score_obs[world_offset]):
                raise RuntimeError("Gate-5 hash control consumed neural-score observations")

        score_observations = tuple(stage_b_score_obs[world_offset])
        results.append(
            Gate5WorldResult(
                generated_terminal_paths=tuple(generated_terminals[world_offset]),
                telemetry=Gate5WorldTelemetry(
                    productive_slots=productive[world_offset],
                    sink_slots=0,
                    total_learned_updates=total_updates,
                    stage_a_frontier_width=GATE5_STAGE_A_FRONTIER,
                    stage_b_live_population_by_slot=tuple(stage_b_live[world_offset]),
                    stage_b_activated_parent_depth_by_slot=tuple(stage_b_depths[world_offset]),
                    stage_b_visible_candidate_count_by_slot=tuple(stage_b_visible[world_offset]),
                    stage_b_score_observation_count_by_slot=score_observations,
                    total_stage_b_score_observations=sum(score_observations),
                    max_stage_b_score_observations=max(score_observations, default=0),
                    selected_visible_score_rank_by_slot=tuple(selected_visible_rank[world_offset]),
                    selected_global_score_rank_by_slot=tuple(selected_global_rank[world_offset]),
                    selected_parent_paths_by_slot=tuple(selected_paths[world_offset]),
                    generated_terminal_count=len(generated_terminals[world_offset]),
                    unique_generated_terminal_count=len(set(generated_terminals[world_offset])),
                ),
            )
        )
    return tuple(results)


def evaluate_gate5_strict_condition(
    model: Gate3V1Scorer,
    *,
    checkpoint_index: int,
    mode: Gate5SchedulerMode,
    device: torch.device | str,
    world_count: int = GATE5_WORLD_COUNT,
    evaluation_batch_size: int = GATE5_EVAL_BATCH_SIZE,
) -> Gate5ConditionEvaluation:
    from .gate5_bounded_score_activation import GATE5_CHECKPOINT_INDICES

    if checkpoint_index not in GATE5_CHECKPOINT_INDICES:
        raise ValueError("Gate-5 checkpoint index must be 0, 1 or 2")
    if mode not in GATE5_CONDITIONS:
        raise ValueError("Gate-5 condition is outside the frozen matrix")
    if world_count != GATE5_WORLD_COUNT:
        raise ValueError("Gate-5 development must use exactly 256 worlds")
    if evaluation_batch_size != GATE5_EVAL_BATCH_SIZE:
        raise ValueError("Gate-5 development batch size is frozen at 64")

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

    for start in range(0, world_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, world_count)
        worlds = tuple(generate_gate5_development_world(world_index=index) for index in range(start, stop))
        batch = run_gate5_strict_world_batch(model, worlds, mode=mode, device=device)
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
        world_count=world_count,
        world_indices=tuple(range(world_count)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / world_count,
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
