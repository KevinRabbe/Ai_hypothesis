"""Batched eager evaluator for the frozen Gate-3 hypothesis-population runtime.

Batching changes only execution organization across independent worlds. Per-world observations,
hypothesis population, control semantics and learned recurrent-update counts are identical to the
single-world reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .gate3_hypothesis_model import (
    GATE3_ACTION_FEATURE_WIDTH,
    GATE3_BIT_FEATURE_WIDTH,
    GATE3_DEPTH_FEATURE_WIDTH,
    GATE3_INPUT_WIDTH,
    GATE3_KIND_FEATURE_WIDTH,
    GATE3_MAX_PHASES,
    Gate3HypothesisScorer,
)
from .gate3_hypothesis_population import (
    GATE3_DEPTHS,
    Gate3ControlMode,
    Gate3ObservationKind,
    Gate3World,
    build_gate3_condition_plan,
    deterministic_tie_break,
    reshuffled_state_permutation,
)


@dataclass(frozen=True, slots=True)
class Gate3BatchRunResult:
    depth: int
    width: int
    mode: Gate3ControlMode
    world_seeds: tuple[int, ...]
    predicted_paths: tuple[tuple[int, ...], ...]
    answer_paths: tuple[tuple[int, ...], ...]
    exact_solved_by_world: tuple[bool, ...]
    bit_accuracy_by_world: tuple[float, ...]
    correct_candidate_present_by_world_by_phase: tuple[tuple[bool, ...], ...]
    unique_candidate_count_by_world_by_phase: tuple[tuple[int, ...], ...]
    learned_updates_per_world: int
    unique_world_observations_per_world: int

    @property
    def exact_solve_rate(self) -> float:
        return sum(self.exact_solved_by_world) / len(self.exact_solved_by_world)

    @property
    def bit_accuracy(self) -> float:
        return sum(self.bit_accuracy_by_world) / len(self.bit_accuracy_by_world)


def _batch_phase_inputs(
    *,
    depth: int,
    phase_index: int,
    kind: Gate3ObservationKind,
    observed_bits: torch.Tensor,
    action_bits: torch.Tensor | None,
    candidate_count: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = observed_bits.shape[0]
    inputs = torch.zeros(
        (batch_size, candidate_count, GATE3_INPUT_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    cursor = 0
    inputs[:, :, cursor + phase_index] = 1.0
    cursor += GATE3_MAX_PHASES

    depth_index = GATE3_DEPTHS.index(depth)
    inputs[:, :, cursor + depth_index] = 1.0
    cursor += GATE3_DEPTH_FEATURE_WIDTH

    kind_index = 0 if kind is Gate3ObservationKind.BRANCH_HINT else 1
    inputs[:, :, cursor + kind_index] = 1.0
    cursor += GATE3_KIND_FEATURE_WIDTH

    bit_index = observed_bits.view(batch_size, 1, 1).expand(-1, candidate_count, 1)
    inputs[:, :, cursor : cursor + GATE3_BIT_FEATURE_WIDTH].scatter_(2, bit_index, 1.0)
    cursor += GATE3_BIT_FEATURE_WIDTH

    if action_bits is None:
        inputs[:, :, cursor + 2] = 1.0
    else:
        if action_bits.shape != (batch_size, candidate_count):
            raise ValueError("Gate-3 batched action tensor has invalid shape")
        action_index = action_bits.unsqueeze(-1)
        inputs[:, :, cursor : cursor + GATE3_ACTION_FEATURE_WIDTH].scatter_(2, action_index, 1.0)
    return inputs


def _gather_candidate_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError("candidate tensor must have at least two dimensions")
    if indices.ndim != 2 or indices.shape[0] != tensor.shape[0]:
        raise ValueError("candidate gather indices have invalid shape")
    expand_shape = list(indices.shape) + list(tensor.shape[2:])
    gather_index = indices
    for _ in tensor.shape[2:]:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(*expand_shape)
    return torch.gather(tensor, 1, gather_index)


def _rank_indices(
    scores: torch.Tensor,
    paths: torch.Tensor,
    *,
    worlds: tuple[Gate3World, ...],
    phase_index: int,
    keep: int,
    prefix_length: int,
) -> torch.Tensor:
    scores_cpu = scores.detach().cpu().tolist()
    paths_cpu = paths.detach().cpu().tolist()
    rows: list[list[int]] = []
    for world_index, world in enumerate(worlds):
        ordering = sorted(
            range(len(scores_cpu[world_index])),
            key=lambda candidate_index: (
                -float(scores_cpu[world_index][candidate_index]),
                deterministic_tie_break(
                    world_seed=world.seed,
                    phase_index=phase_index,
                    candidate_path=tuple(
                        int(bit)
                        for bit in paths_cpu[world_index][candidate_index][:prefix_length]
                    ),
                ),
            ),
        )
        rows.append(ordering[:keep])
    return torch.tensor(rows, dtype=torch.int64, device=scores.device)


def _reshuffle_states(
    states: torch.Tensor,
    *,
    worlds: tuple[Gate3World, ...],
    phase_index: int,
) -> torch.Tensor:
    rows = [
        reshuffled_state_permutation(
            world_seed=world.seed,
            phase_index=phase_index,
            state_count=states.shape[1],
        )
        for world in worlds
    ]
    indices = torch.tensor(rows, dtype=torch.int64, device=states.device)
    return _gather_candidate_axis(states, indices)


def _phase_diagnostics(
    paths: torch.Tensor,
    *,
    worlds: tuple[Gate3World, ...],
    prefix_length: int,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    paths_cpu = paths.detach().cpu().tolist()
    unique_counts: list[int] = []
    correct_present: list[bool] = []
    for world_index, world in enumerate(worlds):
        prefixes = [
            tuple(int(bit) for bit in path[:prefix_length])
            for path in paths_cpu[world_index]
        ]
        unique_counts.append(len(set(prefixes)))
        correct = world.hidden_path[:prefix_length]
        correct_present.append(correct in prefixes)
    return tuple(unique_counts), tuple(correct_present)


def run_gate3_world_batch(
    model: Gate3HypothesisScorer,
    worlds: Iterable[Gate3World],
    *,
    width: int,
    mode: Gate3ControlMode,
    device: torch.device | str = "cpu",
) -> Gate3BatchRunResult:
    materialized = tuple(worlds)
    if not materialized:
        raise ValueError("at least one Gate-3 world is required")
    first = materialized[0]
    first.validate()
    for world in materialized[1:]:
        world.validate()
        if world.depth != first.depth:
            raise ValueError("one Gate-3 batch must use one hidden depth")

    target_device = torch.device(device)
    model = model.to(target_device)
    plans = tuple(build_gate3_condition_plan(world, width=width, mode=mode) for world in materialized)
    reference_plan = plans[0]
    if any(plan.mechanical_signature() != reference_plan.mechanical_signature() for plan in plans[1:]):
        # The world seed and observation signature differ across worlds, so compare only phase mechanics.
        reference_phase_mechanics = tuple(
            (
                phase.kind,
                phase.active_state_slots_before,
                phase.evaluated_state_slots,
                phase.retained_state_slots_after,
                phase.recurrent_updates_per_evaluated_state,
                phase.learned_updates_in_phase,
            )
            for phase in reference_plan.phases
        )
        for plan in plans[1:]:
            phase_mechanics = tuple(
                (
                    phase.kind,
                    phase.active_state_slots_before,
                    phase.evaluated_state_slots,
                    phase.retained_state_slots_after,
                    phase.recurrent_updates_per_evaluated_state,
                    phase.learned_updates_in_phase,
                )
                for phase in plan.phases
            )
            if phase_mechanics != reference_phase_mechanics:
                raise RuntimeError("Gate-3 batched worlds do not share one frozen phase schedule")

    batch_size = len(materialized)
    depth = first.depth
    states = model.initial_state(batch_size, device=target_device).unsqueeze(1)
    paths = torch.full((batch_size, 1, depth), -1, dtype=torch.int64, device=target_device)
    scores = torch.zeros((batch_size, 1), dtype=torch.float32, device=target_device)

    unique_by_phase_per_world: list[list[int]] = [[] for _ in materialized]
    correct_by_phase_per_world: list[list[bool]] = [[] for _ in materialized]

    with torch.inference_mode():
        for phase in reference_plan.phases:
            observations = tuple(world.observations[phase.phase_index] for world in materialized)
            observed_bits = torch.tensor(
                [observation.observed_bit for observation in observations],
                dtype=torch.int64,
                device=target_device,
            )

            if phase.kind is Gate3ObservationKind.BRANCH_HINT:
                active_count = states.shape[1]
                if active_count != phase.active_state_slots_before:
                    raise RuntimeError("Gate-3 batched active population differs from frozen plan")
                states = states.repeat_interleave(2, dim=1)
                paths = paths.repeat_interleave(2, dim=1).clone()
                action_pattern = torch.tensor([0, 1], dtype=torch.int64, device=target_device).repeat(active_count)
                action_bits = action_pattern.unsqueeze(0).expand(batch_size, -1)
                paths[:, :, phase.phase_index] = action_bits
                inputs = _batch_phase_inputs(
                    depth=depth,
                    phase_index=phase.phase_index,
                    kind=phase.kind,
                    observed_bits=observed_bits,
                    action_bits=action_bits,
                    candidate_count=phase.evaluated_state_slots,
                    device=target_device,
                )
                flat_states = states.reshape(batch_size * phase.evaluated_state_slots, -1)
                flat_inputs = inputs.reshape(batch_size * phase.evaluated_state_slots, -1)
                flat_states = model.advance(
                    flat_states,
                    flat_inputs,
                    repeats=phase.recurrent_updates_per_evaluated_state,
                )
                states = flat_states.reshape(batch_size, phase.evaluated_state_slots, -1)
                scores = model.score(flat_states).reshape(batch_size, phase.evaluated_state_slots)
                keep_indices = _rank_indices(
                    scores,
                    paths,
                    worlds=materialized,
                    phase_index=phase.phase_index,
                    keep=phase.retained_state_slots_after,
                    prefix_length=phase.phase_index + 1,
                )
                states = _gather_candidate_axis(states, keep_indices)
                paths = _gather_candidate_axis(paths, keep_indices)
                scores = _gather_candidate_axis(scores, keep_indices)

                if mode is Gate3ControlMode.COLLAPSED_DIVERSITY:
                    states = states[:, :1, :].expand(-1, phase.retained_state_slots_after, -1).clone()
                    paths = paths[:, :1, :].expand(-1, phase.retained_state_slots_after, -1).clone()
                    scores = scores[:, :1].expand(-1, phase.retained_state_slots_after).clone()
            else:
                if states.shape[1] != width:
                    raise RuntimeError("Gate-3 reveal phase must retain the full frozen beam width")
                inputs = _batch_phase_inputs(
                    depth=depth,
                    phase_index=phase.phase_index,
                    kind=phase.kind,
                    observed_bits=observed_bits,
                    action_bits=None,
                    candidate_count=width,
                    device=target_device,
                )
                flat_states = states.reshape(batch_size * width, -1)
                flat_inputs = inputs.reshape(batch_size * width, -1)
                flat_states = model.advance(
                    flat_states,
                    flat_inputs,
                    repeats=phase.recurrent_updates_per_evaluated_state,
                )
                states = flat_states.reshape(batch_size, width, -1)
                scores = model.score(flat_states).reshape(batch_size, width)

            if mode is Gate3ControlMode.RESHUFFLED_CONTINUITY and phase.phase_index < len(reference_plan.phases) - 1:
                states = _reshuffle_states(
                    states,
                    worlds=materialized,
                    phase_index=phase.phase_index,
                )

            prefix_length = min(depth, phase.phase_index + 1)
            if phase.phase_index >= depth:
                prefix_length = depth
            unique_counts, correct_present = _phase_diagnostics(
                paths,
                worlds=materialized,
                prefix_length=prefix_length,
            )
            for world_index in range(batch_size):
                unique_by_phase_per_world[world_index].append(unique_counts[world_index])
                correct_by_phase_per_world[world_index].append(correct_present[world_index])

    final_indices = _rank_indices(
        scores,
        paths,
        worlds=materialized,
        phase_index=len(reference_plan.phases),
        keep=1,
        prefix_length=depth,
    )
    winning_paths = _gather_candidate_axis(paths, final_indices).squeeze(1).detach().cpu().tolist()
    predicted_paths = tuple(tuple(int(bit) for bit in path) for path in winning_paths)
    answer_paths = tuple(world.hidden_path for world in materialized)
    exact_solved = tuple(predicted == answer for predicted, answer in zip(predicted_paths, answer_paths, strict=True))
    bit_accuracy = tuple(
        sum(int(a == b) for a, b in zip(predicted, answer, strict=True)) / depth
        for predicted, answer in zip(predicted_paths, answer_paths, strict=True)
    )

    return Gate3BatchRunResult(
        depth=depth,
        width=width,
        mode=mode,
        world_seeds=tuple(world.seed for world in materialized),
        predicted_paths=predicted_paths,
        answer_paths=answer_paths,
        exact_solved_by_world=exact_solved,
        bit_accuracy_by_world=bit_accuracy,
        correct_candidate_present_by_world_by_phase=tuple(tuple(row) for row in correct_by_phase_per_world),
        unique_candidate_count_by_world_by_phase=tuple(tuple(row) for row in unique_by_phase_per_world),
        learned_updates_per_world=reference_plan.learned_update_count,
        unique_world_observations_per_world=reference_plan.unique_world_observation_count,
    )
