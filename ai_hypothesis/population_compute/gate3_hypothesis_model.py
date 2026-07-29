"""Shared neural hypothesis scorer and matched Gate-3 runtime schedules.

No training or scientific result is defined here. The module establishes one fixed-size learned
function reused across every runtime hypothesis state plus the stable/collapsed/reshuffled
execution semantics frozen by the Gate-3 protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from .gate3_hypothesis_population import (
    GATE3_DEPTHS,
    Gate3ConditionPlan,
    Gate3ControlMode,
    Gate3Observation,
    Gate3ObservationKind,
    Gate3World,
    build_gate3_condition_plan,
    deterministic_tie_break,
    reshuffled_state_permutation,
)


GATE3_MAX_PHASES = 2 * max(GATE3_DEPTHS)
GATE3_DEPTH_FEATURE_WIDTH = len(GATE3_DEPTHS)
GATE3_KIND_FEATURE_WIDTH = 2
GATE3_BIT_FEATURE_WIDTH = 2
GATE3_ACTION_FEATURE_WIDTH = 3  # branch 0 / branch 1 / no branch
GATE3_INPUT_WIDTH = (
    GATE3_MAX_PHASES
    + GATE3_DEPTH_FEATURE_WIDTH
    + GATE3_KIND_FEATURE_WIDTH
    + GATE3_BIT_FEATURE_WIDTH
    + GATE3_ACTION_FEATURE_WIDTH
)


@dataclass(frozen=True, slots=True)
class Gate3HypothesisModelConfig:
    input_projection_width: int = 32
    state_width: int = 64

    def validate(self) -> None:
        if self.input_projection_width <= 0:
            raise ValueError("input_projection_width must be positive")
        if self.state_width <= 0:
            raise ValueError("state_width must be positive")


class Gate3HypothesisScorer(nn.Module):
    """One shared ~20K-parameter recurrent scorer reused across all hypothesis states."""

    def __init__(self, config: Gate3HypothesisModelConfig = Gate3HypothesisModelConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(GATE3_INPUT_WIDTH, config.input_projection_width)
        # Standard eager PyTorch GRU sequence primitive. This is deliberately not torch.compile,
        # CUDA graphs or a compiler experiment; it simply executes repeated recurrent updates
        # without a Python kernel-launch loop.
        self.update = nn.GRU(
            config.input_projection_width,
            config.state_width,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.state_width)
        self.score_head = nn.Linear(config.state_width, 1)

    def advance(
        self,
        state: torch.Tensor,
        phase_input: torch.Tensor,
        *,
        repeats: int,
    ) -> torch.Tensor:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if state.ndim != 2 or phase_input.ndim != 2:
            raise ValueError("Gate-3 recurrent state/input tensors must be rank two")
        if state.shape[0] != phase_input.shape[0]:
            raise ValueError("Gate-3 recurrent state/input batch sizes must match")
        projected = torch.nn.functional.silu(self.input_projection(phase_input))
        sequence = projected.unsqueeze(1).expand(-1, repeats, -1).contiguous()
        _, final_state = self.update(sequence, state.unsqueeze(0))
        return final_state[0]

    def step(self, state: torch.Tensor, phase_input: torch.Tensor) -> torch.Tensor:
        return self.advance(state, phase_input, repeats=1)

    def score(self, state: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.output_norm(state)).squeeze(-1)

    def initial_state(self, count: int, *, device: torch.device | str) -> torch.Tensor:
        if count <= 0:
            raise ValueError("initial state count must be positive")
        return torch.zeros((count, self.config.state_width), dtype=torch.float32, device=device)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            detached = tensor.detach().cpu().contiguous().clone()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("ascii"))
            digest.update(str(tuple(detached.shape)).encode("ascii"))
            digest.update(bytes(detached.untyped_storage()))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Gate3Candidate:
    path: tuple[int, ...]
    state: torch.Tensor
    score: float


@dataclass(frozen=True, slots=True)
class Gate3RunTelemetry:
    depth: int
    width: int
    mode: Gate3ControlMode
    learned_updates_per_phase: tuple[int, ...]
    learned_updates_total: int
    retained_state_slots_by_phase: tuple[int, ...]
    unique_candidate_count_by_phase: tuple[int, ...]
    correct_candidate_present_by_phase: tuple[bool, ...]

    def validate(self, *, plan: Gate3ConditionPlan) -> None:
        if self.depth != plan.depth or self.width != plan.width or self.mode is not plan.mode:
            raise ValueError("Gate-3 telemetry does not belong to the supplied plan")
        if self.learned_updates_per_phase != tuple(phase.learned_updates_in_phase for phase in plan.phases):
            raise ValueError("Gate-3 runtime phase work differs from the frozen plan")
        if self.learned_updates_total != plan.learned_update_count:
            raise ValueError("Gate-3 runtime total learned work differs from the frozen plan")
        if self.retained_state_slots_by_phase != tuple(phase.retained_state_slots_after for phase in plan.phases):
            raise ValueError("Gate-3 runtime state population differs from the frozen plan")
        if len(self.unique_candidate_count_by_phase) != len(plan.phases):
            raise ValueError("Gate-3 unique-candidate telemetry is incomplete")
        if len(self.correct_candidate_present_by_phase) != len(plan.phases):
            raise ValueError("Gate-3 candidate-survival telemetry is incomplete")
        if any(count <= 0 or count > self.width for count in self.unique_candidate_count_by_phase):
            raise ValueError("Gate-3 unique-candidate count is outside the physical population")


@dataclass(frozen=True, slots=True)
class Gate3RunResult:
    predicted_path: tuple[int, ...]
    answer_path: tuple[int, ...]
    exact_solved: bool
    bit_accuracy: float
    final_score: float
    telemetry: Gate3RunTelemetry


def encode_gate3_phase_input(
    *,
    depth: int,
    observation: Gate3Observation,
    branch_action: int | None,
    device: torch.device | str,
) -> torch.Tensor:
    if depth not in GATE3_DEPTHS:
        raise ValueError("depth is outside the frozen Gate-3 ladder")
    observation.validate(depth=depth)
    if branch_action not in {None, 0, 1}:
        raise ValueError("branch_action must be 0, 1 or None")
    if observation.kind is Gate3ObservationKind.BRANCH_HINT and branch_action is None:
        raise ValueError("branch-hint phase requires a proposed branch action")
    if observation.kind is Gate3ObservationKind.DELAYED_REVEAL and branch_action is not None:
        raise ValueError("reveal phase must not carry a branch action")

    vector = torch.zeros(GATE3_INPUT_WIDTH, dtype=torch.float32, device=device)
    cursor = 0

    vector[cursor + observation.phase_index] = 1.0
    cursor += GATE3_MAX_PHASES

    vector[cursor + GATE3_DEPTHS.index(depth)] = 1.0
    cursor += GATE3_DEPTH_FEATURE_WIDTH

    kind_index = 0 if observation.kind is Gate3ObservationKind.BRANCH_HINT else 1
    vector[cursor + kind_index] = 1.0
    cursor += GATE3_KIND_FEATURE_WIDTH

    vector[cursor + observation.observed_bit] = 1.0
    cursor += GATE3_BIT_FEATURE_WIDTH

    action_index = 2 if branch_action is None else branch_action
    vector[cursor + action_index] = 1.0
    return vector


def _repeat_update(
    model: Gate3HypothesisScorer,
    states: torch.Tensor,
    phase_inputs: torch.Tensor,
    *,
    repeats: int,
) -> torch.Tensor:
    return model.advance(states, phase_inputs, repeats=repeats)


def _rank_candidates(
    candidates: list[Gate3Candidate],
    *,
    world_seed: int,
    phase_index: int,
) -> list[Gate3Candidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            deterministic_tie_break(
                world_seed=world_seed,
                phase_index=phase_index,
                candidate_path=candidate.path,
            ),
        ),
    )


def _collapse_candidates(candidates: list[Gate3Candidate], *, count: int) -> list[Gate3Candidate]:
    if not candidates or count <= 0:
        raise ValueError("collapsed Gate-3 population requires a positive retained count")
    top = candidates[0]
    return [
        Gate3Candidate(path=top.path, state=top.state.clone(), score=top.score)
        for _ in range(count)
    ]


def _reshuffle_candidate_states(
    candidates: list[Gate3Candidate],
    *,
    world_seed: int,
    phase_index: int,
) -> list[Gate3Candidate]:
    permutation = reshuffled_state_permutation(
        world_seed=world_seed,
        phase_index=phase_index,
        state_count=len(candidates),
    )
    states = [candidate.state for candidate in candidates]
    return [
        Gate3Candidate(path=candidate.path, state=states[permutation[index]], score=candidate.score)
        for index, candidate in enumerate(candidates)
    ]


def run_gate3_world(
    model: Gate3HypothesisScorer,
    world: Gate3World,
    *,
    width: int,
    mode: Gate3ControlMode,
    device: torch.device | str = "cpu",
) -> Gate3RunResult:
    """Execute one frozen Gate-3 world without observation replay or oracle metadata checks."""

    world.validate()
    target_device = torch.device(device)
    plan = build_gate3_condition_plan(world, width=width, mode=mode)
    plan.validate()
    model = model.to(target_device)

    candidates = [
        Gate3Candidate(
            path=(),
            state=model.initial_state(1, device=target_device)[0],
            score=0.0,
        )
    ]
    phase_updates: list[int] = []
    retained_counts: list[int] = []
    unique_counts: list[int] = []
    correct_present: list[bool] = []

    with torch.inference_mode():
        for phase, observation in zip(plan.phases, world.observations, strict=True):
            if phase.kind is Gate3ObservationKind.BRANCH_HINT:
                expanded_paths: list[tuple[int, ...]] = []
                expanded_states: list[torch.Tensor] = []
                expanded_inputs: list[torch.Tensor] = []
                for candidate in candidates:
                    for branch_action in (0, 1):
                        expanded_paths.append(candidate.path + (branch_action,))
                        expanded_states.append(candidate.state.clone())
                        expanded_inputs.append(
                            encode_gate3_phase_input(
                                depth=world.depth,
                                observation=observation,
                                branch_action=branch_action,
                                device=target_device,
                            )
                        )
                if len(expanded_paths) != phase.evaluated_state_slots:
                    raise RuntimeError("Gate-3 runtime branch fanout differs from frozen phase plan")
                states = torch.stack(expanded_states, dim=0)
                inputs = torch.stack(expanded_inputs, dim=0)
                states = _repeat_update(
                    model,
                    states,
                    inputs,
                    repeats=phase.recurrent_updates_per_evaluated_state,
                )
                scores = model.score(states)
                expanded = [
                    Gate3Candidate(
                        path=path,
                        state=states[index].clone(),
                        score=float(scores[index].item()),
                    )
                    for index, path in enumerate(expanded_paths)
                ]
                ranked = _rank_candidates(
                    expanded,
                    world_seed=world.seed,
                    phase_index=phase.phase_index,
                )
                candidates = ranked[: phase.retained_state_slots_after]
                if mode is Gate3ControlMode.COLLAPSED_DIVERSITY:
                    candidates = _collapse_candidates(candidates, count=phase.retained_state_slots_after)
            else:
                states = torch.stack([candidate.state for candidate in candidates], dim=0)
                phase_input = encode_gate3_phase_input(
                    depth=world.depth,
                    observation=observation,
                    branch_action=None,
                    device=target_device,
                )
                inputs = phase_input.unsqueeze(0).expand(len(candidates), -1)
                states = _repeat_update(
                    model,
                    states,
                    inputs,
                    repeats=phase.recurrent_updates_per_evaluated_state,
                )
                scores = model.score(states)
                candidates = [
                    Gate3Candidate(
                        path=candidate.path,
                        state=states[index].clone(),
                        score=float(scores[index].item()),
                    )
                    for index, candidate in enumerate(candidates)
                ]

            if mode is Gate3ControlMode.RESHUFFLED_CONTINUITY and phase.phase_index < len(plan.phases) - 1:
                candidates = _reshuffle_candidate_states(
                    candidates,
                    world_seed=world.seed,
                    phase_index=phase.phase_index,
                )

            phase_updates.append(
                phase.evaluated_state_slots * phase.recurrent_updates_per_evaluated_state
            )
            retained_counts.append(len(candidates))
            unique_counts.append(len({candidate.path for candidate in candidates}))
            correct_prefix_length = min(world.depth, phase.phase_index + 1)
            if phase.phase_index >= world.depth:
                correct_prefix_length = world.depth
            correct_prefix = world.hidden_path[:correct_prefix_length]
            correct_present.append(
                any(candidate.path[:correct_prefix_length] == correct_prefix for candidate in candidates)
            )

    ranked_final = _rank_candidates(
        candidates,
        world_seed=world.seed,
        phase_index=len(plan.phases),
    )
    winner = ranked_final[0]
    predicted = winner.path
    if len(predicted) != world.depth:
        raise RuntimeError("Gate-3 final hypothesis path is incomplete")
    bit_accuracy = sum(int(a == b) for a, b in zip(predicted, world.hidden_path, strict=True)) / world.depth
    telemetry = Gate3RunTelemetry(
        depth=world.depth,
        width=width,
        mode=mode,
        learned_updates_per_phase=tuple(phase_updates),
        learned_updates_total=sum(phase_updates),
        retained_state_slots_by_phase=tuple(retained_counts),
        unique_candidate_count_by_phase=tuple(unique_counts),
        correct_candidate_present_by_phase=tuple(correct_present),
    )
    telemetry.validate(plan=plan)
    return Gate3RunResult(
        predicted_path=predicted,
        answer_path=world.hidden_path,
        exact_solved=predicted == world.hidden_path,
        bit_accuracy=bit_accuracy,
        final_score=winner.score,
        telemetry=telemetry,
    )
