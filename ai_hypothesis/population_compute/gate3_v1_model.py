"""Shared recurrent scorer and reference runtime for Gate-3 v1 sparse-active search.

No training or development evidence is defined here. The reference runtime intentionally accepts
only ``Gate3V1PublicWorld`` so hidden answers cannot influence search decisions.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import torch
from torch import nn

from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_DEPTHS,
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    Gate3V1ControlMode,
    Gate3V1PublicWorld,
    build_gate3_v1_condition_plan,
    deterministic_gate3_v1_tie_break,
    make_gate3_v1_accounting,
    quantize_gate3_v1_score,
)


GATE3_V1_MAX_DEPTH = max(GATE3_V1_DEPTHS)
GATE3_V1_DEPTH_FEATURE_WIDTH = len(GATE3_V1_DEPTHS)
GATE3_V1_HINT_FEATURE_WIDTH = 3  # bit 0 / bit 1 / sink
GATE3_V1_ACTION_FEATURE_WIDTH = 3  # branch 0 / branch 1 / sink
GATE3_V1_INPUT_WIDTH = (
    GATE3_V1_MAX_DEPTH
    + GATE3_V1_DEPTH_FEATURE_WIDTH
    + GATE3_V1_HINT_FEATURE_WIDTH
    + GATE3_V1_ACTION_FEATURE_WIDTH
)


@dataclass(frozen=True, slots=True)
class Gate3V1ModelConfig:
    input_projection_width: int = 32
    state_width: int = 64

    def validate(self) -> None:
        if self.input_projection_width <= 0:
            raise ValueError("input_projection_width must be positive")
        if self.state_width <= 0:
            raise ValueError("state_width must be positive")


class Gate3V1Scorer(nn.Module):
    """One shared recurrent prefix scorer reused across all reserve capacities."""

    def __init__(self, config: Gate3V1ModelConfig = Gate3V1ModelConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(GATE3_V1_INPUT_WIDTH, config.input_projection_width)
        self.update = nn.GRU(
            config.input_projection_width,
            config.state_width,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.state_width)
        self.score_head = nn.Linear(config.state_width, 1)

    def initial_state(self, count: int, *, device: torch.device | str) -> torch.Tensor:
        if count <= 0:
            raise ValueError("initial-state count must be positive")
        return torch.zeros((count, self.config.state_width), dtype=torch.float32, device=device)

    def advance(
        self,
        state: torch.Tensor,
        phase_input: torch.Tensor,
        *,
        repeats: int = GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    ) -> torch.Tensor:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if state.ndim != 2 or phase_input.ndim != 2:
            raise ValueError("state and input tensors must be rank two")
        if state.shape[0] != phase_input.shape[0]:
            raise ValueError("state/input batch sizes must match")
        projected = torch.nn.functional.silu(self.input_projection(phase_input))
        sequence = projected.unsqueeze(1).expand(-1, repeats, -1).contiguous()
        _, final_state = self.update(sequence, state.unsqueeze(0))
        return final_state[0]

    def score(self, state: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.output_norm(state)).squeeze(-1)

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
class Gate3V1NeuralCandidate:
    path: tuple[int, ...]
    state: torch.Tensor
    score: float

    @property
    def depth(self) -> int:
        return len(self.path)


@dataclass(frozen=True, slots=True)
class Gate3V1RuntimeTelemetry:
    depth: int
    reserve_capacity: int
    mode: Gate3V1ControlMode
    productive_rounds: int
    sink_rounds: int
    productive_learned_updates: int
    sink_learned_updates: int
    total_learned_updates: int
    reserve_population_by_round: tuple[int, ...]
    unique_reserve_population_by_round: tuple[int, ...]
    generated_terminal_count: int
    unique_generated_terminal_count: int


@dataclass(frozen=True, slots=True)
class Gate3V1RuntimeResult:
    generated_terminal_paths: tuple[tuple[int, ...], ...]
    telemetry: Gate3V1RuntimeTelemetry


def encode_gate3_v1_child_input(
    *,
    world: Gate3V1PublicWorld,
    child_depth: int,
    observed_hint: int | None,
    branch_action: int | None,
    sink: bool,
    device: torch.device | str,
) -> torch.Tensor:
    world.validate()
    if not 1 <= child_depth <= world.depth:
        raise ValueError("child depth is outside the public world")
    if sink:
        if observed_hint is not None or branch_action is not None:
            raise ValueError("sink input cannot carry world evidence or branch action")
    else:
        if observed_hint not in (0, 1) or branch_action not in (0, 1):
            raise ValueError("productive child input requires binary hint/action")

    vector = torch.zeros(GATE3_V1_INPUT_WIDTH, dtype=torch.float32, device=device)
    cursor = 0
    vector[cursor + child_depth - 1] = 1.0
    cursor += GATE3_V1_MAX_DEPTH
    vector[cursor + GATE3_V1_DEPTHS.index(world.depth)] = 1.0
    cursor += GATE3_V1_DEPTH_FEATURE_WIDTH
    hint_index = 2 if sink else int(observed_hint)
    vector[cursor + hint_index] = 1.0
    cursor += GATE3_V1_HINT_FEATURE_WIDTH
    action_index = 2 if sink else int(branch_action)
    vector[cursor + action_index] = 1.0
    return vector


def _rank_neural_candidates(
    candidates: tuple[Gate3V1NeuralCandidate, ...],
    *,
    world_seed: int,
    expansion_index: int,
) -> tuple[Gate3V1NeuralCandidate, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -quantize_gate3_v1_score(candidate.score),
                deterministic_gate3_v1_tie_break(
                    world_seed=world_seed,
                    expansion_index=expansion_index,
                    candidate_path=candidate.path,
                ),
            ),
        )
    )


def _reshuffle_neural_histories(
    candidates: tuple[Gate3V1NeuralCandidate, ...],
    *,
    world_seed: int,
    expansion_index: int,
) -> tuple[Gate3V1NeuralCandidate, ...]:
    if len(candidates) <= 1:
        return candidates
    digest = hashlib.sha256(
        f"gate3-v1-neural-reshuffle:{world_seed}:{expansion_index}:{len(candidates)}".encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    permutation = list(range(len(candidates)))
    rng.shuffle(permutation)
    histories = tuple((candidate.state, candidate.score) for candidate in candidates)
    return tuple(
        Gate3V1NeuralCandidate(
            path=candidate.path,
            state=histories[permutation[index]][0].clone(),
            score=histories[permutation[index]][1],
        )
        for index, candidate in enumerate(candidates)
    )


def _apply_neural_control(
    candidates: tuple[Gate3V1NeuralCandidate, ...],
    *,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    world_seed: int,
    expansion_index: int,
) -> tuple[Gate3V1NeuralCandidate, ...]:
    if not candidates:
        return ()
    retained = _rank_neural_candidates(
        candidates,
        world_seed=world_seed,
        expansion_index=expansion_index,
    )[:reserve_capacity]
    if mode is Gate3V1ControlMode.STABLE_RESERVE:
        return retained
    if mode is Gate3V1ControlMode.COLLAPSED_DIVERSITY:
        top = retained[0]
        return tuple(
            Gate3V1NeuralCandidate(path=top.path, state=top.state.clone(), score=top.score)
            for _ in retained
        )
    if mode is Gate3V1ControlMode.RESHUFFLED_CONTINUITY:
        return _reshuffle_neural_histories(
            retained,
            world_seed=world_seed,
            expansion_index=expansion_index,
        )
    raise ValueError(f"unknown Gate-3 v1 control mode: {mode}")


def run_gate3_v1_public_world(
    model: Gate3V1Scorer,
    world: Gate3V1PublicWorld,
    *,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    device: torch.device | str = "cpu",
) -> Gate3V1RuntimeResult:
    """Run answer-blind best-first search on one public world."""

    world.validate()
    target_device = torch.device(device)
    plan = build_gate3_v1_condition_plan(
        depth=world.depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
    )
    model = model.to(target_device)
    reserve: tuple[Gate3V1NeuralCandidate, ...] = (
        Gate3V1NeuralCandidate(
            path=(),
            state=model.initial_state(1, device=target_device)[0],
            score=0.0,
        ),
    )
    generated_terminals: list[tuple[int, ...]] = []
    reserve_counts: list[int] = []
    unique_counts: list[int] = []
    productive_rounds = 0

    with torch.inference_mode():
        for expansion_index in range(plan.search_rounds):
            nonterminal = tuple(candidate for candidate in reserve if candidate.depth < world.depth)
            if not nonterminal:
                sink_state = model.initial_state(2, device=target_device)
                sink_input = torch.stack(
                    [
                        encode_gate3_v1_child_input(
                            world=world,
                            child_depth=world.depth,
                            observed_hint=None,
                            branch_action=None,
                            sink=True,
                            device=target_device,
                        )
                        for _ in range(2)
                    ],
                    dim=0,
                )
                model.advance(
                    sink_state,
                    sink_input,
                    repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
                )
                reserve_counts.append(len(reserve))
                unique_counts.append(len({candidate.path for candidate in reserve}))
                continue

            ranked = _rank_neural_candidates(
                nonterminal,
                world_seed=world.seed,
                expansion_index=expansion_index,
            )
            parent = ranked[0]
            removed = False
            remaining: list[Gate3V1NeuralCandidate] = []
            for candidate in reserve:
                if not removed and candidate is parent:
                    removed = True
                    continue
                remaining.append(candidate)
            if not removed:
                # Identity can be duplicated by the collapsed control. Fall back to exact field
                # identity and remove only one slot.
                remaining = list(reserve)
                remaining.remove(parent)

            next_depth = parent.depth + 1
            hint = world.noisy_hints[next_depth - 1]
            child_states = torch.stack((parent.state.clone(), parent.state.clone()), dim=0)
            child_inputs = torch.stack(
                [
                    encode_gate3_v1_child_input(
                        world=world,
                        child_depth=next_depth,
                        observed_hint=hint,
                        branch_action=action,
                        sink=False,
                        device=target_device,
                    )
                    for action in (0, 1)
                ],
                dim=0,
            )
            child_states = model.advance(
                child_states,
                child_inputs,
                repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            )
            child_scores = model.score(child_states)
            for action in (0, 1):
                path = parent.path + (action,)
                if next_depth == world.depth:
                    generated_terminals.append(path)
                else:
                    remaining.append(
                        Gate3V1NeuralCandidate(
                            path=path,
                            state=child_states[action].clone(),
                            score=float(child_scores[action].item()),
                        )
                    )
            reserve = _apply_neural_control(
                tuple(remaining),
                reserve_capacity=reserve_capacity,
                mode=mode,
                world_seed=world.seed,
                expansion_index=expansion_index,
            )
            productive_rounds += 1
            reserve_counts.append(len(reserve))
            unique_counts.append(len({candidate.path for candidate in reserve}))

    accounting = make_gate3_v1_accounting(
        depth=world.depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
        productive_rounds=productive_rounds,
    )
    telemetry = Gate3V1RuntimeTelemetry(
        depth=world.depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
        productive_rounds=accounting.productive_rounds,
        sink_rounds=accounting.sink_rounds,
        productive_learned_updates=accounting.productive_learned_updates,
        sink_learned_updates=accounting.sink_learned_updates,
        total_learned_updates=accounting.total_learned_updates,
        reserve_population_by_round=tuple(reserve_counts),
        unique_reserve_population_by_round=tuple(unique_counts),
        generated_terminal_count=len(generated_terminals),
        unique_generated_terminal_count=len(set(generated_terminals)),
    )
    return Gate3V1RuntimeResult(
        generated_terminal_paths=tuple(generated_terminals),
        telemetry=telemetry,
    )
