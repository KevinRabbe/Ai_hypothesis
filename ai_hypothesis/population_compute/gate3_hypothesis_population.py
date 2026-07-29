"""Deterministic mechanics for Gate-3 hypothesis-population breadth.

This module contains no training loop and admits no scientific result. It freezes the world,
width ladder, control modes, information accounting and exact learned-update schedule used by
the Gate-3 delayed-hypothesis experiment.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum


GATE3_PROTOCOL_VERSION = "gate3-hypothesis-population-v0"
GATE3_DEPTHS = (4, 6, 8)
GATE3_WIDTHS_BY_DEPTH = {
    4: (1, 4, 16),
    6: (1, 4, 16, 64),
    8: (1, 4, 16, 64, 256),
}
GATE3_HINT_RELIABILITY = 0.70
GATE3_TRAIN_WORLD_START = 0
GATE3_DEVELOPMENT_WORLD_START = 1 << 30
GATE3_CONFIRMATION_WORLD_START = 1 << 31


class Gate3ControlMode(str, Enum):
    """Frozen Gate-3 capability modes."""

    STABLE_DIVERSE = "stable_diverse"
    COLLAPSED_DIVERSITY = "collapsed_diversity"
    RESHUFFLED_CONTINUITY = "reshuffled_continuity"


class Gate3ObservationKind(str, Enum):
    BRANCH_HINT = "branch_hint"
    DELAYED_REVEAL = "delayed_reveal"


@dataclass(frozen=True, slots=True)
class Gate3Observation:
    phase_index: int
    bit_index: int
    kind: Gate3ObservationKind
    observed_bit: int

    def validate(self, *, depth: int) -> None:
        if not 0 <= self.phase_index < 2 * depth:
            raise ValueError("phase_index is outside the frozen Gate-3 phase range")
        if not 0 <= self.bit_index < depth:
            raise ValueError("bit_index is outside the hidden path")
        if self.observed_bit not in {0, 1}:
            raise ValueError("Gate-3 observations must be binary")
        if self.phase_index < depth:
            if self.kind is not Gate3ObservationKind.BRANCH_HINT:
                raise ValueError("the first D phases must contain noisy branch hints")
            if self.bit_index != self.phase_index:
                raise ValueError("branch hints must be emitted in hidden-path order")
        else:
            if self.kind is not Gate3ObservationKind.DELAYED_REVEAL:
                raise ValueError("the final D phases must contain exact delayed reveals")
            if self.bit_index != self.phase_index - depth:
                raise ValueError("delayed reveals must be emitted in hidden-path order")

    def semantic_signature(self) -> tuple[int, int, str, int]:
        return (self.phase_index, self.bit_index, self.kind.value, self.observed_bit)


@dataclass(frozen=True, slots=True)
class Gate3World:
    seed: int
    depth: int
    hidden_path: tuple[int, ...]
    noisy_hints: tuple[int, ...]
    observations: tuple[Gate3Observation, ...]

    def validate(self) -> None:
        if self.depth not in GATE3_DEPTHS:
            raise ValueError("depth is outside the frozen Gate-3 difficulty ladder")
        if len(self.hidden_path) != self.depth:
            raise ValueError("hidden path length does not match depth")
        if len(self.noisy_hints) != self.depth:
            raise ValueError("noisy hint count does not match depth")
        if any(bit not in {0, 1} for bit in self.hidden_path):
            raise ValueError("hidden path must be binary")
        if any(bit not in {0, 1} for bit in self.noisy_hints):
            raise ValueError("noisy hints must be binary")
        if len(self.observations) != 2 * self.depth:
            raise ValueError("Gate-3 world must contain exactly D hints and D reveals")

        for expected_phase, observation in enumerate(self.observations):
            observation.validate(depth=self.depth)
            if observation.phase_index != expected_phase:
                raise ValueError("observations must be ordered by phase")
            if expected_phase < self.depth:
                if observation.observed_bit != self.noisy_hints[expected_phase]:
                    raise ValueError("branch-hint observation disagrees with world hint")
            else:
                bit_index = expected_phase - self.depth
                if observation.observed_bit != self.hidden_path[bit_index]:
                    raise ValueError("delayed reveal must exactly match the hidden path")

    @property
    def answer_path(self) -> tuple[int, ...]:
        return self.hidden_path

    @property
    def unique_world_observation_count(self) -> int:
        return 2 * self.depth

    @property
    def learned_updates_per_phase(self) -> int:
        return 1 << self.depth

    @property
    def learned_update_count(self) -> int:
        return self.learned_updates_per_phase * 2 * self.depth

    def observation_signature(self) -> tuple[tuple[int, int, str, int], ...]:
        return tuple(observation.semantic_signature() for observation in self.observations)


@dataclass(frozen=True, slots=True)
class Gate3PhasePlan:
    phase_index: int
    kind: Gate3ObservationKind
    active_state_slots_before: int
    evaluated_state_slots: int
    retained_state_slots_after: int
    recurrent_updates_per_evaluated_state: int
    learned_updates_in_phase: int

    def validate(self, *, depth: int, width: int) -> None:
        budget = 1 << depth
        if self.learned_updates_in_phase != budget:
            raise ValueError("every Gate-3 phase must consume the frozen learned-update budget")
        if self.active_state_slots_before <= 0:
            raise ValueError("Gate-3 phase cannot start with an empty population")
        if self.evaluated_state_slots <= 0:
            raise ValueError("Gate-3 phase must evaluate at least one state")
        if self.retained_state_slots_after <= 0 or self.retained_state_slots_after > width:
            raise ValueError("retained state count is outside the frozen width")
        if self.recurrent_updates_per_evaluated_state <= 0:
            raise ValueError("each evaluated state must receive at least one learned update")
        if self.evaluated_state_slots * self.recurrent_updates_per_evaluated_state != budget:
            raise ValueError("phase learned-work identity was violated")

        if self.phase_index < depth:
            if self.kind is not Gate3ObservationKind.BRANCH_HINT:
                raise ValueError("branch phase kind mismatch")
            if self.evaluated_state_slots != 2 * self.active_state_slots_before:
                raise ValueError("binary branching must evaluate exactly two children per active slot")
            expected_after = min(width, 1 << (self.phase_index + 1))
            if self.retained_state_slots_after != expected_after:
                raise ValueError("branch-phase retained population differs from frozen schedule")
        else:
            if self.kind is not Gate3ObservationKind.DELAYED_REVEAL:
                raise ValueError("reveal phase kind mismatch")
            if self.active_state_slots_before != width:
                raise ValueError("reveal phases must begin with the full frozen beam width")
            if self.evaluated_state_slots != width or self.retained_state_slots_after != width:
                raise ValueError("reveal phases must update and retain every beam slot")


@dataclass(frozen=True, slots=True)
class Gate3ConditionPlan:
    protocol_version: str
    world_seed: int
    depth: int
    width: int
    mode: Gate3ControlMode
    phases: tuple[Gate3PhasePlan, ...]
    observation_signature: tuple[tuple[int, int, str, int], ...]
    unique_world_observation_count: int
    learned_update_count: int

    def validate(self) -> None:
        if self.protocol_version != GATE3_PROTOCOL_VERSION:
            raise ValueError("unexpected Gate-3 protocol version")
        if self.depth not in GATE3_DEPTHS:
            raise ValueError("depth is outside the frozen Gate-3 ladder")
        if self.width not in GATE3_WIDTHS_BY_DEPTH[self.depth]:
            raise ValueError("width is outside the frozen Gate-3 matrix")
        if len(self.phases) != 2 * self.depth:
            raise ValueError("Gate-3 condition must contain exactly 2D phases")
        for expected_phase, phase in enumerate(self.phases):
            if phase.phase_index != expected_phase:
                raise ValueError("Gate-3 phase plans must be ordered")
            phase.validate(depth=self.depth, width=self.width)

        expected_observations = 2 * self.depth
        if self.unique_world_observation_count != expected_observations:
            raise ValueError("Gate-3 information identity was violated")
        if len(self.observation_signature) != expected_observations:
            raise ValueError("Gate-3 observation signature is incomplete")
        expected_updates = (1 << self.depth) * 2 * self.depth
        if self.learned_update_count != expected_updates:
            raise ValueError("Gate-3 learned-work identity was violated")
        if sum(phase.learned_updates_in_phase for phase in self.phases) != expected_updates:
            raise ValueError("Gate-3 phase schedule does not sum to the frozen work budget")

    def mechanical_signature(self) -> tuple[object, ...]:
        """Mode-independent mechanics used to prove matched-control work identity."""

        return (
            self.protocol_version,
            self.world_seed,
            self.depth,
            self.width,
            tuple(
                (
                    phase.phase_index,
                    phase.kind.value,
                    phase.active_state_slots_before,
                    phase.evaluated_state_slots,
                    phase.retained_state_slots_after,
                    phase.recurrent_updates_per_evaluated_state,
                    phase.learned_updates_in_phase,
                )
                for phase in self.phases
            ),
            self.observation_signature,
            self.unique_world_observation_count,
            self.learned_update_count,
        )


def _rng(seed: int, domain: str) -> random.Random:
    digest = hashlib.sha256(f"{GATE3_PROTOCOL_VERSION}:{domain}:{seed}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def generate_gate3_world(*, seed: int, depth: int) -> Gate3World:
    """Generate one deterministic delayed-hypothesis world."""

    if depth not in GATE3_DEPTHS:
        raise ValueError("depth is outside the frozen Gate-3 difficulty ladder")

    path_rng = _rng(seed, f"path-d{depth}")
    hint_rng = _rng(seed, f"hint-d{depth}")
    hidden_path = tuple(path_rng.randrange(2) for _ in range(depth))

    noisy_hints: list[int] = []
    for hidden_bit in hidden_path:
        if hint_rng.random() < GATE3_HINT_RELIABILITY:
            noisy_hints.append(hidden_bit)
        else:
            noisy_hints.append(1 - hidden_bit)

    observations: list[Gate3Observation] = []
    for bit_index, hint in enumerate(noisy_hints):
        observations.append(
            Gate3Observation(
                phase_index=bit_index,
                bit_index=bit_index,
                kind=Gate3ObservationKind.BRANCH_HINT,
                observed_bit=hint,
            )
        )
    for bit_index, hidden_bit in enumerate(hidden_path):
        observations.append(
            Gate3Observation(
                phase_index=depth + bit_index,
                bit_index=bit_index,
                kind=Gate3ObservationKind.DELAYED_REVEAL,
                observed_bit=hidden_bit,
            )
        )

    world = Gate3World(
        seed=seed,
        depth=depth,
        hidden_path=hidden_path,
        noisy_hints=tuple(noisy_hints),
        observations=tuple(observations),
    )
    world.validate()
    return world


def build_gate3_condition_plan(
    world: Gate3World,
    *,
    width: int,
    mode: Gate3ControlMode,
) -> Gate3ConditionPlan:
    """Build the exact width/control work schedule over an immutable Gate-3 world."""

    world.validate()
    if width not in GATE3_WIDTHS_BY_DEPTH[world.depth]:
        raise ValueError("width is outside the frozen Gate-3 matrix for this depth")

    budget = 1 << world.depth
    phases: list[Gate3PhasePlan] = []

    active = 1
    for phase_index in range(world.depth):
        evaluated = 2 * active
        if budget % evaluated != 0:
            raise RuntimeError("frozen Gate-3 branch budget is not divisible by evaluated states")
        retained = min(width, 1 << (phase_index + 1))
        phases.append(
            Gate3PhasePlan(
                phase_index=phase_index,
                kind=Gate3ObservationKind.BRANCH_HINT,
                active_state_slots_before=active,
                evaluated_state_slots=evaluated,
                retained_state_slots_after=retained,
                recurrent_updates_per_evaluated_state=budget // evaluated,
                learned_updates_in_phase=budget,
            )
        )
        active = retained

    if active != width:
        raise RuntimeError("frozen width ladder must fill the beam by the end of branching")

    for reveal_index in range(world.depth):
        if budget % width != 0:
            raise RuntimeError("frozen Gate-3 reveal budget is not divisible by width")
        phases.append(
            Gate3PhasePlan(
                phase_index=world.depth + reveal_index,
                kind=Gate3ObservationKind.DELAYED_REVEAL,
                active_state_slots_before=width,
                evaluated_state_slots=width,
                retained_state_slots_after=width,
                recurrent_updates_per_evaluated_state=budget // width,
                learned_updates_in_phase=budget,
            )
        )

    plan = Gate3ConditionPlan(
        protocol_version=GATE3_PROTOCOL_VERSION,
        world_seed=world.seed,
        depth=world.depth,
        width=width,
        mode=mode,
        phases=tuple(phases),
        observation_signature=world.observation_signature(),
        unique_world_observation_count=world.unique_world_observation_count,
        learned_update_count=world.learned_update_count,
    )
    plan.validate()
    return plan


def deterministic_tie_break(*, world_seed: int, phase_index: int, candidate_path: tuple[int, ...]) -> int:
    """Stable answer-independent tie-break key for equal neural scores."""

    if any(bit not in {0, 1} for bit in candidate_path):
        raise ValueError("candidate path must be binary")
    payload = (
        f"{GATE3_PROTOCOL_VERSION}:tie:{world_seed}:{phase_index}:"
        + "".join(str(bit) for bit in candidate_path)
    )
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def reshuffled_state_permutation(*, world_seed: int, phase_index: int, state_count: int) -> tuple[int, ...]:
    """Deterministic answer-independent state/history permutation for the continuity control."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    indices = list(range(state_count))
    rng = _rng(world_seed, f"reshuffle-phase-{phase_index}-n{state_count}")
    rng.shuffle(indices)
    return tuple(indices)
