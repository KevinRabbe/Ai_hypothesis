"""Mechanical substrate for Gate-3 v1 sparse-active hypothesis search.

This module intentionally contains no neural model and no scientific result. It freezes public
world generation, fixed-work accounting, reserve capacity/control semantics and answer-blind
ordering before any Gate-3 v1 development data can exist.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable


GATE3_V1_EXPERIMENT_VERSION = "gate3-v1-sparse-active-reserve"
GATE3_V1_DEPTHS = (6, 8, 10)
GATE3_V1_RESERVE_CAPACITIES = {
    6: (1, 4, 16),
    8: (1, 4, 16, 64),
    10: (1, 4, 16, 64, 256),
}
GATE3_V1_SEARCH_ROUNDS = {6: 16, 8: 64, 10: 256}
GATE3_V1_HINT_RELIABILITY = 0.70
GATE3_V1_ACTIVE_CHILD_LANES = 2
GATE3_V1_RECURRENT_UPDATES_PER_CHILD = 8
GATE3_V1_UPDATES_PER_ROUND = (
    GATE3_V1_ACTIVE_CHILD_LANES * GATE3_V1_RECURRENT_UPDATES_PER_CHILD
)
GATE3_V1_SCORE_QUANTIZATION = 1e-3
GATE3_V1_DEVELOPMENT_WORLD_START = 1 << 30
GATE3_V1_CONFIRMATION_WORLD_START = 1 << 31


class Gate3V1ControlMode(str, Enum):
    STABLE_RESERVE = "stable_reserve"
    COLLAPSED_DIVERSITY = "collapsed_diversity"
    RESHUFFLED_CONTINUITY = "reshuffled_continuity"


@dataclass(frozen=True, slots=True)
class Gate3V1PublicWorld:
    """The only world object admissible to the search runtime."""

    seed: int
    depth: int
    noisy_hints: tuple[int, ...]

    def validate(self) -> None:
        if self.depth not in GATE3_V1_DEPTHS:
            raise ValueError("depth is outside the frozen Gate-3 v1 ladder")
        if self.seed < 0:
            raise ValueError("world seed must be non-negative")
        if len(self.noisy_hints) != self.depth:
            raise ValueError("noisy-hint count must equal hidden depth")
        if any(bit not in (0, 1) for bit in self.noisy_hints):
            raise ValueError("noisy hints must be binary")


@dataclass(frozen=True, slots=True)
class Gate3V1EvaluationWorld:
    """Evaluation-only wrapper containing the hidden answer plus runtime-public evidence."""

    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        self.public.validate()
        if len(self.hidden_path) != self.public.depth:
            raise ValueError("hidden path length must equal world depth")
        if any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("hidden path must be binary")


@dataclass(frozen=True, slots=True)
class Gate3V1ConditionPlan:
    depth: int
    reserve_capacity: int
    mode: Gate3V1ControlMode
    search_rounds: int
    active_child_lanes: int
    recurrent_updates_per_child: int
    learned_updates_per_round: int
    total_learned_updates: int

    def validate(self) -> None:
        if self.depth not in GATE3_V1_DEPTHS:
            raise ValueError("plan depth is outside the frozen Gate-3 v1 ladder")
        if self.reserve_capacity not in GATE3_V1_RESERVE_CAPACITIES[self.depth]:
            raise ValueError("reserve capacity is outside the frozen ladder for this depth")
        if self.search_rounds != GATE3_V1_SEARCH_ROUNDS[self.depth]:
            raise ValueError("search-round count differs from frozen Gate-3 v1 budget")
        if self.active_child_lanes != GATE3_V1_ACTIVE_CHILD_LANES:
            raise ValueError("Gate-3 v1 active child lanes must remain fixed at two")
        if self.recurrent_updates_per_child != GATE3_V1_RECURRENT_UPDATES_PER_CHILD:
            raise ValueError("Gate-3 v1 per-child recurrent refinement must remain fixed at eight")
        if self.learned_updates_per_round != GATE3_V1_UPDATES_PER_ROUND:
            raise ValueError("Gate-3 v1 learned updates/round differ from frozen budget")
        if self.total_learned_updates != self.search_rounds * self.learned_updates_per_round:
            raise ValueError("Gate-3 v1 total learned work is inconsistent")

    def mechanical_signature(self) -> tuple[int, int, int, int]:
        return (
            self.search_rounds,
            self.active_child_lanes,
            self.recurrent_updates_per_child,
            self.total_learned_updates,
        )


@dataclass(frozen=True, slots=True)
class Gate3V1Candidate:
    """Answer-blind reserve representation used by mechanical/control tests."""

    path: tuple[int, ...]
    score: float
    state_token: str

    @property
    def depth(self) -> int:
        return len(self.path)

    def validate(self, *, world_depth: int) -> None:
        if not 0 <= self.depth < world_depth:
            raise ValueError("reserve candidates must be nonterminal prefixes")
        if any(bit not in (0, 1) for bit in self.path):
            raise ValueError("candidate path must be binary")
        if not self.state_token:
            raise ValueError("candidate state token must be non-empty")


@dataclass(frozen=True, slots=True)
class Gate3V1SearchAccounting:
    depth: int
    reserve_capacity: int
    mode: Gate3V1ControlMode
    scheduled_rounds: int
    productive_rounds: int
    sink_rounds: int
    productive_learned_updates: int
    sink_learned_updates: int
    total_learned_updates: int

    def validate(self) -> None:
        plan = build_gate3_v1_condition_plan(
            depth=self.depth,
            reserve_capacity=self.reserve_capacity,
            mode=self.mode,
        )
        if self.scheduled_rounds != plan.search_rounds:
            raise ValueError("search accounting round count differs from frozen plan")
        if self.productive_rounds < 0 or self.sink_rounds < 0:
            raise ValueError("productive/sink round counts must be non-negative")
        if self.productive_rounds + self.sink_rounds != self.scheduled_rounds:
            raise ValueError("productive + sink rounds must equal scheduled rounds")
        if self.productive_learned_updates != self.productive_rounds * GATE3_V1_UPDATES_PER_ROUND:
            raise ValueError("productive learned-work accounting is invalid")
        if self.sink_learned_updates != self.sink_rounds * GATE3_V1_UPDATES_PER_ROUND:
            raise ValueError("sink learned-work accounting is invalid")
        if self.total_learned_updates != plan.total_learned_updates:
            raise ValueError("total learned-work accounting differs from frozen plan")
        if self.productive_learned_updates + self.sink_learned_updates != self.total_learned_updates:
            raise ValueError("productive + sink learned work must equal total learned work")


def generate_gate3_v1_world(*, seed: int, depth: int) -> Gate3V1EvaluationWorld:
    if depth not in GATE3_V1_DEPTHS:
        raise ValueError("depth is outside the frozen Gate-3 v1 ladder")
    if seed < 0:
        raise ValueError("world seed must be non-negative")

    hidden_rng = random.Random(_seed_from_parts("gate3-v1-hidden", seed, depth))
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(depth))
    hint_rng = random.Random(_seed_from_parts("gate3-v1-hints", seed, depth))
    noisy_hints = tuple(
        hidden_bit if hint_rng.random() < GATE3_V1_HINT_RELIABILITY else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    world = Gate3V1EvaluationWorld(
        public=Gate3V1PublicWorld(seed=seed, depth=depth, noisy_hints=noisy_hints),
        hidden_path=hidden_path,
    )
    world.validate()
    return world


def build_gate3_v1_condition_plan(
    *,
    depth: int,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
) -> Gate3V1ConditionPlan:
    if depth not in GATE3_V1_DEPTHS:
        raise ValueError("depth is outside the frozen Gate-3 v1 ladder")
    if reserve_capacity not in GATE3_V1_RESERVE_CAPACITIES[depth]:
        raise ValueError("reserve capacity is outside the frozen ladder for this depth")
    plan = Gate3V1ConditionPlan(
        depth=depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
        search_rounds=GATE3_V1_SEARCH_ROUNDS[depth],
        active_child_lanes=GATE3_V1_ACTIVE_CHILD_LANES,
        recurrent_updates_per_child=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
        learned_updates_per_round=GATE3_V1_UPDATES_PER_ROUND,
        total_learned_updates=GATE3_V1_SEARCH_ROUNDS[depth] * GATE3_V1_UPDATES_PER_ROUND,
    )
    plan.validate()
    return plan


def quantize_gate3_v1_score(score: float) -> int:
    # Deliberately avoids any hidden-world dependency. Python round is deterministic for finite
    # scores generated by the same runtime; neural implementation qualification later proves the
    # batched/reference decision path under this exact rule.
    if score != score or score in (float("inf"), float("-inf")):
        raise ValueError("candidate score must be finite")
    return int(round(score / GATE3_V1_SCORE_QUANTIZATION))


def deterministic_gate3_v1_tie_break(
    *,
    world_seed: int,
    expansion_index: int,
    candidate_path: tuple[int, ...],
) -> int:
    digest = hashlib.sha256()
    digest.update(b"gate3-v1-tie-break\0")
    digest.update(str(world_seed).encode("ascii"))
    digest.update(b":")
    digest.update(str(expansion_index).encode("ascii"))
    digest.update(b":")
    digest.update(bytes(candidate_path))
    return int.from_bytes(digest.digest()[:8], "big")


def rank_gate3_v1_candidates(
    candidates: Iterable[Gate3V1Candidate],
    *,
    world_seed: int,
    expansion_index: int,
) -> tuple[Gate3V1Candidate, ...]:
    rows = tuple(candidates)
    return tuple(
        sorted(
            rows,
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


def _deterministic_permutation(*, world_seed: int, expansion_index: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("permutation count must be positive")
    rng = random.Random(_seed_from_parts("gate3-v1-reshuffle", world_seed, expansion_index, count))
    indices = list(range(count))
    rng.shuffle(indices)
    return tuple(indices)


def apply_gate3_v1_reserve_control(
    candidates: Iterable[Gate3V1Candidate],
    *,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    world_seed: int,
    expansion_index: int,
    world_depth: int,
) -> tuple[Gate3V1Candidate, ...]:
    if reserve_capacity <= 0:
        raise ValueError("reserve capacity must be positive")
    rows = tuple(candidates)
    for candidate in rows:
        candidate.validate(world_depth=world_depth)
    if not rows:
        return ()

    ranked = rank_gate3_v1_candidates(
        rows,
        world_seed=world_seed,
        expansion_index=expansion_index,
    )
    retained = ranked[:reserve_capacity]

    if mode is Gate3V1ControlMode.STABLE_RESERVE:
        return retained

    if mode is Gate3V1ControlMode.COLLAPSED_DIVERSITY:
        top = retained[0]
        return tuple(replace(top) for _ in range(len(retained)))

    if mode is Gate3V1ControlMode.RESHUFFLED_CONTINUITY:
        if len(retained) == 1:
            return retained
        permutation = _deterministic_permutation(
            world_seed=world_seed,
            expansion_index=expansion_index,
            count=len(retained),
        )
        histories = tuple((candidate.score, candidate.state_token) for candidate in retained)
        return tuple(
            Gate3V1Candidate(
                path=candidate.path,
                score=histories[permutation[index]][0],
                state_token=histories[permutation[index]][1],
            )
            for index, candidate in enumerate(retained)
        )

    raise ValueError(f"unknown Gate-3 v1 control mode: {mode}")


def score_generated_solution(
    *,
    hidden_path: tuple[int, ...],
    generated_terminal_paths: Iterable[tuple[int, ...]],
) -> bool:
    """Evaluation-only exact search-coverage scoring.

    Search runtime code must never call this function while deciding which hypothesis to expand.
    """

    return hidden_path in set(generated_terminal_paths)


def make_gate3_v1_accounting(
    *,
    depth: int,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    productive_rounds: int,
) -> Gate3V1SearchAccounting:
    plan = build_gate3_v1_condition_plan(
        depth=depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
    )
    if not 0 <= productive_rounds <= plan.search_rounds:
        raise ValueError("productive rounds must be inside the frozen search budget")
    sink_rounds = plan.search_rounds - productive_rounds
    accounting = Gate3V1SearchAccounting(
        depth=depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
        scheduled_rounds=plan.search_rounds,
        productive_rounds=productive_rounds,
        sink_rounds=sink_rounds,
        productive_learned_updates=productive_rounds * GATE3_V1_UPDATES_PER_ROUND,
        sink_learned_updates=sink_rounds * GATE3_V1_UPDATES_PER_ROUND,
        total_learned_updates=plan.total_learned_updates,
    )
    accounting.validate()
    return accounting


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")
