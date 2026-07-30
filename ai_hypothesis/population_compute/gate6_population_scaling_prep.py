"""Preparation-only mechanics for candidate Gate-6 fixed-K population scaling.

This module deliberately contains no scientific world generator, no model runner, no checkpoint loader,
and no admitted result logic.  It exists only to freeze/check work accounting and answer-blind nested
population thinning while Gate-5 confirmation remains unresolved.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product

GATE6_PREPARATION_ONLY = True
GATE6_DEPTH = 10
GATE6_FRONTIER_DEPTH = 8
GATE6_FULL_FRONTIER = 1 << GATE6_FRONTIER_DEPTH
GATE6_POPULATION_LADDER = (64, 128, 256)
GATE6_STAGE_A_PARENT_SLOTS = GATE6_FULL_FRONTIER - 1
GATE6_STAGE_B_PARENT_SLOTS = 128
GATE6_ACTIVE_CHILD_LANES = 2
GATE6_RECURRENT_UPDATES_PER_CHILD = 8
GATE6_UPDATES_PER_PARENT_SLOT = GATE6_ACTIVE_CHILD_LANES * GATE6_RECURRENT_UPDATES_PER_CHILD
GATE6_STAGE_A_LEARNED_UPDATES = GATE6_STAGE_A_PARENT_SLOTS * GATE6_UPDATES_PER_PARENT_SLOT
GATE6_STAGE_B_LEARNED_UPDATES = GATE6_STAGE_B_PARENT_SLOTS * GATE6_UPDATES_PER_PARENT_SLOT
GATE6_TOTAL_LEARNED_UPDATES = GATE6_STAGE_A_LEARNED_UPDATES + GATE6_STAGE_B_LEARNED_UPDATES
GATE6_PRIMARY_K = 16
GATE6_DESCRIPTIVE_K = 8
GATE6_NONINFERIORITY_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class Gate6PreparationPlan:
    population_size: int
    stage_a_parent_slots: int
    stage_b_parent_slots: int
    active_child_lanes: int
    recurrent_updates_per_child: int
    stage_a_learned_updates: int
    stage_b_learned_updates: int
    total_learned_updates: int

    def validate(self) -> None:
        if self.population_size not in GATE6_POPULATION_LADDER:
            raise ValueError("population size is outside the candidate Gate-6 ladder")
        if self.stage_a_parent_slots != GATE6_STAGE_A_PARENT_SLOTS:
            raise ValueError("Stage-A work must build the same complete depth-8 frontier")
        if self.stage_b_parent_slots != GATE6_STAGE_B_PARENT_SLOTS:
            raise ValueError("Stage-B parent-slot budget differs from candidate Gate-6 freeze")
        if self.active_child_lanes != GATE6_ACTIVE_CHILD_LANES:
            raise ValueError("active child lanes must remain fixed at two")
        if self.recurrent_updates_per_child != GATE6_RECURRENT_UPDATES_PER_CHILD:
            raise ValueError("per-child recurrent refinement must remain fixed at eight")
        if self.stage_a_learned_updates != GATE6_STAGE_A_LEARNED_UPDATES:
            raise ValueError("Stage-A learned-work accounting mismatch")
        if self.stage_b_learned_updates != GATE6_STAGE_B_LEARNED_UPDATES:
            raise ValueError("Stage-B learned-work accounting mismatch")
        if self.total_learned_updates != GATE6_TOTAL_LEARNED_UPDATES:
            raise ValueError("total learned-work accounting mismatch")
        if self.total_learned_updates != self.stage_a_learned_updates + self.stage_b_learned_updates:
            raise ValueError("Stage-A + Stage-B work must equal total learned work")


def build_gate6_preparation_plan(population_size: int) -> Gate6PreparationPlan:
    plan = Gate6PreparationPlan(
        population_size=population_size,
        stage_a_parent_slots=GATE6_STAGE_A_PARENT_SLOTS,
        stage_b_parent_slots=GATE6_STAGE_B_PARENT_SLOTS,
        active_child_lanes=GATE6_ACTIVE_CHILD_LANES,
        recurrent_updates_per_child=GATE6_RECURRENT_UPDATES_PER_CHILD,
        stage_a_learned_updates=GATE6_STAGE_A_LEARNED_UPDATES,
        stage_b_learned_updates=GATE6_STAGE_B_LEARNED_UPDATES,
        total_learned_updates=GATE6_TOTAL_LEARNED_UPDATES,
    )
    plan.validate()
    return plan


def complete_depth8_frontier_paths() -> tuple[tuple[int, ...], ...]:
    """Return the structural 256-path depth-8 frontier; no scientific world is involved."""
    return tuple(tuple(bits) for bits in product((0, 1), repeat=GATE6_FRONTIER_DEPTH))


def _thinning_key(runtime_seed: int, path: tuple[int, ...]) -> tuple[bytes, tuple[int, ...]]:
    if runtime_seed < 0:
        raise ValueError("runtime seed must be non-negative")
    if len(path) != GATE6_FRONTIER_DEPTH or any(bit not in (0, 1) for bit in path):
        raise ValueError("Gate-6 thinning accepts only depth-8 binary candidate paths")
    encoded_path = "".join(str(bit) for bit in path)
    digest = hashlib.sha256(
        f"gate6-fixed-k-population-thinning:{runtime_seed}:{encoded_path}".encode("ascii")
    ).digest()
    return digest, path


def nested_answer_blind_thinning(
    paths: tuple[tuple[int, ...], ...], *, runtime_seed: int, population_size: int
) -> tuple[tuple[int, ...], ...]:
    """Select a nested prefix of one deterministic answer-blind permutation.

    The key depends only on runtime_seed and candidate path.  It does not accept neural scores,
    hidden answers, labels, or condition identifiers.  Therefore N64 is a subset of N128, which is
    a subset of N256 for the same incoming frontier and runtime seed.
    """
    if population_size not in GATE6_POPULATION_LADDER:
        raise ValueError("population size is outside the candidate Gate-6 ladder")
    if len(paths) != GATE6_FULL_FRONTIER or len(set(paths)) != GATE6_FULL_FRONTIER:
        raise ValueError("Gate-6 thinning requires exactly 256 unique candidate paths")
    ordered = tuple(sorted(paths, key=lambda path: _thinning_key(runtime_seed, path)))
    return ordered[:population_size]


def validate_nested_thinning(*, runtime_seed: int) -> None:
    frontier = complete_depth8_frontier_paths()
    n64 = set(nested_answer_blind_thinning(frontier, runtime_seed=runtime_seed, population_size=64))
    n128 = set(nested_answer_blind_thinning(frontier, runtime_seed=runtime_seed, population_size=128))
    n256 = set(nested_answer_blind_thinning(frontier, runtime_seed=runtime_seed, population_size=256))
    if not n64 < n128 < n256:
        raise RuntimeError("Gate-6 candidate thinning is not strictly nested")
