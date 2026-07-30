"""Preparation-only planner for the high-scale Gate-7 frontier campaign.

No scientific worlds, model execution, checkpoint loading, training, or outcome assignment live here.
The module only freezes/checks the geometric population ladder and structural work/visibility scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

GATE7_PREPARATION_ONLY = True
GATE7_EXISTING_CHECKPOINT_MAX_POPULATION = 512
GATE7_HIGH_SCALE_LADDER = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
GATE7_PRIMARY_K = 16
GATE7_DESCRIPTIVE_K = 8
GATE7_STAGE_B_PARENT_SLOTS = 128
GATE7_RECURRENT_UPDATES_PER_CHILD = 8
GATE7_ACTIVE_CHILD_LANES = 2
GATE7_UPDATES_PER_PARENT_SLOT = GATE7_RECURRENT_UPDATES_PER_CHILD * GATE7_ACTIVE_CHILD_LANES
GATE7_SCREENING_WORLDS_CANDIDATE = 64
GATE7_NONINFERIORITY_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class Gate7ScalePlan:
    population: int
    frontier_depth: int
    minimum_world_depth: int
    stage_a_parent_slots: int
    stage_a_learned_updates: int
    stage_b_parent_slots: int
    stage_b_learned_updates: int
    k16_score_observations_upper_bound: int
    global_score_observations_nominal: int

    def validate(self) -> None:
        if self.population not in (GATE7_EXISTING_CHECKPOINT_MAX_POPULATION, *GATE7_HIGH_SCALE_LADDER):
            raise ValueError("population is outside the prepared Gate-7 ladder")
        if self.population & (self.population - 1):
            raise ValueError("prepared Gate-7 populations must remain powers of two")
        if (1 << self.frontier_depth) != self.population:
            raise ValueError("frontier depth does not match population")
        if self.minimum_world_depth != self.frontier_depth + 1:
            raise ValueError("world must remain at least one decision deeper than the live frontier")
        if self.stage_a_parent_slots != self.population - 1:
            raise ValueError("complete-frontier parent-slot count mismatch")
        if self.stage_a_learned_updates != self.stage_a_parent_slots * GATE7_UPDATES_PER_PARENT_SLOT:
            raise ValueError("Stage-A learned-work accounting mismatch")
        if self.stage_b_parent_slots != GATE7_STAGE_B_PARENT_SLOTS:
            raise ValueError("Stage-B slot budget changed")
        if self.stage_b_learned_updates != GATE7_STAGE_B_PARENT_SLOTS * GATE7_UPDATES_PER_PARENT_SLOT:
            raise ValueError("Stage-B learned-work accounting mismatch")
        if self.k16_score_observations_upper_bound != GATE7_PRIMARY_K * GATE7_STAGE_B_PARENT_SLOTS:
            raise ValueError("K16 score-observation bound changed")


def _log2_power_of_two(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError("value must be a positive power of two")
    return value.bit_length() - 1


def build_gate7_scale_plan(population: int) -> Gate7ScalePlan:
    depth = _log2_power_of_two(population)
    stage_a_slots = population - 1
    plan = Gate7ScalePlan(
        population=population,
        frontier_depth=depth,
        minimum_world_depth=depth + 1,
        stage_a_parent_slots=stage_a_slots,
        stage_a_learned_updates=stage_a_slots * GATE7_UPDATES_PER_PARENT_SLOT,
        stage_b_parent_slots=GATE7_STAGE_B_PARENT_SLOTS,
        stage_b_learned_updates=GATE7_STAGE_B_PARENT_SLOTS * GATE7_UPDATES_PER_PARENT_SLOT,
        k16_score_observations_upper_bound=GATE7_PRIMARY_K * GATE7_STAGE_B_PARENT_SLOTS,
        global_score_observations_nominal=population * GATE7_STAGE_B_PARENT_SLOTS,
    )
    plan.validate()
    return plan


def prepared_gate7_plans() -> tuple[Gate7ScalePlan, ...]:
    return tuple(
        build_gate7_scale_plan(population)
        for population in (GATE7_EXISTING_CHECKPOINT_MAX_POPULATION, *GATE7_HIGH_SCALE_LADDER)
    )
