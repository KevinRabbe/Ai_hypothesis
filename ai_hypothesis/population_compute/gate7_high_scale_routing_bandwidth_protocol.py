"""Frozen, data-blind protocol contract for Gate-7 high-scale routing-bandwidth screening.

This module contains no scientific world generator, checkpoint loader, neural execution, result artifact,
or confirmation path. It freezes the exact population/K ladders, condition exposure order, work identity,
primary criteria, and stop classes before any N>=1024 Gate-7 world is generated.
"""

from __future__ import annotations

from dataclasses import dataclass

GATE7_HIGH_SCALE_PROTOCOL_FROZEN = True
GATE7_HIGH_SCALE_EXECUTION_OPENED = False
GATE7_HIGH_SCALE_VERSION = "gate7-high-scale-routing-bandwidth-screening-v0"
GATE7_HIGH_SCALE_BRIDGE_EVIDENCE_HEAD = "0d1bd683bae322a11a76b4d885f2efeb3c4a5099"
GATE7_HIGH_SCALE_BRIDGE_OUTCOME = "GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED"

GATE7_HIGH_SCALE_CHECKPOINTS = {
    0: {
        "sha256": "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
        "fingerprint": "0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa",
    },
    1: {
        "sha256": "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
        "fingerprint": "b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46",
    },
    2: {
        "sha256": "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
        "fingerprint": "1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb",
    },
}

GATE7_HIGH_SCALE_POPULATIONS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
GATE7_HIGH_SCALE_K_LADDER = (16, 32, 64, 128, 256, 512)
GATE7_HIGH_SCALE_WORLD_COUNT = 64
GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE = 64
GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES = 2_000
GATE7_HIGH_SCALE_HINT_RELIABILITY = 0.70
GATE7_HIGH_SCALE_NONINFERIORITY_MARGIN = 0.05
GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS = 128
GATE7_HIGH_SCALE_ACTIVE_CHILD_LANES = 2
GATE7_HIGH_SCALE_RECURRENT_UPDATES_PER_CHILD = 8
GATE7_HIGH_SCALE_UPDATES_PER_PARENT = (
    GATE7_HIGH_SCALE_ACTIVE_CHILD_LANES * GATE7_HIGH_SCALE_RECURRENT_UPDATES_PER_CHILD
)
GATE7_HIGH_SCALE_LEARNED_PARAMETER_COUNT = 19_649
GATE7_HIGH_SCALE_CHECKPOINT_INDICES = (0, 1, 2)

GATE7_HIGH_SCALE_GLOBAL_SCORE = "global_score"
GATE7_HIGH_SCALE_GLOBAL_HASH = "global_hash"
GATE7_HIGH_SCALE_SCREENING_INCOMPLETE = "G7_SCREENING_INCOMPLETE"
GATE7_HIGH_SCALE_REFERENCE_FRONTIER_REACHED = "G7_REFERENCE_FRONTIER_REACHED"
GATE7_HIGH_SCALE_ROUTING_FRONTIER_REACHED = "G7_ROUTING_BANDWIDTH_FRONTIER_REACHED"
GATE7_HIGH_SCALE_RESOURCE_FRONTIER_REACHED = "G7_RESOURCE_FRONTIER_REACHED"
GATE7_HIGH_SCALE_CAMPAIGN_CEILING_REACHED = "G7_CAMPAIGN_CEILING_REACHED"
GATE7_HIGH_SCALE_CONTINUE = "G7_CONTINUE_TO_NEXT_POPULATION"


def _log2_power_of_two(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError("Gate-7 population must remain a positive power of two")
    return value.bit_length() - 1


def bounded_score_condition(k: int) -> str:
    if k not in GATE7_HIGH_SCALE_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 ladder")
    return f"bounded_score_k{k}"


def bounded_hash_condition(k: int) -> str:
    if k not in GATE7_HIGH_SCALE_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 ladder")
    return f"bounded_hash_k{k}"


@dataclass(frozen=True, slots=True)
class Gate7HighScaleTierPlan:
    population: int
    frontier_depth: int
    world_depth: int
    stage_a_parent_slots: int
    stage_a_learned_updates: int
    stage_b_parent_slots: int
    stage_b_learned_updates: int
    total_logical_learned_updates: int
    k_ladder: tuple[int, ...]
    full_condition_ladder: tuple[str, ...]

    def validate(self) -> None:
        if self.population not in GATE7_HIGH_SCALE_POPULATIONS:
            raise ValueError("population is outside the frozen Gate-7 high-scale ladder")
        if (1 << self.frontier_depth) != self.population:
            raise ValueError("frontier depth does not match population")
        if self.world_depth != self.frontier_depth + 1:
            raise ValueError("high-scale task must terminate one decision beyond the complete frontier")
        if self.stage_a_parent_slots != self.population - 1:
            raise ValueError("Stage-A complete-frontier parent count changed")
        if self.stage_a_learned_updates != self.stage_a_parent_slots * GATE7_HIGH_SCALE_UPDATES_PER_PARENT:
            raise ValueError("Stage-A learned-work identity changed")
        if self.stage_b_parent_slots != GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS:
            raise ValueError("Stage-B parent-slot budget changed")
        if self.stage_b_learned_updates != self.stage_b_parent_slots * GATE7_HIGH_SCALE_UPDATES_PER_PARENT:
            raise ValueError("Stage-B learned-work identity changed")
        if self.total_logical_learned_updates != self.stage_a_learned_updates + self.stage_b_learned_updates:
            raise ValueError("total logical learned-work identity changed")
        expected_k = tuple(k for k in GATE7_HIGH_SCALE_K_LADDER if k < self.population)
        if self.k_ladder != expected_k:
            raise ValueError("tier K ladder changed")
        expected_conditions: list[str] = [GATE7_HIGH_SCALE_GLOBAL_SCORE, GATE7_HIGH_SCALE_GLOBAL_HASH]
        for k in expected_k:
            expected_conditions.extend((bounded_score_condition(k), bounded_hash_condition(k)))
        if self.full_condition_ladder != tuple(expected_conditions):
            raise ValueError("full condition ladder changed")


def build_gate7_high_scale_tier_plan(population: int) -> Gate7HighScaleTierPlan:
    if population not in GATE7_HIGH_SCALE_POPULATIONS:
        raise ValueError("population is outside the frozen Gate-7 high-scale ladder")
    frontier_depth = _log2_power_of_two(population)
    k_ladder = tuple(k for k in GATE7_HIGH_SCALE_K_LADDER if k < population)
    conditions: list[str] = [GATE7_HIGH_SCALE_GLOBAL_SCORE, GATE7_HIGH_SCALE_GLOBAL_HASH]
    for k in k_ladder:
        conditions.extend((bounded_score_condition(k), bounded_hash_condition(k)))
    stage_a_slots = population - 1
    plan = Gate7HighScaleTierPlan(
        population=population,
        frontier_depth=frontier_depth,
        world_depth=frontier_depth + 1,
        stage_a_parent_slots=stage_a_slots,
        stage_a_learned_updates=stage_a_slots * GATE7_HIGH_SCALE_UPDATES_PER_PARENT,
        stage_b_parent_slots=GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
        stage_b_learned_updates=(
            GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS * GATE7_HIGH_SCALE_UPDATES_PER_PARENT
        ),
        total_logical_learned_updates=(
            (stage_a_slots + GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS)
            * GATE7_HIGH_SCALE_UPDATES_PER_PARENT
        ),
        k_ladder=k_ladder,
        full_condition_ladder=tuple(conditions),
    )
    plan.validate()
    return plan


def prepared_gate7_high_scale_tiers() -> tuple[Gate7HighScaleTierPlan, ...]:
    return tuple(build_gate7_high_scale_tier_plan(population) for population in GATE7_HIGH_SCALE_POPULATIONS)


def condition_prefix_through_k(population: int, k: int) -> tuple[str, ...]:
    plan = build_gate7_high_scale_tier_plan(population)
    if k not in plan.k_ladder:
        raise ValueError("requested K is not valid for this population")
    stop = plan.full_condition_ladder.index(bounded_hash_condition(k)) + 1
    return plan.full_condition_ladder[:stop]


def validate_sequential_k_exposure(population: int, tested_k: tuple[int, ...]) -> None:
    plan = build_gate7_high_scale_tier_plan(population)
    if tested_k != plan.k_ladder[: len(tested_k)]:
        raise ValueError("tested K values must be one contiguous prefix of the frozen ascending ladder")


def reference_is_viable(*, checkpoint_point_deltas: dict[int, float], pooled_ci_low: float) -> bool:
    if set(checkpoint_point_deltas) != set(GATE7_HIGH_SCALE_CHECKPOINT_INDICES):
        raise ValueError("reference viability requires all three checkpoint point deltas")
    return all(checkpoint_point_deltas[index] > 0.0 for index in GATE7_HIGH_SCALE_CHECKPOINT_INDICES) and pooled_ci_low > 0.0


def k_passes_all_checkpoints(*, k: int, primary_ci_lows: dict[str, float]) -> bool:
    if k not in GATE7_HIGH_SCALE_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 ladder")
    required: list[bool] = []
    for checkpoint in GATE7_HIGH_SCALE_CHECKPOINT_INDICES:
        required.extend(
            (
                primary_ci_lows[f"c{checkpoint}_k{k}_score_vs_hash"] > 0.0,
                primary_ci_lows[f"c{checkpoint}_k{k}_score_vs_global"]
                > -GATE7_HIGH_SCALE_NONINFERIORITY_MARGIN,
            )
        )
    return all(required)


def smallest_passing_k(*, population: int, primary_ci_lows_by_k: dict[int, dict[str, float]]) -> int | None:
    tested = tuple(primary_ci_lows_by_k)
    validate_sequential_k_exposure(population, tested)
    for position, k in enumerate(tested):
        if k_passes_all_checkpoints(k=k, primary_ci_lows=primary_ci_lows_by_k[k]):
            if position != len(tested) - 1:
                raise ValueError("larger K values were exposed after the first all-checkpoint passing K")
            return k
    return None


def classify_completed_tier(
    *,
    population: int,
    reference_viable: bool,
    primary_ci_lows_by_k: dict[int, dict[str, float]],
    resource_complete: bool = True,
) -> str:
    plan = build_gate7_high_scale_tier_plan(population)
    if not resource_complete:
        return GATE7_HIGH_SCALE_RESOURCE_FRONTIER_REACHED
    if not reference_viable:
        if primary_ci_lows_by_k:
            raise ValueError("K conditions were exposed after the global reference failed viability")
        return GATE7_HIGH_SCALE_REFERENCE_FRONTIER_REACHED
    passing = smallest_passing_k(population=population, primary_ci_lows_by_k=primary_ci_lows_by_k)
    if passing is not None:
        return f"G7_K_REQUIRED_{passing}"
    if tuple(primary_ci_lows_by_k) == plan.k_ladder:
        return GATE7_HIGH_SCALE_ROUTING_FRONTIER_REACHED
    return GATE7_HIGH_SCALE_SCREENING_INCOMPLETE


def campaign_action_after_tier(*, population: int, tier_outcome: str) -> str:
    build_gate7_high_scale_tier_plan(population)
    if tier_outcome.startswith("G7_K_REQUIRED_"):
        if population == GATE7_HIGH_SCALE_POPULATIONS[-1]:
            return GATE7_HIGH_SCALE_CAMPAIGN_CEILING_REACHED
        return GATE7_HIGH_SCALE_CONTINUE
    if tier_outcome in {
        GATE7_HIGH_SCALE_REFERENCE_FRONTIER_REACHED,
        GATE7_HIGH_SCALE_ROUTING_FRONTIER_REACHED,
        GATE7_HIGH_SCALE_RESOURCE_FRONTIER_REACHED,
    }:
        return tier_outcome
    if tier_outcome == GATE7_HIGH_SCALE_SCREENING_INCOMPLETE:
        return GATE7_HIGH_SCALE_SCREENING_INCOMPLETE
    raise ValueError("unknown Gate-7 tier outcome")
