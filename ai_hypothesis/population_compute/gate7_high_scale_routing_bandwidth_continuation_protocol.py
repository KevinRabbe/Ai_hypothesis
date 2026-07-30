"""Frozen post-confirmation protocol for the remaining Gate-7 high-scale ladder.

This module binds the valid confirmation result and freezes one complete, non-adaptive continuation
matrix for N16384 through N131072. It contains no world generator, checkpoint loader, Torch import,
execution runner, result artifact or continuation-opening path.
"""

from __future__ import annotations

from dataclasses import dataclass

GATE7_CONTINUATION_PROTOCOL_FROZEN = True
GATE7_CONTINUATION_EXECUTION_OPENED = False
GATE7_CONTINUATION_VERSION = "gate7-high-scale-routing-bandwidth-continuation-v0"

GATE7_CONTINUATION_CONFIRMATION_RESULT_HEAD = "ae8bd8544a03e48f4f397d2ca5ae933d9247e430"
GATE7_CONTINUATION_CONFIRMATION_EXECUTION_HEAD = "7afa6f204215bac7da4623e231ec34ef3b7fdc9f"
GATE7_CONTINUATION_CONFIRMATION_RESULT_SHA256 = (
    "725e3749ba5fed7cdcbb6d61df81bcc77a7b69bacfdc82d553efb06f5ff888da"
)
GATE7_CONTINUATION_CONFIRMATION_AUDIT_SHA256 = (
    "27a46ba0feccf6b3322885334819e0e7a07bb02be930122eb1f063c65d69fb99"
)
GATE7_CONTINUATION_CONFIRMATION_MANIFEST_SHA256 = (
    "e7c1823dc59a50b58250cab0f7b18b95ca42b831e90182f07295680b6986b263"
)
GATE7_CONTINUATION_CONFIRMATION_OUTCOME = "G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED"
GATE7_CONTINUATION_CONFIRMED_N8192_PASSING_K = (256, 512)
GATE7_CONTINUATION_CONFIRMED_N8192_K_REQUIRED = 256

GATE7_CONTINUATION_CHECKPOINTS = {
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
GATE7_CONTINUATION_CHECKPOINT_INDICES = (0, 1, 2)
GATE7_CONTINUATION_LEARNED_PARAMETER_COUNT = 19_649

GATE7_CONTINUATION_POPULATIONS = (16_384, 32_768, 65_536, 131_072)
GATE7_CONTINUATION_K_LADDER = (16, 32, 64, 128, 256, 512)
GATE7_CONTINUATION_WORLD_COUNT = 512
GATE7_CONTINUATION_EVALUATION_BATCH_SIZE = 64
GATE7_CONTINUATION_BOOTSTRAP_SAMPLES = 10_000
GATE7_CONTINUATION_HINT_RELIABILITY = 0.70
GATE7_CONTINUATION_NONINFERIORITY_MARGIN = 0.05
GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS = 128
GATE7_CONTINUATION_ACTIVE_CHILD_LANES = 2
GATE7_CONTINUATION_RECURRENT_UPDATES_PER_CHILD = 8
GATE7_CONTINUATION_UPDATES_PER_PARENT = 16

GATE7_CONTINUATION_GLOBAL_SCORE = "global_score"
GATE7_CONTINUATION_GLOBAL_HASH = "global_hash"

GATE7_CONTINUATION_TIER_K_REQUIRED = "G7_CONTINUATION_K_REQUIRED"
GATE7_CONTINUATION_TIER_NO_K_LE_512 = "G7_CONTINUATION_NO_K_LE_512"
GATE7_CONTINUATION_TIER_REFERENCE_NOT_VIABLE = "G7_CONTINUATION_REFERENCE_NOT_VIABLE"
GATE7_CONTINUATION_COMPLETE = "G7_POST_CONFIRMATION_LADDER_COMPLETE"
GATE7_CONTINUATION_RESOURCE_FRONTIER_REACHED = (
    "G7_POST_CONFIRMATION_RESOURCE_FRONTIER_REACHED"
)
GATE7_CONTINUATION_CONTINUE = "G7_CONTINUE_TO_NEXT_POST_CONFIRMATION_POPULATION"


def bounded_score_condition(k: int) -> str:
    if k not in GATE7_CONTINUATION_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 continuation ladder")
    return f"bounded_score_k{k}"


def bounded_hash_condition(k: int) -> str:
    if k not in GATE7_CONTINUATION_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 continuation ladder")
    return f"bounded_hash_k{k}"


def complete_condition_matrix() -> tuple[str, ...]:
    conditions: list[str] = [
        GATE7_CONTINUATION_GLOBAL_SCORE,
        GATE7_CONTINUATION_GLOBAL_HASH,
    ]
    for k in GATE7_CONTINUATION_K_LADDER:
        conditions.extend((bounded_score_condition(k), bounded_hash_condition(k)))
    return tuple(conditions)


@dataclass(frozen=True, slots=True)
class Gate7ContinuationTierPlan:
    population: int
    k_values: tuple[int, ...]
    conditions: tuple[str, ...]
    world_count: int
    evaluation_batch_size: int
    stage_a_parent_slots: int
    stage_b_parent_slots: int
    logical_learned_updates_per_world: int

    def validate(self) -> None:
        if self.population not in GATE7_CONTINUATION_POPULATIONS:
            raise ValueError("population is outside the frozen continuation ladder")
        if self.k_values != GATE7_CONTINUATION_K_LADDER:
            raise ValueError("continuation K matrix changed")
        if self.conditions != complete_condition_matrix():
            raise ValueError("continuation condition matrix changed")
        if self.world_count != GATE7_CONTINUATION_WORLD_COUNT:
            raise ValueError("continuation world count changed")
        if self.evaluation_batch_size != GATE7_CONTINUATION_EVALUATION_BATCH_SIZE:
            raise ValueError("continuation physical batch changed")
        if self.world_count % self.evaluation_batch_size:
            raise ValueError("continuation worlds must divide exactly into physical batches")
        if self.stage_a_parent_slots != self.population - 1:
            raise ValueError("continuation Stage-A work identity changed")
        if self.stage_b_parent_slots != GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS:
            raise ValueError("continuation Stage-B work identity changed")
        expected_updates = (
            self.population - 1 + GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS
        ) * GATE7_CONTINUATION_UPDATES_PER_PARENT
        if self.logical_learned_updates_per_world != expected_updates:
            raise ValueError("continuation logical learned-work identity changed")


def build_continuation_tier_plan(population: int) -> Gate7ContinuationTierPlan:
    if population not in GATE7_CONTINUATION_POPULATIONS:
        raise ValueError("population is outside the frozen continuation ladder")
    plan = Gate7ContinuationTierPlan(
        population=population,
        k_values=GATE7_CONTINUATION_K_LADDER,
        conditions=complete_condition_matrix(),
        world_count=GATE7_CONTINUATION_WORLD_COUNT,
        evaluation_batch_size=GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
        stage_a_parent_slots=population - 1,
        stage_b_parent_slots=GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS,
        logical_learned_updates_per_world=(
            population - 1 + GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS
        )
        * GATE7_CONTINUATION_UPDATES_PER_PARENT,
    )
    plan.validate()
    return plan


def prepared_continuation_tiers() -> tuple[Gate7ContinuationTierPlan, ...]:
    return tuple(
        build_continuation_tier_plan(population)
        for population in GATE7_CONTINUATION_POPULATIONS
    )


def reference_is_viable(*, checkpoint_point_deltas: dict[int, float], pooled_ci_low: float) -> bool:
    if set(checkpoint_point_deltas) != set(GATE7_CONTINUATION_CHECKPOINT_INDICES):
        raise ValueError("continuation reference viability requires all three checkpoints")
    return (
        all(
            checkpoint_point_deltas[index] > 0.0
            for index in GATE7_CONTINUATION_CHECKPOINT_INDICES
        )
        and pooled_ci_low > 0.0
    )


def k_passes_all_checkpoints(*, k: int, primary_ci_lows: dict[str, float]) -> bool:
    if k not in GATE7_CONTINUATION_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 continuation ladder")
    required: list[bool] = []
    for checkpoint in GATE7_CONTINUATION_CHECKPOINT_INDICES:
        required.extend(
            (
                primary_ci_lows[f"c{checkpoint}_k{k}_score_vs_hash"] > 0.0,
                primary_ci_lows[f"c{checkpoint}_k{k}_score_vs_global"]
                > -GATE7_CONTINUATION_NONINFERIORITY_MARGIN,
            )
        )
    return all(required)


@dataclass(frozen=True, slots=True)
class Gate7ContinuationTierClassification:
    population: int
    outcome: str
    reference_viable: bool
    passing_k: tuple[int, ...]
    smallest_passing_k: int | None
    smallest_passing_k_over_n: float | None

    def validate(self) -> None:
        if self.population not in GATE7_CONTINUATION_POPULATIONS:
            raise ValueError("classified population is outside the continuation ladder")
        if self.passing_k != tuple(
            k for k in GATE7_CONTINUATION_K_LADDER if k in self.passing_k
        ):
            raise ValueError("passing continuation K values must remain ordered and unique")
        expected_smallest = self.passing_k[0] if self.passing_k else None
        if self.smallest_passing_k != expected_smallest:
            raise ValueError("smallest passing continuation K is inconsistent")
        expected_ratio = (
            expected_smallest / self.population if expected_smallest is not None else None
        )
        if self.smallest_passing_k_over_n != expected_ratio:
            raise ValueError("continuation K/N ratio is inconsistent")
        valid_outcomes = {
            GATE7_CONTINUATION_TIER_K_REQUIRED,
            GATE7_CONTINUATION_TIER_NO_K_LE_512,
            GATE7_CONTINUATION_TIER_REFERENCE_NOT_VIABLE,
        }
        if self.outcome not in valid_outcomes:
            raise ValueError("unknown Gate-7 continuation tier outcome")
        if not self.reference_viable and self.outcome != GATE7_CONTINUATION_TIER_REFERENCE_NOT_VIABLE:
            raise ValueError("non-viable reference must retain its own tier outcome")
        if self.reference_viable and self.passing_k and self.outcome != GATE7_CONTINUATION_TIER_K_REQUIRED:
            raise ValueError("passing continuation K values require a K-required outcome")
        if self.reference_viable and not self.passing_k and self.outcome != GATE7_CONTINUATION_TIER_NO_K_LE_512:
            raise ValueError("empty continuation pass set requires the no-K<=512 outcome")


def classify_continuation_tier(
    *,
    population: int,
    reference_viable: bool,
    primary_ci_lows_by_k: dict[int, dict[str, float]],
) -> Gate7ContinuationTierClassification:
    if population not in GATE7_CONTINUATION_POPULATIONS:
        raise ValueError("population is outside the frozen continuation ladder")
    if tuple(primary_ci_lows_by_k) != GATE7_CONTINUATION_K_LADDER:
        raise ValueError("continuation tier must expose the complete frozen K ladder")
    passing = tuple(
        k
        for k in GATE7_CONTINUATION_K_LADDER
        if k_passes_all_checkpoints(k=k, primary_ci_lows=primary_ci_lows_by_k[k])
    )
    if not reference_viable:
        outcome = GATE7_CONTINUATION_TIER_REFERENCE_NOT_VIABLE
    elif passing:
        outcome = GATE7_CONTINUATION_TIER_K_REQUIRED
    else:
        outcome = GATE7_CONTINUATION_TIER_NO_K_LE_512
    smallest = passing[0] if passing else None
    result = Gate7ContinuationTierClassification(
        population=population,
        outcome=outcome,
        reference_viable=reference_viable,
        passing_k=passing,
        smallest_passing_k=smallest,
        smallest_passing_k_over_n=(smallest / population if smallest is not None else None),
    )
    result.validate()
    return result


def action_after_continuation_tier(*, population: int) -> str:
    if population not in GATE7_CONTINUATION_POPULATIONS:
        raise ValueError("population is outside the frozen continuation ladder")
    if population == GATE7_CONTINUATION_POPULATIONS[-1]:
        return GATE7_CONTINUATION_COMPLETE
    return GATE7_CONTINUATION_CONTINUE


def validate_completed_population_prefix(completed_populations: tuple[int, ...]) -> None:
    if completed_populations != GATE7_CONTINUATION_POPULATIONS[: len(completed_populations)]:
        raise ValueError("completed continuation populations must remain a contiguous prefix")


def classify_continuation_campaign(
    *,
    completed_populations: tuple[int, ...],
    resource_frontier_population: int | None,
) -> str:
    validate_completed_population_prefix(completed_populations)
    if resource_frontier_population is not None:
        expected_next = GATE7_CONTINUATION_POPULATIONS[len(completed_populations)]
        if resource_frontier_population != expected_next:
            raise ValueError("resource frontier must be the next uncompleted continuation tier")
        return GATE7_CONTINUATION_RESOURCE_FRONTIER_REACHED
    if completed_populations != GATE7_CONTINUATION_POPULATIONS:
        raise ValueError("non-resource continuation campaign cannot stop before N131072")
    return GATE7_CONTINUATION_COMPLETE
