"""Frozen protocol for Gate-7 routing-bandwidth confirmation.

This module is protocol-only. It binds the completed screening result, exact checkpoint family,
fixed confirmation populations/conditions, untouched-world count, bootstrap count and outcome
classifier. It contains no world generator, checkpoint loader, Torch import, execution runner,
result artifact or confirmation-opening path.
"""

from __future__ import annotations

from dataclasses import dataclass

GATE7_CONFIRMATION_PROTOCOL_FROZEN = True
GATE7_CONFIRMATION_EXECUTION_OPENED = False
GATE7_CONFIRMATION_VERSION = "gate7-high-scale-routing-bandwidth-confirmation-v0"
GATE7_CONFIRMATION_SCREENING_RESULT_HEAD = "07b6397f2a9d4f71ed789d6c7011e12b4cbf90e0"
GATE7_CONFIRMATION_SCREENING_RESULT_SHA256 = (
    "d76c8b0753a518b4c61b3ff42c1f3e85902e2e492342f23fa6706459ee13a9b5"
)
GATE7_CONFIRMATION_SCREENING_AUDIT_SHA256 = (
    "7352621ef5c5199cba98070e2f2511674bd2f4aba8b20b48c0ec87436c5204d5"
)
GATE7_CONFIRMATION_SCREENING_OUTCOME = "G7_ROUTING_BANDWIDTH_FRONTIER_REACHED"

GATE7_CONFIRMATION_CHECKPOINTS = {
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
GATE7_CONFIRMATION_CHECKPOINT_INDICES = (0, 1, 2)
GATE7_CONFIRMATION_LEARNED_PARAMETER_COUNT = 19_649

GATE7_CONFIRMATION_POPULATIONS = (4096, 8192)
GATE7_CONFIRMATION_ANCHOR_POPULATION = 4096
GATE7_CONFIRMATION_FRONTIER_POPULATION = 8192
GATE7_CONFIRMATION_K_LADDER = (16, 32, 64, 128, 256, 512)
GATE7_CONFIRMATION_ANCHOR_K = 512
GATE7_CONFIRMATION_WORLD_COUNT = 512
GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE = 64
GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES = 10_000
GATE7_CONFIRMATION_HINT_RELIABILITY = 0.70
GATE7_CONFIRMATION_NONINFERIORITY_MARGIN = 0.05
GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS = 128
GATE7_CONFIRMATION_ACTIVE_CHILD_LANES = 2
GATE7_CONFIRMATION_RECURRENT_UPDATES_PER_CHILD = 8
GATE7_CONFIRMATION_UPDATES_PER_PARENT = 16

GATE7_CONFIRMATION_GLOBAL_SCORE = "global_score"
GATE7_CONFIRMATION_GLOBAL_HASH = "global_hash"

GATE7_CONFIRMATION_FRONTIER_CONFIRMED = "G7_ROUTING_BANDWIDTH_FRONTIER_CONFIRMED"
GATE7_CONFIRMATION_FRONTIER_NOT_CONFIRMED = "G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED"
GATE7_CONFIRMATION_ANCHOR_REFERENCE_NOT_REPLICATED = (
    "G7_CONFIRMATION_ANCHOR_REFERENCE_NOT_REPLICATED"
)
GATE7_CONFIRMATION_ANCHOR_K_NOT_REPLICATED = "G7_CONFIRMATION_ANCHOR_K512_NOT_REPLICATED"
GATE7_CONFIRMATION_FRONTIER_REFERENCE_NOT_REPLICATED = (
    "G7_CONFIRMATION_N8192_REFERENCE_NOT_REPLICATED"
)


def bounded_score_condition(k: int) -> str:
    if k not in GATE7_CONFIRMATION_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 confirmation ladder")
    return f"bounded_score_k{k}"


def bounded_hash_condition(k: int) -> str:
    if k not in GATE7_CONFIRMATION_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 confirmation ladder")
    return f"bounded_hash_k{k}"


@dataclass(frozen=True, slots=True)
class Gate7ConfirmationTierPlan:
    population: int
    k_values: tuple[int, ...]
    conditions: tuple[str, ...]
    world_count: int
    evaluation_batch_size: int
    stage_a_parent_slots: int
    stage_b_parent_slots: int
    logical_learned_updates_per_world: int

    def validate(self) -> None:
        if self.population not in GATE7_CONFIRMATION_POPULATIONS:
            raise ValueError("population is outside the frozen confirmation matrix")
        expected_k = (
            (GATE7_CONFIRMATION_ANCHOR_K,)
            if self.population == GATE7_CONFIRMATION_ANCHOR_POPULATION
            else GATE7_CONFIRMATION_K_LADDER
        )
        if self.k_values != expected_k:
            raise ValueError("confirmation K matrix changed")
        expected_conditions: list[str] = [
            GATE7_CONFIRMATION_GLOBAL_SCORE,
            GATE7_CONFIRMATION_GLOBAL_HASH,
        ]
        for k in expected_k:
            expected_conditions.extend((bounded_score_condition(k), bounded_hash_condition(k)))
        if self.conditions != tuple(expected_conditions):
            raise ValueError("confirmation condition matrix changed")
        if self.world_count != GATE7_CONFIRMATION_WORLD_COUNT:
            raise ValueError("confirmation world count changed")
        if self.evaluation_batch_size != GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE:
            raise ValueError("confirmation physical batch changed")
        if self.world_count % self.evaluation_batch_size:
            raise ValueError("confirmation worlds must divide exactly into physical batches")
        if self.stage_a_parent_slots != self.population - 1:
            raise ValueError("confirmation Stage-A work identity changed")
        if self.stage_b_parent_slots != GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS:
            raise ValueError("confirmation Stage-B work identity changed")
        expected_updates = (
            self.population - 1 + GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS
        ) * GATE7_CONFIRMATION_UPDATES_PER_PARENT
        if self.logical_learned_updates_per_world != expected_updates:
            raise ValueError("confirmation logical learned-work identity changed")


def build_confirmation_tier_plan(population: int) -> Gate7ConfirmationTierPlan:
    if population not in GATE7_CONFIRMATION_POPULATIONS:
        raise ValueError("population is outside the frozen confirmation matrix")
    k_values = (
        (GATE7_CONFIRMATION_ANCHOR_K,)
        if population == GATE7_CONFIRMATION_ANCHOR_POPULATION
        else GATE7_CONFIRMATION_K_LADDER
    )
    conditions: list[str] = [
        GATE7_CONFIRMATION_GLOBAL_SCORE,
        GATE7_CONFIRMATION_GLOBAL_HASH,
    ]
    for k in k_values:
        conditions.extend((bounded_score_condition(k), bounded_hash_condition(k)))
    plan = Gate7ConfirmationTierPlan(
        population=population,
        k_values=k_values,
        conditions=tuple(conditions),
        world_count=GATE7_CONFIRMATION_WORLD_COUNT,
        evaluation_batch_size=GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE,
        stage_a_parent_slots=population - 1,
        stage_b_parent_slots=GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS,
        logical_learned_updates_per_world=(
            population - 1 + GATE7_CONFIRMATION_STAGE_B_PARENT_SLOTS
        )
        * GATE7_CONFIRMATION_UPDATES_PER_PARENT,
    )
    plan.validate()
    return plan


def prepared_confirmation_tiers() -> tuple[Gate7ConfirmationTierPlan, ...]:
    return tuple(build_confirmation_tier_plan(population) for population in GATE7_CONFIRMATION_POPULATIONS)


def reference_is_viable(*, checkpoint_point_deltas: dict[int, float], pooled_ci_low: float) -> bool:
    if set(checkpoint_point_deltas) != set(GATE7_CONFIRMATION_CHECKPOINT_INDICES):
        raise ValueError("confirmation reference viability requires all three checkpoints")
    return (
        all(checkpoint_point_deltas[index] > 0.0 for index in GATE7_CONFIRMATION_CHECKPOINT_INDICES)
        and pooled_ci_low > 0.0
    )


def k_passes_all_checkpoints(*, k: int, primary_ci_lows: dict[str, float]) -> bool:
    if k not in GATE7_CONFIRMATION_K_LADDER:
        raise ValueError("K is outside the frozen Gate-7 confirmation ladder")
    required: list[bool] = []
    for checkpoint in GATE7_CONFIRMATION_CHECKPOINT_INDICES:
        required.extend(
            (
                primary_ci_lows[f"c{checkpoint}_k{k}_score_vs_hash"] > 0.0,
                primary_ci_lows[f"c{checkpoint}_k{k}_score_vs_global"]
                > -GATE7_CONFIRMATION_NONINFERIORITY_MARGIN,
            )
        )
    return all(required)


@dataclass(frozen=True, slots=True)
class Gate7ConfirmationClassification:
    outcome: str
    anchor_k512_passed: bool
    passing_k_at_n8192: tuple[int, ...]
    smallest_passing_k_at_n8192: int | None

    def validate(self) -> None:
        if self.passing_k_at_n8192 != tuple(
            k for k in GATE7_CONFIRMATION_K_LADDER if k in self.passing_k_at_n8192
        ):
            raise ValueError("passing confirmation K values must remain ordered and unique")
        expected_smallest = self.passing_k_at_n8192[0] if self.passing_k_at_n8192 else None
        if self.smallest_passing_k_at_n8192 != expected_smallest:
            raise ValueError("smallest passing confirmation K is inconsistent")
        valid_outcomes = {
            GATE7_CONFIRMATION_FRONTIER_CONFIRMED,
            GATE7_CONFIRMATION_FRONTIER_NOT_CONFIRMED,
            GATE7_CONFIRMATION_ANCHOR_REFERENCE_NOT_REPLICATED,
            GATE7_CONFIRMATION_ANCHOR_K_NOT_REPLICATED,
            GATE7_CONFIRMATION_FRONTIER_REFERENCE_NOT_REPLICATED,
        }
        if self.outcome not in valid_outcomes:
            raise ValueError("unknown Gate-7 confirmation outcome")


def classify_confirmation(
    *,
    anchor_reference_viable: bool,
    anchor_k512_primary_ci_lows: dict[str, float],
    frontier_reference_viable: bool,
    frontier_primary_ci_lows_by_k: dict[int, dict[str, float]],
) -> Gate7ConfirmationClassification:
    if tuple(frontier_primary_ci_lows_by_k) != GATE7_CONFIRMATION_K_LADDER:
        raise ValueError("N8192 confirmation must expose the complete frozen K ladder")

    anchor_passed = k_passes_all_checkpoints(
        k=GATE7_CONFIRMATION_ANCHOR_K,
        primary_ci_lows=anchor_k512_primary_ci_lows,
    )
    passing = tuple(
        k
        for k in GATE7_CONFIRMATION_K_LADDER
        if k_passes_all_checkpoints(k=k, primary_ci_lows=frontier_primary_ci_lows_by_k[k])
    )

    if not anchor_reference_viable:
        outcome = GATE7_CONFIRMATION_ANCHOR_REFERENCE_NOT_REPLICATED
    elif not anchor_passed:
        outcome = GATE7_CONFIRMATION_ANCHOR_K_NOT_REPLICATED
    elif not frontier_reference_viable:
        outcome = GATE7_CONFIRMATION_FRONTIER_REFERENCE_NOT_REPLICATED
    elif passing:
        outcome = GATE7_CONFIRMATION_FRONTIER_NOT_CONFIRMED
    else:
        outcome = GATE7_CONFIRMATION_FRONTIER_CONFIRMED

    result = Gate7ConfirmationClassification(
        outcome=outcome,
        anchor_k512_passed=anchor_passed,
        passing_k_at_n8192=passing,
        smallest_passing_k_at_n8192=passing[0] if passing else None,
    )
    result.validate()
    return result
