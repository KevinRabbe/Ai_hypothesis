"""Independent constants and statistics for auditing Gate-7 continuation artifacts."""

from __future__ import annotations

import hashlib
import math
import random

EXPERIMENT_VERSION = "gate7-high-scale-routing-bandwidth-continuation-v0"
SCIENTIFIC_STATUS = "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_CONTINUATION_EVIDENCE"
PROTOCOL_HEAD = "4f05f8b1f9a33aed712edbf28691b927d2e220d3"
CONFIRMATION_EXECUTION_HEAD = "7afa6f204215bac7da4623e231ec34ef3b7fdc9f"
CONFIRMATION_RESULT_HEAD = "ae8bd8544a03e48f4f397d2ca5ae933d9247e430"
CONFIRMATION_RESULT_SHA256 = "725e3749ba5fed7cdcbb6d61df81bcc77a7b69bacfdc82d553efb06f5ff888da"
CONFIRMATION_AUDIT_SHA256 = "27a46ba0feccf6b3322885334819e0e7a07bb02be930122eb1f063c65d69fb99"
CONFIRMATION_MANIFEST_SHA256 = "e7c1823dc59a50b58250cab0f7b18b95ca42b831e90182f07295680b6986b263"
CONFIRMATION_OUTCOME = "G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED"
CONFIRMED_N8192_PASSING_K = (256, 512)
CONFIRMED_N8192_K_REQUIRED = 256

CHECKPOINTS = {
    0: {
        "sha256": "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
        "fingerprint": "0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa",
        "training_seed": 0,
    },
    1: {
        "sha256": "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
        "fingerprint": "b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46",
        "training_seed": 1,
    },
    2: {
        "sha256": "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
        "fingerprint": "1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb",
        "training_seed": 2,
    },
}
CHECKPOINT_INDICES = (0, 1, 2)
TRAINING_GIT_HEAD = "07307650b2bbbfaa09b80e40caa4419ecdda2947"
TRANSITION_VERSION = "gate7-scale-neutral-scorer-transition-v0"
POPULATIONS = (16_384, 32_768, 65_536, 131_072)
K_LADDER = (16, 32, 64, 128, 256, 512)
WORLD_COUNT = 512
BATCH_SIZE = 64
BATCH_COUNT = 8
BOOTSTRAP_SAMPLES = 10_000
HINT_RELIABILITY = 0.70
NONINFERIORITY_MARGIN = 0.05
STAGE_B_SLOTS = 128
PARAMETER_COUNT = 19_649
GLOBAL_SCORE = "global_score"
GLOBAL_HASH = "global_hash"

TIER_K_REQUIRED = "G7_CONTINUATION_K_REQUIRED"
TIER_NO_K_LE_512 = "G7_CONTINUATION_NO_K_LE_512"
TIER_REFERENCE_NOT_VIABLE = "G7_CONTINUATION_REFERENCE_NOT_VIABLE"
CAMPAIGN_COMPLETE = "G7_POST_CONFIRMATION_LADDER_COMPLETE"
RESOURCE_FRONTIER = "G7_POST_CONFIRMATION_RESOURCE_FRONTIER_REACHED"
VALID_TIER_OUTCOMES = {
    TIER_K_REQUIRED,
    TIER_NO_K_LE_512,
    TIER_REFERENCE_NOT_VIABLE,
}
VALID_CAMPAIGN_OUTCOMES = {CAMPAIGN_COMPLETE, RESOURCE_FRONTIER}


def seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def runtime_seed(population: int, world_index: int) -> int:
    return seed_from_parts(
        "gate7-high-scale-routing-bandwidth-continuation-runtime-v0",
        population,
        world_index,
        population.bit_length(),
    )


def score_condition(k: int) -> str:
    return f"bounded_score_k{k}"


def hash_condition(k: int) -> str:
    return f"bounded_hash_k{k}"


def planned_conditions() -> tuple[str, ...]:
    conditions = [GLOBAL_SCORE, GLOBAL_HASH]
    for k in K_LADDER:
        conditions.extend((score_condition(k), hash_condition(k)))
    return tuple(conditions)


def condition_k(condition: str) -> int | None:
    if condition in (GLOBAL_SCORE, GLOBAL_HASH):
        return None
    if condition.startswith("bounded_score_k") or condition.startswith("bounded_hash_k"):
        value = int(condition.rsplit("k", 1)[1])
        if value not in K_LADDER:
            raise ValueError("condition K is outside the continuation ladder")
        return value
    raise ValueError("unknown continuation condition")


def expected_observations(population: int, condition: str) -> int:
    if condition == GLOBAL_SCORE:
        return STAGE_B_SLOTS * population - (STAGE_B_SLOTS - 1) * STAGE_B_SLOTS // 2
    if condition == GLOBAL_HASH or condition.startswith("bounded_hash_k"):
        return 0
    if condition.startswith("bounded_score_k"):
        return STAGE_B_SLOTS * int(condition.rsplit("k", 1)[1])
    raise ValueError("unknown continuation condition")


def float_equal(expected: float, observed: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def bootstrap_quantiles(estimates: list[float]) -> tuple[float, float]:
    estimates.sort()
    return (
        estimates[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))],
    )


def paired_bootstrap(
    differences: tuple[int, ...], *, population: int, checkpoint: int, comparison: str
) -> tuple[float, float]:
    if len(differences) != WORLD_COUNT:
        raise ValueError("continuation paired bootstrap requires 512 values")
    rng = random.Random(
        seed_from_parts(
            "gate7-high-scale-routing-bandwidth-continuation-paired-bootstrap-v0",
            population,
            checkpoint,
            comparison,
        )
    )
    estimates = [
        sum(differences[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return bootstrap_quantiles(estimates)


def stratified_bootstrap(
    differences_by_checkpoint: dict[int, tuple[int, ...]], *, population: int
) -> tuple[float, float]:
    if set(differences_by_checkpoint) != set(CHECKPOINT_INDICES):
        raise ValueError("continuation stratified bootstrap requires all checkpoints")
    rng = random.Random(
        seed_from_parts(
            "gate7-high-scale-routing-bandwidth-continuation-stratified-bootstrap-v0",
            population,
            "global_score_vs_global_hash",
        )
    )
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        strata = []
        for checkpoint in CHECKPOINT_INDICES:
            vector = differences_by_checkpoint[checkpoint]
            strata.append(
                sum(vector[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
            )
        estimates.append(sum(strata) / len(strata))
    return bootstrap_quantiles(estimates)


def reference_is_viable(*, point_deltas: dict[int, float], pooled_ci_low: float) -> bool:
    return (
        set(point_deltas) == set(CHECKPOINT_INDICES)
        and all(point_deltas[index] > 0.0 for index in CHECKPOINT_INDICES)
        and pooled_ci_low > 0.0
    )


def k_passes(k: int, lows: dict[str, float]) -> bool:
    return all(
        lows[f"c{checkpoint}_k{k}_score_vs_hash"] > 0.0
        and lows[f"c{checkpoint}_k{k}_score_vs_global"] > -NONINFERIORITY_MARGIN
        for checkpoint in CHECKPOINT_INDICES
    )


def classify_tier(
    *, reference_viable: bool, lows_by_k: dict[int, dict[str, float]]
) -> tuple[str, tuple[int, ...], int | None]:
    if tuple(lows_by_k) != K_LADDER:
        raise ValueError("continuation audit requires complete ordered K ladder")
    passing = tuple(k for k in K_LADDER if k_passes(k, lows_by_k[k]))
    if not reference_viable:
        outcome = TIER_REFERENCE_NOT_VIABLE
    elif passing:
        outcome = TIER_K_REQUIRED
    else:
        outcome = TIER_NO_K_LE_512
    return outcome, passing, passing[0] if passing else None


def classify_campaign(
    *, completed_populations: tuple[int, ...], resource_frontier_population: int | None
) -> str:
    if completed_populations != POPULATIONS[: len(completed_populations)]:
        raise ValueError("continuation completed populations are not a contiguous prefix")
    if resource_frontier_population is not None:
        if len(completed_populations) >= len(POPULATIONS):
            raise ValueError("resource frontier cannot follow a completed ladder")
        if resource_frontier_population != POPULATIONS[len(completed_populations)]:
            raise ValueError("resource frontier is not the next uncompleted population")
        return RESOURCE_FRONTIER
    if completed_populations != POPULATIONS:
        raise ValueError("non-resource campaign stopped before N131072")
    return CAMPAIGN_COMPLETE
