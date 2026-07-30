"""Frozen data-blind protocol for the Gate-7 information-ceiling decomposition.

The completed routing continuation showed that bounded learned routing remains viable through
N131072 while absolute global coverage declines. This protocol tests whether that decline is
primarily imposed by the benchmark's noisy-hint information ceiling or by a learned scorer
ranking gap. It contains no Torch import, checkpoint loader, world generator, artifact reader,
runner, result value, communication mechanism, training path, or execution-opening surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log2

GATE7_INFORMATION_CEILING_PROTOCOL_FROZEN = True
GATE7_INFORMATION_CEILING_EXECUTION_OPENED = False
GATE7_INFORMATION_CEILING_RESULT_OPENED = False
GATE7_INFORMATION_CEILING_PROTOCOL_VERSION = (
    "gate7-information-ceiling-decomposition-protocol-v0"
)

GATE7_INFORMATION_CEILING_BASE_RESULT_HEAD = (
    "4591dae55cada819e848ae7f929d5e8f2b8805d6"
)
GATE7_INFORMATION_CEILING_EXECUTION_HEAD = (
    "19ee6b4e228c56b32a11b11b1c61b35bf640e2c8"
)
GATE7_INFORMATION_CEILING_RESULT_SHA256 = (
    "4921ea99b44156f08271d6fb2b2e0bcba98ef6a646ed0aaf040762d47aa03b36"
)
GATE7_INFORMATION_CEILING_AUDIT_SHA256 = (
    "92f52a9e7fad3cb5d8962a9127a0cd7140656a0a8f03cfba08fe7cd5376a03fd"
)
GATE7_INFORMATION_CEILING_MANIFEST_SHA256 = (
    "ee9dcefbaf5efe9a75b20d407cb1a4f47ff0b04bbdce4613f2539b76af2c8cca"
)

GATE7_INFORMATION_CEILING_CHECKPOINTS = {
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
GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES = (0, 1, 2)
GATE7_INFORMATION_CEILING_LEARNED_PARAMETER_COUNT = 19_649
GATE7_INFORMATION_CEILING_POPULATIONS = (16_384, 32_768, 65_536, 131_072)
GATE7_INFORMATION_CEILING_ATTEMPT_LADDER = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1_024,
)
GATE7_INFORMATION_CEILING_PRIMARY_ATTEMPTS = 128
GATE7_INFORMATION_CEILING_WORLD_COUNT = 512
GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE = 64
GATE7_INFORMATION_CEILING_BOOTSTRAP_SAMPLES = 10_000
GATE7_INFORMATION_CEILING_HINT_RELIABILITY = 0.70
GATE7_INFORMATION_CEILING_NEAR_CEILING_MARGIN = 0.02

GATE7_INFORMATION_CEILING_HIDDEN_NAMESPACE = (
    "gate7-information-ceiling-decomposition-hidden-v0"
)
GATE7_INFORMATION_CEILING_HINT_NAMESPACE = (
    "gate7-information-ceiling-decomposition-hints-v0"
)
GATE7_INFORMATION_CEILING_RUNTIME_NAMESPACE = (
    "gate7-information-ceiling-decomposition-runtime-v0"
)
GATE7_INFORMATION_CEILING_TIE_NAMESPACE = (
    "gate7-information-ceiling-decomposition-public-tie-v0"
)
GATE7_INFORMATION_CEILING_BOOTSTRAP_NAMESPACE = (
    "gate7-information-ceiling-decomposition-bootstrap-v0"
)

GATE7_INFORMATION_CEILING_LEARNED = "learned_score_rank"
GATE7_INFORMATION_CEILING_BAYES = "bayes_hint_likelihood_rank"
GATE7_INFORMATION_CEILING_HASH = "public_hash_rank"
GATE7_INFORMATION_CEILING_RANKERS = (
    GATE7_INFORMATION_CEILING_LEARNED,
    GATE7_INFORMATION_CEILING_BAYES,
    GATE7_INFORMATION_CEILING_HASH,
)

GATE7_INFORMATION_CEILING_DOMINANT = "G7_INFORMATION_CEILING_DOMINANT"
GATE7_INFORMATION_CEILING_SCORER_GAP = "G7_SCORER_REPRESENTATION_GAP"
GATE7_INFORMATION_CEILING_MIXED = "G7_INFORMATION_AND_SCORER_GAP_MIXED"
GATE7_INFORMATION_CEILING_INCONCLUSIVE = "G7_INFORMATION_CEILING_INCONCLUSIVE"
GATE7_INFORMATION_CEILING_OUTCOMES = {
    GATE7_INFORMATION_CEILING_DOMINANT,
    GATE7_INFORMATION_CEILING_SCORER_GAP,
    GATE7_INFORMATION_CEILING_MIXED,
    GATE7_INFORMATION_CEILING_INCONCLUSIVE,
}


@dataclass(frozen=True, slots=True)
class Gate7InformationCeilingTierPlan:
    population: int
    frontier_depth: int
    world_count: int
    evaluation_batch_size: int
    attempt_ladder: tuple[int, ...]
    rankers: tuple[str, ...]

    def validate(self) -> None:
        if self.population not in GATE7_INFORMATION_CEILING_POPULATIONS:
            raise ValueError("population is outside the frozen information-ceiling ladder")
        if self.frontier_depth != gate7_information_ceiling_frontier_depth(self.population):
            raise ValueError("frontier depth differs from log2(population)")
        if self.world_count != GATE7_INFORMATION_CEILING_WORLD_COUNT:
            raise ValueError("world count changed")
        if self.evaluation_batch_size != GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE:
            raise ValueError("physical evaluation batch changed")
        if self.attempt_ladder != GATE7_INFORMATION_CEILING_ATTEMPT_LADDER:
            raise ValueError("attempt ladder changed")
        if self.rankers != GATE7_INFORMATION_CEILING_RANKERS:
            raise ValueError("ranker matrix changed")
        if any(attempt >= self.population for attempt in self.attempt_ladder):
            raise ValueError("attempt ladder must remain below population")


@dataclass(frozen=True, slots=True)
class Gate7InformationCeilingComparison:
    checkpoint_index: int
    population: int
    learned_minus_bayes_ci_low: float
    learned_minus_bayes_ci_high: float
    learned_minus_hash_ci_low: float
    bayes_minus_hash_ci_low: float

    def validate(self) -> None:
        if self.checkpoint_index not in GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES:
            raise ValueError("checkpoint index is outside T0/T1/T2")
        if self.population not in GATE7_INFORMATION_CEILING_POPULATIONS:
            raise ValueError("population is outside the frozen ladder")
        if self.learned_minus_bayes_ci_low > self.learned_minus_bayes_ci_high:
            raise ValueError("learned-minus-Bayes interval is reversed")

    def near_ceiling(self) -> bool:
        self.validate()
        return (
            self.learned_minus_bayes_ci_low
            > -GATE7_INFORMATION_CEILING_NEAR_CEILING_MARGIN
            and self.learned_minus_hash_ci_low > 0.0
            and self.bayes_minus_hash_ci_low > 0.0
        )

    def clear_scorer_gap(self) -> bool:
        self.validate()
        return (
            self.learned_minus_bayes_ci_high
            < -GATE7_INFORMATION_CEILING_NEAR_CEILING_MARGIN
            and self.bayes_minus_hash_ci_low > 0.0
        )


def gate7_information_ceiling_frontier_depth(population: int) -> int:
    if population <= 0 or population & (population - 1):
        raise ValueError("population must be a positive power of two")
    return int(log2(population))


def bayes_expected_top_m_coverage(
    *,
    population: int,
    attempts: int,
    hint_reliability: float = GATE7_INFORMATION_CEILING_HINT_RELIABILITY,
) -> float:
    """Return the exact expected Bayes top-M parent coverage under public random tie-breaking.

    Stage A exposes a complete binary frontier of depth log2(N). With independent binary hints
    that are correct with probability r>0.5, posterior parent likelihood is strictly ordered by
    Hamming distance from the public hint prefix. Within one Hamming shell all candidates are
    equiprobable, so an answer-independent public tie-break includes an M/count fraction of the
    boundary shell in expectation.
    """

    depth = gate7_information_ceiling_frontier_depth(population)
    if not 0.5 < hint_reliability < 1.0:
        raise ValueError("hint reliability must be strictly between 0.5 and 1.0")
    if not 1 <= attempts < population:
        raise ValueError("attempts must be in 1..population-1")

    error_probability = 1.0 - hint_reliability
    remaining = attempts
    expected = 0.0
    for distance in range(depth + 1):
        shell_size = comb(depth, distance)
        shell_probability = (
            shell_size
            * (error_probability**distance)
            * (hint_reliability ** (depth - distance))
        )
        admitted = min(remaining, shell_size)
        expected += shell_probability * (admitted / shell_size)
        remaining -= admitted
        if remaining == 0:
            break
    return expected


def expected_primary_ceiling_by_population() -> dict[int, float]:
    return {
        population: bayes_expected_top_m_coverage(
            population=population,
            attempts=GATE7_INFORMATION_CEILING_PRIMARY_ATTEMPTS,
        )
        for population in GATE7_INFORMATION_CEILING_POPULATIONS
    }


def prepared_information_ceiling_tiers() -> tuple[Gate7InformationCeilingTierPlan, ...]:
    tiers = tuple(
        Gate7InformationCeilingTierPlan(
            population=population,
            frontier_depth=gate7_information_ceiling_frontier_depth(population),
            world_count=GATE7_INFORMATION_CEILING_WORLD_COUNT,
            evaluation_batch_size=GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE,
            attempt_ladder=GATE7_INFORMATION_CEILING_ATTEMPT_LADDER,
            rankers=GATE7_INFORMATION_CEILING_RANKERS,
        )
        for population in GATE7_INFORMATION_CEILING_POPULATIONS
    )
    for tier in tiers:
        tier.validate()
    return tiers


def classify_information_ceiling_campaign(
    comparisons: tuple[Gate7InformationCeilingComparison, ...],
) -> str:
    expected_pairs = tuple(
        (checkpoint, population)
        for population in GATE7_INFORMATION_CEILING_POPULATIONS
        for checkpoint in GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES
    )
    observed_pairs = tuple(
        (comparison.checkpoint_index, comparison.population)
        for comparison in comparisons
    )
    if observed_pairs != expected_pairs:
        raise ValueError("comparisons must cover the exact population-major T0/T1/T2 matrix")
    for comparison in comparisons:
        comparison.validate()

    near = tuple(comparison.near_ceiling() for comparison in comparisons)
    gaps = tuple(comparison.clear_scorer_gap() for comparison in comparisons)
    if all(near):
        return GATE7_INFORMATION_CEILING_DOMINANT
    if all(gaps):
        return GATE7_INFORMATION_CEILING_SCORER_GAP
    if any(gaps):
        return GATE7_INFORMATION_CEILING_MIXED
    return GATE7_INFORMATION_CEILING_INCONCLUSIVE


def protocol_surface() -> dict[str, object]:
    return {
        "protocol_version": GATE7_INFORMATION_CEILING_PROTOCOL_VERSION,
        "base_result_head": GATE7_INFORMATION_CEILING_BASE_RESULT_HEAD,
        "execution_head": GATE7_INFORMATION_CEILING_EXECUTION_HEAD,
        "populations": GATE7_INFORMATION_CEILING_POPULATIONS,
        "checkpoints": GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES,
        "world_count": GATE7_INFORMATION_CEILING_WORLD_COUNT,
        "evaluation_batch_size": GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE,
        "bootstrap_samples": GATE7_INFORMATION_CEILING_BOOTSTRAP_SAMPLES,
        "attempt_ladder": GATE7_INFORMATION_CEILING_ATTEMPT_LADDER,
        "primary_attempts": GATE7_INFORMATION_CEILING_PRIMARY_ATTEMPTS,
        "rankers": GATE7_INFORMATION_CEILING_RANKERS,
        "hint_reliability": GATE7_INFORMATION_CEILING_HINT_RELIABILITY,
        "near_ceiling_margin": GATE7_INFORMATION_CEILING_NEAR_CEILING_MARGIN,
        "expected_primary_ceiling": expected_primary_ceiling_by_population(),
        "complete_frontier_required": True,
        "hidden_parent_must_exist_exactly_once": True,
        "terminal_children_executed_per_selected_parent": 2,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "communication_intervention_allowed": False,
        "adaptive_attempt_exposure_allowed": False,
        "result_reuse_as_fresh_evidence_allowed": False,
        "execution_opened": GATE7_INFORMATION_CEILING_EXECUTION_OPENED,
    }
