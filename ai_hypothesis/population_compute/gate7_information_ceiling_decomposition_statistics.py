"""Rank summaries, paired bootstrap intervals, and frozen campaign classification."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from .gate7_information_ceiling_decomposition_protocol import (
    GATE7_INFORMATION_CEILING_ATTEMPT_LADDER,
    GATE7_INFORMATION_CEILING_BOOTSTRAP_NAMESPACE,
    GATE7_INFORMATION_CEILING_BOOTSTRAP_SAMPLES,
    GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES,
    GATE7_INFORMATION_CEILING_PRIMARY_ATTEMPTS,
    GATE7_INFORMATION_CEILING_RANKERS,
    GATE7_INFORMATION_CEILING_WORLD_COUNT,
    Gate7InformationCeilingComparison,
    classify_information_ceiling_campaign,
)
from .gate7_information_ceiling_decomposition_rank import (
    Gate7InformationCeilingCheckpointRanks,
)
from .gate7_information_ceiling_decomposition_worlds import (
    information_ceiling_seed_from_parts,
)


@dataclass(frozen=True, slots=True)
class Gate7InformationCeilingRankSummary:
    checkpoint_index: int
    population: int
    ranker: str
    coverage_by_attempt: dict[str, float]
    mean_rank: float
    rank_quantiles: dict[str, int]
    mean_reciprocal_rank: float
    mean_log2_rank_plus_one: float
    rank_checksum: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7InformationCeilingPairedSummary:
    comparison: str
    checkpoint_index: int
    population: int
    attempts: int
    treatment_ranker: str
    reference_ranker: str
    coverage_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quantile(values: tuple[int, ...], probability: float) -> int:
    ordered = sorted(values)
    return ordered[int(math.floor(probability * (len(ordered) - 1)))]


def summarize_information_ceiling_ranks(
    *, checkpoint: Gate7InformationCeilingCheckpointRanks, ranker: str
) -> Gate7InformationCeilingRankSummary:
    checkpoint.validate()
    if ranker not in GATE7_INFORMATION_CEILING_RANKERS:
        raise ValueError("rank summary requested an unknown ranker")
    ranks = checkpoint.ranks_by_ranker[ranker]
    count = len(ranks)
    coverage = {
        str(attempts): sum(int(rank <= attempts) for rank in ranks) / count
        for attempts in GATE7_INFORMATION_CEILING_ATTEMPT_LADDER
    }
    return Gate7InformationCeilingRankSummary(
        checkpoint_index=checkpoint.checkpoint_index,
        population=checkpoint.population,
        ranker=ranker,
        coverage_by_attempt=coverage,
        mean_rank=sum(ranks) / count,
        rank_quantiles={
            "p25": _quantile(ranks, 0.25),
            "p50": _quantile(ranks, 0.50),
            "p75": _quantile(ranks, 0.75),
            "p90": _quantile(ranks, 0.90),
            "p95": _quantile(ranks, 0.95),
            "p99": _quantile(ranks, 0.99),
        },
        mean_reciprocal_rank=sum(1.0 / rank for rank in ranks) / count,
        mean_log2_rank_plus_one=sum(math.log2(rank + 1) for rank in ranks) / count,
        rank_checksum=sum(ranks),
    )


def _bootstrap_quantiles(estimates: list[float]) -> tuple[float, float]:
    estimates.sort()
    return (
        estimates[int(math.floor(0.025 * (GATE7_INFORMATION_CEILING_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE7_INFORMATION_CEILING_BOOTSTRAP_SAMPLES - 1)))],
    )


def paired_information_ceiling_summary(
    *,
    comparison: str,
    checkpoint: Gate7InformationCeilingCheckpointRanks,
    treatment_ranker: str,
    reference_ranker: str,
    attempts: int = GATE7_INFORMATION_CEILING_PRIMARY_ATTEMPTS,
) -> Gate7InformationCeilingPairedSummary:
    checkpoint.validate()
    if treatment_ranker not in GATE7_INFORMATION_CEILING_RANKERS:
        raise ValueError("paired treatment ranker is unknown")
    if reference_ranker not in GATE7_INFORMATION_CEILING_RANKERS:
        raise ValueError("paired reference ranker is unknown")
    if attempts not in GATE7_INFORMATION_CEILING_ATTEMPT_LADDER:
        raise ValueError("paired attempts are outside the frozen curve")
    treatment = checkpoint.ranks_by_ranker[treatment_ranker]
    reference = checkpoint.ranks_by_ranker[reference_ranker]
    differences = tuple(
        int(left <= attempts) - int(right <= attempts)
        for left, right in zip(treatment, reference, strict=True)
    )
    rng = random.Random(
        information_ceiling_seed_from_parts(
            GATE7_INFORMATION_CEILING_BOOTSTRAP_NAMESPACE,
            checkpoint.population,
            checkpoint.checkpoint_index,
            attempts,
            comparison,
        )
    )
    count = GATE7_INFORMATION_CEILING_WORLD_COUNT
    estimates = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE7_INFORMATION_CEILING_BOOTSTRAP_SAMPLES)
    ]
    low, high = _bootstrap_quantiles(estimates)
    return Gate7InformationCeilingPairedSummary(
        comparison=comparison,
        checkpoint_index=checkpoint.checkpoint_index,
        population=checkpoint.population,
        attempts=attempts,
        treatment_ranker=treatment_ranker,
        reference_ranker=reference_ranker,
        coverage_delta=sum(differences) / count,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def comparison_for_frozen_classifier(
    *,
    learned_vs_bayes: Gate7InformationCeilingPairedSummary,
    learned_vs_hash: Gate7InformationCeilingPairedSummary,
    bayes_vs_hash: Gate7InformationCeilingPairedSummary,
) -> Gate7InformationCeilingComparison:
    identity = (
        learned_vs_bayes.checkpoint_index,
        learned_vs_bayes.population,
        learned_vs_bayes.attempts,
    )
    if (
        learned_vs_hash.checkpoint_index,
        learned_vs_hash.population,
        learned_vs_hash.attempts,
    ) != identity or (
        bayes_vs_hash.checkpoint_index,
        bayes_vs_hash.population,
        bayes_vs_hash.attempts,
    ) != identity:
        raise ValueError("classifier summaries do not share checkpoint/population/attempts")
    if identity[2] != GATE7_INFORMATION_CEILING_PRIMARY_ATTEMPTS:
        raise ValueError("frozen classifier requires M128")
    return Gate7InformationCeilingComparison(
        checkpoint_index=identity[0],
        population=identity[1],
        learned_minus_bayes_ci_low=learned_vs_bayes.bootstrap_ci_low,
        learned_minus_bayes_ci_high=learned_vs_bayes.bootstrap_ci_high,
        learned_minus_hash_ci_low=learned_vs_hash.bootstrap_ci_low,
        bayes_minus_hash_ci_low=bayes_vs_hash.bootstrap_ci_low,
    )


def classify_from_paired_summaries(
    comparisons: tuple[Gate7InformationCeilingComparison, ...],
) -> str:
    return classify_information_ceiling_campaign(comparisons)
