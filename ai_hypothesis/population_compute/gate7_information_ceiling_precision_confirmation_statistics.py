"""Rank summaries and deterministic clustered bootstrap for Gate-7 precision confirmation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .gate7_information_ceiling_precision_confirmation_protocol import (
    GATE7_PRECISION_ATTEMPT_LADDER,
    GATE7_PRECISION_BOOTSTRAP_NAMESPACE,
    GATE7_PRECISION_BOOTSTRAP_SAMPLES,
    GATE7_PRECISION_CHECKPOINT_INDICES,
    GATE7_PRECISION_POPULATIONS,
    GATE7_PRECISION_PRIMARY_ATTEMPTS,
    GATE7_PRECISION_RANKERS,
    GATE7_PRECISION_WORLD_COUNT,
    Gate7PrecisionCellComparison,
    Gate7PrecisionPooledComparison,
    Gate7PrecisionPopulationComparison,
    classify_gate7_precision_confirmation,
)
from .gate7_information_ceiling_precision_confirmation_rank import (
    Gate7PrecisionCheckpointRanks,
)

LEARNED, BAYES, HASH = GATE7_PRECISION_RANKERS
_COMPARISONS = ("learned_vs_bayes", "learned_vs_hash", "bayes_vs_hash")
_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)


@dataclass(frozen=True, slots=True)
class Gate7PrecisionRankSummary:
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
class Gate7PrecisionBootstrapSummary:
    scope: str
    population: int | None
    checkpoint_index: int | None
    comparison: str
    attempts: int
    coverage_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def precision_bootstrap_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (values + _SPLITMIX_GAMMA) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(30))) * _SPLITMIX_M1) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(27))) * _SPLITMIX_M2) & _UINT64_MASK
        return (z ^ (z >> np.uint64(31))) & _UINT64_MASK


def _bootstrap_estimates(values: np.ndarray, *, seed: int) -> np.ndarray:
    if values.shape[0] != GATE7_PRECISION_WORLD_COUNT:
        raise ValueError("precision bootstrap requires exact world clusters")
    if values.ndim != 2:
        raise ValueError("precision bootstrap values must be world-by-comparison")
    sample_count = GATE7_PRECISION_BOOTSTRAP_SAMPLES
    world_count = GATE7_PRECISION_WORLD_COUNT
    estimates = np.empty((sample_count, values.shape[1]), dtype=np.float64)
    draws = np.arange(world_count, dtype=np.uint64)[None, :]
    chunk = 128
    for start in range(0, sample_count, chunk):
        stop = min(start + chunk, sample_count)
        replicates = np.arange(start, stop, dtype=np.uint64)[:, None]
        with np.errstate(over="ignore"):
            counters = np.uint64(seed) + replicates * np.uint64(world_count) + draws
        indices = (_splitmix64(counters) % np.uint64(world_count)).astype(np.int64)
        estimates[start:stop] = values[indices].mean(axis=1)
    return estimates


def _intervals(estimates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(estimates, axis=0)
    low_index = int(
        math.floor(0.025 * (GATE7_PRECISION_BOOTSTRAP_SAMPLES - 1))
    )
    high_index = int(
        math.ceil(0.975 * (GATE7_PRECISION_BOOTSTRAP_SAMPLES - 1))
    )
    return ordered[low_index], ordered[high_index]


def _difference_matrix(checkpoint: Gate7PrecisionCheckpointRanks) -> np.ndarray:
    checkpoint.validate()
    learned = np.asarray(checkpoint.ranks_by_ranker[LEARNED], dtype=np.int64)
    bayes = np.asarray(checkpoint.ranks_by_ranker[BAYES], dtype=np.int64)
    public_hash = np.asarray(checkpoint.ranks_by_ranker[HASH], dtype=np.int64)
    attempts = GATE7_PRECISION_PRIMARY_ATTEMPTS
    learned_hit = learned <= attempts
    bayes_hit = bayes <= attempts
    hash_hit = public_hash <= attempts
    return np.column_stack(
        (
            learned_hit.astype(np.int8) - bayes_hit.astype(np.int8),
            learned_hit.astype(np.int8) - hash_hit.astype(np.int8),
            bayes_hit.astype(np.int8) - hash_hit.astype(np.int8),
        )
    ).astype(np.float64)


def _summaries_from_estimates(
    *,
    scope: str,
    population: int | None,
    checkpoint_index: int | None,
    point: np.ndarray,
    estimates: np.ndarray,
) -> tuple[Gate7PrecisionBootstrapSummary, ...]:
    low, high = _intervals(estimates)
    return tuple(
        Gate7PrecisionBootstrapSummary(
            scope=scope,
            population=population,
            checkpoint_index=checkpoint_index,
            comparison=name,
            attempts=GATE7_PRECISION_PRIMARY_ATTEMPTS,
            coverage_delta=float(point[index]),
            bootstrap_ci_low=float(low[index]),
            bootstrap_ci_high=float(high[index]),
        )
        for index, name in enumerate(_COMPARISONS)
    )


def summarize_gate7_precision_ranks(
    *, checkpoint: Gate7PrecisionCheckpointRanks, ranker: str
) -> Gate7PrecisionRankSummary:
    checkpoint.validate()
    if ranker not in GATE7_PRECISION_RANKERS:
        raise ValueError("precision rank summary requested an unknown ranker")
    ranks = tuple(checkpoint.ranks_by_ranker[ranker])
    ordered = sorted(ranks)

    def quantile(probability: float) -> int:
        return ordered[int(math.floor(probability * (len(ordered) - 1)))]

    return Gate7PrecisionRankSummary(
        checkpoint_index=checkpoint.checkpoint_index,
        population=checkpoint.population,
        ranker=ranker,
        coverage_by_attempt={
            str(attempt): sum(int(rank <= attempt) for rank in ranks) / len(ranks)
            for attempt in GATE7_PRECISION_ATTEMPT_LADDER
        },
        mean_rank=sum(ranks) / len(ranks),
        rank_quantiles={
            "p25": quantile(0.25),
            "p50": quantile(0.50),
            "p75": quantile(0.75),
            "p90": quantile(0.90),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
        },
        mean_reciprocal_rank=sum(1.0 / rank for rank in ranks) / len(ranks),
        mean_log2_rank_plus_one=sum(math.log2(rank + 1) for rank in ranks)
        / len(ranks),
        rank_checksum=sum(ranks),
    )


def gate7_precision_cell_statistics(
    checkpoint: Gate7PrecisionCheckpointRanks,
) -> tuple[
    tuple[Gate7PrecisionBootstrapSummary, ...], Gate7PrecisionCellComparison
]:
    matrix = _difference_matrix(checkpoint)
    estimates = _bootstrap_estimates(
        matrix,
        seed=precision_bootstrap_seed(
            GATE7_PRECISION_BOOTSTRAP_NAMESPACE,
            "cell",
            checkpoint.population,
            checkpoint.checkpoint_index,
        ),
    )
    summaries = _summaries_from_estimates(
        scope="cell",
        population=checkpoint.population,
        checkpoint_index=checkpoint.checkpoint_index,
        point=matrix.mean(axis=0),
        estimates=estimates,
    )
    by_name = {row.comparison: row for row in summaries}
    learned_bayes = by_name["learned_vs_bayes"]
    comparison = Gate7PrecisionCellComparison(
        population=checkpoint.population,
        checkpoint_index=checkpoint.checkpoint_index,
        learned_minus_bayes_delta=learned_bayes.coverage_delta,
        learned_minus_bayes_ci_low=learned_bayes.bootstrap_ci_low,
        learned_minus_bayes_ci_high=learned_bayes.bootstrap_ci_high,
        learned_minus_hash_ci_low=by_name["learned_vs_hash"].bootstrap_ci_low,
        bayes_minus_hash_ci_low=by_name["bayes_vs_hash"].bootstrap_ci_low,
    )
    comparison.validate()
    return summaries, comparison


def gate7_precision_population_statistics(
    checkpoints: tuple[Gate7PrecisionCheckpointRanks, ...],
) -> tuple[
    tuple[Gate7PrecisionBootstrapSummary, ...],
    Gate7PrecisionPopulationComparison,
]:
    if tuple(row.checkpoint_index for row in checkpoints) != (
        GATE7_PRECISION_CHECKPOINT_INDICES
    ):
        raise ValueError("precision population statistics require T0/T1/T2 order")
    population = checkpoints[0].population
    if any(row.population != population for row in checkpoints):
        raise ValueError("precision population statistics mix populations")
    matrices = np.stack([_difference_matrix(row) for row in checkpoints], axis=1)
    clustered = matrices.mean(axis=1)
    estimates = _bootstrap_estimates(
        clustered,
        seed=precision_bootstrap_seed(
            GATE7_PRECISION_BOOTSTRAP_NAMESPACE, "population", population
        ),
    )
    summaries = _summaries_from_estimates(
        scope="population",
        population=population,
        checkpoint_index=None,
        point=clustered.mean(axis=0),
        estimates=estimates,
    )
    by_name = {row.comparison: row for row in summaries}
    learned_bayes = by_name["learned_vs_bayes"]
    comparison = Gate7PrecisionPopulationComparison(
        population=population,
        learned_minus_bayes_delta=learned_bayes.coverage_delta,
        learned_minus_bayes_ci_low=learned_bayes.bootstrap_ci_low,
        learned_minus_bayes_ci_high=learned_bayes.bootstrap_ci_high,
        learned_minus_hash_ci_low=by_name["learned_vs_hash"].bootstrap_ci_low,
        bayes_minus_hash_ci_low=by_name["bayes_vs_hash"].bootstrap_ci_low,
    )
    comparison.validate()
    return summaries, comparison


def gate7_precision_pooled_statistics(
    checkpoints_by_population: dict[
        int, tuple[Gate7PrecisionCheckpointRanks, ...]
    ],
) -> tuple[
    tuple[Gate7PrecisionBootstrapSummary, ...], Gate7PrecisionPooledComparison
]:
    if tuple(checkpoints_by_population) != GATE7_PRECISION_POPULATIONS:
        raise ValueError(
            "precision pooled statistics require population-major ladder order"
        )
    population_points: list[np.ndarray] = []
    population_estimates: list[np.ndarray] = []
    for population in GATE7_PRECISION_POPULATIONS:
        checkpoints = checkpoints_by_population[population]
        if tuple(row.checkpoint_index for row in checkpoints) != (
            GATE7_PRECISION_CHECKPOINT_INDICES
        ):
            raise ValueError("precision pooled statistics require T0/T1/T2 order")
        matrices = np.stack(
            [_difference_matrix(row) for row in checkpoints], axis=1
        )
        clustered = matrices.mean(axis=1)
        population_points.append(clustered.mean(axis=0))
        population_estimates.append(
            _bootstrap_estimates(
                clustered,
                seed=precision_bootstrap_seed(
                    GATE7_PRECISION_BOOTSTRAP_NAMESPACE, "pooled", population
                ),
            )
        )
    point = np.stack(population_points, axis=0).mean(axis=0)
    estimates = np.stack(population_estimates, axis=0).mean(axis=0)
    summaries = _summaries_from_estimates(
        scope="pooled",
        population=None,
        checkpoint_index=None,
        point=point,
        estimates=estimates,
    )
    by_name = {row.comparison: row for row in summaries}
    learned_bayes = by_name["learned_vs_bayes"]
    comparison = Gate7PrecisionPooledComparison(
        learned_minus_bayes_delta=learned_bayes.coverage_delta,
        learned_minus_bayes_ci_low=learned_bayes.bootstrap_ci_low,
        learned_minus_bayes_ci_high=learned_bayes.bootstrap_ci_high,
        learned_minus_hash_ci_low=by_name["learned_vs_hash"].bootstrap_ci_low,
        bayes_minus_hash_ci_low=by_name["bayes_vs_hash"].bootstrap_ci_low,
    )
    comparison.validate()
    return summaries, comparison


def classify_gate7_precision_from_rank_matrix(
    checkpoints_by_population: dict[
        int, tuple[Gate7PrecisionCheckpointRanks, ...]
    ],
) -> tuple[
    str,
    tuple[Gate7PrecisionCellComparison, ...],
    tuple[Gate7PrecisionPopulationComparison, ...],
    Gate7PrecisionPooledComparison,
]:
    cells: list[Gate7PrecisionCellComparison] = []
    populations: list[Gate7PrecisionPopulationComparison] = []
    for population in GATE7_PRECISION_POPULATIONS:
        checkpoints = checkpoints_by_population[population]
        for checkpoint in checkpoints:
            _, comparison = gate7_precision_cell_statistics(checkpoint)
            cells.append(comparison)
        _, population_comparison = gate7_precision_population_statistics(checkpoints)
        populations.append(population_comparison)
    _, pooled = gate7_precision_pooled_statistics(checkpoints_by_population)
    outcome = classify_gate7_precision_confirmation(
        cells=tuple(cells), populations=tuple(populations), pooled=pooled
    )
    return outcome, tuple(cells), tuple(populations), pooled
