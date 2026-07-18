"""Deterministic Step 1 baselines.

These baselines are deliberately explicit. They are controls, not strawmen:
where simple deterministic processing solves a benchmark family reliably and
cheaply, that result counts against using neural computation for that task.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from .schema import DATA_START, BenchmarkSample, TaskFamily

_PATTERN = (
    (1.0, 0.25, -0.75, 0.0),
    (0.25, 1.0, 0.0, -0.75),
    (-0.75, 0.0, 1.0, 0.25),
    (0.0, -0.75, 0.25, 1.0),
)
_SIGNATURE_A = (1.0, -0.5, 0.75, 0.25)
_SIGNATURE_B = (-0.25, 1.0, -0.5, 0.75)


def oracle_label(sample: BenchmarkSample) -> str:
    """Recover the target from latent generator metadata.

    This is used to verify generator-label consistency. It is not an inference
    baseline because the latent metadata is unavailable to the model at runtime.
    """

    m = sample.metadata
    if sample.task is TaskFamily.PATTERN:
        if bool(m["latent_ambiguous"]):
            return "UNCERTAIN"
        return "SIGNAL" if bool(m["latent_present"]) else "NO_SIGNAL"

    if sample.task is TaskFamily.CHANGE:
        delta = float(m["delta_norm"])
        if delta <= float(m["no_change_max"]):
            return "NO_CHANGE"
        if delta >= float(m["change_min"]):
            return "CHANGE"
        return "UNCERTAIN"

    if sample.task is TaskFamily.CONFLICT:
        delta = float(m["delta_norm"])
        if delta <= float(m["compatible_max"]):
            return "COMPATIBLE"
        if delta >= float(m["conflict_min"]):
            return "CONFLICT"
        return "UNCERTAIN"

    if sample.task is TaskFamily.RELATION:
        gap = float(m["gap_b_minus_a"])
        if abs(gap) <= float(m["same_time_max"]):
            return "SAME_TIME"
        if abs(gap) < float(m["decisive_gap_min"]):
            return "UNCERTAIN"
        return "A_BEFORE_B" if gap > 0 else "B_BEFORE_A"

    if sample.task is TaskFamily.RELEVANCE:
        if bool(m["latent_ambiguous"]):
            return "UNCERTAIN"
        pos_a = int(m["pos_a"])
        pos_b = int(m["pos_b"])
        gap = pos_b - pos_a
        return (
            "RELEVANT"
            if pos_b >= 0 and 0 < gap <= int(m["max_relevant_gap"])
            else "NOT_RELEVANT"
        )

    raise AssertionError(f"unhandled task {sample.task}")


def predict_baselines(sample: BenchmarkSample) -> dict[str, str]:
    """Return all applicable non-oracle deterministic baseline predictions."""

    if sample.task is TaskFamily.PATTERN:
        return {"template_correlation": _pattern_baseline(sample)}
    if sample.task is TaskFamily.CHANGE:
        return {
            "unweighted_distance": _change_baseline(sample, weighted=False),
            "reliability_weighted_distance": _change_baseline(sample, weighted=True),
        }
    if sample.task is TaskFamily.CONFLICT:
        return {
            "unweighted_distance": _conflict_baseline(sample, weighted=False),
            "reliability_weighted_distance": _conflict_baseline(sample, weighted=True),
        }
    if sample.task is TaskFamily.RELATION:
        return {
            "unweighted_time_mean": _relation_baseline(sample, weighted=False),
            "reliability_weighted_time_mean": _relation_baseline(
                sample, weighted=True
            ),
        }
    if sample.task is TaskFamily.RELEVANCE:
        return {"ordered_signature_match": _relevance_baseline(sample)}
    raise AssertionError(f"unhandled task {sample.task}")


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine(left: tuple[float, ...] | list[float], right: tuple[float, ...]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator <= 1e-12:
        return 0.0
    return _dot(left, right) / denominator


def _pattern_baseline(sample: BenchmarkSample) -> str:
    rows = [
        sample.features[index][:4]
        for index in range(DATA_START, DATA_START + 24)
        if sample.mask[index]
    ]
    template_energy = sum(_dot(row, row) for row in _PATTERN)
    best_score = -math.inf

    for start in range(0, len(rows) - len(_PATTERN) + 1):
        score = 0.0
        for local_index, template_row in enumerate(_PATTERN):
            score += _dot(rows[start + local_index], template_row)
        best_score = max(best_score, score / template_energy)

    if best_score >= 0.62:
        return "SIGNAL"
    if best_score >= 0.28:
        return "UNCERTAIN"
    return "NO_SIGNAL"


def _mean_observation(
    rows: list[tuple[float, ...]],
    *,
    value_width: int,
    weighted: bool,
) -> list[float]:
    total = [0.0] * value_width
    total_weight = 0.0
    for row in rows:
        reliability = float(row[value_width])
        if reliability < 0.0:
            continue
        weight = reliability if weighted else 1.0
        for index in range(value_width):
            total[index] += float(row[index]) * weight
        total_weight += weight

    if total_weight <= 1e-12:
        return [0.0] * value_width
    return [value / total_weight for value in total]


def _euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _change_baseline(sample: BenchmarkSample, *, weighted: bool) -> str:
    rows_a = [sample.features[DATA_START + index] for index in range(8)]
    rows_b = [sample.features[DATA_START + 8 + index] for index in range(8)]
    mean_a = _mean_observation(rows_a, value_width=4, weighted=weighted)
    mean_b = _mean_observation(rows_b, value_width=4, weighted=weighted)
    distance = _euclidean(mean_a, mean_b)

    if distance <= 0.25:
        return "NO_CHANGE"
    if distance >= 0.55:
        return "CHANGE"
    return "UNCERTAIN"


def _conflict_baseline(sample: BenchmarkSample, *, weighted: bool) -> str:
    rows_a = [sample.features[DATA_START + index] for index in range(6)]
    rows_b = [sample.features[DATA_START + 6 + index] for index in range(6)]
    mean_a = _mean_observation(rows_a, value_width=5, weighted=weighted)
    mean_b = _mean_observation(rows_b, value_width=5, weighted=weighted)
    distance = _euclidean(mean_a, mean_b)

    if distance <= 0.28:
        return "COMPATIBLE"
    if distance >= 0.68:
        return "CONFLICT"
    return "UNCERTAIN"


def _event_time(sample: BenchmarkSample, event_index: int, *, weighted: bool) -> float:
    total = 0.0
    total_weight = 0.0
    start = DATA_START + event_index * 7
    for offset in range(7):
        row = sample.features[start + offset]
        reliability = float(row[1])
        if reliability < 0.0:
            continue
        weight = reliability if weighted else 1.0
        total += float(row[0]) * weight
        total_weight += weight
    return total / total_weight if total_weight > 1e-12 else 0.0


def _relation_baseline(sample: BenchmarkSample, *, weighted: bool) -> str:
    time_a = _event_time(sample, 0, weighted=weighted)
    time_b = _event_time(sample, 1, weighted=weighted)
    gap = time_b - time_a

    if abs(gap) <= 0.15:
        return "SAME_TIME"
    if abs(gap) < 0.50:
        return "UNCERTAIN"
    return "A_BEFORE_B" if gap > 0.0 else "B_BEFORE_A"


def _relevance_baseline(sample: BenchmarkSample) -> str:
    rows = [
        sample.features[index][:4]
        for index in range(DATA_START, DATA_START + 24)
        if sample.mask[index]
    ]
    scores_a = [_cosine(row, _SIGNATURE_A) for row in rows]
    scores_b = [_cosine(row, _SIGNATURE_B) for row in rows]

    pos_a = max(range(len(scores_a)), key=scores_a.__getitem__)
    pos_b = max(range(len(scores_b)), key=scores_b.__getitem__)
    score_a = scores_a[pos_a]
    score_b = scores_b[pos_b]
    minimum = min(score_a, score_b)

    if 0.55 <= minimum < 0.78:
        return "UNCERTAIN"
    if minimum >= 0.78 and 0 < pos_b - pos_a <= 4:
        return "RELEVANT"
    return "NOT_RELEVANT"
