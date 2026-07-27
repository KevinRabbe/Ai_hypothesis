"""Validation and normalized readout for large-scope benchmark result artifacts.

This module validates experiment identity and causal comparability. It deliberately does not
convert paired evidence into a population-win/fail threshold; scientific interpretation remains
separate from artifact integrity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .relevance import LARGE_SCOPE_BENCHMARK_VERSION

_EXPECTED_MODES = ("same_worker", "diverse_workers")
_REQUIRED_PAIRED_FIELDS = (
    "retrieval_given_inspected_delta",
    "mean_target_rank_delta_when_inspected",
    "mean_target_relevant_evidence_delta_when_inspected",
    "mean_strongest_distractor_relevant_evidence_delta",
    "mean_target_minus_distractor_gap_delta_when_inspected",
    "mean_candidate_relevant_evidence_positive_delta",
    "mean_candidate_relevant_evidence_negative_delta",
)


@dataclass(frozen=True, slots=True)
class LargeScopeWidthReadout:
    width: int
    target_coverage_rate: float
    retrieval_given_inspected_same: float | None
    retrieval_given_inspected_diverse: float | None
    retrieval_given_inspected_delta: float | None
    retrieval_discordance_p_value: float | None
    mean_target_rank_delta: float | None
    se_target_rank_delta: float | None
    mean_target_evidence_delta: float | None
    se_target_evidence_delta: float | None
    mean_target_minus_distractor_gap_delta: float | None
    se_target_minus_distractor_gap_delta: float | None
    mean_negative_candidate_evidence_delta: float | None
    se_negative_candidate_evidence_delta: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "target_coverage_rate": self.target_coverage_rate,
            "retrieval_given_inspected_same": self.retrieval_given_inspected_same,
            "retrieval_given_inspected_diverse": self.retrieval_given_inspected_diverse,
            "retrieval_given_inspected_delta": self.retrieval_given_inspected_delta,
            "retrieval_discordance_p_value": self.retrieval_discordance_p_value,
            "mean_target_rank_delta": self.mean_target_rank_delta,
            "se_target_rank_delta": self.se_target_rank_delta,
            "mean_target_evidence_delta": self.mean_target_evidence_delta,
            "se_target_evidence_delta": self.se_target_evidence_delta,
            "mean_target_minus_distractor_gap_delta": self.mean_target_minus_distractor_gap_delta,
            "se_target_minus_distractor_gap_delta": self.se_target_minus_distractor_gap_delta,
            "mean_negative_candidate_evidence_delta": self.mean_negative_candidate_evidence_delta,
            "se_negative_candidate_evidence_delta": self.se_negative_candidate_evidence_delta,
        }


@dataclass(frozen=True, slots=True)
class LargeScopeResultReadout:
    split: str
    world_count: int
    widths: tuple[int, ...]
    population_width: int
    checkpoint_ids: tuple[str, ...]
    elapsed_seconds: float
    local_window_evaluations: int
    per_width: tuple[LargeScopeWidthReadout, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "world_count": self.world_count,
            "widths": list(self.widths),
            "population_width": self.population_width,
            "checkpoint_ids": list(self.checkpoint_ids),
            "elapsed_seconds": self.elapsed_seconds,
            "local_window_evaluations": self.local_window_evaluations,
            "per_width": [row.to_dict() for row in self.per_width],
        }


def validate_large_scope_result(
    payload: Mapping[str, object],
    *,
    expected_split: str = "development",
    expected_widths: Sequence[int] | None = (1, 4, 16),
    numeric_tolerance: float = 1e-6,
) -> LargeScopeResultReadout:
    """Validate one benchmark artifact and return an interpretation-ready readout.

    Validation is intentionally structural/causal. It does not require any scientific effect size.
    """

    if numeric_tolerance < 0.0 or not math.isfinite(numeric_tolerance):
        raise ValueError("numeric_tolerance must be finite and non-negative")
    if payload.get("benchmark_version") != LARGE_SCOPE_BENCHMARK_VERSION:
        raise ValueError("unexpected large-scope benchmark version")

    split = _text(payload, "split")
    if split != expected_split:
        raise ValueError(f"expected split {expected_split!r}, got {split!r}")

    world_count = _positive_int(payload, "world_count")
    population_width = _positive_int(payload, "population_width")
    widths = _positive_int_tuple(payload.get("widths"), "widths")
    if tuple(sorted(set(widths))) != widths:
        raise ValueError("widths must be unique and increasing")
    if expected_widths is not None and widths != tuple(int(value) for value in expected_widths):
        raise ValueError("result widths do not match the expected benchmark widths")
    if widths[-1] > population_width:
        raise ValueError("largest diverse width exceeds result population width")

    modes = _text_tuple(payload.get("modes"), "modes")
    if modes != _EXPECTED_MODES:
        raise ValueError("paired result requires modes ['same_worker', 'diverse_workers']")

    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != population_width:
        raise ValueError("checkpoint metadata count must equal population_width")
    checkpoint_ids = tuple(_checkpoint_id(row) for row in checkpoints)
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("checkpoint weight identities must be unique")

    elapsed_seconds = _finite_float(payload, "elapsed_seconds")
    if elapsed_seconds < 0.0:
        raise ValueError("elapsed_seconds must be non-negative")
    local_window_evaluations = _positive_int(payload, "local_window_evaluations")
    expected_evaluations = world_count * len(modes) * sum(widths)
    if local_window_evaluations != expected_evaluations:
        raise ValueError("local-window evaluation accounting does not match experiment shape")

    summaries = _condition_index(payload.get("summaries"), widths, world_count)
    paired = _paired_index(payload.get("paired_summaries"), widths, world_count)

    per_width: list[LargeScopeWidthReadout] = []
    for width in widths:
        same = summaries[("same_worker", width)]
        diverse = summaries[("diverse_workers", width)]
        pair = paired[width]

        if same["target_inspected_count"] != diverse["target_inspected_count"]:
            raise ValueError("same/diverse modes disagree on target inspection count")
        if not _close(
            _required_float(same, "target_coverage_rate"),
            _required_float(diverse, "target_coverage_rate"),
            numeric_tolerance,
        ):
            raise ValueError("same/diverse modes disagree on deterministic target coverage")
        if pair["target_inspected_count"] != same["target_inspected_count"]:
            raise ValueError("paired target-inspected count disagrees with condition summaries")

        if width == 1:
            _validate_width_one_identity(pair, numeric_tolerance)

        per_width.append(
            LargeScopeWidthReadout(
                width=width,
                target_coverage_rate=_required_float(same, "target_coverage_rate"),
                retrieval_given_inspected_same=_optional_float(
                    pair.get("retrieval_given_inspected_same")
                ),
                retrieval_given_inspected_diverse=_optional_float(
                    pair.get("retrieval_given_inspected_diverse")
                ),
                retrieval_given_inspected_delta=_optional_float(
                    pair.get("retrieval_given_inspected_delta")
                ),
                retrieval_discordance_p_value=_optional_float(
                    pair.get("exact_retrieval_discordance_p_value")
                ),
                mean_target_rank_delta=_optional_float(
                    pair.get("mean_target_rank_delta_when_inspected")
                ),
                se_target_rank_delta=_optional_float(
                    pair.get("se_target_rank_delta_when_inspected")
                ),
                mean_target_evidence_delta=_optional_float(
                    pair.get("mean_target_relevant_evidence_delta_when_inspected")
                ),
                se_target_evidence_delta=_optional_float(
                    pair.get("se_target_relevant_evidence_delta_when_inspected")
                ),
                mean_target_minus_distractor_gap_delta=_optional_float(
                    pair.get("mean_target_minus_distractor_gap_delta_when_inspected")
                ),
                se_target_minus_distractor_gap_delta=_optional_float(
                    pair.get("se_target_minus_distractor_gap_delta_when_inspected")
                ),
                mean_negative_candidate_evidence_delta=_optional_float(
                    pair.get("mean_candidate_relevant_evidence_negative_delta")
                ),
                se_negative_candidate_evidence_delta=_optional_float(
                    pair.get("se_candidate_relevant_evidence_negative_delta")
                ),
            )
        )

    return LargeScopeResultReadout(
        split=split,
        world_count=world_count,
        widths=widths,
        population_width=population_width,
        checkpoint_ids=checkpoint_ids,
        elapsed_seconds=elapsed_seconds,
        local_window_evaluations=local_window_evaluations,
        per_width=tuple(per_width),
    )


def _condition_index(
    value: object,
    widths: tuple[int, ...],
    world_count: int,
) -> dict[tuple[str, int], Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError("summaries must be a list")
    result: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("summary rows must be mappings")
        mode = _text(row, "mode")
        width = _positive_int(row, "width")
        key = (mode, width)
        if key in result:
            raise ValueError("duplicate condition summary")
        if mode not in _EXPECTED_MODES or width not in widths:
            raise ValueError("unexpected condition summary")
        if _positive_int(row, "world_count") != world_count:
            raise ValueError("condition world_count mismatch")
        positive = _non_negative_int(row, "positive_world_count")
        negative = _non_negative_int(row, "negative_world_count")
        if positive + negative != world_count:
            raise ValueError("condition positive/negative world counts do not close")
        result[key] = row
    expected = {(mode, width) for mode in _EXPECTED_MODES for width in widths}
    if set(result) != expected:
        raise ValueError("condition summaries do not form the complete mode/width matrix")
    return result


def _paired_index(
    value: object,
    widths: tuple[int, ...],
    world_count: int,
) -> dict[int, Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError("paired_summaries must be a list")
    result: dict[int, Mapping[str, object]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("paired summary rows must be mappings")
        if row.get("delta_definition") != "diverse_workers_minus_same_worker":
            raise ValueError("unexpected paired delta definition")
        width = _positive_int(row, "width")
        if width not in widths or width in result:
            raise ValueError("unexpected or duplicate paired width")
        if _positive_int(row, "pair_count") != world_count:
            raise ValueError("paired summary must contain one pair per world")
        positive = _non_negative_int(row, "positive_world_count")
        negative = _non_negative_int(row, "negative_world_count")
        if positive + negative != world_count:
            raise ValueError("paired positive/negative world counts do not close")
        for field in _REQUIRED_PAIRED_FIELDS:
            _optional_float(row.get(field))
        result[width] = row
    if set(result) != set(widths):
        raise ValueError("paired summaries must contain every requested width exactly once")
    return result


def _validate_width_one_identity(row: Mapping[str, object], tolerance: float) -> None:
    if _non_negative_int(row, "same_only_retrieved_count") != 0:
        raise ValueError("width-1 same/diverse retrieval must be identical")
    if _non_negative_int(row, "diverse_only_retrieved_count") != 0:
        raise ValueError("width-1 same/diverse retrieval must be identical")
    for field in _REQUIRED_PAIRED_FIELDS:
        value = _optional_float(row.get(field))
        if value is not None and abs(value) > tolerance:
            raise ValueError(f"width-1 paired identity failed for {field}")


def _checkpoint_id(value: object) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint metadata rows must be mappings")
    checkpoint_id = _text(value, "checkpoint_id")
    if not checkpoint_id.startswith("weights-sha256-"):
        raise ValueError("checkpoint_id must use stable learned-state identity")
    digest = checkpoint_id.removeprefix("weights-sha256-")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("checkpoint_id contains an invalid SHA-256 digest")
    return checkpoint_id


def _positive_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_as_int(item, name) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(_as_text(item, name) for item in value)


def _text(mapping: Mapping[str, object], key: str) -> str:
    return _as_text(mapping.get(key), key)


def _as_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _positive_int(mapping: Mapping[str, object], key: str) -> int:
    value = _as_int(mapping.get(key), key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _non_negative_int(mapping: Mapping[str, object], key: str) -> int:
    value = _as_int(mapping.get(key), key)
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = _optional_float(mapping.get(key))
    if value is None:
        raise ValueError(f"{key} must be a finite number")
    return value


def _finite_float(mapping: Mapping[str, object], key: str) -> float:
    value = _required_float(mapping, key)
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("metric values must be numeric or null")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("metric values must be finite")
    return resolved


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance
