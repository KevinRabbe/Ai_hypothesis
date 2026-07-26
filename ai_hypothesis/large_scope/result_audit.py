"""Pure-Python structural audit and rendering for large-scope relevance result JSON."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EXPECTED_BENCHMARK_VERSION = "large-scope-relevance-v0"
_ALLOWED_SPLITS = frozenset({"development", "confirmation", "test"})
_ALLOWED_MODES = ("same_worker", "diverse_workers")
_WIDTH1_ZERO_FIELDS = (
    "retrieval_given_inspected_delta",
    "mean_target_rank_delta_when_inspected",
    "se_target_rank_delta_when_inspected",
    "mean_target_relevant_evidence_delta_when_inspected",
    "se_target_relevant_evidence_delta_when_inspected",
    "mean_strongest_distractor_relevant_evidence_delta",
    "se_strongest_distractor_relevant_evidence_delta",
    "mean_target_minus_distractor_gap_delta_when_inspected",
    "se_target_minus_distractor_gap_delta_when_inspected",
    "mean_candidate_relevant_evidence_positive_delta",
    "se_candidate_relevant_evidence_positive_delta",
    "mean_candidate_relevant_evidence_negative_delta",
    "se_candidate_relevant_evidence_negative_delta",
)


@dataclass(frozen=True, slots=True)
class ResultAuditIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class LargeScopeResultAudit:
    benchmark_version: str
    split: str
    world_count: int
    widths: tuple[int, ...]
    modes: tuple[str, ...]
    errors: tuple[ResultAuditIssue, ...]
    warnings: tuple[ResultAuditIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> "LargeScopeResultAudit":
        if self.errors:
            joined = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
            raise ValueError(f"large-scope result audit failed: {joined}")
        return self


def audit_large_scope_result(
    payload: Mapping[str, Any],
    *,
    allow_test_split: bool = False,
    zero_tolerance: float = 1e-6,
) -> LargeScopeResultAudit:
    """Validate benchmark-integrity invariants without deciding scientific success."""

    if zero_tolerance < 0 or not math.isfinite(zero_tolerance):
        raise ValueError("zero_tolerance must be finite and non-negative")

    errors: list[ResultAuditIssue] = []
    warnings: list[ResultAuditIssue] = []

    benchmark_version = _text(payload, "benchmark_version", errors)
    split = _text(payload, "split", errors)
    world_count = _positive_int(payload, "world_count", errors)
    widths = _positive_int_tuple(payload.get("widths"), "widths", errors)
    modes = _string_tuple(payload.get("modes"), "modes", errors)

    if benchmark_version and benchmark_version != EXPECTED_BENCHMARK_VERSION:
        errors.append(
            ResultAuditIssue(
                "BENCHMARK_VERSION",
                f"expected {EXPECTED_BENCHMARK_VERSION!r}, got {benchmark_version!r}",
            )
        )
    if split and split not in _ALLOWED_SPLITS:
        errors.append(ResultAuditIssue("SPLIT", f"unknown split {split!r}"))
    if split == "test" and not allow_test_split:
        errors.append(
            ResultAuditIssue(
                "TEST_SPLIT_LOCKED",
                "test-split result requires explicit allow_test_split=True",
            )
        )
    if widths and tuple(sorted(set(widths))) != widths:
        errors.append(
            ResultAuditIssue(
                "WIDTH_ORDER",
                "widths must be unique and supplied in strictly increasing order",
            )
        )
    if modes:
        if len(set(modes)) != len(modes):
            errors.append(ResultAuditIssue("MODE_DUPLICATE", "modes must be unique"))
        unknown_modes = tuple(mode for mode in modes if mode not in _ALLOWED_MODES)
        if unknown_modes:
            errors.append(
                ResultAuditIssue("MODE_UNKNOWN", f"unknown modes: {unknown_modes!r}")
            )

    config = payload.get("config")
    if not isinstance(config, Mapping):
        errors.append(ResultAuditIssue("CONFIG", "config must be an object"))
        window_count = 0
    else:
        window_count = _positive_int(config, "window_count", errors, prefix="config.")
    if widths and window_count and widths[-1] > window_count:
        errors.append(
            ResultAuditIssue(
                "WIDTH_WINDOW_COUNT",
                "largest width exceeds configured window_count",
            )
        )

    population_width = _positive_int(payload, "population_width", errors)
    if (
        widths
        and "diverse_workers" in modes
        and population_width
        and widths[-1] > population_width
    ):
        errors.append(
            ResultAuditIssue(
                "DIVERSE_POPULATION_WIDTH",
                "diverse width exceeds loaded checkpoint population",
            )
        )

    local_evaluations = _non_negative_int(payload, "local_window_evaluations", errors)
    if world_count and widths and modes:
        expected_evaluations = world_count * len(modes) * sum(widths)
        if local_evaluations != expected_evaluations:
            errors.append(
                ResultAuditIssue(
                    "LOCAL_EVALUATION_COUNT",
                    f"expected {expected_evaluations}, got {local_evaluations}",
                )
            )

    if "acceptance_threshold" in payload:
        errors.append(
            ResultAuditIssue(
                "UNEXPECTED_THRESHOLD",
                "v0 result unexpectedly contains a world-level acceptance_threshold",
            )
        )

    conditions = _summary_index(
        payload.get("summaries"),
        split=split,
        world_count=world_count,
        widths=widths,
        modes=modes,
        errors=errors,
    )
    _audit_scope_equivalence(
        conditions,
        widths=widths,
        errors=errors,
        tolerance=zero_tolerance,
    )
    _audit_nested_coverage(
        conditions,
        widths=widths,
        errors=errors,
        tolerance=zero_tolerance,
    )

    paired = _paired_index(
        payload.get("paired_summaries"),
        split=split,
        world_count=world_count,
        widths=widths,
        modes=modes,
        errors=errors,
    )
    if {"same_worker", "diverse_workers"}.issubset(set(modes)):
        _audit_paired_against_conditions(
            paired,
            conditions,
            widths=widths,
            errors=errors,
            tolerance=zero_tolerance,
        )
        _audit_width1_control(
            paired.get(1),
            errors=errors,
            tolerance=zero_tolerance,
        )
    elif paired:
        warnings.append(
            ResultAuditIssue(
                "PAIRED_WITH_SINGLE_MODE",
                "paired summaries are present although both worker modes were not requested",
            )
        )

    if not widths:
        warnings.append(ResultAuditIssue("NO_WIDTHS", "no usable width conditions were found"))

    return LargeScopeResultAudit(
        benchmark_version=benchmark_version,
        split=split,
        world_count=world_count,
        widths=widths,
        modes=modes,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def render_large_scope_audit_markdown(
    payload: Mapping[str, Any],
    audit: LargeScopeResultAudit,
) -> str:
    """Render observations only; deliberately does not declare hypothesis success/failure."""

    lines = [
        "# Large-Scope Relevance Result Audit",
        "",
        f"**Integrity:** {'VALID' if audit.valid else 'INVALID'}",
        "",
        f"- benchmark: `{audit.benchmark_version}`",
        f"- split: `{audit.split}`",
        f"- worlds: {audit.world_count}",
        f"- widths: {', '.join(str(width) for width in audit.widths) or 'none'}",
        f"- modes: {', '.join(audit.modes) or 'none'}",
        "",
    ]

    if audit.errors:
        lines.extend(("## Integrity errors", ""))
        lines.extend(f"- `{issue.code}` — {issue.message}" for issue in audit.errors)
        lines.append("")
    if audit.warnings:
        lines.extend(("## Warnings", ""))
        lines.extend(f"- `{issue.code}` — {issue.message}" for issue in audit.warnings)
        lines.append("")

    conditions = _rows_by_key(payload.get("summaries"), "mode", "width")
    if conditions:
        lines.extend(
            (
                "## Condition summaries",
                "",
                "| Width | Mode | Coverage | Retrieval | Retrieval given inspected | Mean target rank | Mean target-gap | Mean negative candidate | Max negative candidate |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for (mode, width), row in sorted(
            conditions.items(), key=lambda item: (int(item[0][1]), str(item[0][0]))
        ):
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(width),
                        str(mode),
                        _fmt_rate(row.get("target_coverage_rate")),
                        _fmt_rate(row.get("target_retrieval_rate")),
                        _fmt_rate(row.get("retrieval_given_inspected")),
                        _fmt_number(row.get("mean_target_rank_when_inspected")),
                        _fmt_number(row.get("mean_target_minus_distractor_evidence")),
                        _fmt_number(row.get("mean_candidate_relevant_evidence_negative")),
                        _fmt_number(row.get("max_candidate_relevant_evidence_negative")),
                    )
                )
                + " |"
            )
        lines.append("")

    paired = _rows_by_key(payload.get("paired_summaries"), "width")
    if paired:
        lines.extend(
            (
                "## Paired diversity summaries",
                "",
                "All deltas are `diverse_workers - same_worker`.",
                "",
                "| Width | Δ retrieval given inspected | Same-only | Diverse-only | Exact discordance p | Δ target rank | Δ target evidence | Δ distractor evidence | Δ target-gap | Δ negative candidate |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for (width,), row in sorted(paired.items(), key=lambda item: int(item[0][0])):
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(width),
                        _fmt_rate(row.get("retrieval_given_inspected_delta"), signed=True),
                        str(row.get("same_only_retrieved_count", "?")),
                        str(row.get("diverse_only_retrieved_count", "?")),
                        _fmt_number(row.get("exact_retrieval_discordance_p_value")),
                        _fmt_number(row.get("mean_target_rank_delta_when_inspected"), signed=True),
                        _fmt_number(
                            row.get("mean_target_relevant_evidence_delta_when_inspected"),
                            signed=True,
                        ),
                        _fmt_number(
                            row.get("mean_strongest_distractor_relevant_evidence_delta"),
                            signed=True,
                        ),
                        _fmt_number(
                            row.get("mean_target_minus_distractor_gap_delta_when_inspected"),
                            signed=True,
                        ),
                        _fmt_number(
                            row.get("mean_candidate_relevant_evidence_negative_delta"),
                            signed=True,
                        ),
                    )
                )
                + " |"
            )
        lines.extend(
            (
                "",
                "Interpretation signs: Δ retrieval > 0, Δ target rank < 0, Δ target evidence > 0, Δ distractor < 0, Δ target-gap > 0, and Δ negative candidate < 0 favor diverse workers on their respective diagnostics.",
                "",
                "This audit intentionally does **not** apply a research-success threshold.",
                "",
            )
        )

    return "\n".join(lines)


def _summary_index(
    raw: object,
    *,
    split: str,
    world_count: int,
    widths: tuple[int, ...],
    modes: tuple[str, ...],
    errors: list[ResultAuditIssue],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    rows = _mapping_rows(raw, "summaries", errors)
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        mode = row.get("mode")
        width = row.get("width")
        if not isinstance(mode, str) or isinstance(width, bool) or not isinstance(width, int):
            errors.append(ResultAuditIssue("SUMMARY_KEY", "summary has invalid mode/width"))
            continue
        key = (mode, width)
        if key in index:
            errors.append(ResultAuditIssue("SUMMARY_DUPLICATE", f"duplicate summary {key!r}"))
            continue
        index[key] = row
        _audit_common_row(row, split, world_count, errors, f"summary {key!r}")
        _audit_rate_fields(
            row,
            (
                "target_coverage_rate",
                "target_retrieval_rate",
                "retrieval_given_inspected",
            ),
            errors,
            f"summary {key!r}",
        )
        _audit_finite_fields(
            row,
            (
                "mean_target_rank_when_inspected",
                "mean_target_relevant_evidence_when_inspected",
                "mean_strongest_distractor_relevant_evidence",
                "mean_target_minus_distractor_evidence",
                "mean_candidate_relevant_evidence_positive",
                "mean_candidate_relevant_evidence_negative",
                "max_candidate_relevant_evidence_negative",
            ),
            errors,
            f"summary {key!r}",
        )
    expected = {(mode, width) for mode in modes for width in widths}
    missing = expected - set(index)
    extra = set(index) - expected
    if missing:
        errors.append(ResultAuditIssue("SUMMARY_MISSING", f"missing summaries: {sorted(missing)!r}"))
    if extra:
        errors.append(ResultAuditIssue("SUMMARY_EXTRA", f"unexpected summaries: {sorted(extra)!r}"))
    return index


def _paired_index(
    raw: object,
    *,
    split: str,
    world_count: int,
    widths: tuple[int, ...],
    modes: tuple[str, ...],
    errors: list[ResultAuditIssue],
) -> dict[int, Mapping[str, Any]]:
    rows = _mapping_rows(raw, "paired_summaries", errors)
    index: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        width = row.get("width")
        if isinstance(width, bool) or not isinstance(width, int):
            errors.append(ResultAuditIssue("PAIRED_KEY", "paired summary has invalid width"))
            continue
        if width in index:
            errors.append(ResultAuditIssue("PAIRED_DUPLICATE", f"duplicate paired width {width}"))
            continue
        index[width] = row
        _audit_common_row(row, split, world_count, errors, f"paired width {width}")
        if row.get("delta_definition") != "diverse_workers_minus_same_worker":
            errors.append(
                ResultAuditIssue(
                    "PAIRED_DELTA_DEFINITION",
                    f"paired width {width} has invalid delta_definition",
                )
            )
        _audit_rate_fields(
            row,
            (
                "retrieval_given_inspected_same",
                "retrieval_given_inspected_diverse",
            ),
            errors,
            f"paired width {width}",
        )
        _audit_finite_fields(
            row,
            (
                "retrieval_given_inspected_delta",
                "exact_retrieval_discordance_p_value",
                "mean_target_rank_delta_when_inspected",
                "se_target_rank_delta_when_inspected",
                "mean_target_relevant_evidence_delta_when_inspected",
                "se_target_relevant_evidence_delta_when_inspected",
                "mean_strongest_distractor_relevant_evidence_delta",
                "se_strongest_distractor_relevant_evidence_delta",
                "mean_target_minus_distractor_gap_delta_when_inspected",
                "se_target_minus_distractor_gap_delta_when_inspected",
                "mean_candidate_relevant_evidence_positive_delta",
                "se_candidate_relevant_evidence_positive_delta",
                "mean_candidate_relevant_evidence_negative_delta",
                "se_candidate_relevant_evidence_negative_delta",
            ),
            errors,
            f"paired width {width}",
        )
        p_value = row.get("exact_retrieval_discordance_p_value")
        if p_value is not None and _finite_number(p_value) and not 0.0 <= float(p_value) <= 1.0:
            errors.append(
                ResultAuditIssue(
                    "PAIRED_P_VALUE",
                    f"paired width {width} exact discordance p-value is outside [0, 1]",
                )
            )
        for key, value in row.items():
            if key.startswith("se_") and value is not None and _finite_number(value) and float(value) < 0:
                errors.append(
                    ResultAuditIssue(
                        "PAIRED_STANDARD_ERROR",
                        f"paired width {width} has negative {key}",
                    )
                )
    if {"same_worker", "diverse_workers"}.issubset(set(modes)):
        missing = set(widths) - set(index)
        extra = set(index) - set(widths)
        if missing:
            errors.append(ResultAuditIssue("PAIRED_MISSING", f"missing paired widths: {sorted(missing)!r}"))
        if extra:
            errors.append(ResultAuditIssue("PAIRED_EXTRA", f"unexpected paired widths: {sorted(extra)!r}"))
    return index


def _audit_scope_equivalence(
    conditions: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    widths: tuple[int, ...],
    errors: list[ResultAuditIssue],
    tolerance: float,
) -> None:
    for width in widths:
        same = conditions.get(("same_worker", width))
        diverse = conditions.get(("diverse_workers", width))
        if same is None or diverse is None:
            continue
        for field in (
            "positive_world_count",
            "negative_world_count",
            "target_inspected_count",
        ):
            if same.get(field) != diverse.get(field):
                errors.append(
                    ResultAuditIssue(
                        "SCOPE_MODE_MISMATCH",
                        f"width {width} mode summaries disagree on {field}",
                    )
                )
        if not _close_optional(
            same.get("target_coverage_rate"),
            diverse.get("target_coverage_rate"),
            tolerance,
        ):
            errors.append(
                ResultAuditIssue(
                    "SCOPE_COVERAGE_MISMATCH",
                    f"width {width} worker modes disagree on deterministic target coverage",
                )
            )


def _audit_nested_coverage(
    conditions: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    widths: tuple[int, ...],
    errors: list[ResultAuditIssue],
    tolerance: float,
) -> None:
    for mode in _ALLOWED_MODES:
        previous_count: int | None = None
        previous_rate: float | None = None
        for width in widths:
            row = conditions.get((mode, width))
            if row is None:
                continue
            count = row.get("target_inspected_count")
            rate = row.get("target_coverage_rate")
            if isinstance(count, bool) or not isinstance(count, int):
                continue
            if previous_count is not None and count < previous_count:
                errors.append(
                    ResultAuditIssue(
                        "COVERAGE_NOT_NESTED",
                        f"{mode} target-inspected count decreased at width {width}",
                    )
                )
            if _finite_number(rate):
                numeric_rate = float(rate)
                if previous_rate is not None and numeric_rate + tolerance < previous_rate:
                    errors.append(
                        ResultAuditIssue(
                            "COVERAGE_RATE_NOT_NESTED",
                            f"{mode} coverage rate decreased at width {width}",
                        )
                    )
                previous_rate = numeric_rate
            previous_count = count


def _audit_paired_against_conditions(
    paired: Mapping[int, Mapping[str, Any]],
    conditions: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    widths: tuple[int, ...],
    errors: list[ResultAuditIssue],
    tolerance: float,
) -> None:
    for width in widths:
        row = paired.get(width)
        same = conditions.get(("same_worker", width))
        diverse = conditions.get(("diverse_workers", width))
        if row is None or same is None or diverse is None:
            continue
        for paired_field, condition_field in (
            ("positive_world_count", "positive_world_count"),
            ("negative_world_count", "negative_world_count"),
            ("target_inspected_count", "target_inspected_count"),
            ("same_target_retrieved_count", "target_retrieved_count"),
        ):
            if row.get(paired_field) != same.get(condition_field):
                errors.append(
                    ResultAuditIssue(
                        "PAIRED_CONDITION_MISMATCH",
                        f"width {width} {paired_field} disagrees with same_worker {condition_field}",
                    )
                )
        if row.get("diverse_target_retrieved_count") != diverse.get("target_retrieved_count"):
            errors.append(
                ResultAuditIssue(
                    "PAIRED_CONDITION_MISMATCH",
                    f"width {width} diverse retrieval count disagrees with condition summary",
                )
            )
        same_rate = row.get("retrieval_given_inspected_same")
        diverse_rate = row.get("retrieval_given_inspected_diverse")
        delta = row.get("retrieval_given_inspected_delta")
        if not _close_optional(same_rate, same.get("retrieval_given_inspected"), tolerance):
            errors.append(ResultAuditIssue("PAIRED_RATE_MISMATCH", f"width {width} same retrieval rate mismatch"))
        if not _close_optional(diverse_rate, diverse.get("retrieval_given_inspected"), tolerance):
            errors.append(ResultAuditIssue("PAIRED_RATE_MISMATCH", f"width {width} diverse retrieval rate mismatch"))
        if _finite_number(same_rate) and _finite_number(diverse_rate):
            expected_delta = float(diverse_rate) - float(same_rate)
            if not _close_optional(delta, expected_delta, tolerance):
                errors.append(ResultAuditIssue("PAIRED_RATE_DELTA", f"width {width} retrieval delta is inconsistent"))


def _audit_width1_control(
    row: Mapping[str, Any] | None,
    *,
    errors: list[ResultAuditIssue],
    tolerance: float,
) -> None:
    if row is None:
        errors.append(ResultAuditIssue("WIDTH1_MISSING", "paired width-1 summary is required"))
        return
    for field in (
        "same_only_retrieved_count",
        "diverse_only_retrieved_count",
        "retrieval_discordant_count",
    ):
        if row.get(field) != 0:
            errors.append(
                ResultAuditIssue(
                    "WIDTH1_CONTROL_RETRIEVAL",
                    f"width-1 shared-worker control has nonzero {field}",
                )
            )
    if row.get("same_target_retrieved_count") != row.get("diverse_target_retrieved_count"):
        errors.append(
            ResultAuditIssue(
                "WIDTH1_CONTROL_RETRIEVAL",
                "width-1 shared-worker retrieval counts differ between modes",
            )
        )
    p_value = row.get("exact_retrieval_discordance_p_value")
    if p_value is not None:
        errors.append(
            ResultAuditIssue(
                "WIDTH1_CONTROL_P_VALUE",
                "width-1 exact discordance p-value must be null when no discordance exists",
            )
        )
    for field in _WIDTH1_ZERO_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        if not _finite_number(value) or abs(float(value)) > tolerance:
            errors.append(
                ResultAuditIssue(
                    "WIDTH1_CONTROL_DELTA",
                    f"width-1 shared-worker control has nonzero {field}: {value!r}",
                )
            )


def _audit_common_row(
    row: Mapping[str, Any],
    split: str,
    world_count: int,
    errors: list[ResultAuditIssue],
    label: str,
) -> None:
    if split and row.get("split") != split:
        errors.append(ResultAuditIssue("ROW_SPLIT", f"{label} has a different split"))
    if world_count and row.get("world_count", row.get("pair_count")) != world_count:
        errors.append(ResultAuditIssue("ROW_WORLD_COUNT", f"{label} does not cover all worlds"))
    positive = row.get("positive_world_count")
    negative = row.get("negative_world_count")
    if (
        isinstance(positive, int)
        and not isinstance(positive, bool)
        and isinstance(negative, int)
        and not isinstance(negative, bool)
        and world_count
        and positive + negative != world_count
    ):
        errors.append(ResultAuditIssue("ROW_CLASS_COUNT", f"{label} positive+negative count mismatch"))


def _audit_rate_fields(
    row: Mapping[str, Any],
    fields: Sequence[str],
    errors: list[ResultAuditIssue],
    label: str,
) -> None:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
            errors.append(ResultAuditIssue("RATE_RANGE", f"{label} has invalid {field}={value!r}"))


def _audit_finite_fields(
    row: Mapping[str, Any],
    fields: Sequence[str],
    errors: list[ResultAuditIssue],
    label: str,
) -> None:
    for field in fields:
        value = row.get(field)
        if value is not None and not _finite_number(value):
            errors.append(ResultAuditIssue("NONFINITE", f"{label} has non-finite {field}"))


def _mapping_rows(
    raw: object,
    label: str,
    errors: list[ResultAuditIssue],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, list):
        errors.append(ResultAuditIssue("ROW_CONTAINER", f"{label} must be a list"))
        return ()
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            errors.append(ResultAuditIssue("ROW_TYPE", f"{label}[{index}] must be an object"))
            continue
        rows.append(row)
    return tuple(rows)


def _rows_by_key(raw: object, *keys: str) -> dict[tuple[object, ...], Mapping[str, Any]]:
    if not isinstance(raw, list):
        return {}
    result: dict[tuple[object, ...], Mapping[str, Any]] = {}
    for row in raw:
        if isinstance(row, Mapping):
            result[tuple(row.get(key) for key in keys)] = row
    return result


def _text(
    payload: Mapping[str, Any],
    key: str,
    errors: list[ResultAuditIssue],
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(ResultAuditIssue("FIELD_TEXT", f"{key} must be non-empty text"))
        return ""
    return value


def _positive_int(
    payload: Mapping[str, Any],
    key: str,
    errors: list[ResultAuditIssue],
    *,
    prefix: str = "",
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(ResultAuditIssue("FIELD_INT", f"{prefix}{key} must be a positive integer"))
        return 0
    return value


def _non_negative_int(
    payload: Mapping[str, Any],
    key: str,
    errors: list[ResultAuditIssue],
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(ResultAuditIssue("FIELD_INT", f"{key} must be a non-negative integer"))
        return -1
    return value


def _positive_int_tuple(
    raw: object,
    label: str,
    errors: list[ResultAuditIssue],
) -> tuple[int, ...]:
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in raw
    ):
        errors.append(ResultAuditIssue("FIELD_LIST", f"{label} must be a list of positive integers"))
        return ()
    return tuple(raw)


def _string_tuple(
    raw: object,
    label: str,
    errors: list[ResultAuditIssue],
) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(
        not isinstance(value, str) or not value for value in raw
    ):
        errors.append(ResultAuditIssue("FIELD_LIST", f"{label} must be a list of non-empty strings"))
        return ()
    return tuple(raw)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _close_optional(left: object, right: object, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    if not _finite_number(left) or not _finite_number(right):
        return False
    return abs(float(left) - float(right)) <= tolerance


def _fmt_number(value: object, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    if not _finite_number(value):
        return "invalid"
    numeric = float(value)
    return f"{numeric:+.4f}" if signed else f"{numeric:.4f}"


def _fmt_rate(value: object, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    if not _finite_number(value):
        return "invalid"
    numeric = float(value) * 100.0
    return f"{numeric:+.2f}%" if signed else f"{numeric:.2f}%"
