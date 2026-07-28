"""Frozen protocol auditor and descriptive report for Gate-1 v1 resource results."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .relay_resource_audit import (
    CANONICAL_BATCH_SIZES,
    CANONICAL_DIFFICULTIES,
    CANONICAL_EXPERIMENT_VERSION,
    CANONICAL_LEARNED_PARAMETER_COUNT,
    CANONICAL_MEASURED_ITERATIONS,
    CANONICAL_POPULATION_SIZES,
    CANONICAL_PROTOCOL_VERSION,
    CANONICAL_RELAY_BENCHMARK_VERSION,
    CANONICAL_RESOURCE_CHECKPOINT_FILE_SHA256,
    CANONICAL_RESOURCE_PARAMETER_FINGERPRINT,
    CANONICAL_RESOURCE_TRAINING_SEED,
    CANONICAL_WARMUP_ITERATIONS,
    CANONICAL_WORLD_SEED,
    RelayResourceBatchSummary,
    _audit_schedule_accounting,
    _expect_equal,
    _expect_exact_sequence,
    _optional_int,
    _optional_text,
    _positive_float,
    _schedule,
    _summarize_batch,
)
from .relay_resource_frontier_v1 import (
    CORRECTNESS_POLICY_V1,
    FP64_EQUIVALENCE_ATOL,
    FP64_EQUIVALENCE_RTOL,
    RESOURCE_FRONTIER_V1_VERSION,
)


_SCHEDULES = (
    "parallel_normalized",
    "serial_normalized",
    "serial_cached_normalized",
)


@dataclass(frozen=True, slots=True)
class RelayResourceAuditV1:
    protocol_valid: bool
    reasons: tuple[str, ...]
    expected_condition_count: int
    observed_condition_count: int
    device_type: str | None
    learned_parameter_count: int | None
    parameter_fingerprint: str | None
    observed_measurement_rotations: tuple[int, ...]
    worst_fp32_logits_drift: float | None
    worst_fp32_shared_drift: float | None
    worst_fp64_logits_drift: float | None
    worst_fp64_shared_drift: float | None
    worst_cross_precision_logits_drift: float | None
    worst_cross_precision_shared_drift: float | None
    batch_summaries: tuple[RelayResourceBatchSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_valid": self.protocol_valid,
            "reasons": list(self.reasons),
            "expected_condition_count": self.expected_condition_count,
            "observed_condition_count": self.observed_condition_count,
            "device_type": self.device_type,
            "learned_parameter_count": self.learned_parameter_count,
            "parameter_fingerprint": self.parameter_fingerprint,
            "observed_measurement_rotations": list(self.observed_measurement_rotations),
            "worst_fp32_logits_drift": self.worst_fp32_logits_drift,
            "worst_fp32_shared_drift": self.worst_fp32_shared_drift,
            "worst_fp64_logits_drift": self.worst_fp64_logits_drift,
            "worst_fp64_shared_drift": self.worst_fp64_shared_drift,
            "worst_cross_precision_logits_drift": self.worst_cross_precision_logits_drift,
            "worst_cross_precision_shared_drift": self.worst_cross_precision_shared_drift,
            "batch_summaries": [asdict(summary) for summary in self.batch_summaries],
            "interpretation_boundary": (
                "protocol_valid means the Gate-1 v1 measurement contract was satisfied; it does not "
                "mean parallel execution won. Speedups remain descriptive over the complete matrix."
            ),
        }


def _nonnegative_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def audit_relay_resource_result_v1(
    payload: Mapping[str, object],
    *,
    require_cuda: bool = True,
    require_canonical_checkpoint: bool = True,
) -> RelayResourceAuditV1:
    reasons: list[str] = []
    expected_condition_count = (
        len(CANONICAL_POPULATION_SIZES)
        * len(CANONICAL_BATCH_SIZES)
        * len(CANONICAL_DIFFICULTIES)
    )

    if payload.get("benchmark_version") != RESOURCE_FRONTIER_V1_VERSION:
        reasons.append("resource benchmark version is not the frozen Gate-1 v1 version")

    policy = payload.get("correctness_policy")
    if not isinstance(policy, Mapping):
        reasons.append("Gate-1 v1 correctness policy is missing or invalid")
        policy = {}
    _expect_equal(reasons, "correctness policy name", policy.get("name"), CORRECTNESS_POLICY_V1)
    _expect_equal(
        reasons,
        "FP64 equivalence rtol",
        policy.get("fp64_equivalence_rtol"),
        FP64_EQUIVALENCE_RTOL,
    )
    _expect_equal(
        reasons,
        "FP64 equivalence atol",
        policy.get("fp64_equivalence_atol"),
        FP64_EQUIVALENCE_ATOL,
    )
    _expect_equal(
        reasons,
        "FP32 tensor allclose gate flag",
        policy.get("fp32_tensor_allclose_is_gate"),
        False,
    )
    _expect_equal(
        reasons,
        "complete-matrix preflight flag",
        policy.get("complete_matrix_preflight_before_timing"),
        True,
    )
    _expect_equal(
        reasons,
        "FP64 offload flag",
        policy.get("fp64_model_offloaded_before_timing"),
        True,
    )

    config = payload.get("config")
    if not isinstance(config, Mapping):
        reasons.append("result config is missing or invalid")
        config = {}
    _expect_exact_sequence(reasons, "population_sizes", config.get("population_sizes"), CANONICAL_POPULATION_SIZES)
    _expect_exact_sequence(reasons, "batch_sizes", config.get("batch_sizes"), CANONICAL_BATCH_SIZES)
    _expect_equal(reasons, "warmup_iterations", config.get("warmup_iterations"), CANONICAL_WARMUP_ITERATIONS)
    _expect_equal(reasons, "measured_iterations", config.get("measured_iterations"), CANONICAL_MEASURED_ITERATIONS)
    _expect_equal(reasons, "world_seed", config.get("world_seed"), CANONICAL_WORLD_SEED)
    _expect_equal(
        reasons,
        "config FP64 equivalence rtol",
        config.get("fp64_equivalence_rtol"),
        FP64_EQUIVALENCE_RTOL,
    )
    _expect_equal(
        reasons,
        "config FP64 equivalence atol",
        config.get("fp64_equivalence_atol"),
        FP64_EQUIVALENCE_ATOL,
    )

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        reasons.append("runtime provenance is missing or invalid")
        provenance = {}
    device_type = _optional_text(provenance.get("device_type"))
    if require_cuda and device_type != "cuda":
        reasons.append("decisive Gate-1 v1 audit requires CUDA target-hardware results")
    if provenance.get("execution_mode") != "eager":
        reasons.append("Gate-1 v1 frontier must use eager execution")
    timing_policy = _optional_text(provenance.get("schedule_timing_policy")) or ""
    if "rotates deterministically" not in timing_policy:
        reasons.append("schedule timing policy does not record deterministic order rotation")
    correctness_preflight = _optional_text(provenance.get("correctness_preflight")) or ""
    if "complete frozen FP32+FP64 matrix passed before timing" not in correctness_preflight:
        reasons.append("provenance does not attest complete FP32+FP64 preflight before timing")
    if "FP64 model offloaded before timing" not in correctness_preflight:
        reasons.append("provenance does not attest FP64 model offload before timing")

    learned_parameter_count = _optional_int(payload.get("learned_parameter_count"))
    parameter_fingerprint = _optional_text(payload.get("parameter_fingerprint"))
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        reasons.append("checkpoint provenance is missing or invalid")
        checkpoint = {}
    if checkpoint.get("parameter_fingerprint") != parameter_fingerprint:
        reasons.append("checkpoint fingerprint does not match measured model fingerprint")
    if require_canonical_checkpoint:
        if learned_parameter_count != CANONICAL_LEARNED_PARAMETER_COUNT:
            reasons.append("learned parameter count is not the canonical relay-v1 count")
        _expect_equal(reasons, "checkpoint training_seed", checkpoint.get("training_seed"), CANONICAL_RESOURCE_TRAINING_SEED)
        _expect_equal(
            reasons,
            "checkpoint parameter_fingerprint",
            checkpoint.get("parameter_fingerprint"),
            CANONICAL_RESOURCE_PARAMETER_FINGERPRINT,
        )
        _expect_equal(
            reasons,
            "checkpoint file_sha256",
            checkpoint.get("file_sha256"),
            CANONICAL_RESOURCE_CHECKPOINT_FILE_SHA256,
        )
        _expect_equal(reasons, "checkpoint experiment_version", checkpoint.get("experiment_version"), CANONICAL_EXPERIMENT_VERSION)
        _expect_equal(reasons, "checkpoint protocol_version", checkpoint.get("protocol_version"), CANONICAL_PROTOCOL_VERSION)
        _expect_equal(reasons, "checkpoint benchmark_version", checkpoint.get("benchmark_version"), CANONICAL_RELAY_BENCHMARK_VERSION)

    raw_comparisons = payload.get("comparisons")
    if not isinstance(raw_comparisons, Sequence) or isinstance(raw_comparisons, (str, bytes)):
        reasons.append("comparisons are missing or invalid")
        raw_comparisons = []
    comparisons = [row for row in raw_comparisons if isinstance(row, Mapping)]
    if len(comparisons) != len(raw_comparisons):
        reasons.append("one or more comparison rows are not objects")
    if len(comparisons) != expected_condition_count:
        reasons.append(
            f"expected {expected_condition_count} frozen conditions, observed {len(comparisons)}"
        )

    observed_cells: set[tuple[str, int, int]] = set()
    observed_rotations: set[int] = set()
    ratios_by_batch: dict[int, list[tuple[float, float]]] = {
        batch_size: [] for batch_size in CANONICAL_BATCH_SIZES
    }
    drift_fields = {
        "max_abs_fp32_logits_difference": [],
        "max_abs_fp32_shared_difference": [],
        "max_abs_fp64_logits_difference": [],
        "max_abs_fp64_shared_difference": [],
        "max_abs_fp32_vs_fp64_logits_difference": [],
        "max_abs_fp32_vs_fp64_shared_difference": [],
    }

    for index, row in enumerate(comparisons):
        prefix = f"comparison[{index}]"
        difficulty = _optional_text(row.get("difficulty"))
        active_workers = _optional_int(row.get("active_workers"))
        batch_size = _optional_int(row.get("batch_size"))
        if difficulty not in CANONICAL_DIFFICULTIES:
            reasons.append(f"{prefix} has unknown difficulty {difficulty!r}")
            continue
        if active_workers not in CANONICAL_POPULATION_SIZES:
            reasons.append(f"{prefix} has noncanonical active_workers {active_workers!r}")
            continue
        if batch_size not in CANONICAL_BATCH_SIZES:
            reasons.append(f"{prefix} has noncanonical batch_size {batch_size!r}")
            continue
        cell = (difficulty, active_workers, batch_size)
        if cell in observed_cells:
            reasons.append(f"duplicate condition {cell!r}")
        observed_cells.add(cell)
        expected_hops = CANONICAL_DIFFICULTIES[difficulty]
        if row.get("relay_hops") != expected_hops:
            reasons.append(f"{prefix} relay_hops does not match {difficulty}")
        if row.get("learned_parameter_count") != learned_parameter_count:
            reasons.append(f"{prefix} learned parameter count differs from result identity")
        if row.get("parameter_fingerprint") != parameter_fingerprint:
            reasons.append(f"{prefix} parameter fingerprint differs from result identity")
        if row.get("outputs_equivalent") is not True or row.get("decoded_predictions_equal") is not True:
            reasons.append(f"{prefix} does not record admitted v1 output equivalence")
        if row.get("recurrent_worker_updates_equal") is not True:
            reasons.append(f"{prefix} recurrent worker-update accounting differs")
        if row.get("parallel_cached_static_projection_work_equal") is not True:
            reasons.append(f"{prefix} cached serial static work differs from parallel")
        if row.get("parallel_learned_span_proxy") != expected_hops:
            reasons.append(f"{prefix} parallel learned-span proxy is invalid")
        expected_serial_span = active_workers * expected_hops
        if row.get("serial_low_memory_learned_span_proxy") != expected_serial_span:
            reasons.append(f"{prefix} low-memory serial learned-span proxy is invalid")
        if row.get("serial_cached_learned_span_proxy") != expected_serial_span:
            reasons.append(f"{prefix} cached serial learned-span proxy is invalid")

        correctness = row.get("correctness_v1")
        if not isinstance(correctness, Mapping):
            reasons.append(f"{prefix} correctness_v1 evidence is missing")
        else:
            for flag in (
                "admissible",
                "fp32_pairwise_decoded_equal",
                "fp64_pairwise_decoded_equal",
                "fp32_vs_fp64_decoded_equal",
                "fp64_pairwise_tensors_close",
            ):
                if correctness.get(flag) is not True:
                    reasons.append(f"{prefix} failed required correctness flag {flag}")
            for field, values in drift_fields.items():
                value = _nonnegative_finite(correctness.get(field))
                if value is None:
                    reasons.append(f"{prefix} {field} is missing or invalid")
                else:
                    values.append(value)

        order = row.get("measurement_order")
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            reasons.append(f"{prefix} measurement_order is invalid")
        else:
            order_tuple = tuple(order)
            if set(order_tuple) != set(_SCHEDULES) or len(order_tuple) != 3:
                reasons.append(f"{prefix} measurement_order is not a schedule permutation")
            elif order_tuple[0] in _SCHEDULES:
                observed_rotations.add(_SCHEDULES.index(order_tuple[0]))

        parallel = _schedule(row, "parallel", prefix, reasons)
        low = _schedule(row, "serial_low_memory", prefix, reasons)
        cached = _schedule(row, "serial_cached", prefix, reasons)
        if parallel is None or low is None or cached is None:
            continue
        _audit_schedule_accounting(
            reasons,
            prefix,
            active_workers=active_workers,
            relay_hops=expected_hops,
            device_type=device_type,
            batch_size=batch_size,
            parallel=parallel,
            low=low,
            cached=cached,
        )

        low_speedup = _positive_float(row.get("low_memory_serial_over_parallel_latency_speedup"))
        cached_speedup = _positive_float(row.get("cached_serial_over_parallel_latency_speedup"))
        if low_speedup is None:
            reasons.append(f"{prefix} low-memory speedup is not finite and positive")
        if cached_speedup is None:
            reasons.append(f"{prefix} cached-serial speedup is not finite and positive")
        if low_speedup is not None and cached_speedup is not None:
            ratios_by_batch[batch_size].append((low_speedup, cached_speedup))

    expected_cells = {
        (difficulty, active_workers, batch_size)
        for difficulty in CANONICAL_DIFFICULTIES
        for active_workers in CANONICAL_POPULATION_SIZES
        for batch_size in CANONICAL_BATCH_SIZES
    }
    missing = sorted(expected_cells - observed_cells)
    if missing:
        reasons.append(f"missing frozen conditions: {missing}")
    if observed_rotations != {0, 1, 2}:
        reasons.append(
            f"frozen matrix must exercise all three schedule-order rotations; observed {sorted(observed_rotations)}"
        )

    summaries = tuple(
        _summarize_batch(batch_size, ratios_by_batch[batch_size])
        for batch_size in CANONICAL_BATCH_SIZES
        if ratios_by_batch[batch_size]
    )

    def worst(field: str) -> float | None:
        values = drift_fields[field]
        return max(values) if values else None

    return RelayResourceAuditV1(
        protocol_valid=not reasons,
        reasons=tuple(reasons),
        expected_condition_count=expected_condition_count,
        observed_condition_count=len(comparisons),
        device_type=device_type,
        learned_parameter_count=learned_parameter_count,
        parameter_fingerprint=parameter_fingerprint,
        observed_measurement_rotations=tuple(sorted(observed_rotations)),
        worst_fp32_logits_drift=worst("max_abs_fp32_logits_difference"),
        worst_fp32_shared_drift=worst("max_abs_fp32_shared_difference"),
        worst_fp64_logits_drift=worst("max_abs_fp64_logits_difference"),
        worst_fp64_shared_drift=worst("max_abs_fp64_shared_difference"),
        worst_cross_precision_logits_drift=worst("max_abs_fp32_vs_fp64_logits_difference"),
        worst_cross_precision_shared_drift=worst("max_abs_fp32_vs_fp64_shared_difference"),
        batch_summaries=summaries,
    )


def render_relay_resource_markdown_v1(
    payload: Mapping[str, object],
    audit: RelayResourceAuditV1,
) -> str:
    lines = [
        "# Relay Gate-1 v1 resource frontier report",
        "",
        f"Protocol valid: **{'YES' if audit.protocol_valid else 'NO'}**",
        "",
        f"Device type: `{audit.device_type}`",
        f"Learned parameters: `{audit.learned_parameter_count}`",
        f"Parameter fingerprint: `{audit.parameter_fingerprint}`",
        f"Conditions: `{audit.observed_condition_count} / {audit.expected_condition_count}`",
        f"Observed measurement rotations: `{list(audit.observed_measurement_rotations)}`",
        "",
        "## Precision-aware correctness envelope",
        "",
        f"- Worst FP32 schedule-pair logits drift: `{audit.worst_fp32_logits_drift}`",
        f"- Worst FP32 schedule-pair shared drift: `{audit.worst_fp32_shared_drift}`",
        f"- Worst FP64 schedule-pair logits drift: `{audit.worst_fp64_logits_drift}`",
        f"- Worst FP64 schedule-pair shared drift: `{audit.worst_fp64_shared_drift}`",
        f"- Worst FP32↔FP64 logits drift: `{audit.worst_cross_precision_logits_drift}`",
        f"- Worst FP32↔FP64 shared drift: `{audit.worst_cross_precision_shared_drift}`",
        "",
        "FP32 tensor drift is descriptive. Admission requires exact decoded agreement across FP32, FP64, and cross-precision executions plus the frozen FP64 tensor corroboration rule.",
        "",
    ]
    if audit.reasons:
        lines.extend(["## Protocol failures", ""])
        lines.extend(f"- {reason}" for reason in audit.reasons)
        lines.append("")

    lines.extend(
        [
            "## Complete-matrix speedup summaries",
            "",
            "Speedup is `serial median latency / parallel median latency`; values above 1 mean parallel was faster for that cell.",
            "No threshold declares Gate 1 passed.",
            "",
            "| Batch | Cells | Parallel faster vs low-memory | Parallel faster vs cached | Geomean low-memory/parallel | Geomean cached/parallel | Cached range |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in audit.batch_summaries:
        lines.append(
            "| {batch} | {count} | {low_wins} | {cached_wins} | {low_geo:.4f}× | {cached_geo:.4f}× | {cached_min:.4f}–{cached_max:.4f}× |".format(
                batch=summary.batch_size,
                count=summary.condition_count,
                low_wins=summary.parallel_faster_than_low_memory_count,
                cached_wins=summary.parallel_faster_than_cached_count,
                low_geo=summary.low_memory_speedup_geometric_mean,
                cached_geo=summary.cached_speedup_geometric_mean,
                cached_min=summary.cached_speedup_min,
                cached_max=summary.cached_speedup_max,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Gate-1 v1 is a revised protocol frozen after the untimed v0 CUDA correctness failure and precision triangulation, but before any admitted timing result. Protocol validity only makes the resource measurements admissible; it does not imply parallel execution is beneficial.",
            "",
        ]
    )
    return "\n".join(lines)
