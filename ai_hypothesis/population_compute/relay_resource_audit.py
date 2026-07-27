"""Protocol audit and descriptive reporting for Gate-1 relay resource results.

The auditor intentionally does not invent a Gate-1 speedup threshold. It verifies that a
resource result is scientifically admissible under the frozen protocol, then summarizes the
complete matrix so interpretation cannot silently cherry-pick favorable widths or relay tiers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .relay_resource_frontier import RESOURCE_FRONTIER_VERSION


CANONICAL_POPULATION_SIZES = (1, 4, 16, 64, 256)
CANONICAL_BATCH_SIZES = (1, 64)
CANONICAL_DIFFICULTIES: dict[str, int] = {
    "relay-2": 2,
    "relay-4": 4,
    "relay-8": 8,
}
CANONICAL_WARMUP_ITERATIONS = 20
CANONICAL_MEASURED_ITERATIONS = 100
CANONICAL_WORLD_SEED = 0
CANONICAL_EQUIVALENCE_RTOL = 2e-5
CANONICAL_EQUIVALENCE_ATOL = 2e-5
CANONICAL_LEARNED_PARAMETER_COUNT = 26_669
CANONICAL_EXPERIMENT_VERSION = "population-compute-relay-training-v1"
CANONICAL_PROTOCOL_VERSION = "relay-protocol-v1-normalized-gate-supervised"
CANONICAL_RELAY_BENCHMARK_VERSION = "collective-relay-v1-answer-frontier"
_SCHEDULES = (
    "parallel_normalized",
    "serial_normalized",
    "serial_cached_normalized",
)


@dataclass(frozen=True, slots=True)
class RelayResourceBatchSummary:
    batch_size: int
    condition_count: int
    parallel_faster_than_low_memory_count: int
    parallel_faster_than_cached_count: int
    low_memory_speedup_geometric_mean: float
    cached_speedup_geometric_mean: float
    low_memory_speedup_min: float
    low_memory_speedup_max: float
    cached_speedup_min: float
    cached_speedup_max: float


@dataclass(frozen=True, slots=True)
class RelayResourceAudit:
    protocol_valid: bool
    reasons: tuple[str, ...]
    expected_condition_count: int
    observed_condition_count: int
    device_type: str | None
    learned_parameter_count: int | None
    parameter_fingerprint: str | None
    observed_measurement_rotations: tuple[int, ...]
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
            "batch_summaries": [asdict(summary) for summary in self.batch_summaries],
            "interpretation_boundary": (
                "protocol_valid means the frozen measurement contract was satisfied; it does not "
                "mean Gate 1 passed. Speedup summaries are descriptive and use the complete matrix."
            ),
        }


def audit_relay_resource_result(
    payload: Mapping[str, object],
    *,
    require_cuda: bool = True,
    require_canonical_checkpoint: bool = True,
) -> RelayResourceAudit:
    """Audit one resource-frontier JSON payload against the frozen Gate-1 protocol."""

    reasons: list[str] = []
    expected_condition_count = (
        len(CANONICAL_POPULATION_SIZES)
        * len(CANONICAL_BATCH_SIZES)
        * len(CANONICAL_DIFFICULTIES)
    )

    if payload.get("benchmark_version") != RESOURCE_FRONTIER_VERSION:
        reasons.append("resource benchmark version is not the frozen Gate-1 version")

    config = payload.get("config")
    if not isinstance(config, Mapping):
        reasons.append("result config is missing or invalid")
        config = {}
    _expect_exact_sequence(
        reasons,
        "population_sizes",
        config.get("population_sizes"),
        CANONICAL_POPULATION_SIZES,
    )
    _expect_exact_sequence(
        reasons,
        "batch_sizes",
        config.get("batch_sizes"),
        CANONICAL_BATCH_SIZES,
    )
    _expect_equal(
        reasons,
        "warmup_iterations",
        config.get("warmup_iterations"),
        CANONICAL_WARMUP_ITERATIONS,
    )
    _expect_equal(
        reasons,
        "measured_iterations",
        config.get("measured_iterations"),
        CANONICAL_MEASURED_ITERATIONS,
    )
    _expect_equal(reasons, "world_seed", config.get("world_seed"), CANONICAL_WORLD_SEED)
    _expect_equal(
        reasons,
        "equivalence_rtol",
        config.get("equivalence_rtol"),
        CANONICAL_EQUIVALENCE_RTOL,
    )
    _expect_equal(
        reasons,
        "equivalence_atol",
        config.get("equivalence_atol"),
        CANONICAL_EQUIVALENCE_ATOL,
    )

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        reasons.append("runtime provenance is missing or invalid")
        provenance = {}
    device_type = _optional_text(provenance.get("device_type"))
    if require_cuda and device_type != "cuda":
        reasons.append("decisive Gate-1 audit requires CUDA target-hardware results")
    if provenance.get("execution_mode") != "eager":
        reasons.append("first Gate-1 frontier must use eager execution")
    timing_policy = _optional_text(provenance.get("schedule_timing_policy")) or ""
    if "rotates deterministically" not in timing_policy:
        reasons.append("schedule timing policy does not record deterministic order rotation")

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
        _expect_equal(
            reasons,
            "checkpoint experiment_version",
            checkpoint.get("experiment_version"),
            CANONICAL_EXPERIMENT_VERSION,
        )
        _expect_equal(
            reasons,
            "checkpoint protocol_version",
            checkpoint.get("protocol_version"),
            CANONICAL_PROTOCOL_VERSION,
        )
        _expect_equal(
            reasons,
            "checkpoint benchmark_version",
            checkpoint.get("benchmark_version"),
            CANONICAL_RELAY_BENCHMARK_VERSION,
        )

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
        if row.get("parallel_learned_span_proxy") != expected_hops:
            reasons.append(f"{prefix} parallel learned-span proxy is invalid")
        expected_serial_span = active_workers * expected_hops
        if row.get("serial_low_memory_learned_span_proxy") != expected_serial_span:
            reasons.append(f"{prefix} low-memory serial learned-span proxy is invalid")
        if row.get("serial_cached_learned_span_proxy") != expected_serial_span:
            reasons.append(f"{prefix} cached serial learned-span proxy is invalid")
        for flag in (
            "outputs_equivalent",
            "decoded_predictions_equal",
            "recurrent_worker_updates_equal",
            "parallel_cached_static_projection_work_equal",
        ):
            if row.get(flag) is not True:
                reasons.append(f"{prefix} failed required invariant {flag}")

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

    batch_summaries = tuple(
        _summarize_batch(batch_size, ratios_by_batch[batch_size])
        for batch_size in CANONICAL_BATCH_SIZES
        if ratios_by_batch[batch_size]
    )
    return RelayResourceAudit(
        protocol_valid=not reasons,
        reasons=tuple(reasons),
        expected_condition_count=expected_condition_count,
        observed_condition_count=len(comparisons),
        device_type=device_type,
        learned_parameter_count=learned_parameter_count,
        parameter_fingerprint=parameter_fingerprint,
        observed_measurement_rotations=tuple(sorted(observed_rotations)),
        batch_summaries=batch_summaries,
    )


def render_relay_resource_markdown(
    payload: Mapping[str, object],
    audit: RelayResourceAudit,
) -> str:
    """Render a compact complete-matrix report without creating a Gate pass threshold."""

    lines = [
        "# Relay Gate-1 resource frontier report",
        "",
        f"Protocol valid: **{'YES' if audit.protocol_valid else 'NO'}**",
        "",
        f"Device type: `{audit.device_type}`",
        f"Learned parameters: `{audit.learned_parameter_count}`",
        f"Parameter fingerprint: `{audit.parameter_fingerprint}`",
        f"Conditions: `{audit.observed_condition_count} / {audit.expected_condition_count}`",
        f"Observed measurement rotations: `{list(audit.observed_measurement_rotations)}`",
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
            "No threshold here declares Gate 1 passed.",
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

    comparisons = payload.get("comparisons")
    if isinstance(comparisons, Sequence) and not isinstance(comparisons, (str, bytes)):
        rows = [row for row in comparisons if isinstance(row, Mapping)]
        rows.sort(
            key=lambda row: (
                CANONICAL_BATCH_SIZES.index(int(row.get("batch_size", -1)))
                if row.get("batch_size") in CANONICAL_BATCH_SIZES
                else 99,
                list(CANONICAL_DIFFICULTIES).index(str(row.get("difficulty")))
                if row.get("difficulty") in CANONICAL_DIFFICULTIES
                else 99,
                int(row.get("active_workers", 0)),
            )
        )
        lines.extend(
            [
                "",
                "## All frozen cells",
                "",
                "| Batch | Difficulty | Workers | Low-memory/parallel | Cached/parallel | Parallel peak Δ alloc | Cached peak Δ alloc | Order |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            parallel = row.get("parallel") if isinstance(row.get("parallel"), Mapping) else {}
            cached = row.get("serial_cached") if isinstance(row.get("serial_cached"), Mapping) else {}
            order = row.get("measurement_order")
            order_text = " → ".join(str(value) for value in order) if isinstance(order, Sequence) and not isinstance(order, (str, bytes)) else "?"
            lines.append(
                "| {batch} | {difficulty} | {workers} | {low:.4f}× | {cached_speed:.4f}× | {p_mem} | {c_mem} | {order} |".format(
                    batch=row.get("batch_size"),
                    difficulty=row.get("difficulty"),
                    workers=row.get("active_workers"),
                    low=float(row.get("low_memory_serial_over_parallel_latency_speedup", float("nan"))),
                    cached_speed=float(row.get("cached_serial_over_parallel_latency_speedup", float("nan"))),
                    p_mem=_format_bytes(parallel.get("cuda_peak_allocated_delta_bytes")),
                    c_mem=_format_bytes(cached.get("cuda_peak_allocated_delta_bytes")),
                    order=order_text,
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This report validates protocol completeness and shows the full descriptive frontier. It intentionally does not define a numeric Gate-1 pass threshold or hide unfavorable cells.",
            "",
        ]
    )
    return "\n".join(lines)


def _audit_schedule_accounting(
    reasons: list[str],
    prefix: str,
    *,
    active_workers: int,
    relay_hops: int,
    device_type: str | None,
    batch_size: int,
    parallel: Mapping[str, object],
    low: Mapping[str, object],
    cached: Mapping[str, object],
) -> None:
    expected_updates = active_workers * relay_hops
    expected_latency_source = "cuda_event" if device_type == "cuda" else "host_perf_counter"
    schedules = (
        ("parallel", "parallel_normalized", parallel),
        ("serial_low_memory", "serial_normalized", low),
        ("serial_cached", "serial_cached_normalized", cached),
    )
    for name, expected_schedule_name, schedule in schedules:
        if schedule.get("schedule") != expected_schedule_name:
            reasons.append(f"{prefix} {name} schedule identity is invalid")
        if schedule.get("batch_size") != batch_size:
            reasons.append(f"{prefix} {name} batch_size differs from condition")
        if schedule.get("warmup_iterations") != CANONICAL_WARMUP_ITERATIONS:
            reasons.append(f"{prefix} {name} warm-up count differs from frozen protocol")
        if schedule.get("measured_iterations") != CANONICAL_MEASURED_ITERATIONS:
            reasons.append(f"{prefix} {name} measured-iteration count differs from frozen protocol")
        if schedule.get("worker_updates_per_sample") != expected_updates:
            reasons.append(f"{prefix} {name} worker-update accounting is invalid")
        if schedule.get("candidate_evaluations_per_sample") != expected_updates:
            reasons.append(f"{prefix} {name} candidate accounting is invalid")
        if _positive_float(schedule.get("median_batch_latency_ms")) is None:
            reasons.append(f"{prefix} {name} median latency is invalid")
        if _positive_float(schedule.get("throughput_samples_per_second")) is None:
            reasons.append(f"{prefix} {name} throughput is invalid")
        if _positive_float(schedule.get("host_enqueue_seconds")) is None:
            reasons.append(f"{prefix} {name} host enqueue time is invalid")
        if schedule.get("latency_source") != expected_latency_source:
            reasons.append(f"{prefix} {name} latency source does not match device type")
        if device_type == "cuda":
            if _positive_float(schedule.get("device_median_latency_ms")) is None:
                reasons.append(f"{prefix} {name} CUDA device latency is missing")
            _audit_cuda_memory(reasons, prefix, name, schedule)

    expected_static_cached = 2 * active_workers
    expected_static_low = 2 * active_workers * relay_hops
    for name, schedule, expected_each in (
        ("parallel", parallel, active_workers),
        ("serial_cached", cached, active_workers),
        ("serial_low_memory", low, active_workers * relay_hops),
    ):
        if schedule.get("input_projection_evaluations_per_sample") != expected_each:
            reasons.append(f"{prefix} {name} input projection accounting is invalid")
        if schedule.get("value_projection_evaluations_per_sample") != expected_each:
            reasons.append(f"{prefix} {name} value projection accounting is invalid")
    if parallel.get("static_projection_evaluations_per_sample") != expected_static_cached:
        reasons.append(f"{prefix} parallel static projection accounting is invalid")
    if cached.get("static_projection_evaluations_per_sample") != expected_static_cached:
        reasons.append(f"{prefix} cached serial static projection accounting is invalid")
    if low.get("static_projection_evaluations_per_sample") != expected_static_low:
        reasons.append(f"{prefix} low-memory serial static projection accounting is invalid")

    if parallel.get("peak_active_neural_states_per_sample") != active_workers:
        reasons.append(f"{prefix} parallel live-state accounting is invalid")
    if low.get("peak_active_neural_states_per_sample") != 1:
        reasons.append(f"{prefix} low-memory serial live-state accounting is invalid")
    if cached.get("peak_active_neural_states_per_sample") != 1:
        reasons.append(f"{prefix} cached serial live-state accounting is invalid")
    if parallel.get("cached_state_vectors_per_sample") != active_workers:
        reasons.append(f"{prefix} parallel state-cache accounting is invalid")
    if parallel.get("cached_message_vectors_per_sample") != active_workers:
        reasons.append(f"{prefix} parallel message-cache accounting is invalid")
    if low.get("cached_state_vectors_per_sample") != 0 or low.get("cached_message_vectors_per_sample") != 0:
        reasons.append(f"{prefix} low-memory serial unexpectedly retains O(N) projection caches")
    if cached.get("cached_state_vectors_per_sample") != active_workers:
        reasons.append(f"{prefix} cached serial state-cache accounting is invalid")
    if cached.get("cached_message_vectors_per_sample") != active_workers:
        reasons.append(f"{prefix} cached serial message-cache accounting is invalid")
    if low.get("communicated_scalars_per_sample") != 0:
        reasons.append(f"{prefix} low-memory serial communication accounting is invalid")
    if cached.get("communicated_scalars_per_sample") != 0:
        reasons.append(f"{prefix} cached serial communication accounting is invalid")
    if _optional_int(parallel.get("communicated_scalars_per_sample")) is None:
        reasons.append(f"{prefix} parallel communication accounting is invalid")


def _audit_cuda_memory(
    reasons: list[str],
    prefix: str,
    name: str,
    schedule: Mapping[str, object],
) -> None:
    fields = (
        "cuda_baseline_allocated_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_allocated_delta_bytes",
        "cuda_baseline_reserved_bytes",
        "cuda_peak_reserved_bytes",
    )
    values: dict[str, int] = {}
    for field in fields:
        value = schedule.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            reasons.append(f"{prefix} {name} {field} is missing or invalid")
        else:
            values[field] = value
    if len(values) != len(fields):
        return
    if values["cuda_peak_allocated_bytes"] < values["cuda_baseline_allocated_bytes"]:
        reasons.append(f"{prefix} {name} CUDA allocated peak is below baseline")
    expected_delta = max(
        0,
        values["cuda_peak_allocated_bytes"] - values["cuda_baseline_allocated_bytes"],
    )
    if values["cuda_peak_allocated_delta_bytes"] != expected_delta:
        reasons.append(f"{prefix} {name} CUDA allocated delta is inconsistent")
    if values["cuda_peak_reserved_bytes"] < values["cuda_baseline_reserved_bytes"]:
        reasons.append(f"{prefix} {name} CUDA reserved peak is below baseline")


def _schedule(
    row: Mapping[str, object],
    name: str,
    prefix: str,
    reasons: list[str],
) -> Mapping[str, object] | None:
    value = row.get(name)
    if not isinstance(value, Mapping):
        reasons.append(f"{prefix} is missing {name}")
        return None
    return value


def _summarize_batch(
    batch_size: int,
    ratios: Sequence[tuple[float, float]],
) -> RelayResourceBatchSummary:
    low = [row[0] for row in ratios]
    cached = [row[1] for row in ratios]
    return RelayResourceBatchSummary(
        batch_size=batch_size,
        condition_count=len(ratios),
        parallel_faster_than_low_memory_count=sum(value > 1.0 for value in low),
        parallel_faster_than_cached_count=sum(value > 1.0 for value in cached),
        low_memory_speedup_geometric_mean=_geometric_mean(low),
        cached_speedup_geometric_mean=_geometric_mean(cached),
        low_memory_speedup_min=min(low),
        low_memory_speedup_max=max(low),
        cached_speedup_min=min(cached),
        cached_speedup_max=max(cached),
    )


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _expect_exact_sequence(
    reasons: list[str],
    name: str,
    actual: object,
    expected: Sequence[int],
) -> None:
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
        reasons.append(f"{name} is missing or invalid")
        return
    if tuple(actual) != tuple(expected):
        reasons.append(f"{name} differs from frozen protocol: {list(actual)!r}")


def _expect_equal(reasons: list[str], name: str, actual: object, expected: object) -> None:
    if actual != expected:
        reasons.append(f"{name} differs from frozen protocol: {actual!r} != {expected!r}")


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _format_bytes(value: object) -> str:
    if not isinstance(value, int) or value < 0:
        return "n/a"
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.1f} KiB"
    if value < 1024**3:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value / 1024**3:.2f} GiB"
