"""Work/span resource measurements for equivalent relay execution schedules.

This module does not change the learned relay function. It benchmarks the already-qualified
parallel schedule plus two serial controls on identical, pre-materialized relay batches with
gradients disabled.

The serial controls expose an important systems trade-off:

* ``serial_normalized`` minimizes live/cached neural state by recomputing immutable learned
  record projections every hop.
* ``serial_cached_normalized`` computes those immutable projections once, matching the
  parallel path's static learned-projection work, then serializes the recurrent updates.

This prevents the Gate-1 comparison from silently treating equal recurrent-update count as
equal total learned work.
"""

from __future__ import annotations

import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch

from .collective_relay import RelayDifficulty, generate_relay_dataset
from .relay_model import RelayPopulationModel, RelayTensorBatch, build_relay_tensor_batch, decode_node_logits
from .relay_serial_control import (
    RelayScheduleOutput,
    normalized_parallel_forward,
    normalized_serial_cached_forward,
    normalized_serial_forward,
)


RESOURCE_FRONTIER_VERSION = "relay-work-span-frontier-v0"


@dataclass(frozen=True, slots=True)
class RelayResourceBenchmarkConfig:
    population_sizes: tuple[int, ...] = (1, 4, 16, 64, 256)
    batch_sizes: tuple[int, ...] = (1, 64)
    warmup_iterations: int = 20
    measured_iterations: int = 100
    world_seed: int = 0
    equivalence_rtol: float = 2e-5
    equivalence_atol: float = 2e-5

    def validate(self) -> None:
        if not self.population_sizes:
            raise ValueError("population_sizes must be non-empty")
        if tuple(sorted(set(self.population_sizes))) != self.population_sizes:
            raise ValueError("population_sizes must be unique and increasing")
        if self.population_sizes[0] <= 0 or self.population_sizes[-1] > 256:
            raise ValueError("population_sizes must stay inside the frozen 256-slot relay world")
        if not self.batch_sizes or any(size <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must be non-empty and positive")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must be unique")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.measured_iterations <= 0:
            raise ValueError("measured_iterations must be positive")
        if self.world_seed < 0:
            raise ValueError("world_seed must be non-negative")
        if self.equivalence_rtol < 0 or self.equivalence_atol < 0:
            raise ValueError("equivalence tolerances must be non-negative")


@dataclass(frozen=True, slots=True)
class RelayScheduleMeasurement:
    schedule: str
    batch_size: int
    warmup_iterations: int
    measured_iterations: int
    median_batch_latency_ms: float
    p95_batch_latency_ms: float
    min_batch_latency_ms: float
    total_measured_seconds: float
    throughput_samples_per_second: float
    device_median_latency_ms: float | None
    cuda_baseline_allocated_bytes: int | None
    cuda_peak_allocated_bytes: int | None
    cuda_peak_allocated_delta_bytes: int | None
    cuda_baseline_reserved_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    worker_updates_per_sample: int
    candidate_evaluations_per_sample: int
    input_projection_evaluations_per_sample: int
    value_projection_evaluations_per_sample: int
    static_projection_evaluations_per_sample: int
    communicated_scalars_per_sample: int
    peak_active_neural_states_per_sample: int
    cached_state_vectors_per_sample: int
    cached_message_vectors_per_sample: int

    def validate(self) -> None:
        if self.schedule not in {
            "parallel_normalized",
            "serial_normalized",
            "serial_cached_normalized",
        }:
            raise ValueError("unknown schedule measurement")
        if self.batch_size <= 0 or self.measured_iterations <= 0:
            raise ValueError("measurement counts must be positive")
        for value in (
            self.median_batch_latency_ms,
            self.p95_batch_latency_ms,
            self.min_batch_latency_ms,
            self.total_measured_seconds,
            self.throughput_samples_per_second,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timing measurements must be finite and positive")
        if self.worker_updates_per_sample <= 0:
            raise ValueError("worker_updates_per_sample must be positive")
        if self.candidate_evaluations_per_sample != self.worker_updates_per_sample:
            raise ValueError("candidate evaluations must match worker updates")
        if self.input_projection_evaluations_per_sample <= 0:
            raise ValueError("input projection count must be positive")
        if self.value_projection_evaluations_per_sample <= 0:
            raise ValueError("value projection count must be positive")
        if self.static_projection_evaluations_per_sample != (
            self.input_projection_evaluations_per_sample
            + self.value_projection_evaluations_per_sample
        ):
            raise ValueError("static learned projection accounting is inconsistent")
        if self.communicated_scalars_per_sample < 0:
            raise ValueError("communicated scalar count must be non-negative")
        if self.peak_active_neural_states_per_sample <= 0:
            raise ValueError("peak active neural state count must be positive")
        if self.cached_state_vectors_per_sample < 0 or self.cached_message_vectors_per_sample < 0:
            raise ValueError("cached vector counts must be non-negative")


@dataclass(frozen=True, slots=True)
class RelayResourceComparison:
    difficulty: str
    relay_hops: int
    active_workers: int
    batch_size: int
    learned_parameter_count: int
    parameter_fingerprint: str
    outputs_equivalent: bool
    decoded_predictions_equal: bool
    max_abs_logits_difference: float
    max_abs_shared_difference: float
    recurrent_worker_updates_equal: bool
    parallel_cached_static_projection_work_equal: bool
    parallel_learned_span_proxy: int
    serial_low_memory_learned_span_proxy: int
    serial_cached_learned_span_proxy: int
    measurement_order: tuple[str, str, str]
    parallel: RelayScheduleMeasurement
    serial_low_memory: RelayScheduleMeasurement
    serial_cached: RelayScheduleMeasurement
    low_memory_serial_over_parallel_latency_speedup: float
    cached_serial_over_parallel_latency_speedup: float

    def validate(self) -> None:
        if not self.difficulty.strip() or self.relay_hops <= 0 or self.active_workers <= 0:
            raise ValueError("comparison identity is invalid")
        if self.learned_parameter_count <= 0 or not self.parameter_fingerprint:
            raise ValueError("comparison must preserve learned model identity")
        if not self.outputs_equivalent or not self.decoded_predictions_equal:
            raise ValueError("resource comparison requires equivalent neural outputs")
        if not self.recurrent_worker_updates_equal:
            raise ValueError("all schedules must perform the same recurrent worker-update work")
        if not self.parallel_cached_static_projection_work_equal:
            raise ValueError("cached serial control must match parallel static projection work")
        if self.parallel_learned_span_proxy != self.relay_hops:
            raise ValueError("parallel span proxy must equal relay hop count")
        expected_serial_span = self.active_workers * self.relay_hops
        if self.serial_low_memory_learned_span_proxy != expected_serial_span:
            raise ValueError("low-memory serial span proxy is invalid")
        if self.serial_cached_learned_span_proxy != expected_serial_span:
            raise ValueError("cached serial span proxy is invalid")
        if set(self.measurement_order) != {
            "parallel_normalized",
            "serial_normalized",
            "serial_cached_normalized",
        }:
            raise ValueError("measurement order must contain all three schedules exactly once")
        self.parallel.validate()
        self.serial_low_memory.validate()
        self.serial_cached.validate()
        updates = {
            self.parallel.worker_updates_per_sample,
            self.serial_low_memory.worker_updates_per_sample,
            self.serial_cached.worker_updates_per_sample,
        }
        if len(updates) != 1:
            raise ValueError("schedule recurrent worker-update counts differ")
        if (
            self.parallel.static_projection_evaluations_per_sample
            != self.serial_cached.static_projection_evaluations_per_sample
        ):
            raise ValueError("parallel and cached serial static projection work differs")
        if (
            self.serial_low_memory.static_projection_evaluations_per_sample
            < self.serial_cached.static_projection_evaluations_per_sample
        ):
            raise ValueError("low-memory serial control unexpectedly performs less static work")
        for speedup in (
            self.low_memory_serial_over_parallel_latency_speedup,
            self.cached_serial_over_parallel_latency_speedup,
        ):
            if not math.isfinite(speedup) or speedup <= 0:
                raise ValueError("latency speedup must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["speedup_definition"] = "serial_median_latency_divided_by_parallel_median_latency"
        payload["span_proxy_note"] = (
            "Learned recurrent-update depth proxy only; reducer/kernel/runtime span is measured separately by latency."
        )
        payload["serial_control_note"] = (
            "serial_low_memory minimizes cached learned state by recomputing immutable projections; "
            "serial_cached matches parallel static projection counts but retains O(N) projection caches."
        )
        return payload


@dataclass(frozen=True, slots=True)
class RelayResourceFrontierResult:
    benchmark_version: str
    parameter_fingerprint: str
    learned_parameter_count: int
    config: RelayResourceBenchmarkConfig
    device: str
    provenance: dict[str, object]
    comparisons: tuple[RelayResourceComparison, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_version": self.benchmark_version,
            "parameter_fingerprint": self.parameter_fingerprint,
            "learned_parameter_count": self.learned_parameter_count,
            "config": asdict(self.config),
            "device": self.device,
            "provenance": dict(self.provenance),
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "interpretation_note": (
                "This benchmark measures execution organization for an already-proven equivalent function. "
                "It does not create a capability difference or a neural-architecture success claim."
            ),
        }


def benchmark_relay_resource_frontier(
    model: RelayPopulationModel,
    *,
    difficulties: Sequence[RelayDifficulty],
    config: RelayResourceBenchmarkConfig = RelayResourceBenchmarkConfig(),
    device: torch.device | str = "cpu",
) -> RelayResourceFrontierResult:
    """Measure equivalent relay schedules outside data-generation/device-transfer time."""

    config.validate()
    if not difficulties:
        raise ValueError("at least one relay difficulty is required")
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    fingerprint = model.parameter_fingerprint()
    parameter_count = model.trainable_parameter_count()
    comparisons: list[RelayResourceComparison] = []

    with torch.inference_mode():
        for difficulty in difficulties:
            difficulty.validate()
            max_batch = max(config.batch_sizes)
            worlds = generate_relay_dataset(
                start_seed=config.world_seed,
                world_count=max_batch,
                difficulty=difficulty,
            )
            for batch_size in config.batch_sizes:
                world_batch = worlds[:batch_size]
                for active_workers in config.population_sizes:
                    batch = build_relay_tensor_batch(
                        world_batch,
                        active_workers=active_workers,
                        device=target_device,
                    )
                    comparisons.append(
                        benchmark_relay_resource_condition(
                            model,
                            batch,
                            difficulty=difficulty,
                            config=config,
                        )
                    )

    if model.parameter_fingerprint() != fingerprint:
        raise RuntimeError("resource benchmark mutated the relay checkpoint")
    provenance = runtime_provenance(target_device)
    provenance["schedule_timing_policy"] = (
        "each schedule independently warmed; measurement order rotates deterministically by condition"
    )
    return RelayResourceFrontierResult(
        benchmark_version=RESOURCE_FRONTIER_VERSION,
        parameter_fingerprint=fingerprint,
        learned_parameter_count=parameter_count,
        config=config,
        device=str(target_device),
        provenance=provenance,
        comparisons=tuple(comparisons),
    )


def benchmark_relay_resource_condition(
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    *,
    difficulty: RelayDifficulty,
    config: RelayResourceBenchmarkConfig,
) -> RelayResourceComparison:
    """Benchmark one exact `(difficulty, width, batch)` condition with three controls."""

    config.validate()
    batch.validate()
    if batch.hop_count != difficulty.hop_count:
        raise ValueError("batch and difficulty disagree on relay hop count")

    references = {
        "parallel_normalized": normalized_parallel_forward(model, batch),
        "serial_normalized": normalized_serial_forward(model, batch),
        "serial_cached_normalized": normalized_serial_cached_forward(model, batch),
    }
    parallel_reference = references["parallel_normalized"]
    decoded_parallel = decode_node_logits(parallel_reference.logits)
    max_logits_difference = 0.0
    max_shared_difference = 0.0
    decoded_equal = True
    outputs_equivalent = True
    for name in ("serial_normalized", "serial_cached_normalized"):
        row = references[name]
        logits_close = torch.allclose(
            parallel_reference.logits,
            row.logits,
            rtol=config.equivalence_rtol,
            atol=config.equivalence_atol,
        )
        shared_close = torch.allclose(
            parallel_reference.final_shared,
            row.final_shared,
            rtol=config.equivalence_rtol,
            atol=config.equivalence_atol,
        )
        row_decoded_equal = torch.equal(decoded_parallel, decode_node_logits(row.logits))
        outputs_equivalent = outputs_equivalent and bool(logits_close and shared_close)
        decoded_equal = decoded_equal and row_decoded_equal
        max_logits_difference = max(
            max_logits_difference,
            float((parallel_reference.logits - row.logits).abs().max().item()),
        )
        max_shared_difference = max(
            max_shared_difference,
            float((parallel_reference.final_shared - row.final_shared).abs().max().item()),
        )
    if not outputs_equivalent or not decoded_equal:
        raise RuntimeError("relay execution schedules diverged before resource timing")

    forwards: dict[str, Callable[[RelayPopulationModel, RelayTensorBatch], RelayScheduleOutput]] = {
        "parallel_normalized": normalized_parallel_forward,
        "serial_normalized": normalized_serial_forward,
        "serial_cached_normalized": normalized_serial_cached_forward,
    }
    names = ("parallel_normalized", "serial_normalized", "serial_cached_normalized")
    rotation = (batch.active_workers + int(batch.local_inputs.shape[0]) + difficulty.hop_count) % 3
    measurement_order = names[rotation:] + names[:rotation]
    measured: dict[str, RelayScheduleMeasurement] = {}
    for name in measurement_order:
        measured[name] = _measure_schedule(
            forwards[name],
            model,
            batch,
            warmup_iterations=config.warmup_iterations,
            measured_iterations=config.measured_iterations,
        )

    parallel = measured["parallel_normalized"]
    serial_low_memory = measured["serial_normalized"]
    serial_cached = measured["serial_cached_normalized"]
    recurrent_updates_equal = len(
        {
            parallel.worker_updates_per_sample,
            serial_low_memory.worker_updates_per_sample,
            serial_cached.worker_updates_per_sample,
        }
    ) == 1
    static_work_equal = (
        parallel.static_projection_evaluations_per_sample
        == serial_cached.static_projection_evaluations_per_sample
    )
    comparison = RelayResourceComparison(
        difficulty=difficulty.name,
        relay_hops=difficulty.hop_count,
        active_workers=batch.active_workers,
        batch_size=int(batch.local_inputs.shape[0]),
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
        outputs_equivalent=outputs_equivalent,
        decoded_predictions_equal=decoded_equal,
        max_abs_logits_difference=max_logits_difference,
        max_abs_shared_difference=max_shared_difference,
        recurrent_worker_updates_equal=recurrent_updates_equal,
        parallel_cached_static_projection_work_equal=static_work_equal,
        parallel_learned_span_proxy=difficulty.hop_count,
        serial_low_memory_learned_span_proxy=batch.active_workers * difficulty.hop_count,
        serial_cached_learned_span_proxy=batch.active_workers * difficulty.hop_count,
        measurement_order=measurement_order,
        parallel=parallel,
        serial_low_memory=serial_low_memory,
        serial_cached=serial_cached,
        low_memory_serial_over_parallel_latency_speedup=(
            serial_low_memory.median_batch_latency_ms / parallel.median_batch_latency_ms
        ),
        cached_serial_over_parallel_latency_speedup=(
            serial_cached.median_batch_latency_ms / parallel.median_batch_latency_ms
        ),
    )
    comparison.validate()
    return comparison


def runtime_provenance(device: torch.device) -> dict[str, object]:
    payload: dict[str, object] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "execution_mode": "eager",
        "device_type": device.type,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        payload.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cuda_device_index": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return payload


def _measure_schedule(
    forward: Callable[[RelayPopulationModel, RelayTensorBatch], RelayScheduleOutput],
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> RelayScheduleMeasurement:
    device = batch.local_inputs.device
    output: RelayScheduleOutput | None = None
    for _ in range(warmup_iterations):
        output = forward(model, batch)
    _synchronize(device)

    baseline_allocated: int | None = None
    baseline_reserved: int | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))

    latencies_ms: list[float] = []
    device_latencies_ms: list[float] = []
    measured_started = time.perf_counter()
    for _ in range(measured_iterations):
        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter()
        output = forward(model, batch)
        if end_event is not None:
            end_event.record()
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        latencies_ms.append(elapsed_ms)
        if start_event is not None and end_event is not None:
            device_latencies_ms.append(float(start_event.elapsed_time(end_event)))
    total_measured_seconds = time.perf_counter() - measured_started

    if output is None:
        raise RuntimeError("schedule measurement produced no output")
    telemetry = output.telemetry
    peak_allocated = peak_reserved = peak_delta = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        assert baseline_allocated is not None
        peak_delta = max(0, peak_allocated - baseline_allocated)

    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    measurement = RelayScheduleMeasurement(
        schedule=telemetry.schedule,
        batch_size=int(batch.local_inputs.shape[0]),
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        median_batch_latency_ms=float(statistics.median(latencies_ms)),
        p95_batch_latency_ms=float(sorted_latencies[p95_index]),
        min_batch_latency_ms=float(sorted_latencies[0]),
        total_measured_seconds=total_measured_seconds,
        throughput_samples_per_second=(
            int(batch.local_inputs.shape[0]) * measured_iterations / total_measured_seconds
        ),
        device_median_latency_ms=(
            float(statistics.median(device_latencies_ms)) if device_latencies_ms else None
        ),
        cuda_baseline_allocated_bytes=baseline_allocated,
        cuda_peak_allocated_bytes=peak_allocated,
        cuda_peak_allocated_delta_bytes=peak_delta,
        cuda_baseline_reserved_bytes=baseline_reserved,
        cuda_peak_reserved_bytes=peak_reserved,
        worker_updates_per_sample=telemetry.worker_updates_per_sample,
        candidate_evaluations_per_sample=telemetry.candidate_evaluations_per_sample,
        input_projection_evaluations_per_sample=(
            telemetry.input_projection_evaluations_per_sample
        ),
        value_projection_evaluations_per_sample=(
            telemetry.value_projection_evaluations_per_sample
        ),
        static_projection_evaluations_per_sample=(
            telemetry.static_projection_evaluations_per_sample
        ),
        communicated_scalars_per_sample=(
            telemetry.inter_state_communicated_scalars_per_sample
        ),
        peak_active_neural_states_per_sample=(
            telemetry.peak_active_neural_states_per_sample
        ),
        cached_state_vectors_per_sample=telemetry.cached_state_vectors_per_sample,
        cached_message_vectors_per_sample=telemetry.cached_message_vectors_per_sample,
    )
    measurement.validate()
    return measurement


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
