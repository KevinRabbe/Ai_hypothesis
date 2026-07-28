"""Gate-1 v1 work/span resource frontier with precision-aware correctness preflight.

Gate-1 v0 correctly rejected target-CUDA timing because mathematically equivalent FP32 schedules
can exceed a fixed tensor-allclose tolerance when their execution shapes select different CUDA
kernels. The untimed precision diagnostic then showed exact decoded agreement across FP32/FP64
and float64 pairwise schedule drift near machine precision.

V1 therefore performs the complete frozen correctness matrix before *any* timing:
- all FP32 schedules must decode identically;
- all FP64 schedules must decode identically;
- each FP32 schedule must decode identically to its FP64 counterpart;
- FP64 parallel/serial tensors must agree under the frozen v1 FP64 tolerance.

FP32 tensor drift is still recorded descriptively, but it is not converted into a post-hoc larger
FP32 tolerance. After the global preflight passes, the FP64 model is moved off CUDA before the
resource matrix is timed so the reference model cannot contaminate memory measurements.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from typing import Callable, Sequence

import torch

from .collective_relay import RelayDifficulty, generate_relay_dataset
from .relay_model import RelayPopulationModel, RelayTensorBatch, build_relay_tensor_batch, decode_node_logits
from .relay_resource_frontier import (
    RelayResourceComparison,
    RelayScheduleMeasurement,
    _measure_schedule,
    runtime_provenance,
)
from .relay_serial_control import (
    RelayScheduleOutput,
    normalized_parallel_forward,
    normalized_serial_cached_forward,
    normalized_serial_forward,
)


RESOURCE_FRONTIER_V1_VERSION = "relay-work-span-frontier-v1"
CORRECTNESS_POLICY_V1 = "fp64-shadow-decoded-equivalence-v1"
FP64_EQUIVALENCE_RTOL = 1e-10
FP64_EQUIVALENCE_ATOL = 1e-10

_SCHEDULES: dict[
    str,
    Callable[[RelayPopulationModel, RelayTensorBatch], RelayScheduleOutput],
] = {
    "parallel_normalized": normalized_parallel_forward,
    "serial_normalized": normalized_serial_forward,
    "serial_cached_normalized": normalized_serial_cached_forward,
}
_SCHEDULE_NAMES = tuple(_SCHEDULES)
_SERIAL_NAMES = ("serial_normalized", "serial_cached_normalized")


@dataclass(frozen=True, slots=True)
class RelayResourceBenchmarkConfigV1:
    population_sizes: tuple[int, ...] = (1, 4, 16, 64, 256)
    batch_sizes: tuple[int, ...] = (1, 64)
    warmup_iterations: int = 20
    measured_iterations: int = 100
    world_seed: int = 0
    fp64_equivalence_rtol: float = FP64_EQUIVALENCE_RTOL
    fp64_equivalence_atol: float = FP64_EQUIVALENCE_ATOL

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
        if self.fp64_equivalence_rtol <= 0 or self.fp64_equivalence_atol <= 0:
            raise ValueError("FP64 equivalence tolerances must be positive")


@dataclass(frozen=True, slots=True)
class RelayCorrectnessEvidenceV1:
    fp32_pairwise_decoded_equal: bool
    fp64_pairwise_decoded_equal: bool
    fp32_vs_fp64_decoded_equal: bool
    fp64_pairwise_tensors_close: bool
    max_abs_fp32_logits_difference: float
    max_abs_fp32_shared_difference: float
    max_abs_fp64_logits_difference: float
    max_abs_fp64_shared_difference: float
    max_abs_fp32_vs_fp64_logits_difference: float
    max_abs_fp32_vs_fp64_shared_difference: float

    @property
    def admissible(self) -> bool:
        return bool(
            self.fp32_pairwise_decoded_equal
            and self.fp64_pairwise_decoded_equal
            and self.fp32_vs_fp64_decoded_equal
            and self.fp64_pairwise_tensors_close
        )

    def validate(self) -> None:
        if not self.admissible:
            raise ValueError("Gate-1 v1 correctness evidence is not admissible")
        for value in (
            self.max_abs_fp32_logits_difference,
            self.max_abs_fp32_shared_difference,
            self.max_abs_fp64_logits_difference,
            self.max_abs_fp64_shared_difference,
            self.max_abs_fp32_vs_fp64_logits_difference,
            self.max_abs_fp32_vs_fp64_shared_difference,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("correctness drift metrics must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["admissible"] = self.admissible
        return payload


@dataclass(frozen=True, slots=True)
class RelayResourceComparisonV1:
    base: RelayResourceComparison
    correctness_v1: RelayCorrectnessEvidenceV1

    def validate(self) -> None:
        self.correctness_v1.validate()
        self.base.validate()
        if not self.base.outputs_equivalent or not self.base.decoded_predictions_equal:
            raise ValueError("v1 base comparison must reflect passed correctness evidence")

    def to_dict(self) -> dict[str, object]:
        payload = self.base.to_dict()
        payload["correctness_v1"] = self.correctness_v1.to_dict()
        payload["outputs_equivalent"] = self.correctness_v1.admissible
        payload["decoded_predictions_equal"] = bool(
            self.correctness_v1.fp32_pairwise_decoded_equal
            and self.correctness_v1.fp64_pairwise_decoded_equal
            and self.correctness_v1.fp32_vs_fp64_decoded_equal
        )
        payload["equivalence_note"] = (
            "Gate-1 v1 uses complete-matrix FP64 shadow corroboration plus exact decoded agreement; "
            "FP32 tensor drift is descriptive and is not gated by a post-hoc enlarged FP32 tolerance."
        )
        return payload


@dataclass(frozen=True, slots=True)
class RelayResourceFrontierResultV1:
    parameter_fingerprint: str
    learned_parameter_count: int
    config: RelayResourceBenchmarkConfigV1
    device: str
    provenance: dict[str, object]
    comparisons: tuple[RelayResourceComparisonV1, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_version": RESOURCE_FRONTIER_V1_VERSION,
            "correctness_policy": {
                "name": CORRECTNESS_POLICY_V1,
                "fp64_equivalence_rtol": self.config.fp64_equivalence_rtol,
                "fp64_equivalence_atol": self.config.fp64_equivalence_atol,
                "fp32_tensor_allclose_is_gate": False,
                "complete_matrix_preflight_before_timing": True,
                "fp64_model_offloaded_before_timing": True,
            },
            "parameter_fingerprint": self.parameter_fingerprint,
            "learned_parameter_count": self.learned_parameter_count,
            "config": asdict(self.config),
            "device": self.device,
            "provenance": dict(self.provenance),
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "interpretation_note": (
                "This benchmark measures execution organization for an already-established relay function. "
                "Gate-1 v1 was frozen after the untimed v0 CUDA precision diagnostic and before any admitted timing result."
            ),
        }


def _float_batch(batch: RelayTensorBatch, *, dtype: torch.dtype) -> RelayTensorBatch:
    return replace(
        batch,
        local_inputs=batch.local_inputs.to(dtype=dtype),
        start_bits=batch.start_bits.to(dtype=dtype),
        target_bits=batch.target_bits.to(dtype=dtype),
    )


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.to(torch.float64) - right.to(torch.float64)).abs().max().item())


def _decoded_equal(left: RelayScheduleOutput, right: RelayScheduleOutput) -> bool:
    return bool(torch.equal(decode_node_logits(left.logits), decode_node_logits(right.logits)))


def _correctness_evidence(
    outputs32: dict[str, RelayScheduleOutput],
    outputs64: dict[str, RelayScheduleOutput],
    *,
    rtol: float,
    atol: float,
) -> RelayCorrectnessEvidenceV1:
    parallel32 = outputs32["parallel_normalized"]
    parallel64 = outputs64["parallel_normalized"]

    fp32_pairwise_decoded_equal = all(
        _decoded_equal(parallel32, outputs32[name]) for name in _SERIAL_NAMES
    )
    fp64_pairwise_decoded_equal = all(
        _decoded_equal(parallel64, outputs64[name]) for name in _SERIAL_NAMES
    )
    fp32_vs_fp64_decoded_equal = all(
        _decoded_equal(outputs32[name], outputs64[name]) for name in _SCHEDULE_NAMES
    )
    fp64_pairwise_tensors_close = all(
        bool(
            torch.allclose(parallel64.logits, outputs64[name].logits, rtol=rtol, atol=atol)
            and torch.allclose(
                parallel64.final_shared,
                outputs64[name].final_shared,
                rtol=rtol,
                atol=atol,
            )
        )
        for name in _SERIAL_NAMES
    )

    return RelayCorrectnessEvidenceV1(
        fp32_pairwise_decoded_equal=fp32_pairwise_decoded_equal,
        fp64_pairwise_decoded_equal=fp64_pairwise_decoded_equal,
        fp32_vs_fp64_decoded_equal=fp32_vs_fp64_decoded_equal,
        fp64_pairwise_tensors_close=fp64_pairwise_tensors_close,
        max_abs_fp32_logits_difference=max(
            _max_difference(parallel32.logits, outputs32[name].logits) for name in _SERIAL_NAMES
        ),
        max_abs_fp32_shared_difference=max(
            _max_difference(parallel32.final_shared, outputs32[name].final_shared)
            for name in _SERIAL_NAMES
        ),
        max_abs_fp64_logits_difference=max(
            _max_difference(parallel64.logits, outputs64[name].logits) for name in _SERIAL_NAMES
        ),
        max_abs_fp64_shared_difference=max(
            _max_difference(parallel64.final_shared, outputs64[name].final_shared)
            for name in _SERIAL_NAMES
        ),
        max_abs_fp32_vs_fp64_logits_difference=max(
            _max_difference(outputs32[name].logits, outputs64[name].logits) for name in _SCHEDULE_NAMES
        ),
        max_abs_fp32_vs_fp64_shared_difference=max(
            _max_difference(outputs32[name].final_shared, outputs64[name].final_shared)
            for name in _SCHEDULE_NAMES
        ),
    )


def _condition_key(difficulty: RelayDifficulty, batch: RelayTensorBatch) -> tuple[str, int, int]:
    return (
        difficulty.name,
        batch.active_workers,
        int(batch.local_inputs.shape[0]),
    )


def _measurement_rotation_v1(
    *,
    difficulty: str,
    active_workers: int,
    batch_size: int,
    relay_hops: int,
) -> int:
    payload = (
        f"{RESOURCE_FRONTIER_V1_VERSION}|{difficulty}|{active_workers}|{batch_size}|{relay_hops}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % 3


def _collect_correctness_preflight(
    model32: RelayPopulationModel,
    model64: RelayPopulationModel,
    *,
    difficulties: Sequence[RelayDifficulty],
    config: RelayResourceBenchmarkConfigV1,
    target_device: torch.device,
) -> dict[tuple[str, int, int], RelayCorrectnessEvidenceV1]:
    evidence: dict[tuple[str, int, int], RelayCorrectnessEvidenceV1] = {}
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
                    batch32 = build_relay_tensor_batch(
                        world_batch,
                        active_workers=active_workers,
                        device=target_device,
                    )
                    batch64 = _float_batch(batch32, dtype=torch.float64)
                    outputs32 = {
                        name: forward(model32, batch32) for name, forward in _SCHEDULES.items()
                    }
                    outputs64 = {
                        name: forward(model64, batch64) for name, forward in _SCHEDULES.items()
                    }
                    row = _correctness_evidence(
                        outputs32,
                        outputs64,
                        rtol=config.fp64_equivalence_rtol,
                        atol=config.fp64_equivalence_atol,
                    )
                    if not row.admissible:
                        raise RuntimeError(
                            "Gate-1 v1 precision-aware correctness preflight failed before timing: "
                            f"difficulty={difficulty.name}, workers={active_workers}, batch={batch_size}"
                        )
                    evidence[_condition_key(difficulty, batch32)] = row
    return evidence


def _measure_condition_v1(
    model: RelayPopulationModel,
    batch: RelayTensorBatch,
    *,
    difficulty: RelayDifficulty,
    config: RelayResourceBenchmarkConfigV1,
    correctness: RelayCorrectnessEvidenceV1,
) -> RelayResourceComparisonV1:
    forwards = _SCHEDULES
    rotation = _measurement_rotation_v1(
        difficulty=difficulty.name,
        active_workers=batch.active_workers,
        batch_size=int(batch.local_inputs.shape[0]),
        relay_hops=difficulty.hop_count,
    )
    measurement_order = _SCHEDULE_NAMES[rotation:] + _SCHEDULE_NAMES[:rotation]
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

    base = RelayResourceComparison(
        difficulty=difficulty.name,
        relay_hops=difficulty.hop_count,
        active_workers=batch.active_workers,
        batch_size=int(batch.local_inputs.shape[0]),
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
        outputs_equivalent=correctness.admissible,
        decoded_predictions_equal=bool(
            correctness.fp32_pairwise_decoded_equal
            and correctness.fp64_pairwise_decoded_equal
            and correctness.fp32_vs_fp64_decoded_equal
        ),
        max_abs_logits_difference=correctness.max_abs_fp32_logits_difference,
        max_abs_shared_difference=correctness.max_abs_fp32_shared_difference,
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
    comparison = RelayResourceComparisonV1(base=base, correctness_v1=correctness)
    comparison.validate()
    return comparison


def benchmark_relay_resource_frontier_v1(
    model32: RelayPopulationModel,
    model64: RelayPopulationModel,
    *,
    difficulties: Sequence[RelayDifficulty],
    config: RelayResourceBenchmarkConfigV1 = RelayResourceBenchmarkConfigV1(),
    device: torch.device | str = "cpu",
) -> RelayResourceFrontierResultV1:
    """Run complete precision-aware correctness preflight, then time only FP32 schedules."""

    config.validate()
    if not difficulties:
        raise ValueError("at least one relay difficulty is required")
    target_device = torch.device(device)
    model32 = model32.to(device=target_device, dtype=torch.float32).eval()
    model64 = model64.to(device=target_device, dtype=torch.float64).eval()
    fingerprint = model32.parameter_fingerprint()
    parameter_count = model32.trainable_parameter_count()

    correctness = _collect_correctness_preflight(
        model32,
        model64,
        difficulties=difficulties,
        config=config,
        target_device=target_device,
    )

    # The FP64 reference exists only for the untimed correctness gate. Remove its CUDA residency
    # before any resource measurement so it cannot inflate allocator baselines or peak memory.
    model64.to(device=torch.device("cpu"))
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        torch.cuda.empty_cache()

    comparisons: list[RelayResourceComparisonV1] = []
    with torch.inference_mode():
        for difficulty in difficulties:
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
                    key = _condition_key(difficulty, batch)
                    comparisons.append(
                        _measure_condition_v1(
                            model32,
                            batch,
                            difficulty=difficulty,
                            config=config,
                            correctness=correctness[key],
                        )
                    )

    if model32.parameter_fingerprint() != fingerprint:
        raise RuntimeError("Gate-1 v1 resource benchmark mutated the relay checkpoint")
    provenance = runtime_provenance(target_device)
    provenance["schedule_timing_policy"] = (
        "each schedule independently warmed; measurement order rotates deterministically by condition"
    )
    provenance["correctness_preflight"] = (
        "complete frozen FP32+FP64 matrix passed before timing; FP64 model offloaded before timing"
    )
    return RelayResourceFrontierResultV1(
        parameter_fingerprint=fingerprint,
        learned_parameter_count=parameter_count,
        config=config,
        device=str(target_device),
        provenance=provenance,
        comparisons=tuple(comparisons),
    )
