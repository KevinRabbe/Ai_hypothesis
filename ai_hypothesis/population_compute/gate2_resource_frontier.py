"""Frozen eager-CUDA Gate-2 persistent-state resource frontier measurement."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from .gate2_persistent_model import (
    Gate2PersistentModelConfig,
    Gate2PersistentStateModel,
    build_gate2_tensor_batch,
    decode_gate2_payload_logits,
    parallel_persistent_forward,
    serial_persistent_forward,
)
from .gate2_persistent_state_capacity import Gate2ControlMode, generate_gate2_world


GATE2_RESOURCE_EXPERIMENT_VERSION = "gate2-persistent-state-resource-frontier-v0"
GATE2_RESOURCE_PRIMARY_CONFIRMATION_SEED = 3
GATE2_RESOURCE_ENTITY_WIDTHS = {64: (1, 4, 16, 64), 256: (1, 4, 16, 64, 256)}
GATE2_RESOURCE_BATCH_SIZES = (1, 64)
GATE2_RESOURCE_WARMUP_ITERATIONS = 10
GATE2_RESOURCE_TIMED_ITERATIONS = 50
GATE2_RESOURCE_WORLD_SEED_START = 4 << 30
GATE2_RESOURCE_EXPECTED_CELL_COUNT = 18

Progress = Callable[[str, int, int, int, int], None]


@dataclass(frozen=True, slots=True)
class Gate2ResourcePreflight:
    entity_count: int
    width: int
    batch_size: int
    world_seeds: tuple[int, ...]
    decoded_identity: bool
    learned_update_identity: bool
    state_bank_identity: bool
    max_abs_logit_drift: float
    max_abs_final_state_drift: float


@dataclass(frozen=True, slots=True)
class Gate2ResourceScheduleTiming:
    schedule: str
    raw_latency_ms: tuple[float, ...]
    median_latency_ms: float
    p25_latency_ms: float
    p75_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    samples_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    learned_updates_per_sample: int
    peak_simultaneous_updates_per_sample: int
    persistent_state_vectors_per_sample: int
    collision_load: int


@dataclass(frozen=True, slots=True)
class Gate2ResourceCellResult:
    entity_count: int
    width: int
    batch_size: int
    preflight: Gate2ResourcePreflight
    parallel: Gate2ResourceScheduleTiming
    serial: Gate2ResourceScheduleTiming
    serial_over_parallel_median_speedup: float


@dataclass(frozen=True, slots=True)
class Gate2ResourceResult:
    experiment_version: str
    checkpoint_path: str
    checkpoint_training_seed: int
    checkpoint_parameter_fingerprint: str
    learned_parameter_count: int
    device: str
    cuda_device_name: str
    warmup_iterations: int
    timed_iterations: int
    cells: tuple[Gate2ResourceCellResult, ...]
    all_preflights_passed: bool
    decision_endpoint_passes: dict[str, bool]
    resource_frontier_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_version": self.experiment_version,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_training_seed": self.checkpoint_training_seed,
            "checkpoint_parameter_fingerprint": self.checkpoint_parameter_fingerprint,
            "learned_parameter_count": self.learned_parameter_count,
            "device": self.device,
            "cuda_device_name": self.cuda_device_name,
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "cells": [
                {
                    "entity_count": cell.entity_count,
                    "width": cell.width,
                    "batch_size": cell.batch_size,
                    "preflight": asdict(cell.preflight),
                    "parallel": asdict(cell.parallel),
                    "serial": asdict(cell.serial),
                    "serial_over_parallel_median_speedup": cell.serial_over_parallel_median_speedup,
                }
                for cell in self.cells
            ],
            "all_preflights_passed": self.all_preflights_passed,
            "decision_endpoint_passes": self.decision_endpoint_passes,
            "resource_frontier_passed": self.resource_frontier_passed,
            "scientific_status": "FROZEN_GATE2_RESOURCE_RESULT",
            "gate2_overall_verdict": "NOT_ASSIGNED_BY_RESOURCE_RUNNER",
        }


def load_seed3_confirmation_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cuda",
) -> Gate2PersistentStateModel:
    target_device = torch.device(device)
    payload = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Gate-2 resource checkpoint must contain one mapping")
    if payload.get("training_seed") != GATE2_RESOURCE_PRIMARY_CONFIRMATION_SEED:
        raise ValueError("Gate-2 resource protocol is bound to confirmation training seed 3")
    if payload.get("evaluation_split") != "confirmation" or payload.get("confirmation_opened") is not True:
        raise ValueError("Gate-2 resource checkpoint must come from the frozen confirmation path")
    model_config = payload.get("model_config")
    if model_config != {"state_width": 64, "query_width": 24}:
        raise ValueError("Gate-2 resource checkpoint model config differs from the frozen protocol")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Gate-2 resource checkpoint is missing state_dict")

    model = Gate2PersistentStateModel(Gate2PersistentModelConfig(state_width=64, query_width=24)).to(target_device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if model.trainable_parameter_count() != payload.get("learned_parameter_count"):
        raise ValueError("Gate-2 resource checkpoint parameter count mismatch")
    if model.parameter_fingerprint() != payload.get("parameter_fingerprint"):
        raise ValueError("Gate-2 resource checkpoint parameter fingerprint mismatch")
    return model


def _world_seeds(entity_count: int, batch_size: int) -> tuple[int, ...]:
    base = GATE2_RESOURCE_WORLD_SEED_START + entity_count * 1_000
    return tuple(base + offset for offset in range(batch_size))


def _batch(entity_count: int, width: int, batch_size: int, device: torch.device) -> object:
    seeds = _world_seeds(entity_count, batch_size)
    worlds = tuple(generate_gate2_world(seed=seed, entity_count=entity_count) for seed in seeds)
    return build_gate2_tensor_batch(
        worlds,
        width=width,
        mode=Gate2ControlMode.STABLE_PERSISTENT,
        device=device,
    )


def _drift(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(a - b)).item())


def _preflight(
    model: Gate2PersistentStateModel,
    *,
    entity_count: int,
    width: int,
    batch_size: int,
    device: torch.device,
) -> Gate2ResourcePreflight:
    batch = _batch(entity_count, width, batch_size, device)
    with torch.inference_mode():
        parallel = parallel_persistent_forward(model, batch)
        serial = serial_persistent_forward(model, batch)
    decoded_identity = torch.equal(
        decode_gate2_payload_logits(parallel.logits),
        decode_gate2_payload_logits(serial.logits),
    )
    learned_update_identity = (
        parallel.telemetry.learned_updates_per_sample
        == serial.telemetry.learned_updates_per_sample
        == 8 * entity_count
    )
    state_bank_identity = (
        parallel.telemetry.persistent_state_vectors_per_sample
        == serial.telemetry.persistent_state_vectors_per_sample
        == width
    )
    return Gate2ResourcePreflight(
        entity_count=entity_count,
        width=width,
        batch_size=batch_size,
        world_seeds=_world_seeds(entity_count, batch_size),
        decoded_identity=decoded_identity,
        learned_update_identity=learned_update_identity,
        state_bank_identity=state_bank_identity,
        max_abs_logit_drift=_drift(parallel.logits, serial.logits),
        max_abs_final_state_drift=_drift(parallel.final_states, serial.final_states),
    )


def _time_once(
    fn: Callable[[Gate2PersistentStateModel, object], object],
    model: Gate2PersistentStateModel,
    batch: object,
) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        fn(model, batch)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _quantile(values: tuple[float, ...], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _memory_peak(
    fn: Callable[[Gate2PersistentStateModel, object], object],
    model: Gate2PersistentStateModel,
    batch: object,
) -> tuple[int, int, object]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = fn(model, batch)
    torch.cuda.synchronize()
    return (
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
        output,
    )


def _schedule_timing(
    *,
    schedule: str,
    raw: tuple[float, ...],
    batch_size: int,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    telemetry: object,
) -> Gate2ResourceScheduleTiming:
    median = float(statistics.median(raw))
    return Gate2ResourceScheduleTiming(
        schedule=schedule,
        raw_latency_ms=raw,
        median_latency_ms=median,
        p25_latency_ms=_quantile(raw, 0.25),
        p75_latency_ms=_quantile(raw, 0.75),
        min_latency_ms=min(raw),
        max_latency_ms=max(raw),
        samples_per_second=batch_size / (median / 1_000.0),
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        learned_updates_per_sample=int(telemetry.learned_updates_per_sample),
        peak_simultaneous_updates_per_sample=int(telemetry.peak_simultaneous_updates_per_sample),
        persistent_state_vectors_per_sample=int(telemetry.persistent_state_vectors_per_sample),
        collision_load=int(telemetry.collision_load),
    )


def run_gate2_resource_frontier(
    *,
    checkpoint_path: Path,
    device: torch.device | str = "cuda",
    progress: Progress | None = None,
) -> Gate2ResourceResult:
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("Gate-2 resource protocol v0 requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    model = load_seed3_confirmation_checkpoint(checkpoint_path, device=target_device)
    expected_fingerprint = model.parameter_fingerprint()
    expected_count = model.trainable_parameter_count()

    matrix = tuple(
        (entity_count, width, batch_size)
        for entity_count, widths in GATE2_RESOURCE_ENTITY_WIDTHS.items()
        for width in widths
        for batch_size in GATE2_RESOURCE_BATCH_SIZES
    )
    if len(matrix) != GATE2_RESOURCE_EXPECTED_CELL_COUNT:
        raise RuntimeError("Gate-2 frozen resource matrix size changed")

    # Stronger-than-minimum ordering: complete the entire correctness preflight before any timing.
    preflights: dict[tuple[int, int, int], Gate2ResourcePreflight] = {}
    for index, (entity_count, width, batch_size) in enumerate(matrix, start=1):
        if progress is not None:
            progress("preflight", index, len(matrix), entity_count, width)
        row = _preflight(
            model,
            entity_count=entity_count,
            width=width,
            batch_size=batch_size,
            device=target_device,
        )
        preflights[(entity_count, width, batch_size)] = row
        if not (row.decoded_identity and row.learned_update_identity and row.state_bank_identity):
            raise RuntimeError(
                f"Gate-2 resource correctness preflight failed at C{entity_count}/W{width}/B{batch_size}"
            )

    cells: list[Gate2ResourceCellResult] = []
    for index, (entity_count, width, batch_size) in enumerate(matrix, start=1):
        if progress is not None:
            progress("timing", index, len(matrix), entity_count, width)
        batch = _batch(entity_count, width, batch_size, target_device)

        # Equal warmup count, deterministically interleaved.
        with torch.inference_mode():
            for warmup in range(GATE2_RESOURCE_WARMUP_ITERATIONS):
                if warmup % 2 == 0:
                    parallel_persistent_forward(model, batch)
                    serial_persistent_forward(model, batch)
                else:
                    serial_persistent_forward(model, batch)
                    parallel_persistent_forward(model, batch)
            torch.cuda.synchronize()

        parallel_raw: list[float] = []
        serial_raw: list[float] = []
        for trial in range(GATE2_RESOURCE_TIMED_ITERATIONS):
            if trial % 2 == 0:
                parallel_raw.append(_time_once(parallel_persistent_forward, model, batch))
                serial_raw.append(_time_once(serial_persistent_forward, model, batch))
            else:
                serial_raw.append(_time_once(serial_persistent_forward, model, batch))
                parallel_raw.append(_time_once(parallel_persistent_forward, model, batch))

        p_alloc, p_reserved, p_output = _memory_peak(parallel_persistent_forward, model, batch)
        s_alloc, s_reserved, s_output = _memory_peak(serial_persistent_forward, model, batch)
        parallel_timing = _schedule_timing(
            schedule="parallel_persistent",
            raw=tuple(parallel_raw),
            batch_size=batch_size,
            peak_allocated_bytes=p_alloc,
            peak_reserved_bytes=p_reserved,
            telemetry=p_output.telemetry,
        )
        serial_timing = _schedule_timing(
            schedule="serial_persistent",
            raw=tuple(serial_raw),
            batch_size=batch_size,
            peak_allocated_bytes=s_alloc,
            peak_reserved_bytes=s_reserved,
            telemetry=s_output.telemetry,
        )
        cells.append(
            Gate2ResourceCellResult(
                entity_count=entity_count,
                width=width,
                batch_size=batch_size,
                preflight=preflights[(entity_count, width, batch_size)],
                parallel=parallel_timing,
                serial=serial_timing,
                serial_over_parallel_median_speedup=(
                    serial_timing.median_latency_ms / parallel_timing.median_latency_ms
                ),
            )
        )

    if model.parameter_fingerprint() != expected_fingerprint or model.trainable_parameter_count() != expected_count:
        raise RuntimeError("Gate-2 resource measurement mutated the frozen checkpoint")

    endpoint_passes: dict[str, bool] = {}
    for entity_count, width in ((64, 64), (256, 256)):
        for batch_size in GATE2_RESOURCE_BATCH_SIZES:
            cell = next(
                cell
                for cell in cells
                if (cell.entity_count, cell.width, cell.batch_size)
                == (entity_count, width, batch_size)
            )
            endpoint_passes[f"c{entity_count}_w{width}_b{batch_size}"] = (
                cell.parallel.median_latency_ms < cell.serial.median_latency_ms
            )

    all_preflights = all(
        row.decoded_identity and row.learned_update_identity and row.state_bank_identity
        for row in preflights.values()
    )
    resource_passed = all_preflights and len(endpoint_passes) == 4 and all(endpoint_passes.values())
    return Gate2ResourceResult(
        experiment_version=GATE2_RESOURCE_EXPERIMENT_VERSION,
        checkpoint_path=str(checkpoint_path),
        checkpoint_training_seed=GATE2_RESOURCE_PRIMARY_CONFIRMATION_SEED,
        checkpoint_parameter_fingerprint=expected_fingerprint,
        learned_parameter_count=expected_count,
        device=str(target_device),
        cuda_device_name=torch.cuda.get_device_name(target_device),
        warmup_iterations=GATE2_RESOURCE_WARMUP_ITERATIONS,
        timed_iterations=GATE2_RESOURCE_TIMED_ITERATIONS,
        cells=tuple(cells),
        all_preflights_passed=all_preflights,
        decision_endpoint_passes=endpoint_passes,
        resource_frontier_passed=resource_passed,
    )
