"""Untimed precision triangulation for Gate-1 relay schedule equivalence.

This diagnostic does not measure latency, throughput, or memory and cannot produce a Gate-1
performance result. It compares float32 and float64 executions of the frozen relay schedules so a
target-device equivalence failure can be distinguished from a schedule-semantic disagreement
without relaxing the frozen Gate-1 v0 tolerance after observing target-hardware data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import torch

from .collective_relay import RELAY_DIFFICULTIES, generate_relay_dataset
from .relay_experiment_v1 import load_relay_checkpoint_v1
from .relay_model import RelayTensorBatch, build_relay_tensor_batch, decode_node_logits
from .relay_resource_frontier import RelayResourceBenchmarkConfig
from .relay_serial_control import (
    RelayScheduleOutput,
    normalized_parallel_forward,
    normalized_serial_cached_forward,
    normalized_serial_forward,
)


DIAGNOSTIC_VERSION = "relay-resource-precision-diagnostic-v0"
SCHEDULES: dict[
    str,
    Callable[[object, RelayTensorBatch], RelayScheduleOutput],
] = {
    "parallel_normalized": normalized_parallel_forward,
    "serial_normalized": normalized_serial_forward,
    "serial_cached_normalized": normalized_serial_cached_forward,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Triangulate frozen Gate-1 schedule equivalence in float32 and float64 without timing. "
            "This mechanics diagnostic cannot establish a performance result."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--population-sizes",
        nargs="+",
        type=int,
        default=(1, 4, 16, 64, 256),
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 64))
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=tuple(row.name for row in RELAY_DIFFICULTIES),
        default=tuple(row.name for row in RELAY_DIFFICULTIES),
    )
    parser.add_argument("--world-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=(
            "results/population_compute_scaling_v0/"
            "gate1_resource_frontier_v0/precision-diagnostic.json"
        ),
    )
    return parser


def _float_batch(batch: RelayTensorBatch, *, dtype: torch.dtype) -> RelayTensorBatch:
    return replace(
        batch,
        local_inputs=batch.local_inputs.to(dtype=dtype),
        start_bits=batch.start_bits.to(dtype=dtype),
        target_bits=batch.target_bits.to(dtype=dtype),
    )


def _difference(left: RelayScheduleOutput, right: RelayScheduleOutput) -> dict[str, object]:
    logits_difference = (left.logits.to(torch.float64) - right.logits.to(torch.float64)).abs()
    shared_difference = (
        left.final_shared.to(torch.float64) - right.final_shared.to(torch.float64)
    ).abs()
    return {
        "decoded_equal": bool(
            torch.equal(decode_node_logits(left.logits), decode_node_logits(right.logits))
        ),
        "max_abs_logits_difference": float(logits_difference.max().item()),
        "max_abs_shared_difference": float(shared_difference.max().item()),
    }


def _worst(rows: list[dict[str, object]], metric: str, section: str) -> dict[str, object] | None:
    candidates: list[tuple[float, dict[str, object], str]] = []
    for row in rows:
        comparisons = row[section]
        for name, values in comparisons.items():
            candidates.append((float(values[metric]), row, name))
    if not candidates:
        return None
    value, row, comparison = max(candidates, key=lambda item: item[0])
    return {
        "value": value,
        "comparison": comparison,
        "difficulty": row["difficulty"],
        "relay_hops": row["relay_hops"],
        "active_workers": row["active_workers"],
        "batch_size": row["batch_size"],
    }


def diagnose_precision(
    model_float32,
    model_float64,
    *,
    difficulties,
    config: RelayResourceBenchmarkConfig,
    device: str,
) -> dict[str, object]:
    target_device = torch.device(device)
    model_float32 = model_float32.to(device=target_device, dtype=torch.float32).eval()
    model_float64 = model_float64.to(device=target_device, dtype=torch.float64).eval()
    rows: list[dict[str, object]] = []

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
                    batch32 = build_relay_tensor_batch(
                        world_batch,
                        active_workers=active_workers,
                        device=target_device,
                    )
                    batch64 = _float_batch(batch32, dtype=torch.float64)
                    outputs32 = {
                        name: forward(model_float32, batch32)
                        for name, forward in SCHEDULES.items()
                    }
                    outputs64 = {
                        name: forward(model_float64, batch64)
                        for name, forward in SCHEDULES.items()
                    }

                    float32_pairwise = {
                        name: _difference(outputs32["parallel_normalized"], outputs32[name])
                        for name in ("serial_normalized", "serial_cached_normalized")
                    }
                    float64_pairwise = {
                        name: _difference(outputs64["parallel_normalized"], outputs64[name])
                        for name in ("serial_normalized", "serial_cached_normalized")
                    }
                    float32_vs_float64 = {
                        name: _difference(outputs32[name], outputs64[name])
                        for name in SCHEDULES
                    }
                    rows.append(
                        {
                            "difficulty": difficulty.name,
                            "relay_hops": difficulty.hop_count,
                            "active_workers": active_workers,
                            "batch_size": batch_size,
                            "float32_pairwise": float32_pairwise,
                            "float64_pairwise": float64_pairwise,
                            "float32_vs_float64": float32_vs_float64,
                        }
                    )

    float32_pair_decoded_equal = all(
        values["decoded_equal"]
        for row in rows
        for values in row["float32_pairwise"].values()
    )
    float64_pair_decoded_equal = all(
        values["decoded_equal"]
        for row in rows
        for values in row["float64_pairwise"].values()
    )
    cross_precision_decoded_equal = all(
        values["decoded_equal"]
        for row in rows
        for values in row["float32_vs_float64"].values()
    )

    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "scientific_status": "MECHANICS_DIAGNOSTIC_ONLY",
        "performance_result": "NOT_MEASURED",
        "device": str(target_device),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if target_device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "world_seed": config.world_seed,
        "condition_count": len(rows),
        "float32_pairwise_decoded_equal": float32_pair_decoded_equal,
        "float64_pairwise_decoded_equal": float64_pair_decoded_equal,
        "float32_vs_float64_decoded_equal": cross_precision_decoded_equal,
        "worst_float32_pair_logits_difference": _worst(
            rows, "max_abs_logits_difference", "float32_pairwise"
        ),
        "worst_float32_pair_shared_difference": _worst(
            rows, "max_abs_shared_difference", "float32_pairwise"
        ),
        "worst_float64_pair_logits_difference": _worst(
            rows, "max_abs_logits_difference", "float64_pairwise"
        ),
        "worst_float64_pair_shared_difference": _worst(
            rows, "max_abs_shared_difference", "float64_pairwise"
        ),
        "worst_float32_vs_float64_logits_difference": _worst(
            rows, "max_abs_logits_difference", "float32_vs_float64"
        ),
        "worst_float32_vs_float64_shared_difference": _worst(
            rows, "max_abs_shared_difference", "float32_vs_float64"
        ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RelayResourceBenchmarkConfig(
        population_sizes=tuple(args.population_sizes),
        batch_sizes=tuple(args.batch_sizes),
        warmup_iterations=0,
        measured_iterations=1,
        world_seed=args.world_seed,
    )
    config.validate()
    difficulty_by_name = {row.name: row for row in RELAY_DIFFICULTIES}
    difficulties = tuple(difficulty_by_name[name] for name in args.difficulties)
    if len(set(args.difficulties)) != len(args.difficulties):
        raise SystemExit("--difficulties must be unique")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    model_float32, checkpoint_payload = load_relay_checkpoint_v1(
        checkpoint_path,
        device=args.device,
    )
    model_float64, _ = load_relay_checkpoint_v1(
        checkpoint_path,
        device=args.device,
    )
    payload = diagnose_precision(
        model_float32,
        model_float64,
        difficulties=difficulties,
        config=config,
        device=args.device,
    )
    payload["checkpoint"] = {
        "path": str(checkpoint_path),
        "file_sha256": checkpoint_sha256,
        "training_seed": checkpoint_payload.get("training_seed"),
        "parameter_fingerprint": checkpoint_payload.get("parameter_fingerprint"),
        "learned_parameter_count": model_float32.trainable_parameter_count(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
