"""Untimed target-device equivalence diagnostic for Gate-1 relay schedules.

This diagnostic does not measure latency, throughput, or memory and therefore cannot produce a
Gate-1 performance result. It exists only to characterize numerical agreement between the frozen
parallel and serial relay schedules on a target device before any resource timing is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .collective_relay import RELAY_DIFFICULTIES, generate_relay_dataset
from .relay_experiment_v1 import load_relay_checkpoint_v1
from .relay_model import build_relay_tensor_batch, decode_node_logits
from .relay_resource_frontier import RelayResourceBenchmarkConfig
from .relay_serial_control import (
    normalized_parallel_forward,
    normalized_serial_cached_forward,
    normalized_serial_forward,
)


DIAGNOSTIC_VERSION = "relay-resource-equivalence-diagnostic-v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Gate-1 schedule-equivalence matrix without timing. "
            "This is a mechanics diagnostic only and cannot establish a performance result."
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
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument(
        "--output",
        default=(
            "results/population_compute_scaling_v0/"
            "gate1_resource_frontier_v0/equivalence-diagnostic.json"
        ),
    )
    return parser


def _pair_metrics(parallel, row, *, rtol: float, atol: float) -> dict[str, object]:
    logits_difference = (parallel.logits - row.logits).abs()
    shared_difference = (parallel.final_shared - row.final_shared).abs()
    logits_close = bool(torch.allclose(parallel.logits, row.logits, rtol=rtol, atol=atol))
    shared_close = bool(
        torch.allclose(parallel.final_shared, row.final_shared, rtol=rtol, atol=atol)
    )
    decoded_equal = bool(
        torch.equal(decode_node_logits(parallel.logits), decode_node_logits(row.logits))
    )
    return {
        "logits_close": logits_close,
        "shared_close": shared_close,
        "decoded_equal": decoded_equal,
        "max_abs_logits_difference": float(logits_difference.max().item()),
        "max_abs_shared_difference": float(shared_difference.max().item()),
    }


def diagnose(
    model,
    *,
    difficulties,
    config: RelayResourceBenchmarkConfig,
    device: str,
) -> dict[str, object]:
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
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
                    batch = build_relay_tensor_batch(
                        world_batch,
                        active_workers=active_workers,
                        device=target_device,
                    )
                    parallel = normalized_parallel_forward(model, batch)
                    serial = normalized_serial_forward(model, batch)
                    cached = normalized_serial_cached_forward(model, batch)
                    serial_metrics = _pair_metrics(
                        parallel,
                        serial,
                        rtol=config.equivalence_rtol,
                        atol=config.equivalence_atol,
                    )
                    cached_metrics = _pair_metrics(
                        parallel,
                        cached,
                        rtol=config.equivalence_rtol,
                        atol=config.equivalence_atol,
                    )
                    rows.append(
                        {
                            "difficulty": difficulty.name,
                            "relay_hops": difficulty.hop_count,
                            "active_workers": active_workers,
                            "batch_size": batch_size,
                            "serial_normalized": serial_metrics,
                            "serial_cached_normalized": cached_metrics,
                            "condition_allclose": bool(
                                serial_metrics["logits_close"]
                                and serial_metrics["shared_close"]
                                and cached_metrics["logits_close"]
                                and cached_metrics["shared_close"]
                            ),
                            "condition_decoded_equal": bool(
                                serial_metrics["decoded_equal"]
                                and cached_metrics["decoded_equal"]
                            ),
                        }
                    )

    failing = [row for row in rows if not row["condition_allclose"]]
    decoded_failures = [row for row in rows if not row["condition_decoded_equal"]]

    def worst(metric: str) -> dict[str, object] | None:
        candidates: list[tuple[float, dict[str, object], str]] = []
        for row in rows:
            for schedule in ("serial_normalized", "serial_cached_normalized"):
                value = float(row[schedule][metric])
                candidates.append((value, row, schedule))
        if not candidates:
            return None
        value, row, schedule = max(candidates, key=lambda item: item[0])
        return {
            "value": value,
            "schedule": schedule,
            "difficulty": row["difficulty"],
            "relay_hops": row["relay_hops"],
            "active_workers": row["active_workers"],
            "batch_size": row["batch_size"],
        }

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
        "equivalence_rtol": config.equivalence_rtol,
        "equivalence_atol": config.equivalence_atol,
        "world_seed": config.world_seed,
        "condition_count": len(rows),
        "allclose_failure_count": len(failing),
        "decoded_failure_count": len(decoded_failures),
        "worst_logits_difference": worst("max_abs_logits_difference"),
        "worst_shared_difference": worst("max_abs_shared_difference"),
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
        equivalence_rtol=args.rtol,
        equivalence_atol=args.atol,
    )
    config.validate()
    difficulty_by_name = {row.name: row for row in RELAY_DIFFICULTIES}
    difficulties = tuple(difficulty_by_name[name] for name in args.difficulties)
    if len(set(args.difficulties)) != len(args.difficulties):
        raise SystemExit("--difficulties must be unique")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    model, checkpoint_payload = load_relay_checkpoint_v1(checkpoint_path, device=args.device)
    payload = diagnose(model, difficulties=difficulties, config=config, device=args.device)
    payload["checkpoint"] = {
        "path": str(checkpoint_path),
        "file_sha256": checkpoint_sha256,
        "training_seed": checkpoint_payload.get("training_seed"),
        "parameter_fingerprint": checkpoint_payload.get("parameter_fingerprint"),
        "learned_parameter_count": model.trainable_parameter_count(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
