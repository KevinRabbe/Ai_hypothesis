"""CLI for Gate-1 relay work/span resource measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .collective_relay import RELAY_DIFFICULTIES
from .relay_experiment_v1 import load_relay_checkpoint_v1
from .relay_resource_frontier import (
    RelayResourceBenchmarkConfig,
    benchmark_relay_resource_frontier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the work/span/latency/memory frontier of the already-equivalent "
            "parallel and serial canonical relay-v1 schedules. No training is performed."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations", type=int, default=100)
    parser.add_argument("--world-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/population_compute_scaling_v0/relay_resource_frontier_v0.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RelayResourceBenchmarkConfig(
        population_sizes=tuple(args.population_sizes),
        batch_sizes=tuple(args.batch_sizes),
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
        world_seed=args.world_seed,
    )
    config.validate()
    difficulty_by_name = {row.name: row for row in RELAY_DIFFICULTIES}
    difficulties = tuple(difficulty_by_name[name] for name in args.difficulties)
    if len(set(args.difficulties)) != len(args.difficulties):
        raise SystemExit("--difficulties must be unique")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_file_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    model, checkpoint_payload = load_relay_checkpoint_v1(
        checkpoint_path,
        device=args.device,
    )
    model.eval()
    result = benchmark_relay_resource_frontier(
        model,
        difficulties=difficulties,
        config=config,
        device=args.device,
    )
    payload = result.to_dict()
    payload["checkpoint"] = {
        "path": str(checkpoint_path),
        "file_sha256": checkpoint_file_sha256,
        "experiment_version": checkpoint_payload.get("experiment_version"),
        "protocol_version": checkpoint_payload.get("protocol_version"),
        "benchmark_version": checkpoint_payload.get("benchmark_version"),
        "training_seed": checkpoint_payload.get("training_seed"),
        "parameter_fingerprint": checkpoint_payload.get("parameter_fingerprint"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
