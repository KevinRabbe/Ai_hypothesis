"""CLI entry point for the first Step 2 population runtime slice."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ai_hypothesis.step01.torch_data import make_loader
from .evaluation import evaluate_population
from .evidence import AggregationConfig
from .population import HomogeneousWorkerBank


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a homogeneous bank of Step 1 checkpoints, execute the population, "
            "construct evidence, aggregate it, and evaluate population controls."
        )
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Best-checkpoint paths for independently trained workers of one architecture.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, or a concrete CUDA device.",
    )
    parser.add_argument(
        "--backend",
        choices=("vmap", "loop"),
        default="vmap",
        help="Population execution backend. vmap is the primary homogeneous backend.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Benchmark split. Validation is the safe default for reducer development.",
    )
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--aggregation-config",
        default=None,
        help=(
            "Optional JSON file containing AggregationConfig fields. A formal test "
            "run requires this so thresholds are explicit and frozen."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/step02/population_runtime/result.json",
        help="Result JSON path.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _load_aggregation_config(path: str | None) -> AggregationConfig:
    if path is None:
        return AggregationConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = AggregationConfig(**payload)
    config.validate()
    return config


def main() -> None:
    args = _parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.split == "test" and args.aggregation_config is None:
        raise ValueError(
            "test evaluation requires --aggregation-config with validation-calibrated, "
            "frozen thresholds; the development defaults must not be tuned on test"
        )

    aggregation_config = _load_aggregation_config(args.aggregation_config)
    device = _resolve_device(args.device)
    bank = HomogeneousWorkerBank.from_checkpoints(
        args.checkpoints,
        device=device,
        execution_backend=args.backend,
    )
    loader = make_loader(
        split=args.split,
        count=args.count,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=0,
    )

    metrics = evaluate_population(
        bank,
        loader,
        aggregation_config=aggregation_config,
    )
    result = {
        "runtime_version": "step02-population-runtime-v0",
        "evidence_contract_version": "step02-evidence-v0",
        "split": args.split,
        "device": device,
        "backend": args.backend,
        "count": args.count,
        "batch_size": args.batch_size,
        "aggregation_config": asdict(aggregation_config),
        "checkpoints": [asdict(checkpoint) for checkpoint in bank.checkpoints],
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "population_evaluation_complete",
                "population_width": bank.population_width,
                "evidence_reducer_accuracy": metrics["evidence_reducer_accuracy"],
                "majority_vote_accuracy": metrics["majority_vote_accuracy"],
                "oracle_any_correct_coverage": metrics["oracle_any_correct_coverage"],
                "result_path": str(output_path),
            }
        )
    )


if __name__ == "__main__":
    main()
