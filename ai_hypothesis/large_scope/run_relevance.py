"""Run the large-scope relevance benchmark against frozen homogeneous checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ai_hypothesis.step01.schema import Difficulty
from ai_hypothesis.step02.evidence import AggregationConfig
from ai_hypothesis.step02.population import HomogeneousWorkerBank

from .evaluate import ScopeWorkerMode, evaluate_scope_sample
from .metrics import ScopeMetricsAccumulator
from .relevance import (
    LARGE_SCOPE_BENCHMARK_VERSION,
    LARGE_SCOPE_SPLIT_SEED_RANGES,
    LargeScopeRelevanceConfig,
    generate_large_scope_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate large-scope relevance with frozen Worker v1 checkpoints."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("vmap", "loop"), default="vmap")
    parser.add_argument(
        "--split",
        choices=tuple(LARGE_SCOPE_SPLIT_SEED_RANGES),
        default="development",
    )
    parser.add_argument("--allow-test-split", action="store_true")
    parser.add_argument("--world-count", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--window-count", type=int, default=16)
    parser.add_argument("--widths", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(mode.value for mode in ScopeWorkerMode),
        default=tuple(mode.value for mode in ScopeWorkerMode),
    )
    parser.add_argument(
        "--target-difficulty",
        choices=("easy", "medium", "hard"),
        default="hard",
    )
    parser.add_argument(
        "--distractor-difficulty",
        choices=("easy", "medium", "hard"),
        default="hard",
    )
    parser.add_argument("--ambiguous-distractor-fraction", type=float, default=0.125)
    parser.add_argument(
        "--output",
        default="results/large_scope_relevance_v0.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split == "test" and not args.allow_test_split:
        raise SystemExit(
            "Refusing to open the frozen test split without --allow-test-split"
        )
    if args.world_count <= 0:
        raise SystemExit("--world-count must be positive")
    if args.start_seed < 0:
        raise SystemExit("--start-seed must be non-negative")

    widths = tuple(args.widths)
    if any(width <= 0 for width in widths):
        raise SystemExit("--widths must be positive")
    if tuple(sorted(set(widths))) != widths:
        raise SystemExit("--widths must be unique and supplied in increasing order")
    if widths[-1] > args.window_count:
        raise SystemExit("largest width cannot exceed --window-count")

    config = LargeScopeRelevanceConfig(
        window_count=args.window_count,
        target_difficulty=Difficulty(args.target_difficulty),
        distractor_difficulty=Difficulty(args.distractor_difficulty),
        ambiguous_distractor_fraction=args.ambiguous_distractor_fraction,
    )
    config.validate()
    modes = tuple(ScopeWorkerMode(value) for value in args.modes)

    bank = HomogeneousWorkerBank.from_checkpoints(
        args.checkpoints,
        device=args.device,
        execution_backend=args.backend,
    )
    if ScopeWorkerMode.DIVERSE_WORKERS in modes and widths[-1] > bank.population_width:
        raise SystemExit(
            "diverse-worker width cannot exceed the loaded checkpoint population"
        )

    evidence_config = AggregationConfig()
    accumulator = ScopeMetricsAccumulator()
    _synchronize(bank)
    started = time.perf_counter()
    for sample in generate_large_scope_dataset(
        args.split,
        args.world_count,
        config,
        start_seed=args.start_seed,
    ):
        for mode in modes:
            for width in widths:
                accumulator.add(
                    evaluate_scope_sample(
                        bank,
                        sample,
                        width=width,
                        mode=mode,
                        evidence_config=evidence_config,
                    )
                )
    _synchronize(bank)
    elapsed = time.perf_counter() - started

    payload = {
        "benchmark_version": LARGE_SCOPE_BENCHMARK_VERSION,
        "split": args.split,
        "world_count": args.world_count,
        "start_seed": args.start_seed,
        "config": {
            "window_count": config.window_count,
            "target_difficulty": config.target_difficulty.value,
            "distractor_difficulty": config.distractor_difficulty.value,
            "ambiguous_distractor_fraction": config.ambiguous_distractor_fraction,
        },
        "widths": list(widths),
        "modes": [mode.value for mode in modes],
        "evidence_config": asdict(evidence_config),
        "population_width": bank.population_width,
        "unit_config": asdict(bank.unit_config),
        "checkpoints": [asdict(checkpoint) for checkpoint in bank.checkpoints],
        "elapsed_seconds": elapsed,
        "local_window_evaluations": (
            args.world_count * len(modes) * sum(widths)
        ),
        "summaries": [summary.to_dict() for summary in accumulator.summaries()],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _synchronize(bank: HomogeneousWorkerBank) -> None:
    if bank.device.type == "cuda":
        torch.cuda.synchronize(bank.device)


if __name__ == "__main__":
    raise SystemExit(main())
