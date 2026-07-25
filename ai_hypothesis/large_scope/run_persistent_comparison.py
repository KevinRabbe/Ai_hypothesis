"""Run normalized direct versus persistent large-scope execution on frozen workers."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ai_hypothesis.step01.schema import Difficulty
from ai_hypothesis.step02.evidence import AggregationConfig
from ai_hypothesis.step02.population import HomogeneousWorkerBank

from .evaluate import ScopeWorkerMode, evaluate_scope_batch
from .metrics import ScopeMetricsAccumulator
from .persistent_batch import PersistentScopeWorldBatchExperiment
from .persistent_comparison import (
    ScopeEquivalenceAccumulator,
    compare_scope_evaluations,
    scope_evaluation_from_persistent,
)
from .relevance import (
    LARGE_SCOPE_SPLIT_SEED_RANGES,
    LargeScopeRelevanceConfig,
    generate_large_scope_dataset,
)
from .runtime_bridge import LargeScopeRuntimeWorkerBank
from ai_hypothesis.runtime import SQLiteResearchLedger


PERSISTENT_COMPARISON_VERSION = "persistent-scope-comparison-v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct batched and persistent large-scope execution under the "
            "same frozen worker/local-evaluation budget."
        )
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
    parser.add_argument("--world-batch-size", type=int, default=64)
    parser.add_argument("--window-count", type=int, default=16)
    parser.add_argument("--step-width", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=4)
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
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        default="results/persistent_scope_comparison_v0.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split == "test" and not args.allow_test_split:
        raise SystemExit(
            "Refusing to open the frozen test split without --allow-test-split"
        )
    for name, value in (
        ("--world-count", args.world_count),
        ("--world-batch-size", args.world_batch_size),
        ("--window-count", args.window_count),
        ("--step-width", args.step_width),
        ("--rounds", args.rounds),
    ):
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    if args.start_seed < 0:
        raise SystemExit("--start-seed must be non-negative")
    if args.equivalence_tolerance < 0.0:
        raise SystemExit("--equivalence-tolerance must be non-negative")

    direct_width = args.step_width * args.rounds
    if direct_width > args.window_count:
        raise SystemExit(
            "step-width × rounds must not exceed --window-count for the neutral "
            "non-redundant equivalence baseline"
        )

    config = LargeScopeRelevanceConfig(
        window_count=args.window_count,
        target_difficulty=Difficulty(args.target_difficulty),
        distractor_difficulty=Difficulty(args.distractor_difficulty),
        ambiguous_distractor_fraction=args.ambiguous_distractor_fraction,
    )
    config.validate()
    modes = tuple(ScopeWorkerMode(value) for value in args.modes)
    if len(set(modes)) != len(modes):
        raise SystemExit("--modes must not contain duplicates")

    bank = HomogeneousWorkerBank.from_checkpoints(
        args.checkpoints,
        device=args.device,
        execution_backend=args.backend,
    )
    if (
        ScopeWorkerMode.DIVERSE_WORKERS in modes
        and direct_width > bank.population_width
    ):
        raise SystemExit(
            "diverse direct width cannot exceed the loaded checkpoint population"
        )

    evidence_config = AggregationConfig()
    worker_bank_id = LargeScopeRuntimeWorkerBank(
        bank,
        evidence_config,
    ).worker_bank_id
    direct_metrics = ScopeMetricsAccumulator()
    persistent_metrics = ScopeMetricsAccumulator()
    equivalence = {
        mode: ScopeEquivalenceAccumulator(
            tolerance=args.equivalence_tolerance,
        )
        for mode in modes
    }
    timing = {
        mode: {
            "direct_seconds": 0.0,
            "persistent_setup_seconds": 0.0,
            "persistent_run_seconds": 0.0,
            "persistent_storage_bytes": 0,
            "persistent_ledger_events": 0,
        }
        for mode in modes
    }
    warmed_modes: set[ScopeWorkerMode] = set()

    for chunk_offset in range(0, args.world_count, args.world_batch_size):
        chunk_count = min(args.world_batch_size, args.world_count - chunk_offset)
        samples = tuple(
            generate_large_scope_dataset(
                args.split,
                chunk_count,
                config,
                start_seed=args.start_seed + chunk_offset,
            )
        )
        if len(samples) != chunk_count:
            raise RuntimeError("large-scope generator returned the wrong chunk size")

        for mode in modes:
            if mode not in warmed_modes:
                evaluate_scope_batch(
                    bank,
                    samples[:1],
                    width=direct_width,
                    mode=mode,
                    evidence_config=evidence_config,
                )
                _synchronize(bank)
                warmed_modes.add(mode)

            _synchronize(bank)
            direct_started = time.perf_counter()
            direct_results = evaluate_scope_batch(
                bank,
                samples,
                width=direct_width,
                mode=mode,
                evidence_config=evidence_config,
            )
            _synchronize(bank)
            timing[mode]["direct_seconds"] += time.perf_counter() - direct_started

            with tempfile.TemporaryDirectory(
                prefix="ai-hypothesis-persistent-scope-"
            ) as directory:
                ledger_path = Path(directory) / "research-ledger.sqlite"
                setup_started = time.perf_counter()
                ledger = SQLiteResearchLedger(ledger_path)
                try:
                    experiment = PersistentScopeWorldBatchExperiment(
                        ledger=ledger,
                        samples=samples,
                        bank=bank,
                        mode=mode,
                        step_width=args.step_width,
                        evidence_config=evidence_config,
                    )
                    timing[mode]["persistent_setup_seconds"] += (
                        time.perf_counter() - setup_started
                    )

                    _synchronize(bank)
                    persistent_started = time.perf_counter()
                    persistent_result = experiment.run_rounds(args.rounds)
                    _synchronize(bank)
                    timing[mode]["persistent_run_seconds"] += (
                        time.perf_counter() - persistent_started
                    )
                    events = ledger.read_all_events()
                    timing[mode]["persistent_ledger_events"] += len(events)
                    timing[mode]["persistent_storage_bytes"] += _sqlite_storage_bytes(
                        ledger_path
                    )
                finally:
                    ledger.close()

            if len(direct_results) != len(persistent_result.worlds):
                raise RuntimeError("direct/persistent world counts differ")
            for direct, persistent_world in zip(
                direct_results,
                persistent_result.worlds,
                strict=True,
            ):
                persistent = scope_evaluation_from_persistent(persistent_world)
                observation = compare_scope_evaluations(direct, persistent)
                equivalence[mode].add(observation)
                direct_metrics.add(direct)
                persistent_metrics.add(persistent)

    local_evaluations_per_mode = args.world_count * direct_width
    mode_payload: dict[str, object] = {}
    all_equivalent = True
    for mode in modes:
        eq = equivalence[mode].summary()
        all_equivalent = all_equivalent and eq.passed
        direct_seconds = float(timing[mode]["direct_seconds"])
        persistent_setup = float(timing[mode]["persistent_setup_seconds"])
        persistent_run = float(timing[mode]["persistent_run_seconds"])
        persistent_total = persistent_setup + persistent_run
        mode_payload[mode.value] = {
            "equivalence": asdict(eq),
            "timing": {
                **timing[mode],
                "persistent_total_seconds": persistent_total,
                "direct_local_evaluations_per_second": _rate(
                    local_evaluations_per_mode,
                    direct_seconds,
                ),
                "persistent_run_local_evaluations_per_second": _rate(
                    local_evaluations_per_mode,
                    persistent_run,
                ),
                "persistent_end_to_end_local_evaluations_per_second": _rate(
                    local_evaluations_per_mode,
                    persistent_total,
                ),
                "persistent_over_direct_time_ratio": (
                    persistent_total / direct_seconds
                    if direct_seconds > 0.0
                    else None
                ),
            },
        }

    payload = {
        "comparison_version": PERSISTENT_COMPARISON_VERSION,
        "split": args.split,
        "world_count": args.world_count,
        "start_seed": args.start_seed,
        "world_batch_size": args.world_batch_size,
        "step_width": args.step_width,
        "rounds": args.rounds,
        "direct_width": direct_width,
        "local_evaluations_per_mode": local_evaluations_per_mode,
        "config": {
            "window_count": config.window_count,
            "target_difficulty": config.target_difficulty.value,
            "distractor_difficulty": config.distractor_difficulty.value,
            "ambiguous_distractor_fraction": config.ambiguous_distractor_fraction,
        },
        "modes": [mode.value for mode in modes],
        "evidence_config": asdict(evidence_config),
        "equivalence_tolerance": args.equivalence_tolerance,
        "population_width": bank.population_width,
        "unit_config": asdict(bank.unit_config),
        "worker_bank_id": worker_bank_id,
        "checkpoints": [asdict(checkpoint) for checkpoint in bank.checkpoints],
        "direct_summaries": [
            summary.to_dict() for summary in direct_metrics.summaries()
        ],
        "persistent_summaries": [
            summary.to_dict() for summary in persistent_metrics.summaries()
        ],
        "mode_results": mode_payload,
        "equivalence_passed": all_equivalent,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_equivalent else 2


def _synchronize(bank: HomogeneousWorkerBank) -> None:
    if bank.device.type == "cuda":
        torch.cuda.synchronize(bank.device)


def _sqlite_storage_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        )
        if candidate.exists()
    )


def _rate(count: int, seconds: float) -> float | None:
    if seconds <= 0.0:
        return None
    return count / seconds


if __name__ == "__main__":
    raise SystemExit(main())
