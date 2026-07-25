"""Profile learned versus non-neural cost for one normalized persistent scope chunk."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ai_hypothesis.runtime import SQLiteResearchLedger
from ai_hypothesis.step01.schema import Difficulty
from ai_hypothesis.step02.evidence import AggregationConfig
from ai_hypothesis.step02.population import HomogeneousWorkerBank

from .evaluate import ScopeWorkerMode, evaluate_scope_batch
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
from .timing import TimedSelectedWorkerBank


PERSISTENT_NEURAL_PROFILE_VERSION = "persistent-neural-profile-v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile selected-worker learned time versus persistent organization cost "
            "for one bounded large-scope world chunk."
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
    parser.add_argument("--world-count", type=int, default=64)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--window-count", type=int, default=16)
    parser.add_argument("--step-width", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in ScopeWorkerMode),
        default=ScopeWorkerMode.DIVERSE_WORKERS.value,
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
        default="results/persistent_neural_profile_v0.json",
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
            "non-redundant profile"
        )

    config = LargeScopeRelevanceConfig(
        window_count=args.window_count,
        target_difficulty=Difficulty(args.target_difficulty),
        distractor_difficulty=Difficulty(args.distractor_difficulty),
        ambiguous_distractor_fraction=args.ambiguous_distractor_fraction,
    )
    config.validate()
    mode = ScopeWorkerMode(args.mode)
    evidence_config = AggregationConfig()

    raw_bank = HomogeneousWorkerBank.from_checkpoints(
        args.checkpoints,
        device=args.device,
        execution_backend=args.backend,
    )
    if mode is ScopeWorkerMode.DIVERSE_WORKERS and direct_width > raw_bank.population_width:
        raise SystemExit(
            "diverse direct width cannot exceed the loaded checkpoint population"
        )
    bank = TimedSelectedWorkerBank(raw_bank)

    samples = tuple(
        generate_large_scope_dataset(
            args.split,
            args.world_count,
            config,
            start_seed=args.start_seed,
        )
    )
    if len(samples) != args.world_count:
        raise RuntimeError("large-scope generator returned the wrong world count")

    # Warm the exact selected-worker path outside both condition timers.
    evaluate_scope_batch(
        bank,
        samples[:1],
        width=direct_width,
        mode=mode,
        evidence_config=evidence_config,
    )
    _synchronize(raw_bank)
    bank.reset_timing()

    direct_started = time.perf_counter()
    direct_results = evaluate_scope_batch(
        bank,
        samples,
        width=direct_width,
        mode=mode,
        evidence_config=evidence_config,
    )
    _synchronize(raw_bank)
    direct_total_seconds = time.perf_counter() - direct_started
    direct_neural = bank.snapshot_after_synchronize()
    bank.reset_timing()

    with tempfile.TemporaryDirectory(
        prefix="ai-hypothesis-persistent-neural-profile-"
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
            persistent_setup_seconds = time.perf_counter() - setup_started

            _synchronize(raw_bank)
            bank.reset_timing()
            persistent_started = time.perf_counter()
            persistent_result = experiment.run_rounds(args.rounds)
            _synchronize(raw_bank)
            persistent_run_seconds = time.perf_counter() - persistent_started
            persistent_neural = bank.snapshot_after_synchronize()

            events = ledger.read_all_events()
            ledger_event_count = len(events)
            storage_bytes = _sqlite_storage_bytes(ledger_path)
            worker_bank_id = experiment.runtime_bank.worker_bank_id
        finally:
            ledger.close()

    equivalence = ScopeEquivalenceAccumulator(
        tolerance=args.equivalence_tolerance
    )
    for direct, persistent_world in zip(
        direct_results,
        persistent_result.worlds,
        strict=True,
    ):
        equivalence.add(
            compare_scope_evaluations(
                direct,
                scope_evaluation_from_persistent(persistent_world),
            )
        )
    equivalence_summary = equivalence.summary()

    local_evaluations = args.world_count * direct_width
    direct_non_neural = max(
        0.0,
        direct_total_seconds - direct_neural.elapsed_seconds,
    )
    persistent_non_neural = max(
        0.0,
        persistent_run_seconds - persistent_neural.elapsed_seconds,
    )
    persistent_total = persistent_setup_seconds + persistent_run_seconds

    payload = {
        "profile_version": PERSISTENT_NEURAL_PROFILE_VERSION,
        "split": args.split,
        "world_count": args.world_count,
        "start_seed": args.start_seed,
        "mode": mode.value,
        "step_width": args.step_width,
        "rounds": args.rounds,
        "direct_width": direct_width,
        "local_evaluations": local_evaluations,
        "config": {
            "window_count": config.window_count,
            "target_difficulty": config.target_difficulty.value,
            "distractor_difficulty": config.distractor_difficulty.value,
            "ambiguous_distractor_fraction": config.ambiguous_distractor_fraction,
        },
        "evidence_config": asdict(evidence_config),
        "equivalence": asdict(equivalence_summary),
        "population_width": raw_bank.population_width,
        "unit_config": asdict(raw_bank.unit_config),
        "worker_bank_id": worker_bank_id,
        "checkpoints": [asdict(checkpoint) for checkpoint in raw_bank.checkpoints],
        "direct": {
            "total_seconds": direct_total_seconds,
            "selected_worker": asdict(direct_neural),
            "non_selected_worker_seconds": direct_non_neural,
            "local_evaluations_per_second": _rate(
                local_evaluations,
                direct_total_seconds,
            ),
        },
        "persistent": {
            "setup_seconds": persistent_setup_seconds,
            "run_seconds": persistent_run_seconds,
            "total_seconds": persistent_total,
            "selected_worker": asdict(persistent_neural),
            "non_selected_worker_run_seconds": persistent_non_neural,
            "run_local_evaluations_per_second": _rate(
                local_evaluations,
                persistent_run_seconds,
            ),
            "end_to_end_local_evaluations_per_second": _rate(
                local_evaluations,
                persistent_total,
            ),
            "ledger_event_count": ledger_event_count,
            "storage_bytes": storage_bytes,
        },
        "derived": {
            "persistent_over_direct_total_time_ratio": _ratio(
                persistent_total,
                direct_total_seconds,
            ),
            "persistent_over_direct_run_time_ratio": _ratio(
                persistent_run_seconds,
                direct_total_seconds,
            ),
            "persistent_over_direct_selected_worker_time_ratio": _ratio(
                persistent_neural.elapsed_seconds,
                direct_neural.elapsed_seconds,
            ),
            "persistent_non_selected_worker_seconds_per_local_evaluation": (
                persistent_non_neural / local_evaluations
            ),
            "direct_non_selected_worker_seconds_per_local_evaluation": (
                direct_non_neural / local_evaluations
            ),
        },
        "equivalence_passed": equivalence_summary.passed,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if equivalence_summary.passed else 2


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


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
