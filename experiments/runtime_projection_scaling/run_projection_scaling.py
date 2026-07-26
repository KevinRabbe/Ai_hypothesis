"""Measure replay-vs-indexed integration partition planning as durable history grows.

This benchmark isolates systems cost only. It does not measure neural quality, uncertainty
reduction, or population advantage.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from ai_hypothesis.runtime.indexed_control import IndexedRuntimeIntegrationTracker
from ai_hypothesis.runtime.indexed_integration_partitions import (
    IndexedIntegrationPartitionPlanner,
)
from ai_hypothesis.runtime.integration_parallelism import IntegrationPartitionAllocator
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger


@dataclass(frozen=True, slots=True)
class ProjectionScalingPoint:
    history_event_count: int
    new_tail_event_count: int
    pending_evidence_count: int
    total_ledger_event_count: int
    partition_count: int
    indexed_catchup_ms: float
    indexed_warm_median_ms: float
    replay_median_ms: float
    replay_to_indexed_warm_ratio: float
    plan_equivalent: bool


@dataclass(frozen=True, slots=True)
class ProjectionScalingResult:
    repeats: int
    points: tuple[ProjectionScalingPoint, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "repeats": self.repeats,
            "points": [asdict(point) for point in self.points],
        }


def run_benchmark(
    *,
    history_event_counts: Sequence[int],
    pending_evidence_count: int = 128,
    repeats: int = 5,
) -> ProjectionScalingResult:
    counts = tuple(history_event_counts)
    if not counts:
        raise ValueError("history_event_counts must not be empty")
    if any(count < 0 for count in counts):
        raise ValueError("history_event_counts must be non-negative")
    if tuple(sorted(set(counts))) != counts:
        raise ValueError("history_event_counts must be strictly increasing")
    if pending_evidence_count <= 0:
        raise ValueError("pending_evidence_count must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        ledger = SQLiteResearchLedger(root / "ledger.sqlite3")
        integration = IndexedRuntimeIntegrationTracker(
            ledger,
            root / "integration.sqlite3",
        )
        indexed = IndexedIntegrationPartitionPlanner(
            ledger=ledger,
            integration=integration,
        )
        replay = IntegrationPartitionAllocator()
        try:
            thread_id = "benchmark-thread"
            ledger.append_event(
                event_type="THREAD_CREATED",
                thread_id=thread_id,
                payload={
                    "objective": "projection scaling benchmark",
                    "purpose": "PROGRESS",
                    "status": "ACTIVE",
                },
            )
            for index in range(pending_evidence_count):
                evidence_id = f"pending-{index:08d}"
                ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id=thread_id,
                    reference_ids=(evidence_id, f"source-{index:08d}"),
                    payload={
                        "evidence_id": evidence_id,
                        "kind": "OBSERVATION",
                        "summary": f"pending observation {index}",
                        "strength": 0.8,
                        "uncertainty": 0.2,
                    },
                )

            points: list[ProjectionScalingPoint] = []
            current_history_count = 0
            for target_history_count in counts:
                new_tail = target_history_count - current_history_count
                for index in range(current_history_count, target_history_count):
                    ledger.append_event(
                        event_type="OBSERVATION_RECORDED",
                        payload={"observation": f"irrelevant-history-{index}"},
                    )
                current_history_count = target_history_count
                revision = ledger.latest_sequence()

                indexed_catchup_ms, indexed_plan = _time_once(
                    lambda: indexed.plan(sequence=revision, thread_id=thread_id)
                )
                replay_plan = replay.plan(ledger)
                equivalent = _indexed_fingerprint(indexed_plan) == _replay_fingerprint(
                    replay,
                    replay_plan,
                    thread_id,
                )

                indexed_samples = _timed_samples(
                    lambda: indexed.plan(sequence=revision, thread_id=thread_id),
                    repeats,
                )
                replay_samples = _timed_samples(
                    lambda: replay.plan(ledger),
                    repeats,
                )
                indexed_median = statistics.median(indexed_samples)
                replay_median = statistics.median(replay_samples)
                ratio = (
                    replay_median / indexed_median
                    if indexed_median > 0.0
                    else float("inf")
                )
                points.append(
                    ProjectionScalingPoint(
                        history_event_count=target_history_count,
                        new_tail_event_count=new_tail,
                        pending_evidence_count=pending_evidence_count,
                        total_ledger_event_count=revision,
                        partition_count=len(indexed_plan.partitions),
                        indexed_catchup_ms=indexed_catchup_ms,
                        indexed_warm_median_ms=indexed_median,
                        replay_median_ms=replay_median,
                        replay_to_indexed_warm_ratio=ratio,
                        plan_equivalent=equivalent,
                    )
                )
            return ProjectionScalingResult(repeats=repeats, points=tuple(points))
        finally:
            integration.close()
            ledger.close()


def _time_once(callable_) -> tuple[float, object]:
    gc.collect()
    started = time.perf_counter()
    value = callable_()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, value


def _timed_samples(callable_, repeats: int) -> tuple[float, ...]:
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        callable_()
        samples.append((time.perf_counter() - started) * 1000.0)
    return tuple(samples)


def _indexed_fingerprint(plan) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            partition.partition_id,
            partition.shard_index,
            partition.shard_count,
            partition.backlog_count,
            partition.oldest_pending_sequence,
            partition.evidence_ids,
        )
        for partition in plan.ordered_for_execution()
    )


def _replay_fingerprint(
    allocator: IntegrationPartitionAllocator,
    plan,
    thread_id: str,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            partition.partition_id,
            partition.shard_index,
            partition.shard_count,
            partition.backlog_count,
            partition.oldest_pending_sequence,
            partition.evidence_ids,
        )
        for partition in allocator.ordered_for_thread(plan, thread_id)
    )


def _parse_history_counts(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("history counts must not be empty")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("history counts must be non-negative")
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("history counts must be strictly increasing")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-events",
        type=_parse_history_counts,
        default=(0, 1_000, 10_000, 50_000),
        help="comma-separated strictly increasing irrelevant-history counts",
    )
    parser.add_argument("--pending-evidence", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_benchmark(
        history_event_counts=args.history_events,
        pending_evidence_count=args.pending_evidence,
        repeats=args.repeats,
    )
    payload = json.dumps(result.to_json_dict(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
