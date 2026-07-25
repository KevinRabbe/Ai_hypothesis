"""Tests deterministic coverage-aware scope planning over persistent ledger history."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ai_hypothesis.large_scope.coverage_planner import CoverageAwareScopePlanner
from ai_hypothesis.large_scope.evaluate import ScopeWorkerMode
from ai_hypothesis.large_scope.relevance import (
    LargeScopeRelevanceConfig,
    generate_large_scope_relevance,
    inspection_order,
)
from ai_hypothesis.large_scope.runtime_bridge import (
    FixedScopeScheduler,
    LargeScopeRuntimeWorkerBank,
    large_scope_region_id,
)
from ai_hypothesis.runtime import (
    RuntimeControlLoop,
    SchedulerSignals,
    SQLiteResearchLedger,
    TracingScheduler,
)
from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output


class _SimpleSelectedBank:
    def __init__(self, population_width: int = 4) -> None:
        self._population_width = population_width
        self.calls: list[tuple[int, ...]] = []

    @property
    def population_width(self) -> int:
        return self._population_width

    def forward_selected(self, worker_indices, features, mask):
        del features, mask
        indices = tuple(int(index) for index in worker_indices)
        self.calls.append(indices)
        batch = len(indices)
        logits = torch.full((batch, len(NON_UNCERTAIN_LABELS)), -3.0)
        logits[:, LABEL_TO_INDEX["NOT_RELEVANT"]] = 3.0
        return Step01Output(
            label_logits=logits,
            uncertainty_logits=torch.full((batch,), -4.0),
        )


def _append_scope_attempt(
    ledger: SQLiteResearchLedger,
    *,
    thread_id: str,
    attempt_id: str,
    worker_id: str,
    region_id: str,
    terminal_event_type: str,
) -> None:
    ledger.append_event(
        event_type="ATTEMPT_STARTED",
        thread_id=thread_id,
        attempt_id=attempt_id,
        payload={
            "worker_id": worker_id,
            "scope_region_ids": [region_id],
        },
    )
    ledger.append_event(
        event_type=terminal_event_type,
        thread_id=thread_id,
        attempt_id=attempt_id,
    )


class CoverageAwareScopePlannerTests(unittest.TestCase):
    def test_initial_plan_matches_deterministic_inspection_prefix(self) -> None:
        sample = generate_large_scope_relevance(
            42,
            LargeScopeRelevanceConfig(window_count=8),
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                planner = CoverageAwareScopePlanner(
                    ledger,
                    sample,
                    ScopeWorkerMode.DIVERSE_WORKERS,
                )
                plan = planner.plan("thread-1", 4)
                order = inspection_order(
                    sample.seed,
                    sample.config.window_count,
                    split=sample.split,
                )
                self.assertEqual(plan.selected_window_indices, order[:4])
                self.assertEqual(plan.missing_region_count, 8)
                self.assertEqual(plan.resolved_region_count, 0)
                self.assertEqual(plan.missing_coverage, 1.0)

    def test_resolved_regions_are_skipped_while_unseen_scope_exists(self) -> None:
        sample = generate_large_scope_relevance(
            44,
            LargeScopeRelevanceConfig(window_count=8),
        )
        order = inspection_order(
            sample.seed,
            sample.config.window_count,
            split=sample.split,
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for offset, window_index in enumerate(order[:2]):
                    _append_scope_attempt(
                        ledger,
                        thread_id="thread-1",
                        attempt_id=f"attempt-{offset}",
                        worker_id=f"worker-{offset}",
                        region_id=large_scope_region_id(sample, window_index),
                        terminal_event_type="ATTEMPT_COMPLETED",
                    )
                planner = CoverageAwareScopePlanner(
                    ledger,
                    sample,
                    ScopeWorkerMode.DIVERSE_WORKERS,
                )
                self.assertEqual(
                    planner.plan("thread-1", 2).selected_window_indices,
                    order[2:4],
                )
                self.assertEqual(planner.plan("thread-1", 1).missing_coverage, 0.75)

    def test_aborted_region_waits_until_never_seen_scope_is_exhausted(self) -> None:
        sample = generate_large_scope_relevance(
            46,
            LargeScopeRelevanceConfig(window_count=4),
        )
        order = inspection_order(
            sample.seed,
            sample.config.window_count,
            split=sample.split,
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _append_scope_attempt(
                    ledger,
                    thread_id="thread-1",
                    attempt_id="crash-0",
                    worker_id="worker-a",
                    region_id=large_scope_region_id(sample, order[0]),
                    terminal_event_type="ATTEMPT_CRASHED",
                )
                planner = CoverageAwareScopePlanner(
                    ledger,
                    sample,
                    ScopeWorkerMode.DIVERSE_WORKERS,
                )
                self.assertEqual(planner.plan("thread-1", 1).selected_window_indices, (order[1],))

                for offset, window_index in enumerate(order[1:], start=1):
                    _append_scope_attempt(
                        ledger,
                        thread_id="thread-1",
                        attempt_id=f"complete-{offset}",
                        worker_id=f"worker-{offset}",
                        region_id=large_scope_region_id(sample, window_index),
                        terminal_event_type="ATTEMPT_COMPLETED",
                    )
                self.assertEqual(planner.plan("thread-1", 1).selected_window_indices, (order[0],))

    def test_after_full_coverage_redundancy_goes_to_least_replicated_regions(self) -> None:
        sample = generate_large_scope_relevance(
            48,
            LargeScopeRelevanceConfig(window_count=4),
        )
        order = inspection_order(
            sample.seed,
            sample.config.window_count,
            split=sample.split,
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for offset, window_index in enumerate(order):
                    _append_scope_attempt(
                        ledger,
                        thread_id="thread-1",
                        attempt_id=f"initial-{offset}",
                        worker_id=f"worker-{offset}",
                        region_id=large_scope_region_id(sample, window_index),
                        terminal_event_type="ATTEMPT_COMPLETED",
                    )
                _append_scope_attempt(
                    ledger,
                    thread_id="thread-1",
                    attempt_id="replicate-first",
                    worker_id="worker-extra",
                    region_id=large_scope_region_id(sample, order[0]),
                    terminal_event_type="ATTEMPT_COMPLETED",
                )
                planner = CoverageAwareScopePlanner(
                    ledger,
                    sample,
                    ScopeWorkerMode.DIVERSE_WORKERS,
                )
                plan = planner.plan("thread-1", 1)
                self.assertEqual(plan.missing_region_count, 0)
                self.assertEqual(plan.selected_window_indices, (order[1],))

    def test_signal_augmentation_uses_observed_missing_fraction_and_preserves_stronger_input(self) -> None:
        sample = generate_large_scope_relevance(
            50,
            LargeScopeRelevanceConfig(window_count=4),
        )
        order = inspection_order(
            sample.seed,
            sample.config.window_count,
            split=sample.split,
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                _append_scope_attempt(
                    ledger,
                    thread_id="thread-1",
                    attempt_id="attempt-1",
                    worker_id="worker-a",
                    region_id=large_scope_region_id(sample, order[0]),
                    terminal_event_type="ATTEMPT_COMPLETED",
                )
                planner = CoverageAwareScopePlanner(
                    ledger,
                    sample,
                    ScopeWorkerMode.DIVERSE_WORKERS,
                )
                from ai_hypothesis.runtime import ProjectedState, WorkPurpose

                state = ProjectedState(
                    revision=0,
                    thread_id="thread-1",
                    objective="inspect",
                    status="ACTIVE",
                    purpose=WorkPurpose.EXPLORE,
                )
                augmented = planner.augment_signals(
                    state,
                    SchedulerSignals(missing_coverage=0.1),
                )
                self.assertEqual(augmented.missing_coverage, 0.75)
                stronger = planner.augment_signals(
                    state,
                    SchedulerSignals(missing_coverage=0.9),
                )
                self.assertEqual(stronger.missing_coverage, 0.9)

    def test_two_persistent_rounds_advance_to_new_scope(self) -> None:
        sample = generate_large_scope_relevance(
            52,
            LargeScopeRelevanceConfig(window_count=8),
        )
        order = inspection_order(
            sample.seed,
            sample.config.window_count,
            split=sample.split,
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                selected_bank = _SimpleSelectedBank(population_width=4)
                runtime_bank = LargeScopeRuntimeWorkerBank(selected_bank)
                planner = CoverageAwareScopePlanner(
                    ledger,
                    sample,
                    ScopeWorkerMode.DIVERSE_WORKERS,
                )
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=TracingScheduler(ledger, FixedScopeScheduler(2)),
                    worker_bank=runtime_bank,
                    worker_ids=runtime_bank.worker_ids,
                )
                loop.create_thread(thread_id="thread-1", objective="inspect world")

                first = loop.run_once(
                    signal_provider=planner.signal_provider(
                        lambda _state: SchedulerSignals(importance=1.0)
                    ),
                    context_provider=planner,
                )
                second = loop.run_once(
                    signal_provider=planner.signal_provider(
                        lambda _state: SchedulerSignals(importance=1.0)
                    ),
                    context_provider=planner,
                )

                first_windows = tuple(
                    assignment.work_item.context["large_scope_window_index"]
                    for assignment in first.assignments
                )
                second_windows = tuple(
                    assignment.work_item.context["large_scope_window_index"]
                    for assignment in second.assignments
                )
                self.assertEqual(first_windows, order[:2])
                self.assertEqual(second_windows, order[2:4])
                self.assertTrue(set(first_windows).isdisjoint(second_windows))
                coverage = planner.coverage_for("thread-1")
                self.assertEqual(len(coverage.resolved_region_ids), 4)
                self.assertEqual(
                    planner.plan("thread-1", 1).missing_coverage,
                    0.5,
                )


if __name__ == "__main__":
    unittest.main()
