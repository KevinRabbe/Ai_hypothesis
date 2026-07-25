"""Tests the large-scope benchmark through the persistent runtime boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ai_hypothesis.large_scope import (
    FixedScopeScheduler,
    LargeScopeRuntimeContextProvider,
    LargeScopeRuntimeWorkerBank,
    ScopeWorkerMode,
    evaluate_scope_sample,
    generate_large_scope_relevance,
    large_scope_region_id,
    planned_scope_worker_selector,
)
from ai_hypothesis.runtime import (
    AllocationOutcomeProjector,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerDecision,
    SchedulerSignals,
    ScopeCoverageProjector,
    SQLiteResearchLedger,
    TracingScheduler,
    WorkPurpose,
)
from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output


class _DeterministicSelectedBank:
    def __init__(self, population_width: int = 16) -> None:
        self._population_width = population_width
        self.calls: list[tuple[int, ...]] = []

    @property
    def population_width(self) -> int:
        return self._population_width

    def forward_selected(self, worker_indices, features, mask):
        del mask
        indices = torch.as_tensor(worker_indices, dtype=torch.float32)
        self.calls.append(tuple(int(index) for index in indices.tolist()))
        batch = features.shape[0]
        logits = torch.full((batch, len(NON_UNCERTAIN_LABELS)), -3.0)
        relevant = LABEL_TO_INDEX["RELEVANT"]
        not_relevant = LABEL_TO_INDEX["NOT_RELEVANT"]
        signal = features[:, 1, 0] + indices * 0.03125
        logits[:, relevant] = signal
        logits[:, not_relevant] = -signal
        uncertainty = torch.full((batch,), -2.0) + indices * 0.005
        return Step01Output(label_logits=logits, uncertainty_logits=uncertainty)


class LargeScopeRuntimeBridgeTests(unittest.TestCase):
    def _run_runtime(self, *, sample, width: int, mode: ScopeWorkerMode):
        selected_bank = _DeterministicSelectedBank()
        runtime_bank = LargeScopeRuntimeWorkerBank(selected_bank)
        selector = planned_scope_worker_selector(
            runtime_bank,
            sample,
            mode=mode,
            max_width=width,
        )
        directory = tempfile.TemporaryDirectory()
        ledger = SQLiteResearchLedger(Path(directory.name) / "ledger.sqlite")
        scheduler = TracingScheduler(ledger, FixedScopeScheduler(width))
        loop = RuntimeControlLoop(
            ledger=ledger,
            scheduler=scheduler,
            worker_bank=runtime_bank,
            worker_ids=runtime_bank.worker_ids,
            worker_selector=selector,
        )
        loop.create_thread(thread_id="scope-thread", objective="Inspect benchmark world")
        step = loop.run_once(
            signal_provider=lambda _state: SchedulerSignals(importance=1.0),
            context_provider=LargeScopeRuntimeContextProvider(sample, mode),
        )
        return directory, ledger, selected_bank, step

    def test_diverse_runtime_matches_direct_windows_workers_evidence_and_provenance(self) -> None:
        sample = generate_large_scope_relevance(42, target_present=True)
        width = 4
        direct_bank = _DeterministicSelectedBank()
        direct = evaluate_scope_sample(
            direct_bank,
            sample,
            width=width,
            mode=ScopeWorkerMode.DIVERSE_WORKERS,
        )

        directory, ledger, runtime_selected_bank, step = self._run_runtime(
            sample=sample,
            width=width,
            mode=ScopeWorkerMode.DIVERSE_WORKERS,
        )
        try:
            self.assertEqual(len(step.assignments), width)
            self.assertEqual(
                runtime_selected_bank.calls,
                [tuple(row.worker_index for row in direct.window_evidence)],
            )
            self.assertEqual(
                tuple(
                    assignment.work_item.scope_region_ids[0]
                    for assignment in step.assignments
                ),
                tuple(
                    large_scope_region_id(sample, window_index)
                    for window_index in direct.inspected_window_indices
                ),
            )

            events = ledger.read_all_events()
            started_events = [
                event for event in events if event.event_type == "ATTEMPT_STARTED"
            ]
            self.assertEqual(len(started_events), width)
            self.assertTrue(
                all(
                    event.payload["scheduler_decision_id"] == step.decision.decision_id
                    for event in started_events
                )
            )

            evidence_events = [
                event
                for event in events
                if event.event_type == "EVIDENCE_ADDED"
                and event.payload.get("kind") == "LARGE_SCOPE_RELEVANCE_WINDOW"
            ]
            self.assertEqual(len(evidence_events), width)
            for row, event in zip(direct.window_evidence, evidence_events, strict=True):
                data = event.payload["data"]
                self.assertEqual(data["window_index"], row.window_index)
                self.assertEqual(data["worker_index"], row.worker_index)
                self.assertEqual(data["local_label"], row.local_label)
                self.assertAlmostEqual(data["relevant_evidence"], row.relevant_evidence, places=6)
                self.assertAlmostEqual(
                    data["not_relevant_evidence"], row.not_relevant_evidence, places=6
                )
                self.assertAlmostEqual(
                    event.payload["uncertainty"], row.uncertainty_probability, places=6
                )
                self.assertAlmostEqual(
                    data["invalid_label_mass"], row.invalid_label_mass, places=6
                )
                self.assertAlmostEqual(data["top_margin"], row.top_margin, places=6)

            coverage = ScopeCoverageProjector().for_thread(events, "scope-thread")
            self.assertEqual(
                coverage.resolved_region_ids,
                tuple(
                    large_scope_region_id(sample, window_index)
                    for window_index in direct.inspected_window_indices
                ),
            )

            allocation = AllocationOutcomeProjector().for_decision(
                events, step.decision.decision_id
            )
            self.assertIsNotNone(allocation)
            assert allocation is not None
            self.assertEqual(allocation.attempt_count, width)
            self.assertEqual(allocation.evidence_count, width)
            self.assertEqual(allocation.thread_id, "scope-thread")
            self.assertEqual(allocation.allocated_width, width)
        finally:
            ledger.close()
            directory.cleanup()

    def test_same_worker_control_reuses_checkpoint_across_distinct_regions(self) -> None:
        sample = generate_large_scope_relevance(64)
        width = 4
        direct_bank = _DeterministicSelectedBank()
        direct = evaluate_scope_sample(
            direct_bank,
            sample,
            width=width,
            mode=ScopeWorkerMode.SAME_WORKER,
        )

        directory, ledger, runtime_selected_bank, step = self._run_runtime(
            sample=sample,
            width=width,
            mode=ScopeWorkerMode.SAME_WORKER,
        )
        try:
            assigned_workers = tuple(assignment.worker_id for assignment in step.assignments)
            self.assertEqual(len(set(assigned_workers)), 1)
            self.assertEqual(
                runtime_selected_bank.calls,
                [tuple(row.worker_index for row in direct.window_evidence)],
            )
            self.assertEqual(len(set(runtime_selected_bank.calls[0])), 1)

            coverage = ScopeCoverageProjector().for_thread(
                ledger.read_all_events(), "scope-thread"
            )
            self.assertEqual(len(coverage.resolved_region_ids), width)
            self.assertEqual(
                coverage.resolved_region_ids,
                tuple(
                    large_scope_region_id(sample, window_index)
                    for window_index in direct.inspected_window_indices
                ),
            )
        finally:
            ledger.close()
            directory.cleanup()

    def test_region_identity_is_stable_opaque_and_unique_inside_world(self) -> None:
        sample = generate_large_scope_relevance(88)
        first = tuple(
            large_scope_region_id(sample, index)
            for index in range(sample.config.window_count)
        )
        second = tuple(
            large_scope_region_id(sample, index)
            for index in range(sample.config.window_count)
        )
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), sample.config.window_count)
        self.assertTrue(all(region.startswith("scope-") for region in first))
        self.assertTrue(all(str(sample.seed) not in region for region in first))

    def test_worker_index_lookup_rejects_negative_and_out_of_range_indices(self) -> None:
        runtime_bank = LargeScopeRuntimeWorkerBank(_DeterministicSelectedBank(population_width=4))
        with self.assertRaisesRegex(IndexError, "outside"):
            runtime_bank.worker_id_for_index(-1)
        with self.assertRaisesRegex(IndexError, "outside"):
            runtime_bank.worker_id_for_index(4)

    def test_context_provider_normalizes_string_worker_mode(self) -> None:
        sample = generate_large_scope_relevance(104)
        provider = LargeScopeRuntimeContextProvider(sample, "same_worker")
        preparation = provider(
            object(),
            SchedulerDecision(
                decision_id="decision-1",
                thread_id="thread-1",
                action=SchedulerAction.CONTINUE,
                purpose=WorkPurpose.EXPLORE,
                width=1,
                projection_revision=0,
            ),
        )
        self.assertEqual(preparation.context["large_scope_mode"], "same_worker")


if __name__ == "__main__":
    unittest.main()
