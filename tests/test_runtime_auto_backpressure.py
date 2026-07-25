"""Tests automatic scheduler redirection from measured integration backlog."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    EvidenceContribution,
    IntegrationBackpressureConfig,
    IntegrationTracker,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SQLiteResearchLedger,
    WorkPreparation,
)


class _EvidenceWorkerBank:
    def execute_batch(self, requests):
        return tuple(
            AttemptResult(
                attempt_id=request.attempt_id,
                work_item_id=request.work_item.work_item_id,
                thread_id=request.work_item.thread_id,
                worker_id=request.worker_id,
                status=AttemptStatus.COMPLETED,
                evidence=(
                    EvidenceContribution(
                        evidence_id=f"{request.attempt_id}:evidence",
                        kind="LOCAL_FINDING",
                        summary="new local evidence",
                        reference_ids=request.work_item.reference_ids,
                        strength=1.0,
                        uncertainty=0.2,
                    ),
                ),
                progress_made=True,
            )
            for request in requests
        )


class AutomaticBackpressureTests(unittest.TestCase):
    def test_pending_evidence_redirects_next_step_to_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                tracker = IntegrationTracker(
                    ledger,
                    IntegrationBackpressureConfig(
                        max_backlog_count=0,
                        max_backlog_age_sequences=100,
                    ),
                )
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=0.0)
                    ),
                    worker_bank=_EvidenceWorkerBank(),
                    worker_ids=("worker-a", "worker-b"),
                    integration_tracker=tracker,
                )
                loop.create_thread(
                    thread_id="thread-1",
                    objective="Search and integrate evidence",
                    reference_ids=("source-1",),
                )
                signals = lambda _state: SchedulerSignals(
                    importance=1.0,
                    recent_progress=1.0,
                )
                prepare = lambda _state, _decision: WorkPreparation(
                    reference_ids=("source-1",)
                )

                first = loop.run_once(
                    signal_provider=signals,
                    context_provider=prepare,
                )
                self.assertEqual(first.decision.action, SchedulerAction.CONTINUE)
                self.assertTrue(tracker.is_backpressured())

                second = loop.run_once(
                    signal_provider=signals,
                    context_provider=prepare,
                )
                self.assertEqual(second.decision.action, SchedulerAction.SYNTHESIZE)
                self.assertEqual(second.assignment.work_item.purpose.value, "SYNTHESIZE")


if __name__ == "__main__":
    unittest.main()
