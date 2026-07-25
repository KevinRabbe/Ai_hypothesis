"""Prove synthesis/integration work uses the ordinary bounded worker path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    EvidenceDisposition,
    EvidenceDispositionKind,
    IntegrationBackpressureConfig,
    IntegrationTracker,
    KnowledgeDelta,
    SQLiteResearchLedger,
    WorkerAssignment,
    WorkerRuntime,
    WorkItem,
    WorkPurpose,
)


class _IntegrationWorkerBank:
    def execute_batch(self, requests):
        results = []
        for request in requests:
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    knowledge_deltas=(
                        KnowledgeDelta(
                            delta_id=f"{request.attempt_id}:delta",
                            kind="EVIDENCE_SYNTHESIZED",
                            summary="Two local findings were integrated",
                            reference_ids=request.work_item.reference_ids,
                            thread_id=request.work_item.thread_id,
                        ),
                    ),
                    evidence_dispositions=(
                        EvidenceDisposition(
                            evidence_ids=request.work_item.reference_ids,
                            disposition=EvidenceDispositionKind.INTEGRATED,
                            reason="Represented by the emitted knowledge delta",
                        ),
                    ),
                    progress_made=True,
                )
            )
        return tuple(results)


class RecursiveIntegrationTests(unittest.TestCase):
    def test_synthesis_attempt_drains_backlog_via_normal_worker_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="integration-thread",
                    payload={
                        "objective": "Integrate pending evidence",
                        "purpose": "SYNTHESIZE",
                    },
                )
                for evidence_id in ("evidence-1", "evidence-2"):
                    ledger.append_event(
                        event_type="EVIDENCE_ADDED",
                        thread_id="source-thread",
                        reference_ids=(evidence_id,),
                        payload={"evidence_id": evidence_id},
                    )

                tracker = IntegrationTracker(
                    ledger,
                    IntegrationBackpressureConfig(
                        max_backlog_count=0,
                        max_backlog_age_sequences=100,
                    ),
                )
                self.assertTrue(tracker.is_backpressured())

                runtime = WorkerRuntime(ledger)
                result = runtime.run_attempt(
                    WorkerAssignment(
                        worker_id="worker-integration",
                        work_item=WorkItem(
                            work_item_id="work-integrate",
                            thread_id="integration-thread",
                            objective="Integrate pending evidence",
                            purpose=WorkPurpose.SYNTHESIZE,
                            projection_revision=3,
                            reference_ids=("evidence-1", "evidence-2"),
                        ),
                    ),
                    _IntegrationWorkerBank(),
                )

                self.assertEqual(len(result.knowledge_deltas), 1)
                snapshot = tracker.snapshot()
                self.assertEqual(snapshot.backlog_count, 0)
                self.assertEqual(snapshot.dispositioned_evidence_count, 2)
                self.assertEqual(snapshot.knowledge_delta_count, 1)
                self.assertFalse(tracker.is_backpressured())

                event_types = tuple(event.event_type for event in ledger.read_events())
                self.assertIn("KNOWLEDGE_DELTA_RECORDED", event_types)
                self.assertIn("INTEGRATION_DISPOSITION_RECORDED", event_types)


if __name__ == "__main__":
    unittest.main()
