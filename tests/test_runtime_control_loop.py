"""End-to-end tests for the thin persistent runtime composition loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    EvidenceContribution,
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SQLiteResearchLedger,
    WorkPreparation,
)


class _RecordingWorkerBank:
    def __init__(self) -> None:
        self.worker_ids: list[str] = []
        self.reference_batches: list[tuple[str, ...]] = []

    def execute_batch(self, requests):
        results = []
        for request in requests:
            self.worker_ids.append(request.worker_id)
            self.reference_batches.append(request.work_item.reference_ids)
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    evidence=(
                        EvidenceContribution(
                            evidence_id=f"{request.attempt_id}:evidence",
                            kind="TEST_EVIDENCE",
                            summary="bounded local finding",
                            reference_ids=request.work_item.reference_ids,
                            strength=1.0,
                            uncertainty=0.1,
                            data={"useful": True},
                        ),
                    ),
                    progress_made=True,
                )
            )
        return tuple(results)


class RuntimeControlLoopTests(unittest.TestCase):
    def test_productive_depth_reuses_worker_but_stagnation_rotates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingWorkerBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=0.0)
                    ),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b"),
                )
                loop.create_thread(
                    thread_id="thread-1",
                    objective="Investigate H2",
                    reference_ids=("global-source-a", "global-source-b"),
                )

                progress = {"value": 1.0}

                def signals(_state):
                    return SchedulerSignals(
                        importance=1.0,
                        recent_progress=progress["value"],
                    )

                def prepare(_state, _decision):
                    # Deliberately choose one bounded source rather than forwarding
                    # the thread's ever-growing global reference set.
                    return WorkPreparation(reference_ids=("bounded-source",))

                first = loop.run_once(signal_provider=signals, context_provider=prepare)
                second = loop.run_once(signal_provider=signals, context_provider=prepare)
                progress["value"] = 0.0
                third = loop.run_once(signal_provider=signals, context_provider=prepare)

                self.assertEqual(first.decision.action, SchedulerAction.CONTINUE)
                self.assertEqual(second.decision.action, SchedulerAction.CONTINUE)
                self.assertEqual(third.decision.action, SchedulerAction.ROTATE_WORKER)
                self.assertEqual(bank.worker_ids, ["worker-a", "worker-a", "worker-b"])
                self.assertEqual(
                    bank.reference_batches,
                    [("bounded-source",), ("bounded-source",), ("bounded-source",)],
                )

    def test_backpressure_redirects_work_to_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingWorkerBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=1.0)
                    ),
                    worker_bank=bank,
                    worker_ids=("worker-a", "worker-b"),
                )
                loop.create_thread(thread_id="thread-1", objective="Integrate findings")

                step = loop.run_once(
                    signal_provider=lambda _state: SchedulerSignals(
                        importance=1.0,
                        missing_coverage=1.0,
                        recent_progress=1.0,
                    ),
                    context_provider=lambda _state, _decision: WorkPreparation(),
                    integration_backpressure=True,
                )

                self.assertEqual(step.decision.action, SchedulerAction.SYNTHESIZE)
                self.assertEqual(step.assignment.work_item.purpose.value, "SYNTHESIZE")
                self.assertEqual(bank.worker_ids, ["worker-a"])

    def test_generated_evidence_changes_global_state_without_expanding_next_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _RecordingWorkerBank()
                loop = RuntimeControlLoop(
                    ledger=ledger,
                    scheduler=SchedulerV0(
                        SchedulerConfig(exploration_probability=0.0)
                    ),
                    worker_bank=bank,
                    worker_ids=("worker-a",),
                )
                loop.create_thread(
                    thread_id="thread-1",
                    objective="Grow evidence while bounding active context",
                    reference_ids=("root-source",),
                )

                provider = lambda _state: SchedulerSignals(
                    importance=1.0, recent_progress=1.0
                )
                prepare = lambda _state, _decision: WorkPreparation(
                    reference_ids=("root-source",)
                )

                first = loop.run_once(signal_provider=provider, context_provider=prepare)
                state_after_first = loop.projector.project(
                    ledger.read_events(), thread_id="thread-1"
                )
                self.assertIn(first.result.evidence[0].evidence_id, state_after_first.reference_ids)

                loop.run_once(signal_provider=provider, context_provider=prepare)
                self.assertEqual(bank.reference_batches[-1], ("root-source",))
                self.assertGreater(len(state_after_first.reference_ids), 1)


if __name__ == "__main__":
    unittest.main()
