"""Tests append-only tracing for schedulers other than SchedulerV0."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    SchedulerAction,
    SchedulerDecision,
    SchedulerSignals,
    SchedulableThread,
    SQLiteResearchLedger,
    ThreadStateProjector,
    TracingScheduler,
    WorkPurpose,
)


class _FixedScheduler:
    def choose(
        self,
        candidates,
        *,
        integration_backpressure: bool = False,
        max_width: int = 1,
    ) -> SchedulerDecision:
        del integration_backpressure
        state = candidates[0].state
        return SchedulerDecision(
            decision_id="fixed-decision",
            thread_id=state.thread_id,
            action=SchedulerAction.ADD_WIDTH,
            purpose=WorkPurpose.EXPLORE,
            width=min(2, max_width),
            reason_codes=("FIXED",),
            projection_revision=state.revision,
        )


class GenericSchedulerTraceTests(unittest.TestCase):
    def test_wrapper_records_decision_without_changing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                created = ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="thread-1",
                    payload={"objective": "trace", "purpose": "EXPLORE"},
                )
                state = ThreadStateProjector().project_all((created,))[0]
                scheduler = TracingScheduler(ledger, _FixedScheduler())
                decision = scheduler.choose(
                    (
                        SchedulableThread(
                            state=state,
                            signals=SchedulerSignals(importance=1.0),
                        ),
                    ),
                    max_width=4,
                )

                self.assertEqual(decision.decision_id, "fixed-decision")
                traces = [
                    event
                    for event in ledger.read_all_events()
                    if event.event_type == "SCHEDULER_DECISION_RECORDED"
                ]
                self.assertEqual(len(traces), 1)
                trace = traces[0]
                self.assertEqual(trace.payload["decision_id"], decision.decision_id)
                self.assertEqual(trace.payload["action"], decision.action.value)
                self.assertEqual(trace.payload["purpose"], decision.purpose.value)
                self.assertEqual(trace.payload["width"], 2)
                self.assertEqual(trace.payload["max_width"], 4)
                self.assertFalse(trace.payload["integration_backpressure"])


if __name__ == "__main__":
    unittest.main()
