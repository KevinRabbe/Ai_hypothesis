"""Tests durable Scheduler v0 allocation traces."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.contracts import ProjectedState, SchedulerAction, WorkPurpose
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.scheduler import SchedulerConfig, SchedulerSignals, SchedulableThread
from ai_hypothesis.runtime.scheduler_trace import TracingSchedulerV0


class SchedulerTraceTests(unittest.TestCase):
    def test_normal_decision_is_recorded_without_changing_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                scheduler = TracingSchedulerV0(
                    ledger,
                    SchedulerConfig(exploration_probability=0.0),
                    rng=random.Random(1),
                )
                candidate = SchedulableThread(
                    state=ProjectedState(
                        revision=7,
                        thread_id="thread-1",
                        objective="Investigate H2",
                        status="ACTIVE",
                        purpose=WorkPurpose.PROGRESS,
                    ),
                    signals=SchedulerSignals(
                        importance=1.0,
                        recent_progress=1.0,
                    ),
                )

                decision = scheduler.choose((candidate,), max_width=4)

                self.assertEqual(decision.action, SchedulerAction.CONTINUE)
                self.assertEqual(decision.width, 1)
                events = ledger.read_all_events()
                self.assertEqual(len(events), 1)
                event = events[0]
                self.assertEqual(event.event_type, "SCHEDULER_DECISION_RECORDED")
                self.assertEqual(event.thread_id, "thread-1")
                self.assertEqual(event.payload["decision_id"], decision.decision_id)
                self.assertEqual(event.payload["action"], "CONTINUE")
                self.assertEqual(event.payload["width"], 1)
                self.assertEqual(event.payload["projection_revision"], 7)
                self.assertFalse(event.payload["integration_backpressure"])

    def test_exploration_trace_records_actual_capped_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                scheduler = TracingSchedulerV0(
                    ledger,
                    SchedulerConfig(
                        exploration_probability=1.0,
                        exploration_width=8,
                    ),
                    rng=random.Random(1),
                )
                candidate = SchedulableThread(
                    state=ProjectedState(
                        revision=3,
                        thread_id="thread-1",
                        objective="Explore alternatives",
                        status="ACTIVE",
                        purpose=WorkPurpose.EXPLORE,
                    ),
                    signals=SchedulerSignals(
                        missing_coverage=1.0,
                        novelty=1.0,
                    ),
                )

                decision = scheduler.choose((candidate,), max_width=3)

                self.assertEqual(decision.action, SchedulerAction.ADD_WIDTH)
                self.assertEqual(decision.width, 3)
                event = ledger.read_all_events()[0]
                self.assertEqual(event.payload["action"], "ADD_WIDTH")
                self.assertEqual(event.payload["width"], 3)
                self.assertEqual(event.payload["max_width"], 3)

    def test_backpressure_state_is_part_of_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                scheduler = TracingSchedulerV0(
                    ledger,
                    SchedulerConfig(exploration_probability=1.0),
                    rng=random.Random(1),
                )
                candidate = SchedulableThread(
                    state=ProjectedState(
                        revision=2,
                        thread_id="thread-1",
                        objective="Integrate backlog",
                        status="ACTIVE",
                        purpose=WorkPurpose.EXPLORE,
                    ),
                    signals=SchedulerSignals(
                        integration_backlog=1.0,
                        recent_progress=1.0,
                    ),
                )

                decision = scheduler.choose(
                    (candidate,),
                    integration_backpressure=True,
                    max_width=4,
                )

                self.assertEqual(decision.action, SchedulerAction.SYNTHESIZE)
                event = ledger.read_all_events()[0]
                self.assertTrue(event.payload["integration_backpressure"])
                self.assertIn("BACKPRESSURE", event.payload["reason_codes"])


if __name__ == "__main__":
    unittest.main()
