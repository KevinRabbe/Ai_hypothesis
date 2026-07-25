"""Tests durable worker follow-up requests becoming persistent child Work Threads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime.contracts import WorkPurpose
from ai_hypothesis.runtime.followups import FollowupMaterializer
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger
from ai_hypothesis.runtime.projector import ThreadStateProjector


class FollowupMaterializerTests(unittest.TestCase):
    def test_oldest_pending_followups_materialize_bounded_child_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="parent",
                    payload={"objective": "root", "purpose": "EXPLORE"},
                )
                requests = [
                    ledger.append_event(
                        event_type="FOLLOWUP_REQUESTED",
                        thread_id="parent",
                        payload={"request": f"follow-up {index}"},
                    )
                    for index in range(3)
                ]
                materializer = FollowupMaterializer(ledger)

                children = materializer.materialize(limit=2)

                self.assertEqual(len(children), 2)
                self.assertEqual(
                    children,
                    tuple(
                        materializer.child_thread_id(event.event_id)
                        for event in requests[:2]
                    ),
                )
                snapshot = materializer.snapshot()
                self.assertEqual(
                    tuple(request.request_event_id for request in snapshot.pending),
                    (requests[2].event_id,),
                )
                states = {
                    state.thread_id: state
                    for state in ThreadStateProjector().project_all(
                        ledger.read_all_events()
                    )
                }
                self.assertEqual(
                    states["parent"].child_thread_ids,
                    children,
                )
                self.assertEqual(states[children[0]].parent_thread_ids, ("parent",))
                self.assertEqual(states[children[0]].objective, "follow-up 0")

    def test_materialization_is_idempotent_after_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="parent",
                    payload={"objective": "root", "purpose": "EXPLORE"},
                )
                request = ledger.append_event(
                    event_type="FOLLOWUP_REQUESTED",
                    thread_id="parent",
                    payload={"request": "inspect anomaly"},
                )
                materializer = FollowupMaterializer(ledger)

                first = materializer.materialize(limit=1)
                sequence_after_first = ledger.latest_sequence()
                second = materializer.materialize(limit=1)

                self.assertEqual(
                    first,
                    (materializer.child_thread_id(request.event_id),),
                )
                self.assertEqual(second, ())
                self.assertEqual(ledger.latest_sequence(), sequence_after_first)

    def test_partial_retry_reuses_deterministic_child_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="parent",
                    payload={"objective": "root", "purpose": "EXPLORE"},
                )
                request = ledger.append_event(
                    event_type="FOLLOWUP_REQUESTED",
                    thread_id="parent",
                    payload={"request": "resume materialization"},
                )
                materializer = FollowupMaterializer(ledger)
                child_id = materializer.child_thread_id(request.event_id)

                # Simulate a crash after child creation but before graph edge/receipt.
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id=child_id,
                    payload={
                        "objective": "resume materialization",
                        "purpose": "EXPLORE",
                    },
                )

                children = materializer.materialize(limit=1)

                self.assertEqual(children, (child_id,))
                events = ledger.read_all_events()
                created = [
                    event
                    for event in events
                    if event.event_type == "THREAD_CREATED"
                    and event.thread_id == child_id
                ]
                self.assertEqual(len(created), 1)
                forked = [
                    event
                    for event in events
                    if event.event_type == "THREAD_FORKED"
                    and child_id in event.reference_ids
                ]
                self.assertEqual(len(forked), 1)

    def test_missing_origin_thread_is_rejected_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="FOLLOWUP_REQUESTED",
                    thread_id="missing-parent",
                    payload={"request": "bad request"},
                )
                materializer = FollowupMaterializer(ledger)
                before = ledger.latest_sequence()

                with self.assertRaisesRegex(ValueError, "missing Work Thread"):
                    materializer.materialize(limit=1)

                self.assertEqual(ledger.latest_sequence(), before)

    def test_materialization_purpose_is_explicit_and_replaceable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                ledger.append_event(
                    event_type="THREAD_CREATED",
                    thread_id="parent",
                    payload={"objective": "root", "purpose": "EXPLORE"},
                )
                request = ledger.append_event(
                    event_type="FOLLOWUP_REQUESTED",
                    thread_id="parent",
                    payload={"request": "verify surprising result"},
                )
                materializer = FollowupMaterializer(ledger)

                child = materializer.materialize(
                    limit=1,
                    purpose=WorkPurpose.VERIFY,
                )[0]
                state = ThreadStateProjector().project(
                    ledger.read_all_events(),
                    thread_id=child,
                )

                self.assertEqual(
                    child,
                    materializer.child_thread_id(request.event_id),
                )
                self.assertEqual(state.purpose, WorkPurpose.VERIFY)

    def test_limit_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                materializer = FollowupMaterializer(ledger)
                with self.assertRaises(ValueError):
                    materializer.materialize(limit=0)


if __name__ == "__main__":
    unittest.main()
