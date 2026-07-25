"""Tests canonical bounded WorkPreparation for synthesis attempts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.runtime import (
    IntegrationTracker,
    ProjectedState,
    SQLiteResearchLedger,
    WorkPurpose,
    prepare_bounded_integration_work,
)


class BoundedIntegrationWorkTests(unittest.TestCase):
    def test_preparation_never_exceeds_requested_evidence_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                for index in range(20):
                    evidence_id = f"evidence-{index}"
                    ledger.append_event(
                        event_type="EVIDENCE_ADDED",
                        thread_id="thread-1",
                        reference_ids=(evidence_id, f"source-{index}"),
                        payload={
                            "evidence_id": evidence_id,
                            "kind": "LOCAL_FINDING",
                            "summary": f"finding {index}",
                            "strength": 0.5,
                            "uncertainty": 0.2,
                            "data": {"index": index},
                        },
                    )
                tracker = IntegrationTracker(ledger)
                state = ProjectedState(
                    revision=ledger.latest_sequence(),
                    thread_id="thread-1",
                    objective="Integrate pending findings",
                    status="ACTIVE",
                    purpose=WorkPurpose.SYNTHESIZE,
                )

                preparation = prepare_bounded_integration_work(
                    tracker,
                    state,
                    limit=3,
                )

                self.assertEqual(
                    preparation.reference_ids,
                    ("evidence-0", "evidence-1", "evidence-2"),
                )
                self.assertEqual(len(preparation.context["pending_evidence"]), 3)
                self.assertEqual(len(preparation.context["causal_event_ids"]), 3)
                self.assertEqual(preparation.constraints["max_pending_evidence"], 3)

    def test_preparation_contains_compact_recoverable_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                event = ledger.append_event(
                    event_type="EVIDENCE_ADDED",
                    thread_id="thread-1",
                    reference_ids=("evidence-1", "source-region-9"),
                    payload={
                        "evidence_id": "evidence-1",
                        "kind": "COUNTEREXAMPLE",
                        "summary": "H2 fails on region 9",
                        "strength": 0.9,
                        "uncertainty": 0.05,
                        "data": {"hypothesis": "H2"},
                    },
                )
                tracker = IntegrationTracker(ledger)
                state = ProjectedState(
                    revision=event.sequence,
                    thread_id="thread-1",
                    objective="Integrate counterexample",
                    status="ACTIVE",
                    purpose=WorkPurpose.SYNTHESIZE,
                )

                preparation = prepare_bounded_integration_work(tracker, state, limit=1)
                record = preparation.context["pending_evidence"][0]

                self.assertEqual(record["evidence_id"], "evidence-1")
                self.assertEqual(record["event_id"], event.event_id)
                self.assertEqual(record["source_reference_ids"], ["source-region-9"])
                self.assertEqual(record["data"], {"hypothesis": "H2"})


if __name__ == "__main__":
    unittest.main()
