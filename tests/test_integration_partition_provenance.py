from __future__ import annotations

import unittest

from ai_hypothesis.runtime import (
    IntegrationPartitionAllocator,
    IntegrationPartitionConfig,
    IntegrationPartitionProjector,
    PartitionedIntegrationContextRouter,
    ProjectedState,
    SchedulerAction,
    SchedulerDecision,
    SQLiteResearchLedger,
    WorkPreparation,
    WorkPurpose,
)


class IntegrationPartitionProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = SQLiteResearchLedger(":memory:")
        self.allocator = IntegrationPartitionAllocator(
            IntegrationPartitionProjector(
                IntegrationPartitionConfig(shard_count=8, batch_limit=2)
            )
        )
        self.router = PartitionedIntegrationContextRouter(
            ledger=self.ledger,
            allocator=self.allocator,
            fallback=lambda state, decision: WorkPreparation(context={"fallback": True}),
        )
        for index in range(32):
            evidence_id = f"evidence-{index}"
            self.ledger.append_event(
                event_type="EVIDENCE_ADDED",
                thread_id="thread-a",
                reference_ids=(evidence_id,),
                payload={
                    "evidence_id": evidence_id,
                    "kind": "OBSERVATION",
                    "summary": f"observation {index}",
                },
            )

    def tearDown(self) -> None:
        self.ledger.close()

    def _state(self) -> ProjectedState:
        return ProjectedState(
            revision=self.ledger.latest_sequence(),
            thread_id="thread-a",
            objective="integrate evidence",
            status="ACTIVE",
            purpose=WorkPurpose.PROGRESS,
        )

    def _decision(self) -> SchedulerDecision:
        return SchedulerDecision(
            decision_id="decision-a",
            thread_id="thread-a",
            action=SchedulerAction.SYNTHESIZE,
            purpose=WorkPurpose.SYNTHESIZE,
            width=3,
            reason_codes=("BACKPRESSURE", "PARTITIONED_INTEGRATION"),
            projection_revision=self.ledger.latest_sequence(),
        )

    def _provenance_events(self):
        return tuple(
            event
            for event in self.ledger.read_all_events()
            if event.event_type == "INTEGRATION_PARTITION_ALLOCATION_RECORDED"
        )

    def test_context_preparation_records_exact_bounded_partition_assignment(self) -> None:
        decision = self._decision()
        batch = self.router(self._state(), decision)
        (event,) = self._provenance_events()

        self.assertEqual(event.thread_id, "thread-a")
        self.assertEqual(event.payload["schema"], "integration-partition-allocation-v0")
        self.assertEqual(event.payload["decision_id"], decision.decision_id)
        self.assertEqual(event.payload["width"], decision.width)
        self.assertEqual(event.payload["shard_count"], 8)
        self.assertEqual(event.payload["batch_limit"], 2)
        self.assertEqual(len(event.reference_ids), decision.width)
        self.assertEqual(len(event.payload["partitions"]), decision.width)

        recorded_evidence = []
        for preparation, recorded in zip(
            batch.items,
            event.payload["partitions"],
            strict=True,
        ):
            self.assertEqual(
                recorded["partition_id"],
                preparation.context["integration_partition"]["partition_id"],
            )
            self.assertEqual(recorded["evidence_ids"], list(preparation.reference_ids))
            recorded_evidence.extend(recorded["evidence_ids"])
        self.assertEqual(len(recorded_evidence), len(set(recorded_evidence)))

    def test_repeated_preparation_is_idempotent_when_assignment_is_identical(self) -> None:
        state = self._state()
        decision = self._decision()

        first = self.router(state, decision)
        # The provenance event itself advances the global ledger revision, but does not
        # change pending evidence or the actual partition assignment.
        second = self.router(self._state(), decision)

        self.assertEqual(
            tuple(item.reference_ids for item in first.items),
            tuple(item.reference_ids for item in second.items),
        )
        self.assertEqual(len(self._provenance_events()), 1)

    def test_same_decision_cannot_be_reused_for_different_evidence_assignment(self) -> None:
        state = self._state()
        decision = self._decision()
        first = self.router(state, decision)
        consumed_id = first.items[0].reference_ids[0]
        self.ledger.append_event(
            event_type="INTEGRATION_DISPOSITION_RECORDED",
            thread_id="thread-a",
            reference_ids=(consumed_id,),
            payload={"disposition": "INTEGRATED"},
        )

        with self.assertRaisesRegex(
            ValueError,
            "conflicting durable partition allocation provenance",
        ):
            self.router(self._state(), decision)

    def test_non_partitioned_context_does_not_record_partition_provenance(self) -> None:
        decision = SchedulerDecision(
            decision_id="decision-final",
            thread_id="thread-a",
            action=SchedulerAction.SYNTHESIZE,
            purpose=WorkPurpose.SYNTHESIZE,
            width=1,
            reason_codes=("FINAL_SYNTHESIS",),
            projection_revision=self.ledger.latest_sequence(),
        )
        preparation = self.router(self._state(), decision)

        self.assertEqual(preparation.context, {"fallback": True})
        self.assertEqual(self._provenance_events(), ())


if __name__ == "__main__":
    unittest.main()
