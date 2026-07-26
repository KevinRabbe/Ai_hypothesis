from __future__ import annotations

import unittest

from ai_hypothesis.runtime.context_views import PurposeContextRouter
from ai_hypothesis.runtime.contracts import (
    ProjectedState,
    SchedulerAction,
    SchedulerDecision,
    WorkPurpose,
)
from ai_hypothesis.runtime.control import WorkPreparation
from ai_hypothesis.runtime.integration import IntegrationTracker
from ai_hypothesis.runtime.knowledge_verification import KnowledgeVerificationTracker
from ai_hypothesis.runtime.ledger import SQLiteResearchLedger


class PurposeContextRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = SQLiteResearchLedger(":memory:")
        self.fallback_calls: list[tuple[str, str]] = []

        def fallback(state: ProjectedState, decision: SchedulerDecision) -> WorkPreparation:
            self.fallback_calls.append((state.thread_id, decision.action.value))
            return WorkPreparation(context={"fallback": True})

        self.fallback = fallback

    def tearDown(self) -> None:
        self.ledger.close()

    def _state(self, purpose: WorkPurpose = WorkPurpose.EXPLORE) -> ProjectedState:
        return ProjectedState(
            revision=self.ledger.latest_sequence(),
            thread_id="thread-a",
            objective="test objective",
            status="ACTIVE",
            purpose=purpose,
        )

    def _decision(
        self,
        action: SchedulerAction,
        purpose: WorkPurpose,
        *,
        reasons: tuple[str, ...] = (),
    ) -> SchedulerDecision:
        return SchedulerDecision(
            decision_id=f"decision-{action.value}-{len(self.fallback_calls)}",
            thread_id="thread-a",
            action=action,
            purpose=purpose,
            reason_codes=reasons,
            projection_revision=self.ledger.latest_sequence(),
        )

    def test_backpressure_synthesis_receives_bounded_pending_evidence(self) -> None:
        for index in range(3):
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

        router = PurposeContextRouter(
            ledger=self.ledger,
            fallback=self.fallback,
            integration_tracker=IntegrationTracker(self.ledger),
            integration_limit=2,
        )
        preparation = router(
            self._state(),
            self._decision(
                SchedulerAction.SYNTHESIZE,
                WorkPurpose.SYNTHESIZE,
                reasons=("BACKPRESSURE",),
            ),
        )

        self.assertEqual(preparation.reference_ids, ("evidence-0", "evidence-1"))
        self.assertEqual(preparation.context["context_view"], "SYNTHESIZE")
        self.assertEqual(
            preparation.context["synthesis_mode"],
            "INTEGRATION_BACKPRESSURE",
        )
        self.assertEqual(len(preparation.context["pending_evidence"]), 2)
        self.assertEqual(preparation.constraints["max_pending_evidence"], 2)
        self.assertTrue(preparation.constraints["emit_structured_knowledge_deltas"])
        self.assertTrue(preparation.constraints["disposition_consumed_evidence"])
        self.assertEqual(self.fallback_calls, [])

    def test_non_backpressure_synthesis_remains_domain_owned(self) -> None:
        router = PurposeContextRouter(
            ledger=self.ledger,
            fallback=self.fallback,
            integration_tracker=IntegrationTracker(self.ledger),
        )
        preparation = router(
            self._state(WorkPurpose.SYNTHESIZE),
            self._decision(
                SchedulerAction.SYNTHESIZE,
                WorkPurpose.SYNTHESIZE,
                reasons=("FINAL_SYNTHESIS",),
            ),
        )

        self.assertEqual(preparation.context, {"fallback": True})
        self.assertEqual(self.fallback_calls, [("thread-a", "SYNTHESIZE")])

    def test_verify_receives_only_unresolved_knowledge_with_identity_hidden(self) -> None:
        for delta_id in ("delta-verified", "delta-disputed"):
            self.ledger.append_event(
                event_type="KNOWLEDGE_DELTA_RECORDED",
                thread_id="thread-a",
                reference_ids=(delta_id, f"source-{delta_id}"),
                payload={
                    "delta_id": delta_id,
                    "kind": "CLAIM",
                    "summary": f"claim {delta_id}",
                    "source_reference_ids": [f"source-{delta_id}"],
                    "causal_event_ids": [],
                },
            )
        self.ledger.append_event(
            event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
            thread_id="thread-a",
            reference_ids=("delta-verified",),
            payload={"assessment": "VERIFIED"},
        )
        self.ledger.append_event(
            event_type="KNOWLEDGE_ASSESSMENT_RECORDED",
            thread_id="thread-a",
            reference_ids=("delta-disputed",),
            payload={"assessment": "DISPUTED", "reason": "counterexample pending"},
        )

        router = PurposeContextRouter(
            ledger=self.ledger,
            fallback=self.fallback,
            verification_tracker=KnowledgeVerificationTracker(self.ledger),
            verification_limit=1,
        )
        preparation = router(
            self._state(),
            self._decision(SchedulerAction.VERIFY, WorkPurpose.VERIFY),
        )

        self.assertEqual(preparation.reference_ids, ("delta-disputed",))
        self.assertEqual(preparation.context["context_view"], "VERIFY")
        self.assertEqual(
            preparation.context["verification_target_delta_ids"],
            ["delta-disputed"],
        )
        self.assertEqual(len(preparation.context["knowledge_records"]), 1)
        record = preparation.context["knowledge_records"][0]
        self.assertEqual(record["delta_id"], "delta-disputed")
        self.assertEqual(record["status"], "DISPUTED")
        self.assertNotIn("worker_id", record)
        self.assertNotIn("vote_count", record)
        self.assertTrue(preparation.constraints["independent_verification"])
        self.assertTrue(preparation.constraints["hide_worker_identity"])
        self.assertTrue(preparation.constraints["hide_vote_counts"])
        self.assertEqual(self.fallback_calls, [])

    def test_empty_backpressure_synthesis_is_rejected(self) -> None:
        router = PurposeContextRouter(
            ledger=self.ledger,
            fallback=self.fallback,
            integration_tracker=IntegrationTracker(self.ledger),
        )
        with self.assertRaisesRegex(ValueError, "no pending evidence"):
            router(
                self._state(),
                self._decision(
                    SchedulerAction.SYNTHESIZE,
                    WorkPurpose.SYNTHESIZE,
                    reasons=("BACKPRESSURE",),
                ),
            )

    def test_verify_without_unresolved_knowledge_is_rejected(self) -> None:
        router = PurposeContextRouter(
            ledger=self.ledger,
            fallback=self.fallback,
            verification_tracker=KnowledgeVerificationTracker(self.ledger),
        )
        with self.assertRaisesRegex(ValueError, "no unresolved knowledge"):
            router(
                self._state(),
                self._decision(SchedulerAction.VERIFY, WorkPurpose.VERIFY),
            )

    def test_trackers_must_share_exact_ledger_instance(self) -> None:
        other = SQLiteResearchLedger(":memory:")
        self.addCleanup(other.close)
        with self.assertRaisesRegex(ValueError, "same Research Ledger"):
            PurposeContextRouter(
                ledger=self.ledger,
                fallback=self.fallback,
                integration_tracker=IntegrationTracker(other),
            )


if __name__ == "__main__":
    unittest.main()
