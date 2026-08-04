from __future__ import annotations

import dataclasses
import unittest

from ai_hypothesis.population_language import post_training_learning_l0_stack_manifest as stack


class PostTrainingLearningL0StackAuditTests(unittest.TestCase):
    def test_exact_qualified_stack_is_contiguous(self) -> None:
        report = stack.validate_stack_audit()
        self.assertTrue(report["valid"], report["checks"])
        self.assertEqual(
            report["merge_order"],
            [203, 204, 207, 208, 209, 210, 211, 212, 215],
        )
        self.assertEqual(stack.STACK[0].base_sha, stack.FIRST_BASE_SHA)
        self.assertEqual(stack.STACK[-1].head_sha, stack.SOURCE_ORCHESTRATOR_HEAD)
        self.assertEqual(len(stack.STACK), 9)

    def test_review_before_merge_and_protected_execution_defaults_are_locked(self) -> None:
        self.assertEqual(
            stack.MERGE_POLICY,
            "REVIEW_BEFORE_MERGE_NO_AUTOMATIC_MERGE",
        )
        self.assertFalse(stack.DEFAULT_PROTECTED_STATE["automatic_merge_allowed"])
        self.assertFalse(
            stack.DEFAULT_PROTECTED_STATE["active_reference_output_access_allowed"]
        )
        self.assertFalse(
            stack.DEFAULT_PROTECTED_STATE["calibration_execution_authorized"]
        )
        self.assertFalse(
            stack.DEFAULT_PROTECTED_STATE["final_world_access_authorized"]
        )
        self.assertFalse(
            stack.DEFAULT_PROTECTED_STATE["final_execution_authorized"]
        )
        self.assertFalse(
            stack.DEFAULT_PROTECTED_STATE["fifty_m_architecture_frozen"]
        )
        self.assertFalse(
            stack.DEFAULT_PROTECTED_STATE["hundred_m_architecture_frozen"]
        )
        self.assertFalse(
            stack.DEFAULT_PROTECTED_STATE["three_hundred_m_architecture_frozen"]
        )

    def test_missing_operational_components_are_explicit(self) -> None:
        self.assertEqual(
            stack.MISSING_OPERATIONAL_COMPONENTS,
            (
                "REAL_AUTHORIZATION_GATED_CALIBRATION_ROW_EXECUTOR",
                "FINAL_EVALUATION_CANDIDATE_LOCK_AND_PLAN",
                "FINAL_RESULT_VERIFIER",
                "POWERSHELL_OPERATOR_RUNBOOK",
            ),
        )
        self.assertEqual(
            stack.NEXT_HANDOFF_ORDER[0],
            "WAIT_FOR_REFERENCE_TRAINING_TO_COMPLETE",
        )
        self.assertEqual(
            stack.NEXT_HANDOFF_ORDER[-1],
            "REQUEST_SEPARATE_EXPLICIT_FINAL_EXECUTION_AUTHORIZATION",
        )

    def test_reordering_or_breaking_sha_chain_fails_closed(self) -> None:
        reordered = list(stack.STACK)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        checks = stack.validate_entries(reordered)
        self.assertFalse(checks["orders_are_contiguous"])
        self.assertFalse(checks["pull_requests_are_exact"])
        self.assertFalse(checks["first_base_is_exact"])
        self.assertFalse(checks["stack_is_sha_contiguous"])

        broken = list(stack.STACK)
        broken[4] = dataclasses.replace(broken[4], base_sha="0" * 40)
        checks = stack.validate_entries(broken)
        self.assertFalse(checks["stack_is_sha_contiguous"])

    def test_no_entry_can_claim_merge_or_architecture_freeze(self) -> None:
        for entry in stack.STACK:
            self.assertEqual(entry.state, stack.STACK_STATE)
            self.assertFalse(entry.merge_allowed)
        manifest = stack.stack_manifest()
        self.assertEqual(
            manifest["merge_policy"],
            "REVIEW_BEFORE_MERGE_NO_AUTOMATIC_MERGE",
        )
        self.assertEqual(
            manifest["missing_operational_components"],
            list(stack.MISSING_OPERATIONAL_COMPONENTS),
        )
        self.assertEqual(len(stack.stack_manifest_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
