"""Tests threshold-free streaming summaries for large-scope evaluations."""

from __future__ import annotations

import unittest

from ai_hypothesis.large_scope import (
    ScopeEvaluation,
    ScopeMetricsAccumulator,
    ScopeWorkerMode,
    WindowEvidence,
    summarize_scope_evaluations,
)


def result(
    seed: int,
    *,
    target_present: bool,
    target_inspected: bool,
    candidate_is_target: bool,
    candidate_evidence: float,
    target_evidence: float | None = None,
    target_rank: int | None = None,
) -> ScopeEvaluation:
    target_index = 0 if target_present else None
    return ScopeEvaluation(
        split="development",
        seed=seed,
        width=1,
        mode=ScopeWorkerMode.SAME_WORKER,
        inspected_window_indices=(0 if target_inspected else 1,),
        worker_indices=(0,),
        target_present=target_present,
        target_index=target_index,
        target_inspected=target_inspected,
        candidate_window_index=0 if target_inspected else 1,
        candidate_is_target=candidate_is_target,
        candidate_relevant_evidence=candidate_evidence,
        target_relevant_evidence=target_evidence,
        target_rank=target_rank,
        strongest_distractor_relevant_evidence=None,
        window_evidence=(
            WindowEvidence(
                window_index=0 if target_inspected else 1,
                worker_index=0,
                local_label="RELEVANT" if candidate_evidence > 1.0 else "NOT_RELEVANT",
                relevant_evidence=candidate_evidence,
                not_relevant_evidence=-candidate_evidence,
                uncertainty_probability=0.1,
                invalid_label_mass=0.0,
                top_margin=0.8,
            ),
        ),
    )


class LargeScopeMetricsTests(unittest.TestCase):
    def test_streaming_summary_matches_batch_helper(self) -> None:
        evaluations = (
            result(
                0,
                target_present=True,
                target_inspected=True,
                candidate_is_target=True,
                candidate_evidence=2.0,
                target_evidence=2.0,
                target_rank=1,
            ),
            result(
                2,
                target_present=True,
                target_inspected=False,
                candidate_is_target=False,
                candidate_evidence=0.8,
            ),
            result(
                1,
                target_present=False,
                target_inspected=False,
                candidate_is_target=False,
                candidate_evidence=0.5,
            ),
            result(
                3,
                target_present=False,
                target_inspected=False,
                candidate_is_target=False,
                candidate_evidence=1.5,
            ),
        )
        accumulator = ScopeMetricsAccumulator()
        for evaluation in evaluations:
            accumulator.add(evaluation)

        streaming = accumulator.summaries()
        batch = summarize_scope_evaluations(evaluations)
        self.assertEqual(streaming, batch)
        summary = streaming[0]
        self.assertEqual(summary.world_count, 4)
        self.assertEqual(summary.positive_world_count, 2)
        self.assertEqual(summary.negative_world_count, 2)
        self.assertEqual(summary.target_coverage_rate, 0.5)
        self.assertEqual(summary.target_retrieval_rate, 0.5)
        self.assertEqual(summary.retrieval_given_inspected, 1.0)
        self.assertEqual(summary.mean_target_rank_when_inspected, 1.0)
        self.assertEqual(summary.mean_candidate_relevant_evidence_negative, 1.0)
        self.assertEqual(summary.max_candidate_relevant_evidence_negative, 1.5)


if __name__ == "__main__":
    unittest.main()
