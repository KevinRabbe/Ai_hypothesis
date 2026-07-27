from __future__ import annotations

import unittest

from ai_hypothesis.large_scope.evaluate import (
    ScopeEvaluation,
    ScopeWorkerMode,
    WindowEvidence,
)
from ai_hypothesis.large_scope.paired_metrics import ScopePairedMetricsAccumulator


def evaluation(
    seed: int,
    *,
    mode: ScopeWorkerMode,
    target_present: bool,
    target_inspected: bool,
    candidate_is_target: bool,
    candidate_evidence: float,
    target_evidence: float | None = None,
    target_rank: int | None = None,
    distractor_evidence: float | None = None,
    inspected: tuple[int, ...] = (0, 1, 2, 3),
) -> ScopeEvaluation:
    width = len(inspected)
    target_index = 0 if target_present else None
    candidate_index = 0 if candidate_is_target else inspected[-1]
    return ScopeEvaluation(
        split="development",
        seed=seed,
        width=width,
        mode=mode,
        inspected_window_indices=inspected,
        worker_indices=(0,) * width if mode is ScopeWorkerMode.SAME_WORKER else tuple(range(width)),
        target_present=target_present,
        target_index=target_index,
        target_inspected=target_inspected,
        candidate_window_index=candidate_index,
        candidate_is_target=candidate_is_target,
        candidate_relevant_evidence=candidate_evidence,
        target_relevant_evidence=target_evidence,
        target_rank=target_rank,
        strongest_distractor_relevant_evidence=distractor_evidence,
        window_evidence=tuple(
            WindowEvidence(
                window_index=index,
                worker_index=0 if mode is ScopeWorkerMode.SAME_WORKER else offset,
                local_label="NOT_RELEVANT",
                relevant_evidence=0.0,
                not_relevant_evidence=0.0,
                uncertainty_probability=0.1,
                invalid_label_mass=0.0,
                top_margin=0.5,
            )
            for offset, index in enumerate(inspected)
        ),
    )


class ScopePairedMetricsTests(unittest.TestCase):
    def test_reports_diverse_minus_same_retrieval_and_evidence_deltas(self) -> None:
        accumulator = ScopePairedMetricsAccumulator()
        accumulator.add(
            evaluation(
                2,
                mode=ScopeWorkerMode.SAME_WORKER,
                target_present=True,
                target_inspected=True,
                candidate_is_target=False,
                candidate_evidence=0.7,
                target_evidence=0.6,
                target_rank=2,
                distractor_evidence=0.7,
            )
        )
        accumulator.add(
            evaluation(
                2,
                mode=ScopeWorkerMode.DIVERSE_WORKERS,
                target_present=True,
                target_inspected=True,
                candidate_is_target=True,
                candidate_evidence=0.9,
                target_evidence=0.9,
                target_rank=1,
                distractor_evidence=0.5,
            )
        )
        summary = accumulator.summaries()[0]
        self.assertEqual(summary.pair_count, 1)
        self.assertEqual(summary.diverse_only_retrieved_count, 1)
        self.assertEqual(summary.same_only_retrieved_count, 0)
        self.assertEqual(summary.retrieval_given_inspected_delta, 1.0)
        self.assertAlmostEqual(summary.mean_target_rank_delta_when_inspected, -1.0)
        self.assertAlmostEqual(
            summary.mean_target_relevant_evidence_delta_when_inspected,
            0.3,
        )
        self.assertAlmostEqual(
            summary.mean_strongest_distractor_relevant_evidence_delta,
            -0.2,
        )
        self.assertAlmostEqual(
            summary.mean_target_minus_distractor_gap_delta_when_inspected,
            0.5,
        )
        self.assertIsNone(summary.se_target_rank_delta_when_inspected)

    def test_negative_world_candidate_delta_is_paired(self) -> None:
        accumulator = ScopePairedMetricsAccumulator()
        accumulator.add(
            evaluation(
                1,
                mode=ScopeWorkerMode.SAME_WORKER,
                target_present=False,
                target_inspected=False,
                candidate_is_target=False,
                candidate_evidence=0.8,
                distractor_evidence=0.8,
            )
        )
        accumulator.add(
            evaluation(
                1,
                mode=ScopeWorkerMode.DIVERSE_WORKERS,
                target_present=False,
                target_inspected=False,
                candidate_is_target=False,
                candidate_evidence=0.6,
                distractor_evidence=0.6,
            )
        )
        summary = accumulator.summaries()[0]
        self.assertAlmostEqual(
            summary.mean_candidate_relevant_evidence_negative_delta,
            -0.2,
        )

    def test_exact_retrieval_discordance_detects_one_sided_six_world_difference(self) -> None:
        accumulator = ScopePairedMetricsAccumulator()
        for seed in range(0, 12, 2):
            accumulator.add(
                evaluation(
                    seed,
                    mode=ScopeWorkerMode.SAME_WORKER,
                    target_present=True,
                    target_inspected=True,
                    candidate_is_target=False,
                    candidate_evidence=0.7,
                    target_evidence=0.6,
                    target_rank=2,
                    distractor_evidence=0.7,
                )
            )
            accumulator.add(
                evaluation(
                    seed,
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    target_present=True,
                    target_inspected=True,
                    candidate_is_target=True,
                    candidate_evidence=0.9,
                    target_evidence=0.9,
                    target_rank=1,
                    distractor_evidence=0.5,
                )
            )
        summary = accumulator.summaries()[0]
        self.assertEqual(summary.retrieval_discordant_count, 6)
        self.assertEqual(summary.diverse_only_retrieved_count, 6)
        self.assertEqual(summary.same_only_retrieved_count, 0)
        self.assertAlmostEqual(summary.exact_retrieval_discordance_p_value, 0.03125)

    def test_pairing_rejects_different_inspected_scope(self) -> None:
        accumulator = ScopePairedMetricsAccumulator()
        accumulator.add(
            evaluation(
                2,
                mode=ScopeWorkerMode.SAME_WORKER,
                target_present=True,
                target_inspected=True,
                candidate_is_target=True,
                candidate_evidence=1.0,
                target_evidence=1.0,
                target_rank=1,
                distractor_evidence=0.2,
            )
        )
        with self.assertRaisesRegex(ValueError, "identical windows"):
            accumulator.add(
                evaluation(
                    2,
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    target_present=True,
                    target_inspected=True,
                    candidate_is_target=True,
                    candidate_evidence=1.0,
                    target_evidence=1.0,
                    target_rank=1,
                    distractor_evidence=0.2,
                    inspected=(0, 1, 2, 4),
                )
            )

    def test_finalize_rejects_unmatched_condition(self) -> None:
        accumulator = ScopePairedMetricsAccumulator()
        accumulator.add(
            evaluation(
                1,
                mode=ScopeWorkerMode.SAME_WORKER,
                target_present=False,
                target_inspected=False,
                candidate_is_target=False,
                candidate_evidence=0.5,
            )
        )
        with self.assertRaisesRegex(ValueError, "unmatched"):
            accumulator.summaries()


if __name__ == "__main__":
    unittest.main()
