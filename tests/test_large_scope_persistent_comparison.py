"""Tests normalized direct/persistent comparison without real checkpoints."""

from __future__ import annotations

import unittest

from ai_hypothesis.large_scope import ScopeWorkerMode, WindowEvidence
from ai_hypothesis.large_scope.evaluate import ScopeEvaluation
from ai_hypothesis.large_scope.persistent_comparison import (
    ScopeEquivalenceAccumulator,
    compare_scope_evaluations,
    scope_evaluation_from_persistent,
)
from ai_hypothesis.large_scope.persistent_experiment import PersistentScopeEvaluation
from ai_hypothesis.large_scope.run_persistent_comparison import main


def _rows(*, delta: float = 0.0, label: str = "NOT_RELEVANT"):
    return (
        WindowEvidence(
            window_index=2,
            worker_index=5,
            local_label=label,
            relevant_evidence=0.25 + delta,
            not_relevant_evidence=1.2 - delta,
            uncertainty_probability=0.1 + delta,
            invalid_label_mass=0.02,
            top_margin=0.7,
        ),
        WindowEvidence(
            window_index=7,
            worker_index=6,
            local_label="RELEVANT",
            relevant_evidence=2.0 + delta,
            not_relevant_evidence=-1.1 - delta,
            uncertainty_probability=0.05,
            invalid_label_mass=0.01,
            top_margin=0.9,
        ),
    )


def _persistent(*, rows=None) -> PersistentScopeEvaluation:
    rows = rows or _rows()
    return PersistentScopeEvaluation(
        thread_id="thread-1",
        split="development",
        seed=42,
        mode=ScopeWorkerMode.DIVERSE_WORKERS,
        worker_bank_id="worker-bank-sha256-test",
        step_width=1,
        step_count=2,
        attempt_count=2,
        evidence_count=2,
        scheduler_decision_count=2,
        distinct_worker_count=2,
        resolved_region_count=2,
        expected_region_count=16,
        coverage_fraction=0.125,
        window_evidence=rows,
        target_present=True,
        target_index=7,
        target_resolved=True,
        candidate_window_index=7,
        candidate_is_target=True,
        candidate_relevant_evidence=rows[1].relevant_evidence,
        target_relevant_evidence=rows[1].relevant_evidence,
        target_rank=1,
        strongest_distractor_relevant_evidence=rows[0].relevant_evidence,
        ledger_event_count=20,
    )


class PersistentComparisonTests(unittest.TestCase):
    def test_persistent_projection_reuses_direct_scope_metric_contract(self) -> None:
        projected = scope_evaluation_from_persistent(_persistent())
        self.assertEqual(projected.width, 2)
        self.assertEqual(projected.inspected_window_indices, (2, 7))
        self.assertEqual(projected.worker_indices, (5, 6))
        self.assertTrue(projected.target_inspected)
        self.assertEqual(projected.candidate_window_index, 7)
        projected.validate()

    def test_equivalence_tracks_numeric_drift_without_structural_failure(self) -> None:
        direct = scope_evaluation_from_persistent(_persistent())
        persistent = scope_evaluation_from_persistent(
            _persistent(rows=_rows(delta=2e-6))
        )
        observation = compare_scope_evaluations(direct, persistent)
        self.assertGreater(observation.max_abs_evidence_delta, 0.0)
        self.assertEqual(observation.local_label_mismatch_count, 0)
        self.assertTrue(observation.within_tolerance(1e-5))
        self.assertFalse(observation.within_tolerance(1e-7))

    def test_label_flip_is_reported_as_equivalence_failure_not_structure_corruption(self) -> None:
        direct = scope_evaluation_from_persistent(_persistent())
        rows = list(_rows())
        rows[0] = WindowEvidence(
            window_index=rows[0].window_index,
            worker_index=rows[0].worker_index,
            local_label="UNCERTAIN",
            relevant_evidence=rows[0].relevant_evidence,
            not_relevant_evidence=rows[0].not_relevant_evidence,
            uncertainty_probability=rows[0].uncertainty_probability,
            invalid_label_mass=rows[0].invalid_label_mass,
            top_margin=rows[0].top_margin,
        )
        persistent = scope_evaluation_from_persistent(
            _persistent(rows=tuple(rows))
        )
        observation = compare_scope_evaluations(direct, persistent)
        self.assertEqual(observation.local_label_mismatch_count, 1)
        self.assertFalse(observation.within_tolerance(1e-5))

        accumulator = ScopeEquivalenceAccumulator(tolerance=1e-5)
        accumulator.add(observation)
        summary = accumulator.summary()
        self.assertEqual(summary.world_count, 1)
        self.assertEqual(summary.mismatch_count, 1)
        self.assertEqual(summary.local_label_mismatch_world_count, 1)
        self.assertEqual(summary.local_label_mismatch_count, 1)

    def test_structural_window_mismatch_is_rejected(self) -> None:
        direct = scope_evaluation_from_persistent(_persistent())
        persistent = ScopeEvaluation(
            split=direct.split,
            seed=direct.seed,
            width=direct.width,
            mode=direct.mode,
            inspected_window_indices=(3, 7),
            worker_indices=direct.worker_indices,
            target_present=direct.target_present,
            target_index=direct.target_index,
            target_inspected=direct.target_inspected,
            candidate_window_index=7,
            candidate_is_target=True,
            candidate_relevant_evidence=direct.candidate_relevant_evidence,
            target_relevant_evidence=direct.target_relevant_evidence,
            target_rank=direct.target_rank,
            strongest_distractor_relevant_evidence=(
                direct.strongest_distractor_relevant_evidence
            ),
            window_evidence=(
                WindowEvidence(
                    window_index=3,
                    worker_index=direct.window_evidence[0].worker_index,
                    local_label=direct.window_evidence[0].local_label,
                    relevant_evidence=direct.window_evidence[0].relevant_evidence,
                    not_relevant_evidence=(
                        direct.window_evidence[0].not_relevant_evidence
                    ),
                    uncertainty_probability=(
                        direct.window_evidence[0].uncertainty_probability
                    ),
                    invalid_label_mass=direct.window_evidence[0].invalid_label_mass,
                    top_margin=direct.window_evidence[0].top_margin,
                ),
                direct.window_evidence[1],
            ),
        )
        persistent.validate()
        with self.assertRaisesRegex(ValueError, "inspected_window_indices"):
            compare_scope_evaluations(direct, persistent)

    def test_redundant_persistent_budget_cannot_be_projected_as_direct_baseline(self) -> None:
        repeated = (_rows()[0], _rows()[0])
        with self.assertRaisesRegex(ValueError, "non-redundant"):
            scope_evaluation_from_persistent(_persistent(rows=repeated))

    def test_test_split_requires_explicit_opt_in_before_checkpoint_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Refusing to open the frozen test split"):
            main(
                [
                    "--checkpoints",
                    "does-not-exist.pt",
                    "--split",
                    "test",
                ]
            )

    def test_budget_must_fit_inside_nonredundant_world_before_checkpoint_loading(self) -> None:
        with self.assertRaisesRegex(SystemExit, "must not exceed"):
            main(
                [
                    "--checkpoints",
                    "does-not-exist.pt",
                    "--window-count",
                    "4",
                    "--step-width",
                    "2",
                    "--rounds",
                    "3",
                ]
            )


if __name__ == "__main__":
    unittest.main()
