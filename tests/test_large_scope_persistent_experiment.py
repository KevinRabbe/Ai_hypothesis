"""Tests normalized persistent-vs-direct large-scope execution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ai_hypothesis.large_scope.evaluate import ScopeWorkerMode, evaluate_scope_sample
from ai_hypothesis.large_scope.persistent_experiment import PersistentScopeExperiment
from ai_hypothesis.large_scope.relevance import (
    LargeScopeRelevanceConfig,
    generate_large_scope_relevance,
)
from ai_hypothesis.runtime import SQLiteResearchLedger
from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output


class _DeterministicSelectedBank:
    def __init__(self, population_width: int = 16) -> None:
        self._population_width = population_width
        self.calls: list[tuple[int, ...]] = []

    @property
    def population_width(self) -> int:
        return self._population_width

    def forward_selected(self, worker_indices, features, mask):
        del mask
        indices = torch.as_tensor(worker_indices, dtype=torch.float32)
        self.calls.append(tuple(int(index) for index in indices.tolist()))
        batch = features.shape[0]
        logits = torch.full((batch, len(NON_UNCERTAIN_LABELS)), -3.0)
        relevant = LABEL_TO_INDEX["RELEVANT"]
        not_relevant = LABEL_TO_INDEX["NOT_RELEVANT"]
        signal = features[:, 1, 0] + features[:, 2, 1] * 0.2 + indices * 0.03125
        logits[:, relevant] = signal
        logits[:, not_relevant] = -signal
        uncertainty = torch.full((batch,), -2.0) + indices * 0.005
        return Step01Output(label_logits=logits, uncertainty_logits=uncertainty)


class PersistentScopeExperimentTests(unittest.TestCase):
    def assert_window_evidence_equal(self, direct, persistent) -> None:
        self.assertEqual(len(direct), len(persistent))
        for expected, actual in zip(direct, persistent, strict=True):
            self.assertEqual(actual.window_index, expected.window_index)
            self.assertEqual(actual.worker_index, expected.worker_index)
            self.assertEqual(actual.local_label, expected.local_label)
            self.assertAlmostEqual(actual.relevant_evidence, expected.relevant_evidence, places=6)
            self.assertAlmostEqual(
                actual.not_relevant_evidence,
                expected.not_relevant_evidence,
                places=6,
            )
            self.assertAlmostEqual(
                actual.uncertainty_probability,
                expected.uncertainty_probability,
                places=6,
            )
            self.assertAlmostEqual(actual.invalid_label_mass, expected.invalid_label_mass, places=6)
            self.assertAlmostEqual(actual.top_margin, expected.top_margin, places=6)

    def test_diverse_equal_budget_matches_direct_width_when_scope_has_not_deviated(self) -> None:
        sample = generate_large_scope_relevance(42, target_present=True)
        direct_bank = _DeterministicSelectedBank(population_width=16)
        direct = evaluate_scope_sample(
            direct_bank,
            sample,
            width=8,
            mode=ScopeWorkerMode.DIVERSE_WORKERS,
        )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                persistent_bank = _DeterministicSelectedBank(population_width=16)
                experiment = PersistentScopeExperiment(
                    ledger=ledger,
                    sample=sample,
                    bank=persistent_bank,
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    step_width=2,
                )
                result = experiment.run_steps(4)

                self.assertEqual(result.requested_window_evaluations, 8)
                self.assertEqual(result.attempt_count, 8)
                self.assertEqual(result.evidence_count, 8)
                self.assertEqual(result.scheduler_decision_count, 4)
                self.assertEqual(result.distinct_worker_count, 8)
                self.assertEqual(result.resolved_region_count, 8)
                self.assertEqual(result.coverage_fraction, 0.5)
                self.assertEqual(result.duplicate_evidence_count, 0)
                self.assert_window_evidence_equal(direct.window_evidence, result.window_evidence)
                self.assertEqual(result.candidate_window_index, direct.candidate_window_index)
                self.assertEqual(result.candidate_is_target, direct.candidate_is_target)
                self.assertEqual(result.target_rank, direct.target_rank)
                self.assertAlmostEqual(
                    result.candidate_relevant_evidence,
                    direct.candidate_relevant_evidence,
                    places=6,
                )
                if direct.target_relevant_evidence is None:
                    self.assertIsNone(result.target_relevant_evidence)
                else:
                    self.assertAlmostEqual(
                        result.target_relevant_evidence,
                        direct.target_relevant_evidence,
                        places=6,
                    )

    def test_same_worker_equal_budget_matches_direct_and_uses_one_checkpoint(self) -> None:
        sample = generate_large_scope_relevance(64)
        direct_bank = _DeterministicSelectedBank(population_width=16)
        direct = evaluate_scope_sample(
            direct_bank,
            sample,
            width=8,
            mode=ScopeWorkerMode.SAME_WORKER,
        )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                persistent_bank = _DeterministicSelectedBank(population_width=16)
                experiment = PersistentScopeExperiment(
                    ledger=ledger,
                    sample=sample,
                    bank=persistent_bank,
                    mode=ScopeWorkerMode.SAME_WORKER,
                    step_width=2,
                )
                result = experiment.run_steps(4)
                self.assertEqual(result.distinct_worker_count, 1)
                self.assert_window_evidence_equal(direct.window_evidence, result.window_evidence)
                self.assertEqual(result.candidate_window_index, direct.candidate_window_index)

    def test_recreated_experiment_resumes_worker_sequence_and_scope_from_ledger(self) -> None:
        sample = generate_large_scope_relevance(70)
        direct_bank = _DeterministicSelectedBank(population_width=16)
        direct = evaluate_scope_sample(
            direct_bank,
            sample,
            width=8,
            mode=ScopeWorkerMode.DIVERSE_WORKERS,
        )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                first_bank = _DeterministicSelectedBank(population_width=16)
                first = PersistentScopeExperiment(
                    ledger=ledger,
                    sample=sample,
                    bank=first_bank,
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    step_width=2,
                )
                first.run_steps(2)

                second_bank = _DeterministicSelectedBank(population_width=16)
                resumed = PersistentScopeExperiment(
                    ledger=ledger,
                    sample=sample,
                    bank=second_bank,
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    step_width=2,
                )
                result = resumed.run_steps(2)

                self.assertEqual(result.attempt_count, 8)
                self.assertEqual(result.scheduler_decision_count, 4)
                self.assert_window_evidence_equal(direct.window_evidence, result.window_evidence)

    def test_same_worker_can_repeat_one_checkpoint_across_wider_step(self) -> None:
        sample = generate_large_scope_relevance(
            72,
            LargeScopeRelevanceConfig(window_count=4),
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _DeterministicSelectedBank(population_width=1)
                experiment = PersistentScopeExperiment(
                    ledger=ledger,
                    sample=sample,
                    bank=bank,
                    mode=ScopeWorkerMode.SAME_WORKER,
                    step_width=4,
                )
                result = experiment.run_steps(1)
                self.assertEqual(result.attempt_count, 4)
                self.assertEqual(result.distinct_worker_count, 1)
                self.assertEqual(bank.calls, [(0, 0, 0, 0)])
                self.assertEqual(result.coverage_fraction, 1.0)

    def test_budget_beyond_full_coverage_turns_into_balanced_redundancy(self) -> None:
        sample = generate_large_scope_relevance(
            74,
            LargeScopeRelevanceConfig(window_count=4),
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                experiment = PersistentScopeExperiment(
                    ledger=ledger,
                    sample=sample,
                    bank=_DeterministicSelectedBank(population_width=4),
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    step_width=2,
                )
                result = experiment.run_steps(3)
                self.assertEqual(result.attempt_count, 6)
                self.assertEqual(result.resolved_region_count, 4)
                self.assertEqual(result.coverage_fraction, 1.0)
                self.assertEqual(result.duplicate_evidence_count, 2)

    def test_existing_thread_metadata_prevents_cross_world_resume(self) -> None:
        first_sample = generate_large_scope_relevance(76)
        other_sample = generate_large_scope_relevance(78)
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                first = PersistentScopeExperiment(
                    ledger=ledger,
                    sample=first_sample,
                    bank=_DeterministicSelectedBank(),
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    step_width=2,
                )
                first.run_steps(1)
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    PersistentScopeExperiment(
                        ledger=ledger,
                        sample=other_sample,
                        bank=_DeterministicSelectedBank(),
                        mode=ScopeWorkerMode.DIVERSE_WORKERS,
                        step_width=2,
                    )


if __name__ == "__main__":
    unittest.main()
