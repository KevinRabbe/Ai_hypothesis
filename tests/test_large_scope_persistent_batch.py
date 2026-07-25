"""Tests persistent multi-world batching against direct large-scope controls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ai_hypothesis.large_scope import (
    ScopeWorkerMode,
    evaluate_scope_sample,
    generate_large_scope_relevance,
)
from ai_hypothesis.large_scope.persistent_batch import (
    PersistentScopeWorldBatchExperiment,
    persistent_scope_world_thread_id,
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
        signal = (
            features[:, 1, 0]
            + features[:, 2, 1] * 0.2
            + indices * 0.03125
        )
        logits[:, relevant] = signal
        logits[:, not_relevant] = -signal
        return Step01Output(
            label_logits=logits,
            uncertainty_logits=torch.full((batch,), -2.0) + indices * 0.005,
        )


class PersistentScopeWorldBatchTests(unittest.TestCase):
    def _assert_window_evidence_equal(self, expected, actual) -> None:
        self.assertEqual(len(expected), len(actual))
        for direct, persistent in zip(expected, actual, strict=True):
            self.assertEqual(persistent.window_index, direct.window_index)
            self.assertEqual(persistent.worker_index, direct.worker_index)
            self.assertEqual(persistent.local_label, direct.local_label)
            self.assertAlmostEqual(
                persistent.relevant_evidence,
                direct.relevant_evidence,
                places=6,
            )
            self.assertAlmostEqual(
                persistent.not_relevant_evidence,
                direct.not_relevant_evidence,
                places=6,
            )
            self.assertAlmostEqual(
                persistent.uncertainty_probability,
                direct.uncertainty_probability,
                places=6,
            )
            self.assertAlmostEqual(
                persistent.invalid_label_mass,
                direct.invalid_label_mass,
                places=6,
            )
            self.assertAlmostEqual(persistent.top_margin, direct.top_margin, places=6)

    def test_three_worlds_two_rounds_use_two_worker_bank_calls_and_match_direct_width_four(self) -> None:
        samples = tuple(
            generate_large_scope_relevance(seed)
            for seed in (100, 102, 104)
        )
        direct_results = tuple(
            evaluate_scope_sample(
                _DeterministicSelectedBank(),
                sample,
                width=4,
                mode=ScopeWorkerMode.DIVERSE_WORKERS,
            )
            for sample in samples
        )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _DeterministicSelectedBank()
                experiment = PersistentScopeWorldBatchExperiment(
                    ledger=ledger,
                    samples=samples,
                    bank=bank,
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    step_width=2,
                )
                result = experiment.run_rounds(2)

                self.assertEqual(len(bank.calls), 2)
                self.assertTrue(all(len(call) == 6 for call in bank.calls))
                self.assertEqual(result.rounds, 2)
                self.assertEqual(result.world_count, 3)
                self.assertEqual(result.local_window_evaluations, 12)
                self.assertEqual(len(result.worlds), 3)

                for direct, persistent in zip(
                    direct_results,
                    result.worlds,
                    strict=True,
                ):
                    self.assertEqual(persistent.step_count, 2)
                    self.assertEqual(persistent.attempt_count, 4)
                    self.assertEqual(persistent.scheduler_decision_count, 2)
                    self.assertEqual(persistent.duplicate_evidence_count, 0)
                    self._assert_window_evidence_equal(
                        direct.window_evidence,
                        persistent.window_evidence,
                    )
                    self.assertEqual(
                        persistent.candidate_window_index,
                        direct.candidate_window_index,
                    )
                    self.assertEqual(
                        persistent.candidate_is_target,
                        direct.candidate_is_target,
                    )
                    self.assertEqual(persistent.target_rank, direct.target_rank)

    def test_recreated_batch_continues_each_world_scope_and_worker_sequence(self) -> None:
        samples = tuple(
            generate_large_scope_relevance(seed)
            for seed in (110, 112, 114)
        )
        direct_results = tuple(
            evaluate_scope_sample(
                _DeterministicSelectedBank(),
                sample,
                width=4,
                mode=ScopeWorkerMode.DIVERSE_WORKERS,
            )
            for sample in samples
        )

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                first_bank = _DeterministicSelectedBank()
                first = PersistentScopeWorldBatchExperiment(
                    ledger=ledger,
                    samples=samples,
                    bank=first_bank,
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    step_width=2,
                )
                first.run_rounds(1)

                second_bank = _DeterministicSelectedBank()
                resumed = PersistentScopeWorldBatchExperiment(
                    ledger=ledger,
                    samples=samples,
                    bank=second_bank,
                    mode=ScopeWorkerMode.DIVERSE_WORKERS,
                    step_width=2,
                )
                result = resumed.run_rounds(1)

                self.assertEqual(len(first_bank.calls), 1)
                self.assertEqual(len(second_bank.calls), 1)
                self.assertEqual(result.rounds, 2)
                for direct, persistent in zip(
                    direct_results,
                    result.worlds,
                    strict=True,
                ):
                    self._assert_window_evidence_equal(
                        direct.window_evidence,
                        persistent.window_evidence,
                    )

    def test_same_worker_mode_keeps_one_checkpoint_per_world_while_batching_worlds(self) -> None:
        samples = tuple(
            generate_large_scope_relevance(seed)
            for seed in (120, 122, 124)
        )
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                bank = _DeterministicSelectedBank()
                experiment = PersistentScopeWorldBatchExperiment(
                    ledger=ledger,
                    samples=samples,
                    bank=bank,
                    mode=ScopeWorkerMode.SAME_WORKER,
                    step_width=2,
                )
                result = experiment.run_rounds(2)

                self.assertEqual(len(bank.calls), 2)
                self.assertTrue(all(len(call) == 6 for call in bank.calls))
                for world in result.worlds:
                    self.assertEqual(world.distinct_worker_count, 1)
                    self.assertEqual(world.attempt_count, 4)

    def test_thread_identity_is_stable_opaque_and_mode_specific(self) -> None:
        sample = generate_large_scope_relevance(130)
        diverse_a = persistent_scope_world_thread_id(
            sample,
            ScopeWorkerMode.DIVERSE_WORKERS,
        )
        diverse_b = persistent_scope_world_thread_id(
            sample,
            ScopeWorkerMode.DIVERSE_WORKERS,
        )
        same = persistent_scope_world_thread_id(
            sample,
            ScopeWorkerMode.SAME_WORKER,
        )
        self.assertEqual(diverse_a, diverse_b)
        self.assertNotEqual(diverse_a, same)
        self.assertTrue(diverse_a.startswith("scope-world-"))
        self.assertNotIn(str(sample.seed), diverse_a)

    def test_duplicate_world_identity_is_rejected_before_thread_creation(self) -> None:
        sample = generate_large_scope_relevance(140)
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteResearchLedger(Path(directory) / "ledger.sqlite") as ledger:
                with self.assertRaisesRegex(ValueError, "unique split/seed worlds"):
                    PersistentScopeWorldBatchExperiment(
                        ledger=ledger,
                        samples=(sample, sample),
                        bank=_DeterministicSelectedBank(),
                        mode=ScopeWorkerMode.DIVERSE_WORKERS,
                        step_width=2,
                    )
                self.assertEqual(ledger.read_all_events(), ())


if __name__ == "__main__":
    unittest.main()
