"""Tests large-scope execution without requiring real checkpoint files."""

from __future__ import annotations

import unittest

import torch

from ai_hypothesis.large_scope import (
    ScopeWorkerMode,
    diverse_worker_indices,
    evaluate_scope_batch,
    evaluate_scope_sample,
    evaluate_scope_widths,
    generate_large_scope_relevance,
    inspection_prefix,
    same_worker_indices,
)
from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output


class _FakeSelectedBank:
    def __init__(self, *, population_width: int = 16, relevant_batch_index: int = 0) -> None:
        self._population_width = population_width
        self.relevant_batch_index = relevant_batch_index
        self.calls: list[tuple[int, ...]] = []

    @property
    def population_width(self) -> int:
        return self._population_width

    def forward_selected(self, worker_indices, features, mask):
        indices = tuple(int(index) for index in worker_indices)
        self.calls.append(indices)
        batch = features.shape[0]
        logits = torch.full((batch, len(NON_UNCERTAIN_LABELS)), -4.0)
        not_relevant = LABEL_TO_INDEX["NOT_RELEVANT"]
        relevant = LABEL_TO_INDEX["RELEVANT"]
        logits[:, not_relevant] = 4.0
        if 0 <= self.relevant_batch_index < batch:
            logits[self.relevant_batch_index, not_relevant] = -4.0
            logits[self.relevant_batch_index, relevant] = 6.0
        return Step01Output(
            label_logits=logits,
            uncertainty_logits=torch.full((batch,), -5.0),
        )


class _FeatureDrivenBank:
    """Deterministic row-local fake bank whose output is independent of batch shape."""

    def __init__(self, population_width: int = 16) -> None:
        self._population_width = population_width
        self.batch_sizes: list[int] = []

    @property
    def population_width(self) -> int:
        return self._population_width

    def forward_selected(self, worker_indices, features, mask):
        self.batch_sizes.append(int(features.shape[0]))
        batch = features.shape[0]
        logits = torch.full((batch, len(NON_UNCERTAIN_LABELS)), -8.0)
        relevant = LABEL_TO_INDEX["RELEVANT"]
        not_relevant = LABEL_TO_INDEX["NOT_RELEVANT"]
        signal = features[:, :, 0].sum(dim=1) / 8.0
        logits[:, relevant] = signal
        logits[:, not_relevant] = -signal
        return Step01Output(
            label_logits=logits,
            uncertainty_logits=torch.full((batch,), -3.0),
        )


class LargeScopeEvaluationTests(unittest.TestCase):
    def test_candidate_mapping_uses_inspected_window_coordinates(self) -> None:
        sample = generate_large_scope_relevance(42, target_present=True)
        inspected = inspection_prefix(sample, 16)
        assert sample.target_index is not None
        target_batch_index = inspected.index(sample.target_index)
        bank = _FakeSelectedBank(relevant_batch_index=target_batch_index)

        result = evaluate_scope_sample(
            bank,
            sample,
            width=16,
            mode=ScopeWorkerMode.DIVERSE_WORKERS,
        )

        self.assertTrue(result.target_inspected)
        self.assertTrue(result.candidate_is_target)
        self.assertEqual(result.candidate_window_index, sample.target_index)
        self.assertEqual(result.target_rank, 1)
        self.assertEqual(
            result.worker_indices,
            diverse_worker_indices(seed=sample.seed, width=16, population_width=16),
        )
        self.assertEqual(bank.calls, [result.worker_indices])

    def test_scope_only_and_diverse_conditions_inspect_identical_windows(self) -> None:
        sample = generate_large_scope_relevance(64)
        same_bank = _FakeSelectedBank(relevant_batch_index=0)
        diverse_bank = _FakeSelectedBank(relevant_batch_index=0)

        same = evaluate_scope_sample(
            same_bank,
            sample,
            width=4,
            mode=ScopeWorkerMode.SAME_WORKER,
        )
        diverse = evaluate_scope_sample(
            diverse_bank,
            sample,
            width=4,
            mode=ScopeWorkerMode.DIVERSE_WORKERS,
        )

        self.assertEqual(same.inspected_window_indices, diverse.inspected_window_indices)
        self.assertEqual(
            same.worker_indices,
            same_worker_indices(seed=sample.seed, width=4, population_width=16),
        )
        self.assertEqual(len(set(same.worker_indices)), 1)
        self.assertEqual(len(set(diverse.worker_indices)), 4)

    def test_batched_worlds_match_individual_evaluation_with_one_forward_call(self) -> None:
        samples = tuple(
            generate_large_scope_relevance(seed) for seed in (40, 41, 42)
        )
        sequential_bank = _FeatureDrivenBank()
        sequential = tuple(
            evaluate_scope_sample(
                sequential_bank,
                sample,
                width=4,
                mode=ScopeWorkerMode.DIVERSE_WORKERS,
            )
            for sample in samples
        )
        batched_bank = _FeatureDrivenBank()
        batched = evaluate_scope_batch(
            batched_bank,
            samples,
            width=4,
            mode=ScopeWorkerMode.DIVERSE_WORKERS,
        )

        self.assertEqual(batched, sequential)
        self.assertEqual(sequential_bank.batch_sizes, [4, 4, 4])
        self.assertEqual(batched_bank.batch_sizes, [12])

    def test_width_runner_preserves_nested_prefixes_in_both_modes(self) -> None:
        sample = generate_large_scope_relevance(88)
        bank = _FakeSelectedBank(relevant_batch_index=0)
        results = evaluate_scope_widths(
            bank,
            sample,
            widths=(1, 4, 16),
        )
        self.assertEqual(len(results), 6)

        by_mode = {
            mode: [result for result in results if result.mode is mode]
            for mode in ScopeWorkerMode
        }
        for mode_results in by_mode.values():
            width_1, width_4, width_16 = mode_results
            self.assertEqual(width_4.inspected_window_indices[:1], width_1.inspected_window_indices)
            self.assertEqual(width_16.inspected_window_indices[:4], width_4.inspected_window_indices)

    def test_target_outside_prefix_remains_uninspected_even_with_strong_candidate(self) -> None:
        sample = None
        target_position = 0
        for seed in range(90, 130, 2):
            candidate = generate_large_scope_relevance(seed, target_present=True)
            assert candidate.target_index is not None
            order = inspection_prefix(candidate, 16)
            target_position = order.index(candidate.target_index)
            if target_position > 0:
                sample = candidate
                break
        self.assertIsNotNone(sample)
        assert sample is not None

        result = evaluate_scope_sample(
            _FakeSelectedBank(relevant_batch_index=0),
            sample,
            width=target_position,
            mode=ScopeWorkerMode.SAME_WORKER,
        )
        self.assertFalse(result.target_inspected)
        self.assertIsNone(result.target_relevant_evidence)
        self.assertIsNone(result.target_rank)
        self.assertFalse(result.candidate_is_target)


if __name__ == "__main__":
    unittest.main()
