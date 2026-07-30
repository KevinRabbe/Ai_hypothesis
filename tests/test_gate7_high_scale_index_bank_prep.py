from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate7_high_scale_index_bank_prep import (
    Gate7HighScaleImmutableFrontier,
    clone_gate7_high_scale_live_index_bank,
    delete_gate7_high_scale_selected,
    gather_gate7_high_scale_selected_states,
    gate7_high_scale_index_bank_storage_bytes,
    initialize_gate7_high_scale_live_index_bank,
    select_gate7_high_scale_bounded_hash,
    select_gate7_high_scale_bounded_score,
    select_gate7_high_scale_global_hash,
    select_gate7_high_scale_global_score,
)


class Gate7HighScaleIndexBankPreparationTests(unittest.TestCase):
    @staticmethod
    def _frontier(*, batch: int = 3, population: int = 1024) -> Gate7HighScaleImmutableFrontier:
        states = torch.arange(
            batch * population * 64,
            dtype=torch.float32,
        ).reshape(batch, population, 64)
        base = torch.arange(population, dtype=torch.float32)
        scores = torch.stack((base, -base, torch.remainder(base * 37.0, 997.0)))[:batch]
        frontier = Gate7HighScaleImmutableFrontier(
            states=states,
            scores=scores,
            population=population,
        )
        frontier.validate()
        return frontier

    def test_global_score_matches_direct_live_argmax_without_ties(self) -> None:
        frontier = self._frontier()
        bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=frontier.batch_size,
            population=frontier.population,
            device="cpu",
        )
        seeds = torch.tensor((11, 22, 33), dtype=torch.int64)
        selection = select_gate7_high_scale_global_score(
            frontier,
            bank,
            public_seeds=seeds,
            slot_index=0,
        )
        expected = frontier.scores.argmax(dim=1)
        torch.testing.assert_close(selection.selected_original_indices, expected)
        torch.testing.assert_close(
            selection.neural_scores_observed_per_world,
            torch.full((3,), 1024, dtype=torch.int64),
        )

    def test_global_hash_reads_zero_neural_scores_and_is_deterministic(self) -> None:
        frontier = self._frontier()
        bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=3,
            population=1024,
            device="cpu",
        )
        seeds = torch.tensor((101, 202, 303), dtype=torch.int64)
        first = select_gate7_high_scale_global_hash(
            frontier,
            bank,
            public_seeds=seeds,
            slot_index=17,
        )
        second = select_gate7_high_scale_global_hash(
            frontier,
            bank,
            public_seeds=seeds,
            slot_index=17,
        )
        torch.testing.assert_close(first.selected_original_indices, second.selected_original_indices)
        torch.testing.assert_close(
            first.neural_scores_observed_per_world,
            torch.zeros(3, dtype=torch.int64),
        )

    def test_bounded_score_and_hash_use_identical_sample_positions(self) -> None:
        frontier = self._frontier()
        bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=3,
            population=1024,
            device="cpu",
        )
        seeds = torch.tensor((7, 19, 31), dtype=torch.int64)
        learned = select_gate7_high_scale_bounded_score(
            frontier,
            bank,
            k=64,
            public_seeds=seeds,
            slot_index=9,
        )
        control = select_gate7_high_scale_bounded_hash(
            frontier,
            bank,
            k=64,
            public_seeds=seeds,
            slot_index=9,
        )
        self.assertIsNotNone(learned.sampled_live_positions)
        self.assertIsNotNone(control.sampled_live_positions)
        torch.testing.assert_close(learned.sampled_live_positions, control.sampled_live_positions)
        torch.testing.assert_close(
            learned.neural_scores_observed_per_world,
            torch.full((3,), 64, dtype=torch.int64),
        )
        torch.testing.assert_close(
            control.neural_scores_observed_per_world,
            torch.zeros(3, dtype=torch.int64),
        )

    def test_bounded_score_selects_the_best_score_inside_its_sample(self) -> None:
        frontier = self._frontier()
        bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=3,
            population=1024,
            device="cpu",
        )
        seeds = torch.tensor((43, 47, 53), dtype=torch.int64)
        selection = select_gate7_high_scale_bounded_score(
            frontier,
            bank,
            k=32,
            public_seeds=seeds,
            slot_index=5,
        )
        assert selection.sampled_live_positions is not None
        sampled_original = bank.live_indices.gather(1, selection.sampled_live_positions)
        sampled_scores = frontier.scores.gather(1, sampled_original)
        chosen_scores = frontier.scores.gather(1, selection.selected_original_indices[:, None]).squeeze(1)
        expected_scores = sampled_scores.max(dim=1).values
        torch.testing.assert_close(chosen_scores, expected_scores)

    def test_swap_delete_removes_only_selected_indices_and_frontier_is_immutable(self) -> None:
        frontier = self._frontier()
        original_states = frontier.states.clone()
        original_scores = frontier.scores.clone()
        bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=3,
            population=1024,
            device="cpu",
        )
        seeds = torch.tensor((61, 67, 71), dtype=torch.int64)
        selection = select_gate7_high_scale_bounded_hash(
            frontier,
            bank,
            k=16,
            public_seeds=seeds,
            slot_index=3,
        )
        removed = selection.selected_original_indices.clone()
        selected_states = gather_gate7_high_scale_selected_states(frontier, selection)
        torch.testing.assert_close(selected_states, frontier.states[torch.arange(3), removed])

        updated = delete_gate7_high_scale_selected(bank, selection)
        self.assertEqual(updated.live_counts.tolist(), [1023, 1023, 1023])
        for row in range(3):
            live = set(updated.live_indices[row, : updated.live_counts[row]].tolist())
            self.assertNotIn(int(removed[row]), live)
            self.assertEqual(len(live), 1023)
        torch.testing.assert_close(frontier.states, original_states)
        torch.testing.assert_close(frontier.scores, original_scores)

    def test_condition_clone_copies_only_index_storage(self) -> None:
        bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=3,
            population=1024,
            device="cpu",
        )
        clone = clone_gate7_high_scale_live_index_bank(bank)
        self.assertNotEqual(bank.live_indices.data_ptr(), clone.live_indices.data_ptr())
        self.assertNotEqual(bank.live_counts.data_ptr(), clone.live_counts.data_ptr())
        torch.testing.assert_close(bank.live_indices, clone.live_indices)
        torch.testing.assert_close(bank.live_counts, clone.live_counts)
        clone.live_indices[0, 0] = 999
        self.assertEqual(int(bank.live_indices[0, 0]), 0)

    def test_largest_tier_index_bank_uses_about_sixty_four_mib(self) -> None:
        bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=64,
            population=131072,
            device="cpu",
        )
        self.assertEqual(gate7_high_scale_index_bank_storage_bytes(bank), 67_109_376)

    def test_hot_selectors_do_not_extract_tensor_values_to_python(self) -> None:
        for function in (
            select_gate7_high_scale_global_score,
            select_gate7_high_scale_global_hash,
            select_gate7_high_scale_bounded_score,
            select_gate7_high_scale_bounded_hash,
            delete_gate7_high_scale_selected,
        ):
            source = inspect.getsource(function)
            for forbidden in (".item(", ".cpu(", ".tolist(", "float(", "bool("):
                self.assertNotIn(forbidden, source, function.__name__)

    def test_execution_preparation_contains_no_scientific_namespace(self) -> None:
        import ai_hypothesis.population_compute.gate7_high_scale_index_bank_prep as module

        source = inspect.getsource(module)
        self.assertNotIn("generate_gate7_high_scale_world", source)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("hidden_path", source)
        self.assertNotIn("G7_K_REQUIRED_", source)


if __name__ == "__main__":
    unittest.main()
