from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate7_high_scale_index_bank_prep import (
    Gate7HighScaleImmutableFrontier,
    initialize_gate7_high_scale_live_index_bank,
)
from ai_hypothesis.population_compute.gate7_high_scale_terminal_stage_b_prep import (
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
    run_gate7_high_scale_terminal_stage_b_preparation,
)
from ai_hypothesis.population_compute.gate7_scale_neutral_model_prep import (
    Gate7ScaleNeutralScorer,
    encode_gate7_scale_neutral_child_input,
)


class Gate7HighScaleTerminalStageBPreparationTests(unittest.TestCase):
    @staticmethod
    def _frontier(*, score_variant: int = 0) -> Gate7HighScaleImmutableFrontier:
        torch.manual_seed(91)
        states = torch.randn((2, 1024, 64), dtype=torch.float32)
        base = torch.arange(1024, dtype=torch.float32)
        if score_variant == 0:
            scores = torch.stack((base, -base))
        else:
            scores = torch.stack((-base, base))
        frontier = Gate7HighScaleImmutableFrontier(states=states, scores=scores, population=1024)
        frontier.validate()
        return frontier

    @staticmethod
    def _model() -> Gate7ScaleNeutralScorer:
        torch.manual_seed(1234)
        return Gate7ScaleNeutralScorer()

    def test_global_score_matches_direct_unique_score_order_and_accounting(self) -> None:
        model = self._model()
        frontier = self._frontier()
        transcript = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            frontier,
            terminal_hints_by_world=(1, 0),
            public_seeds=torch.tensor((17, 29), dtype=torch.int64),
            mode=GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
            k=None,
            stage_b_slots=4,
        )
        self.assertEqual(
            transcript.selected_frontier_indices.tolist(),
            [[1023, 1022, 1021, 1020], [0, 1, 2, 3]],
        )
        self.assertEqual(
            transcript.neural_score_observations_by_slot.tolist(),
            [[1024, 1023, 1022, 1021], [1024, 1023, 1022, 1021]],
        )
        expected_paths = transcript.selected_frontier_indices[:, :, None] * 2 + torch.tensor((0, 1))
        torch.testing.assert_close(transcript.terminal_path_ids, expected_paths)
        self.assertEqual(transcript.final_bank.live_counts.tolist(), [1020, 1020])

    def test_first_terminal_scores_match_direct_serial_transition(self) -> None:
        model = self._model()
        frontier = self._frontier()
        transcript = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            frontier,
            terminal_hints_by_world=(1, 0),
            public_seeds=torch.tensor((31, 37), dtype=torch.int64),
            mode=GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
            k=None,
            stage_b_slots=1,
        )
        selected = transcript.selected_frontier_indices[:, 0]
        parent_states = frontier.states[torch.arange(2), selected]
        direct_scores: list[torch.Tensor] = []
        for row, hint in enumerate((1, 0)):
            child_inputs = torch.stack(
                [
                    encode_gate7_scale_neutral_child_input(
                        world_depth=11,
                        child_depth=11,
                        observed_hint=hint,
                        branch_action=action,
                        sink=False,
                        device="cpu",
                    )
                    for action in (0, 1)
                ]
            )
            child_states = model.advance(
                parent_states[row : row + 1].expand(2, 64),
                child_inputs,
                repeats=8,
            )
            direct_scores.append(model.score(child_states))
        torch.testing.assert_close(
            transcript.terminal_child_scores[:, 0],
            torch.stack(direct_scores),
            rtol=0.0,
            atol=1e-6,
        )

    def test_global_hash_is_score_blind_and_observes_zero_scores(self) -> None:
        model = self._model()
        seeds = torch.tensor((41, 43), dtype=torch.int64)
        first = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            self._frontier(score_variant=0),
            terminal_hints_by_world=(0, 1),
            public_seeds=seeds,
            mode=GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
            k=None,
            stage_b_slots=8,
        )
        second = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            self._frontier(score_variant=1),
            terminal_hints_by_world=(0, 1),
            public_seeds=seeds,
            mode=GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
            k=None,
            stage_b_slots=8,
        )
        torch.testing.assert_close(first.selected_frontier_indices, second.selected_frontier_indices)
        torch.testing.assert_close(
            first.neural_score_observations_by_slot,
            torch.zeros((2, 8), dtype=torch.int64),
        )

    def test_bounded_score_and_hash_have_exact_frozen_observation_counts(self) -> None:
        model = self._model()
        frontier = self._frontier()
        seeds = torch.tensor((47, 53), dtype=torch.int64)
        learned = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            frontier,
            terminal_hints_by_world=(1, 1),
            public_seeds=seeds,
            mode=GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
            k=64,
            stage_b_slots=5,
        )
        control = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            frontier,
            terminal_hints_by_world=(1, 1),
            public_seeds=seeds,
            mode=GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
            k=64,
            stage_b_slots=5,
        )
        torch.testing.assert_close(
            learned.neural_score_observations_by_slot,
            torch.full((2, 5), 64, dtype=torch.int64),
        )
        torch.testing.assert_close(
            learned.total_neural_score_observations_per_world(),
            torch.full((2,), 320, dtype=torch.int64),
        )
        torch.testing.assert_close(
            control.neural_score_observations_by_slot,
            torch.zeros((2, 5), dtype=torch.int64),
        )

    def test_terminal_hint_changes_terminal_evaluation_but_not_parent_selection(self) -> None:
        model = self._model()
        frontier = self._frontier()
        seeds = torch.tensor((59, 61), dtype=torch.int64)
        zero_hint = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            frontier,
            terminal_hints_by_world=(0, 0),
            public_seeds=seeds,
            mode=GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
            k=16,
            stage_b_slots=4,
        )
        one_hint = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            frontier,
            terminal_hints_by_world=(1, 1),
            public_seeds=seeds,
            mode=GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
            k=16,
            stage_b_slots=4,
        )
        torch.testing.assert_close(
            zero_hint.selected_frontier_indices,
            one_hint.selected_frontier_indices,
        )
        self.assertFalse(torch.equal(zero_hint.terminal_child_scores, one_hint.terminal_child_scores))

    def test_initial_bank_is_cloned_and_each_selected_frontier_index_is_unique(self) -> None:
        model = self._model()
        frontier = self._frontier()
        initial = initialize_gate7_high_scale_live_index_bank(
            batch_size=2,
            population=1024,
            device="cpu",
        )
        before_indices = initial.live_indices.clone()
        before_counts = initial.live_counts.clone()
        transcript = run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            frontier,
            terminal_hints_by_world=(0, 1),
            public_seeds=torch.tensor((67, 71), dtype=torch.int64),
            mode=GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
            k=None,
            stage_b_slots=16,
            initial_bank=initial,
        )
        torch.testing.assert_close(initial.live_indices, before_indices)
        torch.testing.assert_close(initial.live_counts, before_counts)
        for row in transcript.selected_frontier_indices:
            self.assertEqual(len(set(row.tolist())), 16)

    def test_mode_and_k_boundaries_are_strict(self) -> None:
        model = self._model()
        frontier = self._frontier()
        common = {
            "terminal_hints_by_world": (0, 1),
            "public_seeds": torch.tensor((73, 79), dtype=torch.int64),
            "stage_b_slots": 1,
        }
        with self.assertRaisesRegex(ValueError, "cannot carry K"):
            run_gate7_high_scale_terminal_stage_b_preparation(
                model,
                frontier,
                mode=GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
                k=16,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "requires one frozen K"):
            run_gate7_high_scale_terminal_stage_b_preparation(
                model,
                frontier,
                mode=GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
                k=None,
                **common,
            )

    def test_terminal_executor_has_no_cuda_scalar_or_hidden_answer_dependency(self) -> None:
        source = inspect.getsource(run_gate7_high_scale_terminal_stage_b_preparation)
        for forbidden in (".item(", ".cpu(", ".tolist(", "float(", "bool("):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("hidden_path", source)
        self.assertNotIn("covered", source)

    def test_terminal_module_contains_no_scientific_execution_surface(self) -> None:
        import ai_hypothesis.population_compute.gate7_high_scale_terminal_stage_b_prep as module

        source = inspect.getsource(module)
        self.assertNotIn("generate_gate7_high_scale_world", source)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("G7_K_REQUIRED_", source)


if __name__ == "__main__":
    unittest.main()
