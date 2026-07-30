from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate7_high_scale_index_bank_prep import (
    Gate7HighScaleImmutableFrontier,
)
from ai_hypothesis.population_compute.gate7_high_scale_terminal_stage_b_prep import (
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
    run_gate7_high_scale_terminal_stage_b_preparation,
)
from ai_hypothesis.population_compute.gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer


class Gate7HighScaleExecutionIsolationPreparationTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[Gate7ScaleNeutralScorer, Gate7HighScaleImmutableFrontier]:
        torch.manual_seed(818)
        model = Gate7ScaleNeutralScorer()
        states = torch.randn((1, 1024, 64), dtype=torch.float32)
        scores = torch.arange(1024, dtype=torch.float32)[None, :]
        frontier = Gate7HighScaleImmutableFrontier(states=states, scores=scores, population=1024)
        frontier.validate()
        return model, frontier

    @staticmethod
    def _run(
        model: Gate7ScaleNeutralScorer,
        frontier: Gate7HighScaleImmutableFrontier,
        *,
        mode: str,
        k: int | None,
        slots: int,
    ):
        return run_gate7_high_scale_terminal_stage_b_preparation(
            model,
            frontier,
            terminal_hints_by_world=(1,),
            public_seeds=torch.tensor((97,), dtype=torch.int64),
            mode=mode,
            k=k,
            stage_b_slots=slots,
        )

    def test_condition_order_cannot_change_transcripts_or_frontier_bytes(self) -> None:
        model, frontier = self._fixture()
        states_before = frontier.states.clone()
        scores_before = frontier.scores.clone()
        fingerprint_before = model.parameter_fingerprint()

        hash_first = self._run(
            model,
            frontier,
            mode=GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
            k=64,
            slots=12,
        )
        global_second = self._run(
            model,
            frontier,
            mode=GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
            k=None,
            slots=12,
        )
        global_first = self._run(
            model,
            frontier,
            mode=GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
            k=None,
            slots=12,
        )
        hash_second = self._run(
            model,
            frontier,
            mode=GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
            k=64,
            slots=12,
        )

        torch.testing.assert_close(
            hash_first.selected_frontier_indices,
            hash_second.selected_frontier_indices,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            hash_first.terminal_child_scores,
            hash_second.terminal_child_scores,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            global_first.selected_frontier_indices,
            global_second.selected_frontier_indices,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            global_first.terminal_child_scores,
            global_second.terminal_child_scores,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(frontier.states, states_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(frontier.scores, scores_before, rtol=0.0, atol=0.0)
        self.assertEqual(model.parameter_fingerprint(), fingerprint_before)

    def test_largest_k_completes_full_frozen_stage_b_horizon_at_n1024(self) -> None:
        model, frontier = self._fixture()
        transcript = self._run(
            model,
            frontier,
            mode=GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
            k=512,
            slots=128,
        )
        self.assertEqual(transcript.final_bank.live_counts.tolist(), [896])
        self.assertEqual(
            transcript.total_neural_score_observations_per_world().tolist(),
            [65_536],
        )
        self.assertEqual(len(set(transcript.selected_frontier_indices[0].tolist())), 128)


if __name__ == "__main__":
    unittest.main()
