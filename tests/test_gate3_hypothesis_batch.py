from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate3_hypothesis_batch import run_gate3_world_batch
from ai_hypothesis.population_compute.gate3_hypothesis_model import Gate3HypothesisScorer, run_gate3_world
from ai_hypothesis.population_compute.gate3_hypothesis_population import Gate3ControlMode, generate_gate3_world


class Gate3HypothesisBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(456)
        self.model = Gate3HypothesisScorer()

    def test_batched_runtime_matches_single_world_reference_on_predictions_and_work(self) -> None:
        worlds = tuple(generate_gate3_world(seed=900 + index, depth=6) for index in range(3))
        for width in (1, 4, 16, 64):
            for mode in Gate3ControlMode:
                batched = run_gate3_world_batch(
                    self.model,
                    worlds,
                    width=width,
                    mode=mode,
                    device="cpu",
                )
                singles = tuple(
                    run_gate3_world(
                        self.model,
                        world,
                        width=width,
                        mode=mode,
                        device="cpu",
                    )
                    for world in worlds
                )
                self.assertEqual(batched.predicted_paths, tuple(row.predicted_path for row in singles))
                self.assertEqual(batched.answer_paths, tuple(row.answer_path for row in singles))
                self.assertEqual(batched.exact_solved_by_world, tuple(row.exact_solved for row in singles))
                self.assertEqual(batched.bit_accuracy_by_world, tuple(row.bit_accuracy for row in singles))
                self.assertEqual(batched.learned_updates_per_world, singles[0].telemetry.learned_updates_total)
                self.assertEqual(
                    batched.correct_candidate_present_by_world_by_phase,
                    tuple(row.telemetry.correct_candidate_present_by_phase for row in singles),
                )
                self.assertEqual(
                    batched.unique_candidate_count_by_world_by_phase,
                    tuple(row.telemetry.unique_candidate_count_by_phase for row in singles),
                )

    def test_batched_h8_largest_width_preserves_all_256_hypotheses_before_reveals(self) -> None:
        worlds = tuple(generate_gate3_world(seed=1000 + index, depth=8) for index in range(2))
        output = run_gate3_world_batch(
            self.model,
            worlds,
            width=256,
            mode=Gate3ControlMode.STABLE_DIVERSE,
            device="cpu",
        )
        self.assertEqual(output.learned_updates_per_world, 4096)
        for world_index in range(len(worlds)):
            self.assertEqual(output.unique_candidate_count_by_world_by_phase[world_index][7], 256)
            self.assertTrue(output.correct_candidate_present_by_world_by_phase[world_index][7])

    def test_batched_width_one_controls_are_exactly_identical(self) -> None:
        worlds = tuple(generate_gate3_world(seed=1200 + index, depth=4) for index in range(4))
        outputs = {
            mode: run_gate3_world_batch(
                self.model,
                worlds,
                width=1,
                mode=mode,
                device="cpu",
            )
            for mode in Gate3ControlMode
        }
        stable = outputs[Gate3ControlMode.STABLE_DIVERSE]
        for mode, output in outputs.items():
            self.assertEqual(output.predicted_paths, stable.predicted_paths, mode)
            self.assertEqual(output.exact_solved_by_world, stable.exact_solved_by_world, mode)
            self.assertEqual(
                output.correct_candidate_present_by_world_by_phase,
                stable.correct_candidate_present_by_world_by_phase,
                mode,
            )
            self.assertEqual(
                output.unique_candidate_count_by_world_by_phase,
                stable.unique_candidate_count_by_world_by_phase,
                mode,
            )


if __name__ == "__main__":
    unittest.main()
