from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate3_hypothesis_model import (
    GATE3_INPUT_WIDTH,
    Gate3HypothesisScorer,
    encode_gate3_phase_input,
    run_gate3_world,
)
from ai_hypothesis.population_compute.gate3_hypothesis_population import (
    Gate3ControlMode,
    build_gate3_condition_plan,
    generate_gate3_world,
)


class Gate3HypothesisModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.model = Gate3HypothesisScorer()

    def test_default_model_is_fixed_and_about_twenty_thousand_parameters(self) -> None:
        self.assertEqual(GATE3_INPUT_WIDTH, 26)
        self.assertEqual(self.model.trainable_parameter_count(), 19_873)
        self.assertEqual(len(self.model.parameter_fingerprint()), 64)

    def test_phase_encoding_has_fixed_width_and_no_population_width_feature(self) -> None:
        world = generate_gate3_world(seed=3, depth=8)
        branch = encode_gate3_phase_input(
            depth=8,
            observation=world.observations[0],
            branch_action=1,
            device="cpu",
        )
        reveal = encode_gate3_phase_input(
            depth=8,
            observation=world.observations[8],
            branch_action=None,
            device="cpu",
        )
        self.assertEqual(branch.shape, (26,))
        self.assertEqual(reveal.shape, (26,))
        self.assertEqual(float(branch.sum().item()), 5.0)
        self.assertEqual(float(reveal.sum().item()), 5.0)

    def test_width_one_controls_are_exactly_identical_with_same_random_model(self) -> None:
        world = generate_gate3_world(seed=77, depth=6)
        outputs = {
            mode: run_gate3_world(self.model, world, width=1, mode=mode, device="cpu")
            for mode in Gate3ControlMode
        }
        stable = outputs[Gate3ControlMode.STABLE_DIVERSE]
        for mode, output in outputs.items():
            self.assertEqual(output.predicted_path, stable.predicted_path, mode)
            self.assertEqual(output.exact_solved, stable.exact_solved, mode)
            self.assertEqual(output.bit_accuracy, stable.bit_accuracy, mode)
            self.assertEqual(output.final_score, stable.final_score, mode)
            self.assertEqual(output.telemetry.learned_updates_total, stable.telemetry.learned_updates_total, mode)
            self.assertEqual(output.telemetry.unique_candidate_count_by_phase, stable.telemetry.unique_candidate_count_by_phase, mode)
            self.assertEqual(output.telemetry.correct_candidate_present_by_phase, stable.telemetry.correct_candidate_present_by_phase, mode)

    def test_h8_runtime_executes_exactly_4096_learned_updates_for_every_width_and_mode(self) -> None:
        world = generate_gate3_world(seed=19, depth=8)
        for width in (1, 4, 16, 64, 256):
            for mode in Gate3ControlMode:
                output = run_gate3_world(self.model, world, width=width, mode=mode, device="cpu")
                self.assertEqual(output.telemetry.learned_updates_total, 4096)
                self.assertEqual(output.telemetry.learned_updates_per_phase, (256,) * 16)
                plan = build_gate3_condition_plan(world, width=width, mode=mode)
                output.telemetry.validate(plan=plan)

    def test_largest_stable_beam_retains_all_hypotheses_before_reveals(self) -> None:
        world = generate_gate3_world(seed=101, depth=8)
        output = run_gate3_world(
            self.model,
            world,
            width=256,
            mode=Gate3ControlMode.STABLE_DIVERSE,
            device="cpu",
        )
        self.assertEqual(output.telemetry.retained_state_slots_by_phase[7], 256)
        self.assertEqual(output.telemetry.unique_candidate_count_by_phase[7], 256)
        self.assertTrue(output.telemetry.correct_candidate_present_by_phase[7])

    def test_collapsed_control_spends_same_work_but_removes_hypothesis_diversity(self) -> None:
        world = generate_gate3_world(seed=202, depth=8)
        stable = run_gate3_world(
            self.model,
            world,
            width=64,
            mode=Gate3ControlMode.STABLE_DIVERSE,
            device="cpu",
        )
        collapsed = run_gate3_world(
            self.model,
            world,
            width=64,
            mode=Gate3ControlMode.COLLAPSED_DIVERSITY,
            device="cpu",
        )
        self.assertEqual(stable.telemetry.learned_updates_total, collapsed.telemetry.learned_updates_total)
        self.assertEqual(stable.telemetry.retained_state_slots_by_phase, collapsed.telemetry.retained_state_slots_by_phase)
        self.assertEqual(collapsed.telemetry.unique_candidate_count_by_phase[-1], 1)
        self.assertGreater(stable.telemetry.unique_candidate_count_by_phase[7], 1)

    def test_runtime_never_exceeds_frozen_physical_width(self) -> None:
        world = generate_gate3_world(seed=303, depth=6)
        for width in (1, 4, 16, 64):
            output = run_gate3_world(
                self.model,
                world,
                width=width,
                mode=Gate3ControlMode.RESHUFFLED_CONTINUITY,
                device="cpu",
            )
            self.assertTrue(all(count <= width for count in output.telemetry.retained_state_slots_by_phase))
            self.assertTrue(all(count <= width for count in output.telemetry.unique_candidate_count_by_phase))


if __name__ == "__main__":
    unittest.main()
