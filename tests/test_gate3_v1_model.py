from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import (
    GATE3_V1_INPUT_WIDTH,
    Gate3V1Scorer,
    encode_gate3_v1_child_input,
    run_gate3_v1_public_world,
)
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import (
    Gate3V1ControlMode,
    generate_gate3_v1_world,
    score_generated_solution,
)


class Gate3V1ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(321)
        self.model = Gate3V1Scorer()

    def test_shared_model_is_fixed_size_and_has_no_capacity_input(self) -> None:
        self.assertEqual(GATE3_V1_INPUT_WIDTH, 19)
        self.assertEqual(self.model.trainable_parameter_count(), 19_649)
        self.assertEqual(len(self.model.parameter_fingerprint()), 64)
        parameters = inspect.signature(encode_gate3_v1_child_input).parameters
        self.assertNotIn("reserve_capacity", parameters)
        self.assertNotIn("hidden_path", parameters)

    def test_public_runtime_cannot_accept_hidden_answer(self) -> None:
        parameters = inspect.signature(run_gate3_v1_public_world).parameters
        self.assertIn("world", parameters)
        self.assertNotIn("hidden_path", parameters)
        self.assertNotIn("answer", parameters)

    def test_l1_modes_are_exactly_identical_with_same_random_model(self) -> None:
        world = generate_gate3_v1_world(seed=77, depth=6)
        outputs = {
            mode: run_gate3_v1_public_world(
                self.model,
                world.public,
                reserve_capacity=1,
                mode=mode,
                device="cpu",
            )
            for mode in Gate3V1ControlMode
        }
        stable = outputs[Gate3V1ControlMode.STABLE_RESERVE]
        for mode, output in outputs.items():
            self.assertEqual(output.generated_terminal_paths, stable.generated_terminal_paths, mode)
            self.assertEqual(output.telemetry.productive_rounds, stable.telemetry.productive_rounds, mode)
            self.assertEqual(output.telemetry.sink_rounds, stable.telemetry.sink_rounds, mode)
            self.assertEqual(output.telemetry.total_learned_updates, stable.telemetry.total_learned_updates, mode)
            self.assertEqual(output.telemetry.unique_reserve_population_by_round, stable.telemetry.unique_reserve_population_by_round, mode)

    def test_l1_commits_then_uses_matched_sink_while_large_reserve_can_continue_search(self) -> None:
        world = generate_gate3_v1_world(seed=91, depth=6)
        narrow = run_gate3_v1_public_world(
            self.model,
            world.public,
            reserve_capacity=1,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            device="cpu",
        )
        wide = run_gate3_v1_public_world(
            self.model,
            world.public,
            reserve_capacity=16,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            device="cpu",
        )
        self.assertEqual(narrow.telemetry.productive_rounds, 6)
        self.assertEqual(narrow.telemetry.sink_rounds, 10)
        self.assertEqual(narrow.telemetry.total_learned_updates, 256)
        self.assertEqual(wide.telemetry.productive_rounds, 16)
        self.assertEqual(wide.telemetry.sink_rounds, 0)
        self.assertEqual(wide.telemetry.total_learned_updates, 256)
        self.assertGreater(wide.telemetry.unique_generated_terminal_count, narrow.telemetry.unique_generated_terminal_count)

    def test_generated_solution_scoring_occurs_only_after_public_runtime(self) -> None:
        world = generate_gate3_v1_world(seed=101, depth=6)
        output = run_gate3_v1_public_world(
            self.model,
            world.public,
            reserve_capacity=16,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            device="cpu",
        )
        self.assertTrue(all(len(path) == 6 for path in output.generated_terminal_paths))
        solved = score_generated_solution(
            hidden_path=world.hidden_path,
            generated_terminal_paths=output.generated_terminal_paths,
        )
        self.assertIsInstance(solved, bool)

    def test_sink_input_contains_no_world_hint_or_branch_action(self) -> None:
        world = generate_gate3_v1_world(seed=202, depth=8)
        sink = encode_gate3_v1_child_input(
            world=world.public,
            child_depth=8,
            observed_hint=None,
            branch_action=None,
            sink=True,
            device="cpu",
        )
        productive = encode_gate3_v1_child_input(
            world=world.public,
            child_depth=1,
            observed_hint=world.public.noisy_hints[0],
            branch_action=1,
            sink=False,
            device="cpu",
        )
        self.assertEqual(sink.shape, (19,))
        self.assertEqual(productive.shape, (19,))
        self.assertEqual(float(sink.sum().item()), 4.0)
        self.assertEqual(float(productive.sum().item()), 4.0)


if __name__ == "__main__":
    unittest.main()
