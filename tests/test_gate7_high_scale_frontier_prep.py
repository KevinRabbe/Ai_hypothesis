from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate7_high_scale_frontier_prep import (
    build_gate7_high_scale_immutable_frontier,
    validate_gate7_high_scale_public_hints,
)
from ai_hypothesis.population_compute.gate7_scale_neutral_model_prep import (
    Gate7ScaleNeutralScorer,
    encode_gate7_scale_neutral_child_input,
)


class Gate7HighScaleFrontierPreparationTests(unittest.TestCase):
    @staticmethod
    def _hints(last: int = 1) -> tuple[tuple[int, ...], ...]:
        return ((0, 1, 1, 0, 1, 0, 0, 1, 0, 1, last),)

    def test_public_hint_boundary_requires_exact_binary_world_depth(self) -> None:
        self.assertEqual(
            validate_gate7_high_scale_public_hints(
                population=1024,
                noisy_hints_by_world=self._hints(),
            ),
            11,
        )
        with self.assertRaisesRegex(ValueError, "row length"):
            validate_gate7_high_scale_public_hints(
                population=1024,
                noisy_hints_by_world=((0,) * 10,),
            )
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_gate7_high_scale_public_hints(
                population=1024,
                noisy_hints_by_world=((0,) * 10 + (2,),),
            )

    def test_complete_frontier_shape_order_and_parameter_identity(self) -> None:
        torch.manual_seed(123)
        model = Gate7ScaleNeutralScorer()
        before = model.parameter_fingerprint()
        frontier = build_gate7_high_scale_immutable_frontier(
            model,
            population=1024,
            noisy_hints_by_world=self._hints(),
            device="cpu",
        )
        self.assertEqual(frontier.states.shape, (1, 1024, 64))
        self.assertEqual(frontier.scores.shape, (1, 1024))
        self.assertEqual(model.trainable_parameter_count(), 19_649)
        self.assertEqual(model.parameter_fingerprint(), before)

        for path_value in (0, 341, 1023):
            actions = tuple(
                (path_value >> shift) & 1
                for shift in reversed(range(10))
            )
            state = model.initial_state(1, device="cpu")
            for child_depth, action in enumerate(actions, start=1):
                phase_input = encode_gate7_scale_neutral_child_input(
                    world_depth=11,
                    child_depth=child_depth,
                    observed_hint=self._hints()[0][child_depth - 1],
                    branch_action=action,
                    sink=False,
                    device="cpu",
                )[None, :]
                state = model.advance(state, phase_input, repeats=8)
            score = model.score(state)
            torch.testing.assert_close(
                frontier.states[0, path_value],
                state[0],
                rtol=0.0,
                atol=1e-6,
            )
            torch.testing.assert_close(
                frontier.scores[0, path_value],
                score[0],
                rtol=0.0,
                atol=1e-6,
            )

    def test_terminal_hint_is_reserved_for_stage_b(self) -> None:
        torch.manual_seed(321)
        model = Gate7ScaleNeutralScorer()
        first = build_gate7_high_scale_immutable_frontier(
            model,
            population=1024,
            noisy_hints_by_world=self._hints(last=0),
            device="cpu",
        )
        second = build_gate7_high_scale_immutable_frontier(
            model,
            population=1024,
            noisy_hints_by_world=self._hints(last=1),
            device="cpu",
        )
        torch.testing.assert_close(first.states, second.states, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first.scores, second.scores, rtol=0.0, atol=0.0)

    def test_two_public_worlds_remain_independent_in_one_physical_batch(self) -> None:
        torch.manual_seed(777)
        model = Gate7ScaleNeutralScorer()
        hints = (
            (0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1),
            (1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0),
        )
        batched = build_gate7_high_scale_immutable_frontier(
            model,
            population=1024,
            noisy_hints_by_world=hints,
            device="cpu",
        )
        first = build_gate7_high_scale_immutable_frontier(
            model,
            population=1024,
            noisy_hints_by_world=(hints[0],),
            device="cpu",
        )
        second = build_gate7_high_scale_immutable_frontier(
            model,
            population=1024,
            noisy_hints_by_world=(hints[1],),
            device="cpu",
        )
        torch.testing.assert_close(batched.states[0], first.states[0], rtol=0.0, atol=1e-6)
        torch.testing.assert_close(batched.states[1], second.states[0], rtol=0.0, atol=1e-6)
        torch.testing.assert_close(batched.scores[0], first.scores[0], rtol=0.0, atol=1e-6)
        torch.testing.assert_close(batched.scores[1], second.scores[0], rtol=0.0, atol=1e-6)

    def test_builder_has_no_candidate_objects_or_cuda_scalar_extraction(self) -> None:
        source = inspect.getsource(build_gate7_high_scale_immutable_frontier)
        for forbidden in (".item(", ".cpu(", ".tolist(", "float(", "bool("):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("Candidate", source)
        self.assertNotIn("hidden", source)

    def test_frontier_module_contains_no_scientific_execution_surface(self) -> None:
        import ai_hypothesis.population_compute.gate7_high_scale_frontier_prep as module

        source = inspect.getsource(module)
        self.assertNotIn("generate_gate7_high_scale_world", source)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("hidden_path", source)
        self.assertNotIn("G7_K_REQUIRED_", source)


if __name__ == "__main__":
    unittest.main()
