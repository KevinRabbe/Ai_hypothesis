from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate7_scale_neutral_memory_bounded_prep import (
    advance_gate7_scale_neutral_memory_bounded,
)
from ai_hypothesis.population_compute.gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer


class Gate7ScaleNeutralMemoryBoundedPreparationTests(unittest.TestCase):
    def test_matches_reference_eight_step_advance_and_scores(self) -> None:
        torch.manual_seed(404)
        model = Gate7ScaleNeutralScorer()
        state = torch.randn((37, 64), dtype=torch.float32)
        phase_input = torch.randn((37, 19), dtype=torch.float32)
        reference = model.advance(state.clone(), phase_input.clone(), repeats=8)
        bounded = advance_gate7_scale_neutral_memory_bounded(
            model,
            state.clone(),
            phase_input.clone(),
            repeats=8,
        )
        torch.testing.assert_close(bounded, reference, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(
            model.score(bounded),
            model.score(reference),
            rtol=0.0,
            atol=1e-6,
        )

    def test_single_step_is_exactly_the_reference_single_step(self) -> None:
        torch.manual_seed(405)
        model = Gate7ScaleNeutralScorer()
        state = torch.randn((11, 64), dtype=torch.float32)
        phase_input = torch.randn((11, 19), dtype=torch.float32)
        reference = model.advance(state.clone(), phase_input.clone(), repeats=1)
        bounded = advance_gate7_scale_neutral_memory_bounded(
            model,
            state.clone(),
            phase_input.clone(),
            repeats=1,
        )
        torch.testing.assert_close(bounded, reference, rtol=0.0, atol=0.0)

    def test_execution_does_not_mutate_learned_parameters(self) -> None:
        torch.manual_seed(406)
        model = Gate7ScaleNeutralScorer()
        before = model.parameter_fingerprint()
        advance_gate7_scale_neutral_memory_bounded(
            model,
            torch.zeros((5, 64), dtype=torch.float32),
            torch.zeros((5, 19), dtype=torch.float32),
            repeats=8,
        )
        self.assertEqual(model.parameter_fingerprint(), before)
        self.assertEqual(model.trainable_parameter_count(), 19_649)

    def test_helper_has_no_repeated_sequence_materialization(self) -> None:
        source = inspect.getsource(advance_gate7_scale_neutral_memory_bounded)
        self.assertNotIn(".expand(", source)
        self.assertNotIn(".repeat(", source)
        self.assertNotIn(".contiguous(", source)
        self.assertNotIn("[batch,repeats", source)
        self.assertIn("for _ in range(repeats)", source)

    def test_helper_has_no_cuda_scalar_extraction_or_scientific_surface(self) -> None:
        import ai_hypothesis.population_compute.gate7_scale_neutral_memory_bounded_prep as module

        source = inspect.getsource(module)
        for forbidden in (".item(", ".cpu(", ".tolist(", "float(", "bool("):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("hidden_path", source)
        self.assertNotIn("generate_gate7_high_scale_world", source)


if __name__ == "__main__":
    unittest.main()
