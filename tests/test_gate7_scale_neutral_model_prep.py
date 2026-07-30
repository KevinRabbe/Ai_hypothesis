from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1Scorer
from ai_hypothesis.population_compute.gate7_scale_neutral_model_prep import (
    GATE7_SCALE_NEUTRAL_INPUT_WIDTH,
    GATE7_SCALE_NEUTRAL_MODEL_PREPARATION_ONLY,
    GATE7_SCALE_NEUTRAL_PARAMETER_COUNT,
    GATE7_SCALE_NEUTRAL_POSITION_WIDTH,
    Gate7ScaleNeutralScorer,
    encode_gate7_scale_neutral_child_input,
    gate7_scale_neutral_position_features,
)


class Gate7ScaleNeutralModelPreparationTests(unittest.TestCase):
    def test_preparation_only_boundary(self) -> None:
        self.assertTrue(GATE7_SCALE_NEUTRAL_MODEL_PREPARATION_ONLY)

    def test_input_width_and_parameter_count_are_unchanged(self) -> None:
        self.assertEqual(GATE7_SCALE_NEUTRAL_INPUT_WIDTH, 19)
        model = Gate7ScaleNeutralScorer()
        self.assertEqual(model.trainable_parameter_count(), GATE7_SCALE_NEUTRAL_PARAMETER_COUNT)
        self.assertEqual(model.trainable_parameter_count(), 19_649)

    def test_parameter_tensor_geometry_matches_gate3_v1(self) -> None:
        old = Gate3V1Scorer()
        new = Gate7ScaleNeutralScorer()
        old_shapes = {name: tuple(tensor.shape) for name, tensor in old.state_dict().items()}
        new_shapes = {name: tuple(tensor.shape) for name, tensor in new.state_dict().items()}
        self.assertEqual(new_shapes, old_shapes)

    def test_position_encoding_has_no_maximum_depth_dependency(self) -> None:
        for world_depth, child_depth in (
            (6, 1),
            (10, 8),
            (18, 17),
            (1_000, 999),
            (1_000_000, 1_000_000),
        ):
            features = gate7_scale_neutral_position_features(
                child_depth=child_depth,
                world_depth=world_depth,
                device="cpu",
            )
            self.assertEqual(tuple(features.shape), (GATE7_SCALE_NEUTRAL_POSITION_WIDTH,))
            self.assertTrue(bool(torch.isfinite(features).all()))
            self.assertTrue(bool((features <= 1.0 + 1e-6).all()))
            self.assertTrue(bool((features >= -1.0 - 1e-6).all()))

    def test_productive_and_sink_token_semantics(self) -> None:
        productive = encode_gate7_scale_neutral_child_input(
            world_depth=18,
            child_depth=11,
            observed_hint=1,
            branch_action=0,
            sink=False,
            device="cpu",
        )
        sink = encode_gate7_scale_neutral_child_input(
            world_depth=18,
            child_depth=11,
            observed_hint=None,
            branch_action=None,
            sink=True,
            device="cpu",
        )
        self.assertEqual(tuple(productive.shape), (19,))
        self.assertEqual(tuple(sink.shape), (19,))
        hint = slice(13, 16)
        action = slice(16, 19)
        torch.testing.assert_close(productive[hint], torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(productive[action], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(sink[hint], torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(sink[action], torch.tensor([0.0, 0.0, 1.0]))

    def test_encoder_signature_exposes_no_population_or_routing_feature(self) -> None:
        parameters = set(inspect.signature(encode_gate7_scale_neutral_child_input).parameters)
        for forbidden in ("population", "reserve_capacity", "k", "slot", "hidden_answer"):
            self.assertNotIn(forbidden, parameters)

    def test_recurrent_shapes(self) -> None:
        model = Gate7ScaleNeutralScorer()
        state = model.initial_state(4, device="cpu")
        inputs = torch.stack(
            [
                encode_gate7_scale_neutral_child_input(
                    world_depth=18,
                    child_depth=3,
                    observed_hint=index % 2,
                    branch_action=(index + 1) % 2,
                    sink=False,
                    device="cpu",
                )
                for index in range(4)
            ]
        )
        advanced = model.advance(state, inputs, repeats=8)
        scores = model.score(advanced)
        self.assertEqual(tuple(advanced.shape), (4, 64))
        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(bool(torch.isfinite(advanced).all()))
        self.assertTrue(bool(torch.isfinite(scores).all()))


if __name__ == "__main__":
    unittest.main()
