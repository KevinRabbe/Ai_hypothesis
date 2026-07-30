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
    encode_gate7_scale_neutral_child_inputs_batch,
    gate7_scale_neutral_position_features,
    gate7_scale_neutral_position_features_batch,
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

    def test_tensorized_position_encoding_matches_frozen_scalar_reference(self) -> None:
        pairs = (
            (6, 1),
            (10, 8),
            (18, 17),
            (1_000, 999),
            (1_000_000, 1_000_000),
        )
        world_depths = torch.tensor([world for world, _child in pairs], dtype=torch.int64)
        child_depths = torch.tensor([child for _world, child in pairs], dtype=torch.int64)
        batched = gate7_scale_neutral_position_features_batch(
            child_depths=child_depths,
            world_depths=world_depths,
        )
        scalar = torch.stack(
            [
                gate7_scale_neutral_position_features(
                    child_depth=child,
                    world_depth=world,
                    device="cpu",
                )
                for world, child in pairs
            ]
        )
        self.assertEqual(tuple(batched.shape), (len(pairs), GATE7_SCALE_NEUTRAL_POSITION_WIDTH))
        torch.testing.assert_close(batched, scalar, rtol=1e-6, atol=3e-7)

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

    def test_tensorized_child_encoder_matches_scalar_reference(self) -> None:
        world_depths = torch.tensor([18, 18, 10, 6], dtype=torch.int64)
        child_depths = torch.tensor([11, 11, 8, 1], dtype=torch.int64)
        observed_hints = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
        branch_actions = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        sink = torch.tensor([False, True, False, True], dtype=torch.bool)
        batched = encode_gate7_scale_neutral_child_inputs_batch(
            world_depths=world_depths,
            child_depths=child_depths,
            observed_hints=observed_hints,
            branch_actions=branch_actions,
            sink=sink,
        )
        scalar = torch.stack(
            [
                encode_gate7_scale_neutral_child_input(
                    world_depth=int(world_depths[index]),
                    child_depth=int(child_depths[index]),
                    observed_hint=None if bool(sink[index]) else int(observed_hints[index]),
                    branch_action=None if bool(sink[index]) else int(branch_actions[index]),
                    sink=bool(sink[index]),
                    device="cpu",
                )
                for index in range(world_depths.shape[0])
            ]
        )
        self.assertEqual(tuple(batched.shape), (4, GATE7_SCALE_NEUTRAL_INPUT_WIDTH))
        torch.testing.assert_close(batched, scalar, rtol=1e-6, atol=3e-7)

    def test_tensorized_encoder_rejects_mismatched_metadata(self) -> None:
        with self.assertRaises(ValueError):
            encode_gate7_scale_neutral_child_inputs_batch(
                world_depths=torch.tensor([18, 18], dtype=torch.int64),
                child_depths=torch.tensor([11], dtype=torch.int64),
                observed_hints=torch.tensor([1, 0], dtype=torch.int64),
                branch_actions=torch.tensor([0, 1], dtype=torch.int64),
                sink=torch.tensor([False, False], dtype=torch.bool),
            )

    def test_encoder_signatures_expose_no_population_or_routing_feature(self) -> None:
        for encoder in (
            encode_gate7_scale_neutral_child_input,
            encode_gate7_scale_neutral_child_inputs_batch,
            gate7_scale_neutral_position_features_batch,
        ):
            parameters = set(inspect.signature(encoder).parameters)
            for forbidden in ("population", "reserve_capacity", "k", "slot", "hidden_answer"):
                self.assertNotIn(forbidden, parameters)

    def test_tensorized_hot_encoders_have_no_cuda_to_python_scalar_extraction(self) -> None:
        for encoder in (
            gate7_scale_neutral_position_features_batch,
            encode_gate7_scale_neutral_child_inputs_batch,
        ):
            source = inspect.getsource(encoder)
            for forbidden in (".item(", ".cpu(", ".tolist(", "float("):
                self.assertNotIn(forbidden, source)

    def test_recurrent_shapes(self) -> None:
        model = Gate7ScaleNeutralScorer()
        state = model.initial_state(4, device="cpu")
        inputs = encode_gate7_scale_neutral_child_inputs_batch(
            world_depths=torch.full((4,), 18, dtype=torch.int64),
            child_depths=torch.full((4,), 3, dtype=torch.int64),
            observed_hints=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            branch_actions=torch.tensor([1, 0, 1, 0], dtype=torch.int64),
            sink=torch.zeros(4, dtype=torch.bool),
        )
        advanced = model.advance(state, inputs, repeats=8)
        scores = model.score(advanced)
        self.assertEqual(tuple(advanced.shape), (4, 64))
        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(bool(torch.isfinite(advanced).all()))
        self.assertTrue(bool(torch.isfinite(scores).all()))


if __name__ == "__main__":
    unittest.main()
