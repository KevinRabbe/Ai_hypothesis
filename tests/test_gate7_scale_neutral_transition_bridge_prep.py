from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import encode_gate3_v1_child_input
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from ai_hypothesis.population_compute.gate7_scale_neutral_model_prep import (
    Gate7ScaleNeutralScorer,
    encode_gate7_scale_neutral_child_inputs_batch,
)
from ai_hypothesis.population_compute.gate7_scale_neutral_transition_bridge_prep import (
    GATE7_TRANSITION_BRIDGE_BATCH_SIZE,
    GATE7_TRANSITION_BRIDGE_BOOTSTRAP_SAMPLES,
    GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES,
    GATE7_TRANSITION_BRIDGE_DEPTH,
    GATE7_TRANSITION_BRIDGE_NONINFERIORITY_MARGIN,
    GATE7_TRANSITION_BRIDGE_POPULATIONS,
    GATE7_TRANSITION_BRIDGE_PREPARATION_ONLY,
    GATE7_TRANSITION_BRIDGE_WORLD_COUNT,
    Gate7ScaleNeutralGate6Adapter,
    classify_gate7_scale_neutral_transition_bridge,
    generate_gate7_transition_bridge_world,
)


class Gate7ScaleNeutralTransitionBridgePreparationTests(unittest.TestCase):
    def test_preparation_only_and_frozen_bridge_constants(self) -> None:
        self.assertTrue(GATE7_TRANSITION_BRIDGE_PREPARATION_ONLY)
        self.assertEqual(GATE7_TRANSITION_BRIDGE_WORLD_COUNT, 256)
        self.assertEqual(GATE7_TRANSITION_BRIDGE_BATCH_SIZE, 64)
        self.assertEqual(GATE7_TRANSITION_BRIDGE_BOOTSTRAP_SAMPLES, 2_000)
        self.assertEqual(GATE7_TRANSITION_BRIDGE_DEPTH, 10)
        self.assertEqual(GATE7_TRANSITION_BRIDGE_POPULATIONS, (128, 256))
        self.assertEqual(GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES, (0, 1, 2))
        self.assertEqual(GATE7_TRANSITION_BRIDGE_NONINFERIORITY_MARGIN, 0.05)

    def test_bridge_world_namespaces_are_fresh_but_not_instantiated(self) -> None:
        source = inspect.getsource(generate_gate7_transition_bridge_world)
        self.assertIn("gate7-scale-neutral-transition-bridge-hidden", source)
        self.assertIn("gate7-scale-neutral-transition-bridge-hints", source)
        self.assertNotIn("gate6-fixed-k-population-scaling-development-hidden", source)
        self.assertNotIn("gate6-fixed-k-population-scaling-development-hints", source)

    def test_adapter_adds_no_learned_parameters(self) -> None:
        scorer = Gate7ScaleNeutralScorer()
        adapter = Gate7ScaleNeutralGate6Adapter(scorer)
        self.assertEqual(scorer.trainable_parameter_count(), 19_649)
        self.assertEqual(adapter.trainable_parameter_count(), 19_649)
        self.assertEqual(sum(p.numel() for p in adapter.parameters() if p.requires_grad), 19_649)

    def test_adapter_matches_direct_scale_neutral_reencoding(self) -> None:
        torch.manual_seed(42)
        scorer = Gate7ScaleNeutralScorer()
        adapter = Gate7ScaleNeutralGate6Adapter(scorer)
        public = Gate3V1PublicWorld(
            seed=12345,
            depth=10,
            noisy_hints=(0, 1, 0, 1, 1, 0, 0, 1, 0, 1),
        )
        depths = (1, 4, 8, 10)
        hints = (0, 1, 1, 1)
        actions = (1, 0, 1, 0)
        old_inputs = torch.stack(
            [
                encode_gate3_v1_child_input(
                    world=public,
                    child_depth=depth,
                    observed_hint=hint,
                    branch_action=action,
                    sink=False,
                    device="cpu",
                )
                for depth, hint, action in zip(depths, hints, actions, strict=True)
            ]
        )
        neutral_inputs = encode_gate7_scale_neutral_child_inputs_batch(
            world_depths=torch.full((4,), 10, dtype=torch.int64),
            child_depths=torch.tensor(depths, dtype=torch.int64),
            observed_hints=torch.tensor(hints, dtype=torch.int64),
            branch_actions=torch.tensor(actions, dtype=torch.int64),
            sink=torch.zeros(4, dtype=torch.bool),
        )
        state = scorer.initial_state(4, device="cpu")
        through_adapter = adapter.advance(state.clone(), old_inputs, repeats=8)
        direct = scorer.advance(state.clone(), neutral_inputs, repeats=8)
        torch.testing.assert_close(through_adapter, direct, rtol=0.0, atol=0.0)
        torch.testing.assert_close(adapter.score(through_adapter), scorer.score(direct), rtol=0.0, atol=0.0)

    @staticmethod
    def _passing_lows() -> dict[str, float]:
        result: dict[str, float] = {}
        for checkpoint in (0, 1, 2):
            result[f"t{checkpoint}_n128_k16_vs_hash"] = 0.01
            result[f"t{checkpoint}_n256_k16_vs_hash"] = 0.01
            result[f"t{checkpoint}_n128_k16_vs_global"] = -0.049
            result[f"t{checkpoint}_n256_transition_global_vs_original_global"] = -0.049
        return result

    def test_classifier_requires_all_twelve_primary_criteria(self) -> None:
        lows = self._passing_lows()
        self.assertEqual(
            classify_gate7_scale_neutral_transition_bridge(lows),
            "GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED",
        )
        for key in tuple(lows):
            broken = dict(lows)
            broken[key] = 0.0 if "vs_hash" in key else -0.05
            self.assertEqual(
                classify_gate7_scale_neutral_transition_bridge(broken),
                "GATE7_SCALE_NEUTRAL_TRANSITION_NOT_QUALIFIED",
                key,
            )

    def test_n256_k16_global_cannot_rescue_or_fail_primary_classifier(self) -> None:
        lows = self._passing_lows()
        lows["t0_n256_k16_vs_global"] = -1.0
        lows["t1_n256_k16_vs_global"] = 1.0
        lows["t2_n256_k16_vs_global"] = -0.5
        self.assertEqual(
            classify_gate7_scale_neutral_transition_bridge(lows),
            "GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED",
        )

    def test_adapter_hot_reencoding_does_not_extract_cuda_values_to_python(self) -> None:
        source = inspect.getsource(Gate7ScaleNeutralGate6Adapter.advance)
        for forbidden in (".item(", ".cpu(", ".tolist(", "float(", "bool("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
