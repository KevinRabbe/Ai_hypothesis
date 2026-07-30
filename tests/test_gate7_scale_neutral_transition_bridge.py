from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute import analyze_gate7_scale_neutral_transition_bridge as audit
from ai_hypothesis.population_compute import run_gate7_scale_neutral_transition_bridge as runner
from ai_hypothesis.population_compute.gate7_scale_neutral_transition_bridge import (
    GATE7_TRANSITION_BRIDGE_EXECUTION_ADMITTED,
    GATE7_TRANSITION_BRIDGE_EXPECTED,
    GATE7_TRANSITION_BRIDGE_HIGH_SCALE_OPENED,
    GATE7_TRANSITION_TRAINING_GIT_HEAD,
    load_verified_gate7_transition_checkpoint,
)


class Gate7ScaleNeutralTransitionBridgeTests(unittest.TestCase):
    def test_exact_transition_checkpoint_identities_are_bound(self) -> None:
        self.assertTrue(GATE7_TRANSITION_BRIDGE_EXECUTION_ADMITTED)
        self.assertFalse(GATE7_TRANSITION_BRIDGE_HIGH_SCALE_OPENED)
        self.assertEqual(
            GATE7_TRANSITION_TRAINING_GIT_HEAD,
            "07307650b2bbbfaa09b80e40caa4419ecdda2947",
        )
        self.assertEqual(
            {index: row["sha256"] for index, row in GATE7_TRANSITION_BRIDGE_EXPECTED.items()},
            {
                0: "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
                1: "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
                2: "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
            },
        )
        self.assertEqual(
            {index: row["fingerprint"] for index, row in GATE7_TRANSITION_BRIDGE_EXPECTED.items()},
            {
                0: "0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa",
                1: "b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46",
                2: "1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb",
            },
        )

    def test_checkpoint_loader_rejects_unbound_bytes_before_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            path.write_bytes(b"not one of the bound transition checkpoints")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                load_verified_gate7_transition_checkpoint(
                    checkpoint_index=0,
                    checkpoint_path=path,
                    device="cpu",
                )

    def test_runner_exposes_only_frozen_low_scale_matrix(self) -> None:
        source = inspect.getsource(runner.run_gate7_scale_neutral_transition_bridge)
        self.assertIn("total_cells = len(GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES) * 7", source)
        self.assertIn("(\"transition\", 128", source)
        self.assertIn("(\"transition\", 256", source)
        self.assertIn("(\"original\", 256", source)
        self.assertNotIn("512", source)
        self.assertNotIn("1024", source)
        self.assertNotIn("train_gate7", source)
        self.assertIn('"high_scale_gate7_opened": False', source)

    def test_auditor_freezes_twenty_one_conditions_and_fifteen_pairs(self) -> None:
        self.assertEqual(audit.EXPECTED_CONDITION_COUNT, 21)
        self.assertEqual(audit.EXPECTED_PAIR_COUNT, 15)
        pair_names = {
            name
            for checkpoint in (0, 1, 2)
            for name in audit._expected_pairs(checkpoint)
        }
        self.assertEqual(len(pair_names), 15)
        self.assertEqual(
            sum(not name.endswith("_n256_k16_vs_global") for name in pair_names),
            12,
        )

    def test_independent_classifier_requires_all_twelve_primary_criteria(self) -> None:
        lows: dict[str, float] = {}
        for checkpoint in (0, 1, 2):
            lows[f"t{checkpoint}_n128_k16_vs_hash"] = 0.01
            lows[f"t{checkpoint}_n256_k16_vs_hash"] = 0.01
            lows[f"t{checkpoint}_n128_k16_vs_global"] = -0.049
            lows[f"t{checkpoint}_n256_transition_global_vs_original_global"] = -0.049
        self.assertEqual(audit._classify(lows), "GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED")
        for key in tuple(lows):
            broken = dict(lows)
            broken[key] = 0.0 if "vs_hash" in key else -0.05
            self.assertEqual(
                audit._classify(broken),
                "GATE7_SCALE_NEUTRAL_TRANSITION_NOT_QUALIFIED",
                key,
            )

    def test_auditor_duplicates_checkpoint_identities_independently(self) -> None:
        self.assertEqual(
            audit.TRANSITION_CHECKPOINTS,
            GATE7_TRANSITION_BRIDGE_EXPECTED,
        )
        source = inspect.getsource(audit)
        self.assertNotIn("from .gate7_scale_neutral_transition_bridge import", source)
        self.assertNotIn("generate_gate7_transition_bridge_world", source)


if __name__ == "__main__":
    unittest.main()
