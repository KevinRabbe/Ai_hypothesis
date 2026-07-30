from __future__ import annotations

import inspect
import unittest

from ai_hypothesis.population_compute import gate7_high_scale_routing_bandwidth_protocol as protocol


class Gate7HighScaleRoutingBandwidthProtocolTests(unittest.TestCase):
    def test_protocol_is_frozen_but_execution_remains_closed(self) -> None:
        self.assertTrue(protocol.GATE7_HIGH_SCALE_PROTOCOL_FROZEN)
        self.assertFalse(protocol.GATE7_HIGH_SCALE_EXECUTION_OPENED)
        self.assertEqual(
            protocol.GATE7_HIGH_SCALE_BRIDGE_EVIDENCE_HEAD,
            "0d1bd683bae322a11a76b4d885f2efeb3c4a5099",
        )
        self.assertEqual(
            protocol.GATE7_HIGH_SCALE_BRIDGE_OUTCOME,
            "GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED",
        )
        self.assertEqual(protocol.GATE7_HIGH_SCALE_WORLD_COUNT, 64)
        self.assertEqual(protocol.GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE, 64)
        self.assertEqual(protocol.GATE7_HIGH_SCALE_BOOTSTRAP_SAMPLES, 2_000)
        self.assertEqual(protocol.GATE7_HIGH_SCALE_NONINFERIORITY_MARGIN, 0.05)

    def test_exact_checkpoint_family_is_bound(self) -> None:
        self.assertEqual(
            [protocol.GATE7_HIGH_SCALE_CHECKPOINTS[index]["sha256"] for index in (0, 1, 2)],
            [
                "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
                "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
                "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
            ],
        )
        self.assertEqual(protocol.GATE7_HIGH_SCALE_LEARNED_PARAMETER_COUNT, 19_649)

    def test_geometric_tier_plans_and_work_identity(self) -> None:
        tiers = protocol.prepared_gate7_high_scale_tiers()
        self.assertEqual(tuple(tier.population for tier in tiers), protocol.GATE7_HIGH_SCALE_POPULATIONS)
        first = tiers[0]
        self.assertEqual((first.frontier_depth, first.world_depth), (10, 11))
        self.assertEqual(first.stage_a_parent_slots, 1023)
        self.assertEqual(first.stage_a_learned_updates, 1023 * 16)
        self.assertEqual(first.stage_b_learned_updates, 128 * 16)
        last = tiers[-1]
        self.assertEqual((last.frontier_depth, last.world_depth), (17, 18))
        self.assertEqual(last.stage_a_parent_slots, 131071)
        self.assertEqual(last.total_logical_learned_updates, (131071 + 128) * 16)

    def test_every_k_has_an_exact_matched_hash_control(self) -> None:
        for tier in protocol.prepared_gate7_high_scale_tiers():
            self.assertEqual(tier.full_condition_ladder[:2], ("global_score", "global_hash"))
            for k in tier.k_ladder:
                score = protocol.bounded_score_condition(k)
                matched_hash = protocol.bounded_hash_condition(k)
                self.assertIn(score, tier.full_condition_ladder)
                self.assertEqual(
                    tier.full_condition_ladder.index(matched_hash),
                    tier.full_condition_ladder.index(score) + 1,
                )

    def test_condition_prefix_stops_after_the_requested_hash_pair(self) -> None:
        prefix = protocol.condition_prefix_through_k(1024, 64)
        self.assertEqual(
            prefix,
            (
                "global_score",
                "global_hash",
                "bounded_score_k16",
                "bounded_hash_k16",
                "bounded_score_k32",
                "bounded_hash_k32",
                "bounded_score_k64",
                "bounded_hash_k64",
            ),
        )

    def test_sequential_exposure_rejects_skips_or_reordering(self) -> None:
        protocol.validate_sequential_k_exposure(1024, (16, 32, 64))
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            protocol.validate_sequential_k_exposure(1024, (16, 64))
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            protocol.validate_sequential_k_exposure(1024, (32, 16))

    def test_reference_viability_requires_all_checkpoint_points_and_pooled_ci(self) -> None:
        points = {0: 0.1, 1: 0.05, 2: 0.02}
        self.assertTrue(protocol.reference_is_viable(checkpoint_point_deltas=points, pooled_ci_low=0.01))
        self.assertFalse(
            protocol.reference_is_viable(
                checkpoint_point_deltas={0: 0.1, 1: 0.0, 2: 0.02},
                pooled_ci_low=0.01,
            )
        )
        self.assertFalse(protocol.reference_is_viable(checkpoint_point_deltas=points, pooled_ci_low=0.0))

    @staticmethod
    def _lows(k: int, *, learned_low: float, global_low: float) -> dict[str, float]:
        result: dict[str, float] = {}
        for checkpoint in (0, 1, 2):
            result[f"c{checkpoint}_k{k}_score_vs_hash"] = learned_low
            result[f"c{checkpoint}_k{k}_score_vs_global"] = global_low
        return result

    def test_first_all_checkpoint_pass_selects_k_and_forbids_later_exposure(self) -> None:
        lows = {
            16: self._lows(16, learned_low=0.01, global_low=-0.05),
            32: self._lows(32, learned_low=0.01, global_low=-0.049),
        }
        self.assertEqual(protocol.smallest_passing_k(population=1024, primary_ci_lows_by_k=lows), 32)
        self.assertEqual(
            protocol.classify_completed_tier(
                population=1024,
                reference_viable=True,
                primary_ci_lows_by_k=lows,
            ),
            "G7_K_REQUIRED_32",
        )

        invalid = dict(lows)
        invalid[64] = self._lows(64, learned_low=0.01, global_low=-0.049)
        with self.assertRaisesRegex(ValueError, "after the first"):
            protocol.smallest_passing_k(population=1024, primary_ci_lows_by_k=invalid)

    def test_reference_failure_forbids_k_exposure(self) -> None:
        self.assertEqual(
            protocol.classify_completed_tier(
                population=1024,
                reference_viable=False,
                primary_ci_lows_by_k={},
            ),
            protocol.GATE7_HIGH_SCALE_REFERENCE_FRONTIER_REACHED,
        )
        with self.assertRaisesRegex(ValueError, "after the global reference failed"):
            protocol.classify_completed_tier(
                population=1024,
                reference_viable=False,
                primary_ci_lows_by_k={
                    16: self._lows(16, learned_low=0.01, global_low=-0.049)
                },
            )

    def test_full_ladder_failure_and_campaign_actions_are_frozen(self) -> None:
        failed = {
            k: self._lows(k, learned_low=0.0, global_low=-0.05)
            for k in protocol.GATE7_HIGH_SCALE_K_LADDER
        }
        outcome = protocol.classify_completed_tier(
            population=1024,
            reference_viable=True,
            primary_ci_lows_by_k=failed,
        )
        self.assertEqual(outcome, protocol.GATE7_HIGH_SCALE_ROUTING_FRONTIER_REACHED)
        self.assertEqual(
            protocol.campaign_action_after_tier(population=1024, tier_outcome=outcome),
            protocol.GATE7_HIGH_SCALE_ROUTING_FRONTIER_REACHED,
        )
        self.assertEqual(
            protocol.campaign_action_after_tier(
                population=1024,
                tier_outcome="G7_K_REQUIRED_16",
            ),
            protocol.GATE7_HIGH_SCALE_CONTINUE,
        )
        self.assertEqual(
            protocol.campaign_action_after_tier(
                population=131072,
                tier_outcome="G7_K_REQUIRED_512",
            ),
            protocol.GATE7_HIGH_SCALE_CAMPAIGN_CEILING_REACHED,
        )

    def test_protocol_module_contains_no_scientific_execution_surface(self) -> None:
        source = inspect.getsource(protocol)
        self.assertNotIn("import torch", source)
        self.assertNotIn("generate_gate7_high_scale_world", source)
        self.assertNotIn("run_gate7_high_scale", source)
        self.assertNotIn("torch.load", source)


if __name__ == "__main__":
    unittest.main()
