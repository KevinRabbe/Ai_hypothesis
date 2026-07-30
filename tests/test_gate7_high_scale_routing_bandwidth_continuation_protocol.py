from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ai_hypothesis"
    / "population_compute"
    / "gate7_high_scale_routing_bandwidth_continuation_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("gate7_continuation_protocol_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


class Gate7HighScaleRoutingBandwidthContinuationProtocolTests(unittest.TestCase):
    @staticmethod
    def _lows(k: int, *, learned_low: float, global_low: float) -> dict[str, float]:
        result: dict[str, float] = {}
        for checkpoint in protocol.GATE7_CONTINUATION_CHECKPOINT_INDICES:
            result[f"c{checkpoint}_k{k}_score_vs_hash"] = learned_low
            result[f"c{checkpoint}_k{k}_score_vs_global"] = global_low
        return result

    @classmethod
    def _ladder(
        cls,
        *,
        passing_k: tuple[int, ...] = (),
        learned_low: float = 0.01,
    ) -> dict[int, dict[str, float]]:
        return {
            k: cls._lows(
                k,
                learned_low=learned_low,
                global_low=-0.049 if k in passing_k else -0.05,
            )
            for k in protocol.GATE7_CONTINUATION_K_LADDER
        }

    def test_protocol_binds_the_valid_confirmation_result(self) -> None:
        self.assertTrue(protocol.GATE7_CONTINUATION_PROTOCOL_FROZEN)
        self.assertFalse(protocol.GATE7_CONTINUATION_EXECUTION_OPENED)
        self.assertEqual(
            protocol.GATE7_CONTINUATION_CONFIRMATION_RESULT_HEAD,
            "ae8bd8544a03e48f4f397d2ca5ae933d9247e430",
        )
        self.assertEqual(
            protocol.GATE7_CONTINUATION_CONFIRMATION_RESULT_SHA256,
            "725e3749ba5fed7cdcbb6d61df81bcc77a7b69bacfdc82d553efb06f5ff888da",
        )
        self.assertEqual(
            protocol.GATE7_CONTINUATION_CONFIRMATION_AUDIT_SHA256,
            "27a46ba0feccf6b3322885334819e0e7a07bb02be930122eb1f063c65d69fb99",
        )
        self.assertEqual(
            protocol.GATE7_CONTINUATION_CONFIRMATION_OUTCOME,
            "G7_ROUTING_BANDWIDTH_FRONTIER_NOT_CONFIRMED",
        )
        self.assertEqual(protocol.GATE7_CONTINUATION_CONFIRMED_N8192_PASSING_K, (256, 512))
        self.assertEqual(protocol.GATE7_CONTINUATION_CONFIRMED_N8192_K_REQUIRED, 256)

    def test_exact_checkpoint_family_remains_bound(self) -> None:
        self.assertEqual(
            [protocol.GATE7_CONTINUATION_CHECKPOINTS[index]["sha256"] for index in (0, 1, 2)],
            [
                "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
                "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
                "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
            ],
        )
        self.assertEqual(protocol.GATE7_CONTINUATION_LEARNED_PARAMETER_COUNT, 19_649)

    def test_complete_remaining_population_and_condition_matrix_is_frozen(self) -> None:
        self.assertEqual(
            protocol.GATE7_CONTINUATION_POPULATIONS,
            (16_384, 32_768, 65_536, 131_072),
        )
        self.assertEqual(protocol.GATE7_CONTINUATION_K_LADDER, (16, 32, 64, 128, 256, 512))
        self.assertEqual(protocol.GATE7_CONTINUATION_WORLD_COUNT, 512)
        self.assertEqual(protocol.GATE7_CONTINUATION_EVALUATION_BATCH_SIZE, 64)
        self.assertEqual(protocol.GATE7_CONTINUATION_BOOTSTRAP_SAMPLES, 10_000)
        self.assertEqual(len(protocol.complete_condition_matrix()), 14)
        for tier in protocol.prepared_continuation_tiers():
            self.assertEqual(tier.k_values, protocol.GATE7_CONTINUATION_K_LADDER)
            self.assertEqual(tier.conditions, protocol.complete_condition_matrix())
            self.assertEqual(tier.world_count // tier.evaluation_batch_size, 8)
            self.assertEqual(tier.stage_a_parent_slots, tier.population - 1)
            self.assertEqual(tier.stage_b_parent_slots, 128)
            self.assertEqual(
                tier.logical_learned_updates_per_world,
                (tier.population - 1 + 128) * 16,
            )

    def test_classifier_preserves_complete_nonmonotonic_passing_set(self) -> None:
        result = protocol.classify_continuation_tier(
            population=16_384,
            reference_viable=True,
            primary_ci_lows_by_k=self._ladder(passing_k=(64, 256)),
        )
        self.assertEqual(result.outcome, protocol.GATE7_CONTINUATION_TIER_K_REQUIRED)
        self.assertEqual(result.passing_k, (64, 256))
        self.assertEqual(result.smallest_passing_k, 64)
        self.assertEqual(result.smallest_passing_k_over_n, 64 / 16_384)

    def test_no_pass_and_reference_failure_are_tier_results_not_campaign_stops(self) -> None:
        no_pass = protocol.classify_continuation_tier(
            population=32_768,
            reference_viable=True,
            primary_ci_lows_by_k=self._ladder(),
        )
        self.assertEqual(no_pass.outcome, protocol.GATE7_CONTINUATION_TIER_NO_K_LE_512)
        self.assertEqual(no_pass.passing_k, ())

        reference_failure = protocol.classify_continuation_tier(
            population=65_536,
            reference_viable=False,
            primary_ci_lows_by_k=self._ladder(passing_k=(256,)),
        )
        self.assertEqual(
            reference_failure.outcome,
            protocol.GATE7_CONTINUATION_TIER_REFERENCE_NOT_VIABLE,
        )
        self.assertEqual(reference_failure.passing_k, (256,))

        self.assertEqual(
            protocol.action_after_continuation_tier(population=32_768),
            protocol.GATE7_CONTINUATION_CONTINUE,
        )
        self.assertEqual(
            protocol.action_after_continuation_tier(population=65_536),
            protocol.GATE7_CONTINUATION_CONTINUE,
        )
        self.assertEqual(
            protocol.action_after_continuation_tier(population=131_072),
            protocol.GATE7_CONTINUATION_COMPLETE,
        )

    def test_campaign_requires_full_ladder_unless_next_tier_hits_resource_frontier(self) -> None:
        self.assertEqual(
            protocol.classify_continuation_campaign(
                completed_populations=protocol.GATE7_CONTINUATION_POPULATIONS,
                resource_frontier_population=None,
            ),
            protocol.GATE7_CONTINUATION_COMPLETE,
        )
        self.assertEqual(
            protocol.classify_continuation_campaign(
                completed_populations=(16_384, 32_768),
                resource_frontier_population=65_536,
            ),
            protocol.GATE7_CONTINUATION_RESOURCE_FRONTIER_REACHED,
        )
        with self.assertRaisesRegex(ValueError, "cannot stop before N131072"):
            protocol.classify_continuation_campaign(
                completed_populations=(16_384, 32_768),
                resource_frontier_population=None,
            )
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            protocol.classify_continuation_campaign(
                completed_populations=(16_384, 65_536),
                resource_frontier_population=None,
            )

    def test_complete_k_ladder_is_required_before_tier_classification(self) -> None:
        incomplete = self._ladder()
        del incomplete[128]
        with self.assertRaisesRegex(ValueError, "complete frozen K ladder"):
            protocol.classify_continuation_tier(
                population=16_384,
                reference_viable=True,
                primary_ci_lows_by_k=incomplete,
            )

    def test_protocol_module_contains_no_execution_or_world_surface(self) -> None:
        source = inspect.getsource(protocol)
        self.assertNotIn("import torch", source)
        self.assertNotIn("from torch", source)
        self.assertNotIn("generate_gate7", source)
        self.assertNotIn("run_gate7", source)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("hidden_path", source)


if __name__ == "__main__":
    unittest.main()
