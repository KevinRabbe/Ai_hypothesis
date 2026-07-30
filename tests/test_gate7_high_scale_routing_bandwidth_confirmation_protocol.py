from __future__ import annotations

import inspect
import unittest

from ai_hypothesis.population_compute import (
    gate7_high_scale_routing_bandwidth_confirmation_protocol as protocol,
)


class Gate7HighScaleRoutingBandwidthConfirmationProtocolTests(unittest.TestCase):
    @staticmethod
    def _lows(k: int, *, learned_low: float, global_low: float) -> dict[str, float]:
        result: dict[str, float] = {}
        for checkpoint in protocol.GATE7_CONFIRMATION_CHECKPOINT_INDICES:
            result[f"c{checkpoint}_k{k}_score_vs_hash"] = learned_low
            result[f"c{checkpoint}_k{k}_score_vs_global"] = global_low
        return result

    @classmethod
    def _frontier_lows(
        cls,
        *,
        default_learned_low: float = 0.01,
        default_global_low: float = -0.05,
        passing_k: tuple[int, ...] = (),
    ) -> dict[int, dict[str, float]]:
        return {
            k: cls._lows(
                k,
                learned_low=default_learned_low,
                global_low=-0.049 if k in passing_k else default_global_low,
            )
            for k in protocol.GATE7_CONFIRMATION_K_LADDER
        }

    def test_protocol_is_frozen_and_confirmation_execution_remains_closed(self) -> None:
        self.assertTrue(protocol.GATE7_CONFIRMATION_PROTOCOL_FROZEN)
        self.assertFalse(protocol.GATE7_CONFIRMATION_EXECUTION_OPENED)
        self.assertEqual(
            protocol.GATE7_CONFIRMATION_SCREENING_RESULT_HEAD,
            "07b6397f2a9d4f71ed789d6c7011e12b4cbf90e0",
        )
        self.assertEqual(
            protocol.GATE7_CONFIRMATION_SCREENING_RESULT_SHA256,
            "d76c8b0753a518b4c61b3ff42c1f3e85902e2e492342f23fa6706459ee13a9b5",
        )
        self.assertEqual(
            protocol.GATE7_CONFIRMATION_SCREENING_AUDIT_SHA256,
            "7352621ef5c5199cba98070e2f2511674bd2f4aba8b20b48c0ec87436c5204d5",
        )
        self.assertEqual(
            protocol.GATE7_CONFIRMATION_SCREENING_OUTCOME,
            "G7_ROUTING_BANDWIDTH_FRONTIER_REACHED",
        )

    def test_exact_checkpoint_family_remains_bound(self) -> None:
        self.assertEqual(
            [protocol.GATE7_CONFIRMATION_CHECKPOINTS[index]["sha256"] for index in (0, 1, 2)],
            [
                "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
                "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
                "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
            ],
        )
        self.assertEqual(protocol.GATE7_CONFIRMATION_LEARNED_PARAMETER_COUNT, 19_649)

    def test_confirmation_matrix_is_fixed_before_exposure(self) -> None:
        self.assertEqual(protocol.GATE7_CONFIRMATION_POPULATIONS, (4096, 8192))
        self.assertEqual(protocol.GATE7_CONFIRMATION_WORLD_COUNT, 512)
        self.assertEqual(protocol.GATE7_CONFIRMATION_EVALUATION_BATCH_SIZE, 64)
        self.assertEqual(protocol.GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES, 10_000)
        self.assertEqual(protocol.GATE7_CONFIRMATION_NONINFERIORITY_MARGIN, 0.05)

        anchor, frontier = protocol.prepared_confirmation_tiers()
        self.assertEqual(anchor.population, 4096)
        self.assertEqual(anchor.k_values, (512,))
        self.assertEqual(
            anchor.conditions,
            (
                "global_score",
                "global_hash",
                "bounded_score_k512",
                "bounded_hash_k512",
            ),
        )
        self.assertEqual(frontier.population, 8192)
        self.assertEqual(frontier.k_values, (16, 32, 64, 128, 256, 512))
        self.assertEqual(len(frontier.conditions), 14)
        self.assertEqual(frontier.conditions[:2], ("global_score", "global_hash"))
        self.assertEqual(frontier.conditions[-2:], ("bounded_score_k512", "bounded_hash_k512"))

    def test_work_identity_and_physical_batches_are_frozen(self) -> None:
        for tier in protocol.prepared_confirmation_tiers():
            self.assertEqual(tier.world_count // tier.evaluation_batch_size, 8)
            self.assertEqual(tier.stage_a_parent_slots, tier.population - 1)
            self.assertEqual(tier.stage_b_parent_slots, 128)
            self.assertEqual(
                tier.logical_learned_updates_per_world,
                (tier.population - 1 + 128) * 16,
            )

    def test_frontier_confirmation_requires_anchor_and_no_passing_n8192_k(self) -> None:
        result = protocol.classify_confirmation(
            anchor_reference_viable=True,
            anchor_k512_primary_ci_lows=self._lows(
                512,
                learned_low=0.01,
                global_low=-0.049,
            ),
            frontier_reference_viable=True,
            frontier_primary_ci_lows_by_k=self._frontier_lows(),
        )
        self.assertEqual(result.outcome, protocol.GATE7_CONFIRMATION_FRONTIER_CONFIRMED)
        self.assertTrue(result.anchor_k512_passed)
        self.assertEqual(result.passing_k_at_n8192, ())
        self.assertIsNone(result.smallest_passing_k_at_n8192)

    def test_any_fixed_passing_n8192_k_prevents_frontier_confirmation(self) -> None:
        result = protocol.classify_confirmation(
            anchor_reference_viable=True,
            anchor_k512_primary_ci_lows=self._lows(
                512,
                learned_low=0.01,
                global_low=-0.049,
            ),
            frontier_reference_viable=True,
            frontier_primary_ci_lows_by_k=self._frontier_lows(passing_k=(128, 512)),
        )
        self.assertEqual(result.outcome, protocol.GATE7_CONFIRMATION_FRONTIER_NOT_CONFIRMED)
        self.assertEqual(result.passing_k_at_n8192, (128, 512))
        self.assertEqual(result.smallest_passing_k_at_n8192, 128)

    def test_outcome_hierarchy_never_converts_anchor_failure_into_confirmation(self) -> None:
        passing_anchor = self._lows(512, learned_low=0.01, global_low=-0.049)
        failing_anchor = self._lows(512, learned_low=0.01, global_low=-0.05)
        frontier = self._frontier_lows()

        anchor_reference_failure = protocol.classify_confirmation(
            anchor_reference_viable=False,
            anchor_k512_primary_ci_lows=passing_anchor,
            frontier_reference_viable=True,
            frontier_primary_ci_lows_by_k=frontier,
        )
        self.assertEqual(
            anchor_reference_failure.outcome,
            protocol.GATE7_CONFIRMATION_ANCHOR_REFERENCE_NOT_REPLICATED,
        )

        anchor_k_failure = protocol.classify_confirmation(
            anchor_reference_viable=True,
            anchor_k512_primary_ci_lows=failing_anchor,
            frontier_reference_viable=True,
            frontier_primary_ci_lows_by_k=frontier,
        )
        self.assertEqual(
            anchor_k_failure.outcome,
            protocol.GATE7_CONFIRMATION_ANCHOR_K_NOT_REPLICATED,
        )

        frontier_reference_failure = protocol.classify_confirmation(
            anchor_reference_viable=True,
            anchor_k512_primary_ci_lows=passing_anchor,
            frontier_reference_viable=False,
            frontier_primary_ci_lows_by_k=frontier,
        )
        self.assertEqual(
            frontier_reference_failure.outcome,
            protocol.GATE7_CONFIRMATION_FRONTIER_REFERENCE_NOT_REPLICATED,
        )

    def test_complete_n8192_ladder_is_required_in_frozen_order(self) -> None:
        incomplete = self._frontier_lows()
        del incomplete[64]
        with self.assertRaisesRegex(ValueError, "complete frozen K ladder"):
            protocol.classify_confirmation(
                anchor_reference_viable=True,
                anchor_k512_primary_ci_lows=self._lows(
                    512,
                    learned_low=0.01,
                    global_low=-0.049,
                ),
                frontier_reference_viable=True,
                frontier_primary_ci_lows_by_k=incomplete,
            )

    def test_protocol_module_contains_no_confirmation_execution_surface(self) -> None:
        source = inspect.getsource(protocol)
        self.assertNotIn("import torch", source)
        self.assertNotIn("generate_gate7", source)
        self.assertNotIn("run_gate7", source)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("hidden_path", source)


if __name__ == "__main__":
    unittest.main()
