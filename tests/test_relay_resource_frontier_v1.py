from __future__ import annotations

import copy
import unittest

import torch

from ai_hypothesis.population_compute.collective_relay import RELAY_DIFFICULTIES
from ai_hypothesis.population_compute.relay_model import RelayPopulationConfig, RelayPopulationModel
from ai_hypothesis.population_compute.relay_resource_audit_v1 import audit_relay_resource_result_v1
from ai_hypothesis.population_compute.relay_resource_frontier_v1 import (
    CORRECTNESS_POLICY_V1,
    FP64_EQUIVALENCE_ATOL,
    FP64_EQUIVALENCE_RTOL,
    RESOURCE_FRONTIER_V1_VERSION,
    RelayResourceBenchmarkConfigV1,
    _measurement_rotation_v1,
    benchmark_relay_resource_frontier_v1,
)


class RelayResourceFrontierV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        self.model = RelayPopulationModel(
            RelayPopulationConfig(state_width=8, message_width=4)
        )
        self.difficulty = RELAY_DIFFICULTIES[0]

    def test_precision_aware_frontier_runs_correctness_before_timing_contract(self) -> None:
        model64 = copy.deepcopy(self.model)
        config = RelayResourceBenchmarkConfigV1(
            population_sizes=(1, 4),
            batch_sizes=(1, 2),
            warmup_iterations=0,
            measured_iterations=1,
            world_seed=1200,
        )
        fingerprint = self.model.parameter_fingerprint()

        result = benchmark_relay_resource_frontier_v1(
            self.model,
            model64,
            difficulties=(self.difficulty,),
            config=config,
            device="cpu",
        )
        payload = result.to_dict()

        self.assertEqual(payload["benchmark_version"], RESOURCE_FRONTIER_V1_VERSION)
        self.assertEqual(payload["correctness_policy"]["name"], CORRECTNESS_POLICY_V1)
        self.assertEqual(
            payload["correctness_policy"]["fp64_equivalence_rtol"],
            FP64_EQUIVALENCE_RTOL,
        )
        self.assertEqual(
            payload["correctness_policy"]["fp64_equivalence_atol"],
            FP64_EQUIVALENCE_ATOL,
        )
        self.assertFalse(payload["correctness_policy"]["fp32_tensor_allclose_is_gate"])
        self.assertTrue(payload["correctness_policy"]["complete_matrix_preflight_before_timing"])
        self.assertTrue(payload["correctness_policy"]["fp64_model_offloaded_before_timing"])
        self.assertEqual(len(payload["comparisons"]), 4)
        for row in payload["comparisons"]:
            correctness = row["correctness_v1"]
            self.assertTrue(correctness["admissible"])
            self.assertTrue(correctness["fp32_pairwise_decoded_equal"])
            self.assertTrue(correctness["fp64_pairwise_decoded_equal"])
            self.assertTrue(correctness["fp32_vs_fp64_decoded_equal"])
            self.assertTrue(correctness["fp64_pairwise_tensors_close"])
            self.assertGreaterEqual(correctness["max_abs_fp32_logits_difference"], 0.0)
            self.assertGreaterEqual(correctness["max_abs_fp64_logits_difference"], 0.0)
            self.assertTrue(row["recurrent_worker_updates_equal"])
            self.assertTrue(row["parallel_cached_static_projection_work_equal"])
        self.assertIn(
            "complete frozen FP32+FP64 matrix passed before timing",
            payload["provenance"]["correctness_preflight"],
        )
        self.assertEqual(self.model.parameter_fingerprint(), fingerprint)
        self.assertEqual(next(model64.parameters()).device.type, "cpu")
        self.assertEqual(next(model64.parameters()).dtype, torch.float64)

    def test_v1_rotation_exercises_all_orders_on_frozen_matrix(self) -> None:
        rotations = {
            _measurement_rotation_v1(
                difficulty=difficulty.name,
                active_workers=active_workers,
                batch_size=batch_size,
                relay_hops=difficulty.hop_count,
            )
            for difficulty in RELAY_DIFFICULTIES
            for active_workers in (1, 4, 16, 64, 256)
            for batch_size in (1, 64)
        }
        self.assertEqual(rotations, {0, 1, 2})

    def test_v1_auditor_rejects_noncanonical_partial_result(self) -> None:
        model64 = copy.deepcopy(self.model)
        config = RelayResourceBenchmarkConfigV1(
            population_sizes=(1,),
            batch_sizes=(1,),
            warmup_iterations=0,
            measured_iterations=1,
            world_seed=0,
        )
        payload = benchmark_relay_resource_frontier_v1(
            self.model,
            model64,
            difficulties=(self.difficulty,),
            config=config,
            device="cpu",
        ).to_dict()
        payload["checkpoint"] = {
            "parameter_fingerprint": payload["parameter_fingerprint"],
        }
        audit = audit_relay_resource_result_v1(
            payload,
            require_cuda=False,
            require_canonical_checkpoint=False,
        )
        self.assertFalse(audit.protocol_valid)
        self.assertTrue(any("population_sizes" in reason for reason in audit.reasons))
        self.assertTrue(any("expected 30 frozen conditions" in reason for reason in audit.reasons))


if __name__ == "__main__":
    unittest.main()
