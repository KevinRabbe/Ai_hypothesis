from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.collective_relay import RELAY_DIFFICULTIES
from ai_hypothesis.population_compute.diagnose_relay_resource_precision import diagnose_precision
from ai_hypothesis.population_compute.relay_model import RelayPopulationConfig, RelayPopulationModel
from ai_hypothesis.population_compute.relay_resource_frontier import RelayResourceBenchmarkConfig


class RelayPrecisionDiagnosticTests(unittest.TestCase):
    def test_precision_diagnostic_is_untimed_and_preserves_decoded_outputs(self) -> None:
        torch.manual_seed(23)
        model32 = RelayPopulationModel(RelayPopulationConfig(state_width=8, message_width=4))
        model64 = RelayPopulationModel(RelayPopulationConfig(state_width=8, message_width=4))
        model64.load_state_dict(model32.state_dict())
        config = RelayResourceBenchmarkConfig(
            population_sizes=(1, 4),
            batch_sizes=(1, 2),
            warmup_iterations=0,
            measured_iterations=1,
            world_seed=44,
        )

        payload = diagnose_precision(
            model32,
            model64,
            difficulties=(RELAY_DIFFICULTIES[0],),
            config=config,
            device="cpu",
        )

        self.assertEqual(payload["scientific_status"], "MECHANICS_DIAGNOSTIC_ONLY")
        self.assertEqual(payload["performance_result"], "NOT_MEASURED")
        self.assertEqual(payload["condition_count"], 4)
        self.assertTrue(payload["float32_pairwise_decoded_equal"])
        self.assertTrue(payload["float64_pairwise_decoded_equal"])
        self.assertTrue(payload["float32_vs_float64_decoded_equal"])
        self.assertEqual(len(payload["rows"]), 4)
        self.assertIn("worst_float32_pair_logits_difference", payload)
        self.assertIn("worst_float64_pair_logits_difference", payload)
        self.assertIn("worst_float32_vs_float64_logits_difference", payload)


if __name__ == "__main__":
    unittest.main()
