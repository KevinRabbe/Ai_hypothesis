from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.collective_relay import RELAY_DIFFICULTIES
from ai_hypothesis.population_compute.diagnose_relay_resource_equivalence import diagnose
from ai_hypothesis.population_compute.relay_model import RelayPopulationConfig, RelayPopulationModel
from ai_hypothesis.population_compute.relay_resource_frontier import RelayResourceBenchmarkConfig


class RelayResourceEquivalenceDiagnosticTests(unittest.TestCase):
    def test_diagnostic_scans_conditions_without_timing(self) -> None:
        model = RelayPopulationModel(RelayPopulationConfig(state_width=8, message_width=4))
        config = RelayResourceBenchmarkConfig(
            population_sizes=(1, 4),
            batch_sizes=(1, 2),
            warmup_iterations=0,
            measured_iterations=1,
            world_seed=17,
        )

        payload = diagnose(
            model,
            difficulties=(RELAY_DIFFICULTIES[0],),
            config=config,
            device="cpu",
        )

        self.assertEqual(payload["scientific_status"], "MECHANICS_DIAGNOSTIC_ONLY")
        self.assertEqual(payload["performance_result"], "NOT_MEASURED")
        self.assertEqual(payload["condition_count"], 4)
        self.assertEqual(len(payload["rows"]), 4)
        self.assertEqual(
            {(row["active_workers"], row["batch_size"]) for row in payload["rows"]},
            {(1, 1), (4, 1), (1, 2), (4, 2)},
        )
        self.assertIn("worst_logits_difference", payload)
        self.assertIn("worst_shared_difference", payload)
        for row in payload["rows"]:
            self.assertIn("serial_normalized", row)
            self.assertIn("serial_cached_normalized", row)
            for schedule in ("serial_normalized", "serial_cached_normalized"):
                metrics = row[schedule]
                self.assertIn("logits_close", metrics)
                self.assertIn("shared_close", metrics)
                self.assertIn("decoded_equal", metrics)
                self.assertGreaterEqual(metrics["max_abs_logits_difference"], 0.0)
                self.assertGreaterEqual(metrics["max_abs_shared_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
