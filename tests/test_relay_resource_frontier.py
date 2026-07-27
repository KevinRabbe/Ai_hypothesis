from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute.collective_relay import RELAY_DIFFICULTIES, generate_relay_dataset
from ai_hypothesis.population_compute.relay_experiment_v1 import (
    RelayTrainingConfigV1,
    RelayTrainingSummaryV1,
    save_relay_checkpoint_v1,
)
from ai_hypothesis.population_compute.relay_model import (
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
)
from ai_hypothesis.population_compute.relay_resource_frontier import (
    RESOURCE_FRONTIER_VERSION,
    RelayResourceBenchmarkConfig,
    benchmark_relay_resource_condition,
    benchmark_relay_resource_frontier,
)
from ai_hypothesis.population_compute.run_relay_resource_frontier import main as run_frontier


class RelayResourceFrontierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = RelayPopulationModel(
            RelayPopulationConfig(state_width=8, message_width=4)
        )
        self.difficulty = RELAY_DIFFICULTIES[0]

    def test_condition_preserves_equivalence_and_exposes_serial_tradeoff(self) -> None:
        worlds = generate_relay_dataset(
            start_seed=1200,
            world_count=2,
            difficulty=self.difficulty,
        )
        batch = build_relay_tensor_batch(worlds, active_workers=4)
        config = RelayResourceBenchmarkConfig(
            population_sizes=(1, 4),
            batch_sizes=(1, 2),
            warmup_iterations=0,
            measured_iterations=1,
            world_seed=1200,
        )
        fingerprint = self.model.parameter_fingerprint()

        comparison = benchmark_relay_resource_condition(
            self.model,
            batch,
            difficulty=self.difficulty,
            config=config,
        )

        self.assertTrue(comparison.outputs_equivalent)
        self.assertTrue(comparison.decoded_predictions_equal)
        self.assertTrue(comparison.recurrent_worker_updates_equal)
        self.assertTrue(comparison.parallel_cached_static_projection_work_equal)
        self.assertEqual(comparison.parallel.worker_updates_per_sample, 4 * 2)
        self.assertEqual(comparison.serial_low_memory.worker_updates_per_sample, 4 * 2)
        self.assertEqual(comparison.serial_cached.worker_updates_per_sample, 4 * 2)
        self.assertEqual(comparison.parallel_learned_span_proxy, 2)
        self.assertEqual(comparison.serial_low_memory_learned_span_proxy, 8)
        self.assertEqual(comparison.serial_cached_learned_span_proxy, 8)
        self.assertEqual(comparison.parallel.peak_active_neural_states_per_sample, 4)
        self.assertEqual(comparison.serial_low_memory.peak_active_neural_states_per_sample, 1)
        self.assertEqual(comparison.serial_cached.peak_active_neural_states_per_sample, 1)
        self.assertEqual(comparison.serial_low_memory.communicated_scalars_per_sample, 0)
        self.assertEqual(comparison.serial_cached.communicated_scalars_per_sample, 0)
        self.assertEqual(
            comparison.parallel.static_projection_evaluations_per_sample,
            comparison.serial_cached.static_projection_evaluations_per_sample,
        )
        self.assertGreater(
            comparison.serial_low_memory.static_projection_evaluations_per_sample,
            comparison.serial_cached.static_projection_evaluations_per_sample,
        )
        self.assertEqual(comparison.serial_cached.cached_state_vectors_per_sample, 4)
        self.assertEqual(comparison.serial_cached.cached_message_vectors_per_sample, 4)
        self.assertGreater(comparison.parallel.median_batch_latency_ms, 0.0)
        self.assertGreater(comparison.serial_low_memory.median_batch_latency_ms, 0.0)
        self.assertGreater(comparison.serial_cached.median_batch_latency_ms, 0.0)
        self.assertGreater(comparison.low_memory_serial_over_parallel_latency_speedup, 0.0)
        self.assertGreater(comparison.cached_serial_over_parallel_latency_speedup, 0.0)
        self.assertIsNone(comparison.parallel.device_median_latency_ms)
        self.assertEqual(
            set(comparison.measurement_order),
            {"parallel_normalized", "serial_normalized", "serial_cached_normalized"},
        )
        self.assertEqual(self.model.parameter_fingerprint(), fingerprint)

    def test_frontier_runs_multiple_widths_and_batches_without_mutating_model(self) -> None:
        config = RelayResourceBenchmarkConfig(
            population_sizes=(1, 4),
            batch_sizes=(1, 2),
            warmup_iterations=0,
            measured_iterations=1,
            world_seed=42,
        )
        fingerprint = self.model.parameter_fingerprint()
        result = benchmark_relay_resource_frontier(
            self.model,
            difficulties=(self.difficulty,),
            config=config,
            device="cpu",
        )

        self.assertEqual(result.benchmark_version, RESOURCE_FRONTIER_VERSION)
        self.assertEqual(len(result.comparisons), 4)
        self.assertEqual(
            {(row.active_workers, row.batch_size) for row in result.comparisons},
            {(1, 1), (4, 1), (1, 2), (4, 2)},
        )
        self.assertTrue(all(row.recurrent_worker_updates_equal for row in result.comparisons))
        self.assertTrue(
            all(row.parallel_cached_static_projection_work_equal for row in result.comparisons)
        )
        self.assertEqual(result.provenance["execution_mode"], "eager")
        self.assertEqual(result.provenance["device_type"], "cpu")
        self.assertIn("rotates deterministically", result.provenance["schedule_timing_policy"])
        self.assertEqual(self.model.parameter_fingerprint(), fingerprint)

    def test_config_rejects_ambiguous_population_or_batch_lists(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            RelayResourceBenchmarkConfig(population_sizes=(1, 4, 4)).validate()
        with self.assertRaisesRegex(ValueError, "batch_sizes must be unique"):
            RelayResourceBenchmarkConfig(batch_sizes=(1, 1)).validate()
        with self.assertRaisesRegex(ValueError, "measured_iterations must be positive"):
            RelayResourceBenchmarkConfig(measured_iterations=0).validate()

    def test_checkpoint_only_cli_writes_resource_artifact_without_training(self) -> None:
        config = RelayTrainingConfigV1(
            steps=1,
            batch_size=1,
            model=self.model.config,
        )
        summary = RelayTrainingSummaryV1(
            training_seed=99,
            steps=1,
            examples_seen=1,
            initial_total_loss=1.0,
            final_total_loss=1.0,
            mean_last_50_total_loss=1.0,
            final_relay_loss=1.0,
            final_gate_loss=0.0,
            learned_parameter_count=self.model.trainable_parameter_count(),
            parameter_fingerprint=self.model.parameter_fingerprint(),
        )

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            checkpoint = save_relay_checkpoint_v1(
                self.model,
                summary,
                config,
                root / "model-v1.pt",
            )
            output = root / "frontier.json"
            exit_code = run_frontier(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cpu",
                    "--population-sizes",
                    "1",
                    "--batch-sizes",
                    "1",
                    "--difficulties",
                    "relay-2",
                    "--warmup-iterations",
                    "0",
                    "--measured-iterations",
                    "1",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark_version"], RESOURCE_FRONTIER_VERSION)
            self.assertEqual(payload["checkpoint"]["training_seed"], 99)
            self.assertEqual(len(payload["comparisons"]), 1)
            row = payload["comparisons"][0]
            self.assertTrue(row["outputs_equivalent"])
            self.assertTrue(row["recurrent_worker_updates_equal"])
            self.assertTrue(row["parallel_cached_static_projection_work_equal"])
            self.assertIn("serial_low_memory", row)
            self.assertIn("serial_cached", row)
            self.assertEqual(
                row["speedup_definition"],
                "serial_median_latency_divided_by_parallel_median_latency",
            )


if __name__ == "__main__":
    unittest.main()
