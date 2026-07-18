from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.dashboard.indexer import DashboardIndexer


def step01_payload() -> dict:
    validation = {
        "count": 10,
        "loss": 0.2,
        "accuracy": 0.94,
        "macro_task_accuracy": 0.94,
        "invalid_output_rate": 0.0,
        "uncertainty_precision": 0.8,
        "uncertainty_recall": 0.7,
        "by_task": {"A_PATTERN": 0.9},
        "by_difficulty": {"easy": 1.0},
        "by_task_difficulty": {"A_PATTERN/easy": 1.0},
    }
    return {
        "experiment_name": "step01_test",
        "architecture_version": "step01-unit-v0",
        "benchmark_version": "step01-v0",
        "git_revision": "abc123",
        "device": "cuda",
        "parameter_count": 50268,
        "unit_config": {
            "d_model": 64,
            "block_count": 2,
            "attention_heads": 4,
            "feed_forward_width": 128,
            "dropout": 0.1,
            "sequence_length": 32,
            "feature_width": 16,
        },
        "train_config": {
            "seed": 1,
            "train_count": 100,
            "validation_count": 10,
            "test_count": 10,
            "batch_size": 4,
            "max_training_steps": 2,
            "eval_interval": 1,
            "output_dir": "results/step01/test/seed_1",
        },
        "best_step": 1,
        "best_validation_score": 0.94,
        "training_duration_seconds": 12.5,
        "checkpoint_size_bytes": 100,
        "validation_history": [
            {"step": 1, "train_loss": {"total_loss": 1.0}, "validation": validation},
            {
                "step": 2,
                "train_loss": {"total_loss": 0.8},
                "validation": {**validation, "accuracy": 0.93, "macro_task_accuracy": 0.93},
            },
        ],
        "test": {**validation, "accuracy": 0.95, "macro_task_accuracy": 0.95},
        "deterministic_baselines": {},
        "inference": {
            "1": {
                "batch_width": 1,
                "timed_runs": 1,
                "batch_latency_ms": 2.0,
                "unit_evaluations_per_second": 500.0,
            }
        },
    }


class DashboardStep01AdapterTests(unittest.TestCase):
    def test_step01_result_normalizes_best_and_latest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "step01" / "run" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(step01_payload()), encoding="utf-8")

            snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.indexed_experiment_count, 1)
            run = snapshot.experiments[0]
            self.assertEqual(run.training.best_validation_status, "AVAILABLE")
            self.assertEqual(run.training.latest_validation_status, "AVAILABLE")
            self.assertEqual(run.training.best_validation.accuracy, 0.94)
            self.assertEqual(run.training.latest_validation.accuracy, 0.93)
            self.assertEqual(run.provenance.artifact.artifact_ref, "step01/run/result.json")
            self.assertIsNone(run.architecture.population_width)

    def test_missing_best_step_history_does_not_substitute_latest(self) -> None:
        payload = step01_payload()
        payload["best_step"] = 999
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            run = DashboardIndexer().build(root).experiments[0]

            self.assertEqual(run.training.best_validation_score, 0.94)
            self.assertIsNone(run.training.best_validation)
            self.assertEqual(run.training.best_validation_status, "NOT_AVAILABLE")
            self.assertEqual(run.training.latest_validation.accuracy, 0.93)


if __name__ == "__main__":
    unittest.main()
