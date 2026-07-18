from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_hypothesis.dashboard.indexer import DashboardIndexer


def step02_payload() -> dict:
    return {
        "runtime_version": "step02-population-runtime-v0",
        "evidence_contract_version": "step02-evidence-v0",
        "split": "validation",
        "device": "cuda",
        "backend": "vmap",
        "count": 20,
        "batch_size": 4,
        "aggregation_config": {"top_k": 3},
        "checkpoints": [
            {"path": "worker_1.pt", "step": 10, "validation_score": 0.9},
            {"path": "worker_2.pt", "step": 10, "validation_score": 0.91},
        ],
        "metrics": {
            "count": 20,
            "population_width": 2,
            "execution_backend": "vmap",
            "unit_config": {
                "d_model": 64,
                "block_count": 2,
                "attention_heads": 4,
                "feed_forward_width": 128,
                "dropout": 0.1,
                "sequence_length": 32,
                "feature_width": 16,
            },
            "evidence_reducer_accuracy": 0.8,
            "majority_vote_accuracy": 0.7,
            "mean_logit_accuracy": 0.75,
            "mean_probability_accuracy": 0.76,
            "oracle_any_correct_coverage": 0.9,
            "all_wrong_rate": 0.1,
            "minority_rescue_opportunity_rate": 0.2,
            "minority_rescue_rate": 0.5,
            "minority_suppression_rate": 0.25,
            "majority_harm_rate": 0.05,
            "evidence_utilization_gap": 0.1,
            "mean_disagreement_entropy": 0.3,
            "mean_population_uncertainty": 0.4,
            "mean_invalid_label_mass": 0.05,
            "single_worker_accuracy": {"values": [0.7, 0.8], "min": 0.7, "max": 0.8, "mean": 0.75},
            "by_task": {
                "A_PATTERN": {
                    "count": 10,
                    "evidence_accuracy": 0.8,
                    "majority_vote_accuracy": 0.7,
                    "mean_logit_accuracy": 0.75,
                    "mean_probability_accuracy": 0.76,
                    "oracle_any_correct_coverage": 0.9,
                }
            },
        },
    }


class DashboardStep02AdapterTests(unittest.TestCase):
    def test_step02_population_result_normalizes_without_checkpoint_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "step02" / "population" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(step02_payload()), encoding="utf-8")

            with patch("torch.load", side_effect=AssertionError("checkpoint loaded")):
                snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.indexed_experiment_count, 1)
            run = snapshot.experiments[0]
            self.assertEqual(run.identity.experiment_type, "population")
            self.assertEqual(run.runtime_version, "step02-population-runtime-v0")
            self.assertEqual(run.evidence_contract_version, "step02-evidence-v0")
            self.assertEqual(run.population_metrics.population_width, 2)
            self.assertEqual(run.population_metrics.evidence_reducer_accuracy, 0.8)
            self.assertEqual(run.architecture.worker_parameter_count_status, "NOT_AVAILABLE")

    def test_malformed_step02_result_is_isolated(self) -> None:
        payload = step02_payload()
        del payload["metrics"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.indexed_experiment_count, 0)
            self.assertEqual(snapshot.status.indexing_error_count, 1)


if __name__ == "__main__":
    unittest.main()
