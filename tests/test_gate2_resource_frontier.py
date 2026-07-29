from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ai_hypothesis.population_compute.gate2_persistent_model import Gate2PersistentStateModel
from ai_hypothesis.population_compute.gate2_resource_frontier import (
    GATE2_RESOURCE_BATCH_SIZES,
    GATE2_RESOURCE_ENTITY_WIDTHS,
    GATE2_RESOURCE_EXPECTED_CELL_COUNT,
    GATE2_RESOURCE_PRIMARY_CONFIRMATION_SEED,
    GATE2_RESOURCE_TIMED_ITERATIONS,
    GATE2_RESOURCE_WARMUP_ITERATIONS,
    GATE2_RESOURCE_WORLD_SEED_START,
    _quantile,
    _world_seeds,
    load_seed3_confirmation_checkpoint,
)


class Gate2ResourceFrontierTests(unittest.TestCase):
    def test_frozen_resource_matrix_has_eighteen_cells(self) -> None:
        matrix = [
            (c, w, b)
            for c, widths in GATE2_RESOURCE_ENTITY_WIDTHS.items()
            for w in widths
            for b in GATE2_RESOURCE_BATCH_SIZES
        ]
        self.assertEqual(GATE2_RESOURCE_ENTITY_WIDTHS, {64: (1, 4, 16, 64), 256: (1, 4, 16, 64, 256)})
        self.assertEqual(GATE2_RESOURCE_BATCH_SIZES, (1, 64))
        self.assertEqual(len(matrix), GATE2_RESOURCE_EXPECTED_CELL_COUNT)
        self.assertEqual(len(matrix), 18)
        self.assertEqual(GATE2_RESOURCE_WARMUP_ITERATIONS, 10)
        self.assertEqual(GATE2_RESOURCE_TIMED_ITERATIONS, 50)
        self.assertEqual(GATE2_RESOURCE_PRIMARY_CONFIRMATION_SEED, 3)
        self.assertEqual(GATE2_RESOURCE_WORLD_SEED_START, 4 << 30)

    def test_resource_world_corpus_is_deterministic_and_entity_separated(self) -> None:
        first = _world_seeds(64, 64)
        second = _world_seeds(64, 64)
        other = _world_seeds(256, 64)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(len(set(first)), 64)
        self.assertTrue(set(first).isdisjoint(other))
        self.assertGreaterEqual(min(first), 4 << 30)

    def test_quantile_is_deterministic_linear_interpolation(self) -> None:
        values = (1.0, 2.0, 3.0, 4.0, 5.0)
        self.assertEqual(_quantile(values, 0.25), 2.0)
        self.assertEqual(_quantile(values, 0.5), 3.0)
        self.assertEqual(_quantile(values, 0.75), 4.0)

    def test_seed3_confirmation_checkpoint_loads_with_exact_fingerprint(self) -> None:
        torch.manual_seed(3)
        model = Gate2PersistentStateModel()
        fingerprint = model.parameter_fingerprint()
        count = model.trainable_parameter_count()
        payload = {
            "experiment_version": "gate2-persistent-state-confirmation-v0",
            "evaluation_split": "confirmation",
            "confirmation_opened": True,
            "training_seed": 3,
            "model_config": {"state_width": 64, "query_width": 24},
            "learned_parameter_count": count,
            "parameter_fingerprint": fingerprint,
            "state_dict": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed3.pt"
            torch.save(payload, path)
            restored = load_seed3_confirmation_checkpoint(path, device="cpu")
            self.assertEqual(restored.trainable_parameter_count(), count)
            self.assertEqual(restored.parameter_fingerprint(), fingerprint)

    def test_non_seed3_checkpoint_is_rejected(self) -> None:
        model = Gate2PersistentStateModel()
        payload = {
            "experiment_version": "gate2-persistent-state-confirmation-v0",
            "evaluation_split": "confirmation",
            "confirmation_opened": True,
            "training_seed": 4,
            "model_config": {"state_width": 64, "query_width": 24},
            "learned_parameter_count": model.trainable_parameter_count(),
            "parameter_fingerprint": model.parameter_fingerprint(),
            "state_dict": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed4.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "seed 3"):
                load_seed3_confirmation_checkpoint(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
