"""Tests learned-execution timing without requiring CUDA."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from ai_hypothesis.large_scope.timing import TimedSelectedWorkerBank
from ai_hypothesis.step01.model import Step01Output


class _FakeCpuBank:
    population_width = 4
    device = torch.device("cpu")
    checkpoint_ids = ("a", "b", "c", "d")

    def forward_selected(self, worker_indices, features, mask):
        del worker_indices, mask
        batch = features.shape[0]
        return Step01Output(
            label_logits=torch.zeros((batch, 11)),
            uncertainty_logits=torch.zeros((batch,)),
        )


class SelectedWorkerTimingTests(unittest.TestCase):
    def test_cpu_timing_tracks_calls_samples_and_elapsed_time(self) -> None:
        bank = TimedSelectedWorkerBank(_FakeCpuBank())
        features = torch.zeros((3, 32, 16))
        mask = torch.ones((3, 32), dtype=torch.bool)

        with patch(
            "ai_hypothesis.large_scope.timing.time.perf_counter",
            side_effect=(10.0, 10.25, 20.0, 20.5),
        ):
            bank.forward_selected((0, 1, 2), features, mask)
            bank.forward_selected((1, 2, 3), features, mask)

        timing = bank.snapshot_after_synchronize()
        self.assertEqual(timing.call_count, 2)
        self.assertEqual(timing.sample_count, 6)
        self.assertAlmostEqual(timing.elapsed_seconds, 0.75)
        self.assertAlmostEqual(timing.samples_per_second, 8.0)

    def test_reset_discards_previous_measurement(self) -> None:
        bank = TimedSelectedWorkerBank(_FakeCpuBank())
        features = torch.zeros((2, 32, 16))
        mask = torch.ones((2, 32), dtype=torch.bool)

        with patch(
            "ai_hypothesis.large_scope.timing.time.perf_counter",
            side_effect=(1.0, 1.1),
        ):
            bank.forward_selected((0, 1), features, mask)
        bank.reset_timing()

        timing = bank.snapshot_after_synchronize()
        self.assertEqual(timing.call_count, 0)
        self.assertEqual(timing.sample_count, 0)
        self.assertEqual(timing.elapsed_seconds, 0.0)
        self.assertIsNone(timing.samples_per_second)

    def test_wrapper_delegates_checkpoint_and_device_metadata(self) -> None:
        wrapped = _FakeCpuBank()
        bank = TimedSelectedWorkerBank(wrapped)
        self.assertEqual(bank.population_width, wrapped.population_width)
        self.assertEqual(bank.checkpoint_ids, wrapped.checkpoint_ids)
        self.assertEqual(bank.device, wrapped.device)


if __name__ == "__main__":
    unittest.main()
