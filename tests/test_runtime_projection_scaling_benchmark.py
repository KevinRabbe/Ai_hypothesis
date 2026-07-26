from __future__ import annotations

import unittest

from experiments.runtime_projection_scaling.run_projection_scaling import run_benchmark


class ProjectionScalingBenchmarkTests(unittest.TestCase):
    def test_tiny_run_preserves_plan_equivalence_and_metrics(self) -> None:
        result = run_benchmark(
            history_event_counts=(0, 20, 50),
            pending_evidence_count=12,
            repeats=2,
        )

        self.assertEqual(result.repeats, 2)
        self.assertEqual(
            tuple(point.history_event_count for point in result.points),
            (0, 20, 50),
        )
        self.assertEqual(
            tuple(point.new_tail_event_count for point in result.points),
            (0, 20, 30),
        )
        self.assertTrue(all(point.plan_equivalent for point in result.points))
        self.assertTrue(all(point.pending_evidence_count == 12 for point in result.points))
        self.assertTrue(all(point.partition_count > 0 for point in result.points))
        self.assertTrue(all(point.indexed_catchup_ms >= 0.0 for point in result.points))
        self.assertTrue(all(point.indexed_warm_median_ms >= 0.0 for point in result.points))
        self.assertTrue(all(point.replay_median_ms >= 0.0 for point in result.points))
        self.assertTrue(
            all(point.replay_to_indexed_warm_ratio >= 0.0 for point in result.points)
        )
        totals = tuple(point.total_ledger_event_count for point in result.points)
        self.assertEqual(totals, tuple(sorted(totals)))

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            run_benchmark(history_event_counts=(), pending_evidence_count=1, repeats=1)
        with self.assertRaises(ValueError):
            run_benchmark(history_event_counts=(10, 5), pending_evidence_count=1, repeats=1)
        with self.assertRaises(ValueError):
            run_benchmark(history_event_counts=(0,), pending_evidence_count=0, repeats=1)
        with self.assertRaises(ValueError):
            run_benchmark(history_event_counts=(0,), pending_evidence_count=1, repeats=0)


if __name__ == "__main__":
    unittest.main()
