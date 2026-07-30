from __future__ import annotations

import pathlib
import unittest

from ai_hypothesis.population_compute.analyze_gate7_information_ceiling_decomposition import (
    exact_public_ranks,
)
from ai_hypothesis.population_compute.gate7_high_scale_routing_bandwidth_continuation_worlds import (
    generate_gate7_continuation_world,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_decomposition_protocol import (
    GATE7_INFORMATION_CEILING_BAYES,
    GATE7_INFORMATION_CEILING_HASH,
    GATE7_INFORMATION_CEILING_LEARNED,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_decomposition_rank import (
    Gate7InformationCeilingBatchRanks,
    aggregate_information_ceiling_rank_batches,
    evaluate_information_ceiling_rank_batch,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_decomposition_statistics import (
    paired_information_ceiling_summary,
    summarize_information_ceiling_ranks,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_decomposition_worlds import (
    generate_gate7_information_ceiling_world,
    information_ceiling_world_batch,
)


class Gate7InformationCeilingExecutionTests(unittest.TestCase):
    def test_fresh_world_namespace_is_deterministic_and_disjoint(self) -> None:
        left = generate_gate7_information_ceiling_world(population=16_384, world_index=7)
        right = generate_gate7_information_ceiling_world(population=16_384, world_index=7)
        old = generate_gate7_continuation_world(population=16_384, world_index=7)
        self.assertEqual(left, right)
        self.assertNotEqual(left.runtime_seed, old.runtime_seed)
        self.assertNotEqual(left.hidden_path, old.hidden_path)
        self.assertEqual(left.hidden_terminal_path_id // 2, left.hidden_parent_index)
        left.validate()

    def test_public_affine_orders_are_exact_permutations(self) -> None:
        world = generate_gate7_information_ceiling_world(population=16_384, world_index=3)
        mask = world.population - 1
        tie = {
            (candidate * world.tie_multiplier + world.tie_offset) & mask
            for candidate in range(world.population)
        }
        hash_order = {
            (candidate * world.hash_multiplier + world.hash_offset) & mask
            for candidate in range(world.population)
        }
        self.assertEqual(len(tie), world.population)
        self.assertEqual(len(hash_order), world.population)

    def test_independent_public_rank_reconstruction_is_bounded(self) -> None:
        runtime_seed, bayes_rank, hash_rank = exact_public_ranks(16_384, 11)
        world = generate_gate7_information_ceiling_world(population=16_384, world_index=11)
        self.assertEqual(runtime_seed, world.runtime_seed)
        self.assertTrue(1 <= bayes_rank <= world.population)
        self.assertEqual(
            hash_rank,
            1
            + (
                world.hidden_parent_index * world.hash_multiplier + world.hash_offset
            )
            % world.population,
        )

    def _batch_row(self, batch: int) -> Gate7InformationCeilingBatchRanks:
        start = batch * 64
        ranks = tuple((index % 1024) + 1 for index in range(start, start + 64))
        return Gate7InformationCeilingBatchRanks(
            checkpoint_index=0,
            population=16_384,
            world_indices=tuple(range(start, start + 64)),
            runtime_seeds=tuple(range(100_000 + start, 100_000 + start + 64)),
            learned_ranks=ranks,
            bayes_ranks=tuple(max(1, rank - 8) for rank in ranks),
            hash_ranks=tuple(16_384 - index for index in range(start, start + 64)),
            learned_parameter_count=19_649,
            parameter_fingerprint="synthetic-fingerprint",
            frontier_wall_seconds=1.0,
            frontier_peak_allocated_bytes=100,
            frontier_storage_bytes=200,
            frontier_score_checksum=3.0,
            hidden_score_checksum=4.0,
            tie_priority_checksum=5,
            hash_priority_checksum=6,
        )

    def test_exact_eight_batch_aggregation_and_rank_summaries(self) -> None:
        checkpoint = aggregate_information_ceiling_rank_batches(
            tuple(self._batch_row(batch) for batch in range(8))
        )
        checkpoint.validate()
        self.assertEqual(checkpoint.world_indices, tuple(range(512)))
        self.assertEqual(tuple(checkpoint.ranks_by_ranker), (
            GATE7_INFORMATION_CEILING_LEARNED,
            GATE7_INFORMATION_CEILING_BAYES,
            GATE7_INFORMATION_CEILING_HASH,
        ))
        learned = summarize_information_ceiling_ranks(
            checkpoint=checkpoint,
            ranker=GATE7_INFORMATION_CEILING_LEARNED,
        )
        self.assertEqual(learned.coverage_by_attempt["128"], 128 / 512)
        self.assertEqual(learned.rank_checksum, sum(range(1, 513)))

    def test_paired_bootstrap_is_deterministic(self) -> None:
        checkpoint = aggregate_information_ceiling_rank_batches(
            tuple(self._batch_row(batch) for batch in range(8))
        )
        first = paired_information_ceiling_summary(
            comparison="synthetic_learned_vs_bayes_m128",
            checkpoint=checkpoint,
            treatment_ranker=GATE7_INFORMATION_CEILING_LEARNED,
            reference_ranker=GATE7_INFORMATION_CEILING_BAYES,
        )
        second = paired_information_ceiling_summary(
            comparison="synthetic_learned_vs_bayes_m128",
            checkpoint=checkpoint,
            treatment_ranker=GATE7_INFORMATION_CEILING_LEARNED,
            reference_ranker=GATE7_INFORMATION_CEILING_BAYES,
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first.bootstrap_ci_low, first.coverage_delta)
        self.assertLessEqual(first.coverage_delta, first.bootstrap_ci_high)

    def test_rank_execution_rejects_cpu_before_frontier_construction(self) -> None:
        worlds = information_ceiling_world_batch(population=16_384, batch_start=0)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            evaluate_information_ceiling_rank_batch(
                object(),
                checkpoint_index=0,
                worlds=worlds,
                device="cpu",
            )

    def test_runner_and_auditor_preserve_intervention_boundary(self) -> None:
        runner = pathlib.Path(
            "ai_hypothesis/population_compute/run_gate7_information_ceiling_decomposition.py"
        ).read_text(encoding="utf-8")
        auditor = pathlib.Path(
            "ai_hypothesis/population_compute/analyze_gate7_information_ceiling_decomposition.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"training_performed": False', runner)
        self.assertIn('"communication_intervention_performed": False', runner)
        self.assertIn('"continuation_worlds_reused": False', runner)
        self.assertNotIn("import torch", auditor)
        self.assertNotIn("gate7_information_ceiling_decomposition import", auditor)


if __name__ == "__main__":
    unittest.main()
