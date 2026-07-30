from __future__ import annotations

import json
import pathlib
import unittest

from ai_hypothesis.population_compute.analyze_gate7_information_ceiling_precision_confirmation import (
    exact_public_ranks,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_decomposition_worlds import (
    generate_gate7_information_ceiling_world,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_precision_confirmation_protocol import (
    GATE7_PRECISION_CHECKPOINT_INDICES,
    GATE7_PRECISION_POPULATIONS,
    GATE7_PRECISION_RANKERS,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_precision_confirmation_rank import (
    Gate7PrecisionBatchRanks,
    Gate7PrecisionCheckpointRanks,
    aggregate_gate7_precision_rank_batches,
    evaluate_gate7_precision_rank_batch,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_precision_confirmation_statistics import (
    gate7_precision_population_statistics,
    summarize_gate7_precision_ranks,
)
from ai_hypothesis.population_compute.gate7_information_ceiling_precision_confirmation_worlds import (
    generate_gate7_precision_world,
    precision_world_batch,
)

LEARNED, BAYES, HASH = GATE7_PRECISION_RANKERS


class Gate7PrecisionExecutionTests(unittest.TestCase):
    def test_fresh_world_namespace_is_deterministic_and_disjoint(self) -> None:
        left = generate_gate7_precision_world(population=16_384, world_index=7)
        right = generate_gate7_precision_world(population=16_384, world_index=7)
        previous = generate_gate7_information_ceiling_world(
            population=16_384, world_index=7
        )
        self.assertEqual(left, right)
        self.assertNotEqual(left.runtime_seed, previous.runtime_seed)
        self.assertNotEqual(left.hidden_path, previous.hidden_path)
        self.assertEqual(
            left.hidden_terminal_path_id // 2, left.hidden_parent_index
        )
        left.validate()

    def test_independent_public_rank_reconstruction_matches_world(self) -> None:
        runtime_seed, bayes_rank, hash_rank = exact_public_ranks(16_384, 11)
        world = generate_gate7_precision_world(
            population=16_384, world_index=11
        )
        self.assertEqual(runtime_seed, world.runtime_seed)
        self.assertTrue(1 <= bayes_rank <= world.population)
        self.assertEqual(
            hash_rank,
            1
            + (
                world.hidden_parent_index * world.hash_multiplier
                + world.hash_offset
            )
            % world.population,
        )

    def _batch_row(
        self, checkpoint: int, batch: int
    ) -> Gate7PrecisionBatchRanks:
        start = batch * 64
        ranks = tuple(
            (index % 1024) + 1 for index in range(start, start + 64)
        )
        return Gate7PrecisionBatchRanks(
            checkpoint_index=checkpoint,
            population=16_384,
            world_indices=tuple(range(start, start + 64)),
            runtime_seeds=tuple(
                range(100_000 + start, 100_000 + start + 64)
            ),
            learned_ranks=ranks,
            bayes_ranks=tuple(max(1, rank - 8) for rank in ranks),
            hash_ranks=tuple(
                16_384 - index for index in range(start, start + 64)
            ),
            learned_parameter_count=19_649,
            parameter_fingerprint=f"synthetic-{checkpoint}",
            frontier_wall_seconds=1.0,
            frontier_peak_allocated_bytes=100,
            frontier_storage_bytes=200,
            frontier_score_checksum=3.0,
            hidden_score_checksum=4.0,
            tie_priority_checksum=5,
            hash_priority_checksum=6,
        )

    def _checkpoint(self, checkpoint: int) -> Gate7PrecisionCheckpointRanks:
        return aggregate_gate7_precision_rank_batches(
            tuple(self._batch_row(checkpoint, batch) for batch in range(32))
        )

    def test_exact_thirty_two_batch_aggregation_and_json_order_independence(
        self,
    ) -> None:
        checkpoint = self._checkpoint(0)
        checkpoint.validate()
        self.assertEqual(checkpoint.world_indices, tuple(range(2048)))
        learned = summarize_gate7_precision_ranks(
            checkpoint=checkpoint, ranker=LEARNED
        )
        self.assertEqual(learned.coverage_by_attempt["128"], 256 / 2048)
        payload = json.loads(json.dumps(checkpoint.to_dict(), sort_keys=True))
        rebuilt = Gate7PrecisionCheckpointRanks(
            checkpoint_index=payload["checkpoint_index"],
            population=payload["population"],
            world_indices=tuple(payload["world_indices"]),
            runtime_seeds=tuple(payload["runtime_seeds"]),
            ranks_by_ranker={
                key: tuple(value)
                for key, value in payload["ranks_by_ranker"].items()
            },
            learned_parameter_count=payload["learned_parameter_count"],
            parameter_fingerprint=payload["parameter_fingerprint"],
            batch_count=payload["batch_count"],
            frontier_wall_seconds=payload["frontier_wall_seconds"],
            frontier_peak_allocated_bytes=payload[
                "frontier_peak_allocated_bytes"
            ],
            frontier_storage_bytes=payload["frontier_storage_bytes"],
            frontier_score_checksum=payload["frontier_score_checksum"],
            hidden_score_checksum=payload["hidden_score_checksum"],
            tie_priority_checksum=payload["tie_priority_checksum"],
            hash_priority_checksum=payload["hash_priority_checksum"],
        )
        rebuilt.validate()

    def test_population_bootstrap_preserves_shared_world_clusters(self) -> None:
        checkpoints = tuple(
            self._checkpoint(index)
            for index in GATE7_PRECISION_CHECKPOINT_INDICES
        )
        summaries, comparison = gate7_precision_population_statistics(
            checkpoints
        )
        self.assertEqual(len(summaries), 3)
        self.assertLessEqual(
            comparison.learned_minus_bayes_ci_low,
            comparison.learned_minus_bayes_delta,
        )
        self.assertLessEqual(
            comparison.learned_minus_bayes_delta,
            comparison.learned_minus_bayes_ci_high,
        )

    def test_rank_execution_rejects_cpu_before_frontier_construction(self) -> None:
        worlds = precision_world_batch(population=16_384, batch_start=0)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            evaluate_gate7_precision_rank_batch(
                object(), checkpoint_index=0, worlds=worlds, device="cpu"
            )

    def test_exact_frozen_matrix_constants(self) -> None:
        self.assertEqual(
            GATE7_PRECISION_POPULATIONS,
            (16_384, 32_768, 65_536, 131_072),
        )
        self.assertEqual(GATE7_PRECISION_CHECKPOINT_INDICES, (0, 1, 2))

    def test_runner_and_auditor_preserve_intervention_boundary(self) -> None:
        runner = pathlib.Path(
            "ai_hypothesis/population_compute/"
            "run_gate7_information_ceiling_precision_confirmation.py"
        ).read_text(encoding="utf-8")
        auditor = pathlib.Path(
            "ai_hypothesis/population_compute/"
            "analyze_gate7_information_ceiling_precision_confirmation.py"
        ).read_text(encoding="utf-8")
        for token in (
            '"training_performed": False',
            '"communication_intervention_performed": False',
            '"prior_worlds_reused": False',
        ):
            self.assertIn(token, runner)
        self.assertNotIn("import torch", auditor)
        self.assertNotIn("precision_confirmation import", auditor)
        self.assertNotIn("precision_confirmation_rank import", auditor)
        self.assertNotIn("precision_confirmation_statistics import", auditor)


if __name__ == "__main__":
    unittest.main()
