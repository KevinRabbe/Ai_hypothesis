from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_hypothesis.population_compute import (
    analyze_gate7_high_scale_routing_bandwidth_continuation as audit,
)
from ai_hypothesis.population_compute import (
    gate7_high_scale_routing_bandwidth_continuation as execution,
)
from ai_hypothesis.population_compute import (
    gate7_high_scale_routing_bandwidth_continuation_audit_spec as audit_spec,
)
from ai_hypothesis.population_compute import (
    gate7_high_scale_routing_bandwidth_continuation_campaign as campaign,
)
from ai_hypothesis.population_compute import (
    gate7_high_scale_routing_bandwidth_continuation_statistics as statistics,
)
from ai_hypothesis.population_compute import (
    run_gate7_high_scale_routing_bandwidth_continuation as runner,
)
from ai_hypothesis.population_compute.gate7_high_scale_routing_bandwidth_confirmation import (
    gate7_confirmation_runtime_seed,
)
from ai_hypothesis.population_compute.gate7_high_scale_routing_bandwidth_continuation_protocol import (
    GATE7_CONTINUATION_CHECKPOINTS,
    GATE7_CONTINUATION_POPULATIONS,
)


class Gate7HighScaleRoutingBandwidthContinuationExecutionTests(unittest.TestCase):
    def test_exact_continuation_execution_bindings(self) -> None:
        self.assertTrue(execution.GATE7_CONTINUATION_EXECUTION_ADMITTED)
        self.assertEqual(
            execution.GATE7_CONTINUATION_PROTOCOL_HEAD,
            "4f05f8b1f9a33aed712edbf28691b927d2e220d3",
        )
        self.assertEqual(
            execution.GATE7_CONTINUATION_SCIENTIFIC_STATUS,
            "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_CONTINUATION_EVIDENCE",
        )
        provenance = execution.continuation_provenance()
        self.assertEqual(
            provenance["confirmation_result_head"],
            "ae8bd8544a03e48f4f397d2ca5ae933d9247e430",
        )
        self.assertEqual(
            provenance["confirmation_result_sha256"],
            "725e3749ba5fed7cdcbb6d61df81bcc77a7b69bacfdc82d553efb06f5ff888da",
        )
        self.assertEqual(provenance["confirmed_n8192_passing_k"], [256, 512])
        self.assertEqual(provenance["confirmed_n8192_k_required"], 256)

    def test_checkpoint_loader_rejects_unbound_bytes_before_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            path.write_bytes(b"not one of the exact transition checkpoints")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                execution.load_verified_gate7_continuation_checkpoint(
                    checkpoint_index=0,
                    checkpoint_path=path,
                    device="cpu",
                )

    def test_continuation_runtime_namespace_is_deterministic_and_distinct(self) -> None:
        first = execution.gate7_continuation_runtime_seed(population=16_384, world_index=0)
        repeated = execution.gate7_continuation_runtime_seed(population=16_384, world_index=0)
        next_world = execution.gate7_continuation_runtime_seed(population=16_384, world_index=1)
        confirmation = gate7_confirmation_runtime_seed(population=8192, world_index=0)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_world)
        self.assertNotEqual(first, confirmation)

    def test_eight_b64_rows_aggregate_to_exact_512_world_condition(self) -> None:
        rows = []
        for batch_index, start in enumerate(range(0, 512, 64)):
            covered = tuple((index % 2) == 0 for index in range(start, start + 64))
            rows.append(
                execution.Gate7ContinuationBatchCondition(
                    checkpoint_index=0,
                    population=16_384,
                    condition="global_hash",
                    k=None,
                    world_indices=tuple(range(start, start + 64)),
                    runtime_seeds=tuple(range(10_000 + start, 10_000 + start + 64)),
                    covered_by_world=covered,
                    score_observations_per_world=(0,) * 64,
                    logical_stage_a_parent_slots=16_383,
                    logical_stage_b_parent_slots=128,
                    logical_learned_updates_per_world=(16_383 + 128) * 16,
                    learned_parameter_count=19_649,
                    parameter_fingerprint=GATE7_CONTINUATION_CHECKPOINTS[0]["fingerprint"],
                    wall_seconds=float(batch_index + 1),
                    peak_allocated_bytes=100 + batch_index,
                    selected_frontier_index_checksum=batch_index,
                    terminal_score_checksum=float(batch_index) / 10.0,
                )
            )
        result = execution.aggregate_gate7_continuation_condition(tuple(rows))
        self.assertEqual(result.world_indices, tuple(range(512)))
        self.assertEqual(len(result.runtime_seeds), 512)
        self.assertEqual(len(result.covered_by_world), 512)
        self.assertEqual(result.coverage_rate, 0.5)
        self.assertEqual(result.batch_count, 8)
        self.assertEqual(result.wall_seconds, 36.0)
        self.assertEqual(result.peak_allocated_bytes, 107)
        self.assertEqual(result.selected_frontier_index_checksum, 28)

    @staticmethod
    def _condition(
        *,
        name: str,
        k: int | None,
        covered: tuple[bool, ...],
    ) -> execution.Gate7ContinuationCondition:
        observations = 0
        if name == "global_score":
            observations = 128 * 16_384 - 127 * 128 // 2
        elif name.startswith("bounded_score_k"):
            assert k is not None
            observations = 128 * k
        return execution.Gate7ContinuationCondition(
            checkpoint_index=0,
            population=16_384,
            condition=name,
            k=k,
            world_indices=tuple(range(512)),
            runtime_seeds=tuple(range(20_000, 20_512)),
            covered_by_world=covered,
            coverage_rate=sum(int(value) for value in covered) / 512,
            score_observations_per_world=(observations,) * 512,
            logical_stage_a_parent_slots=16_383,
            logical_stage_b_parent_slots=128,
            logical_learned_updates_per_world=(16_383 + 128) * 16,
            learned_parameter_count=19_649,
            parameter_fingerprint=GATE7_CONTINUATION_CHECKPOINTS[0]["fingerprint"],
            batch_count=8,
            wall_seconds=1.0,
            peak_allocated_bytes=1,
            selected_frontier_index_checksum=1,
            terminal_score_checksum=1.0,
        )

    def test_paired_bootstrap_is_deterministic_on_synthetic_vectors(self) -> None:
        treatment = self._condition(
            name="bounded_score_k512",
            k=512,
            covered=(True,) * 256 + (False,) * 256,
        )
        reference = self._condition(
            name="bounded_hash_k512",
            k=512,
            covered=(False,) * 512,
        )
        with patch.object(statistics, "GATE7_CONTINUATION_BOOTSTRAP_SAMPLES", 100):
            first = execution.paired_gate7_continuation_summary(
                comparison="c0_k512_score_vs_hash",
                treatment=treatment,
                reference=reference,
            )
            second = execution.paired_gate7_continuation_summary(
                comparison="c0_k512_score_vs_hash",
                treatment=treatment,
                reference=reference,
            )
        self.assertEqual(first, second)
        self.assertEqual(first.coverage_delta, 0.5)
        self.assertGreater(first.bootstrap_ci_low, 0.0)

    def test_runner_and_campaign_use_complete_fixed_matrix_with_only_resource_break(self) -> None:
        runner_source = inspect.getsource(
            runner.run_gate7_high_scale_routing_bandwidth_continuation
        )
        campaign_source = inspect.getsource(campaign.execute_gate7_continuation_tier)
        self.assertIn("for population in GATE7_CONTINUATION_POPULATIONS", runner_source)
        self.assertIn("except torch.cuda.OutOfMemoryError", runner_source)
        self.assertNotIn("tier_outcome", runner_source)
        self.assertNotIn("reference_viable", runner_source)
        self.assertIn("for condition in plan.conditions", campaign_source)
        self.assertIn("for k in GATE7_CONTINUATION_K_LADDER", campaign_source)
        self.assertNotIn("break", campaign_source)
        self.assertNotIn("first passing", runner_source + campaign_source)
        self.assertIn('"continuation_opened": True', runner_source)
        self.assertIn('"second_continuation_opened": False', runner_source)
        self.assertNotIn("train_gate7", runner_source + campaign_source)

        tree = ast.parse(runner_source)
        breaks = [node for node in ast.walk(tree) if isinstance(node, ast.Break)]
        self.assertEqual(len(breaks), 1)

    def test_independent_auditor_duplicates_bindings_without_execution_imports(self) -> None:
        self.assertEqual(audit_spec.PROTOCOL_HEAD, execution.GATE7_CONTINUATION_PROTOCOL_HEAD)
        self.assertEqual(audit_spec.POPULATIONS, GATE7_CONTINUATION_POPULATIONS)
        self.assertEqual(
            audit_spec.CHECKPOINTS,
            {
                index: {**row, "training_seed": index}
                for index, row in GATE7_CONTINUATION_CHECKPOINTS.items()
            },
        )
        source = inspect.getsource(audit) + inspect.getsource(audit_spec)
        self.assertNotIn("import torch", source)
        self.assertNotIn(
            "from .gate7_high_scale_routing_bandwidth_continuation import",
            source,
        )
        self.assertNotIn("generate_gate7_continuation_world", source)

    def test_invalid_artifact_returns_structured_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("[]", encoding="utf-8")
            result = audit.audit_gate7_high_scale_routing_bandwidth_continuation(path)
        self.assertFalse(result.artifact_valid)
        self.assertEqual(result.scientific_status, "INVALID_ARTIFACT")
        self.assertTrue(result.errors)

    def test_tests_never_generate_worlds_or_invoke_runner(self) -> None:
        tree = ast.parse(inspect.getsource(type(self)))
        forbidden = {
            "generate_gate7_continuation_world",
            "continuation_world_batch",
            "run_gate7_high_scale_routing_bandwidth_continuation",
        }
        called: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
        self.assertTrue(forbidden.isdisjoint(called))


if __name__ == "__main__":
    unittest.main()
