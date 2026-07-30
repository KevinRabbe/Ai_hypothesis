from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_hypothesis.population_compute import (
    analyze_gate7_high_scale_routing_bandwidth_confirmation as audit,
)
from ai_hypothesis.population_compute import (
    gate7_high_scale_routing_bandwidth_confirmation as execution,
)
from ai_hypothesis.population_compute import (
    run_gate7_high_scale_routing_bandwidth_confirmation as runner,
)
from ai_hypothesis.population_compute.gate7_high_scale_routing_bandwidth import (
    gate7_high_scale_runtime_seed,
)
from ai_hypothesis.population_compute.gate7_high_scale_routing_bandwidth_confirmation_protocol import (
    GATE7_CONFIRMATION_CHECKPOINTS,
)


class Gate7HighScaleRoutingBandwidthConfirmationExecutionTests(unittest.TestCase):
    def test_exact_confirmation_execution_bindings(self) -> None:
        self.assertTrue(execution.GATE7_CONFIRMATION_EXECUTION_ADMITTED)
        self.assertEqual(
            execution.GATE7_CONFIRMATION_PROTOCOL_HEAD,
            "b0f0cfca736186b9400f82a7539a54f888dc59e5",
        )
        self.assertEqual(
            execution.GATE7_CONFIRMATION_SCIENTIFIC_STATUS,
            "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_CONFIRMATION_EVIDENCE",
        )
        self.assertEqual(
            execution.confirmation_provenance()["screening_result_sha256"],
            "d76c8b0753a518b4c61b3ff42c1f3e85902e2e492342f23fa6706459ee13a9b5",
        )

    def test_checkpoint_loader_rejects_unbound_bytes_before_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            path.write_bytes(b"not one of the exact transition checkpoints")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                execution.load_verified_gate7_confirmation_checkpoint(
                    checkpoint_index=0,
                    checkpoint_path=path,
                    device="cpu",
                )

    def test_confirmation_runtime_namespace_is_deterministic_and_distinct(self) -> None:
        first = execution.gate7_confirmation_runtime_seed(population=4096, world_index=0)
        repeated = execution.gate7_confirmation_runtime_seed(population=4096, world_index=0)
        next_world = execution.gate7_confirmation_runtime_seed(population=4096, world_index=1)
        screening = gate7_high_scale_runtime_seed(population=4096, world_index=0)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_world)
        self.assertNotEqual(first, screening)

    def test_eight_b64_rows_aggregate_to_exact_512_world_condition(self) -> None:
        rows = []
        for batch_index, start in enumerate(range(0, 512, 64)):
            covered = tuple((index % 2) == 0 for index in range(start, start + 64))
            rows.append(
                execution.Gate7ConfirmationBatchCondition(
                    checkpoint_index=0,
                    population=4096,
                    condition="global_hash",
                    k=None,
                    world_indices=tuple(range(start, start + 64)),
                    runtime_seeds=tuple(range(10_000 + start, 10_000 + start + 64)),
                    covered_by_world=covered,
                    score_observations_per_world=(0,) * 64,
                    logical_stage_a_parent_slots=4095,
                    logical_stage_b_parent_slots=128,
                    logical_learned_updates_per_world=(4095 + 128) * 16,
                    learned_parameter_count=19_649,
                    parameter_fingerprint=GATE7_CONFIRMATION_CHECKPOINTS[0]["fingerprint"],
                    wall_seconds=float(batch_index + 1),
                    peak_allocated_bytes=100 + batch_index,
                    selected_frontier_index_checksum=batch_index,
                    terminal_score_checksum=float(batch_index) / 10.0,
                )
            )
        result = execution.aggregate_gate7_confirmation_condition(tuple(rows))
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
    ) -> execution.Gate7ConfirmationCondition:
        observations = 0
        if name == "global_score":
            observations = 128 * 4096 - 127 * 128 // 2
        elif name.startswith("bounded_score_k"):
            assert k is not None
            observations = 128 * k
        return execution.Gate7ConfirmationCondition(
            checkpoint_index=0,
            population=4096,
            condition=name,
            k=k,
            world_indices=tuple(range(512)),
            runtime_seeds=tuple(range(20_000, 20_512)),
            covered_by_world=covered,
            coverage_rate=sum(int(value) for value in covered) / 512,
            score_observations_per_world=(observations,) * 512,
            logical_stage_a_parent_slots=4095,
            logical_stage_b_parent_slots=128,
            logical_learned_updates_per_world=(4095 + 128) * 16,
            learned_parameter_count=19_649,
            parameter_fingerprint=GATE7_CONFIRMATION_CHECKPOINTS[0]["fingerprint"],
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
        with patch.object(execution, "GATE7_CONFIRMATION_BOOTSTRAP_SAMPLES", 100):
            first = execution.paired_gate7_confirmation_summary(
                comparison="c0_k512_score_vs_hash",
                treatment=treatment,
                reference=reference,
            )
            second = execution.paired_gate7_confirmation_summary(
                comparison="c0_k512_score_vs_hash",
                treatment=treatment,
                reference=reference,
            )
        self.assertEqual(first, second)
        self.assertEqual(first.coverage_delta, 0.5)
        self.assertGreater(first.bootstrap_ci_low, 0.0)

    def test_runner_uses_complete_fixed_matrix_without_first_pass(self) -> None:
        source = inspect.getsource(runner.run_gate7_high_scale_routing_bandwidth_confirmation)
        self.assertIn("for population in GATE7_CONFIRMATION_POPULATIONS", source)
        self.assertIn("for condition in plan.conditions", source)
        self.assertIn("for k in plan.k_values", source)
        self.assertIn("classify_confirmation", source)
        self.assertNotIn("break", source)
        self.assertNotIn("first passing", source)
        self.assertIn('"confirmation_opened": True', source)
        self.assertIn('"second_confirmation_opened": False', source)
        self.assertNotIn("train_gate7", source)

    def test_independent_auditor_duplicates_bindings_without_execution_imports(self) -> None:
        self.assertEqual(audit.PROTOCOL_HEAD, execution.GATE7_CONFIRMATION_PROTOCOL_HEAD)
        self.assertEqual(audit.CHECKPOINTS, {
            index: {
                **row,
                "training_seed": index,
            }
            for index, row in GATE7_CONFIRMATION_CHECKPOINTS.items()
        })
        source = inspect.getsource(audit)
        self.assertNotIn("import torch", source)
        self.assertNotIn("from .gate7_high_scale_routing_bandwidth_confirmation import", source)
        self.assertNotIn("generate_gate7_confirmation_world", source)

    def test_tests_never_generate_worlds_or_invoke_runner(self) -> None:
        tree = ast.parse(inspect.getsource(type(self)))
        forbidden = {
            "generate_gate7_confirmation_world",
            "confirmation_world_batch",
            "run_gate7_high_scale_routing_bandwidth_confirmation",
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
