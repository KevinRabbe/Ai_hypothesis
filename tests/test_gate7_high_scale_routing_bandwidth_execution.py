from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute import analyze_gate7_high_scale_routing_bandwidth as audit
from ai_hypothesis.population_compute import gate7_high_scale_routing_bandwidth as execution
from ai_hypothesis.population_compute import run_gate7_high_scale_routing_bandwidth as runner
from ai_hypothesis.population_compute.gate7_high_scale_routing_bandwidth import (
    Gate7HighScaleCondition,
    paired_gate7_high_scale_summary,
    stratified_gate7_high_scale_global_summary,
)


class Gate7HighScaleRoutingBandwidthExecutionTests(unittest.TestCase):
    @staticmethod
    def _condition(
        *, checkpoint: int, population: int, condition: str, covered: tuple[bool, ...], k: int | None
    ) -> Gate7HighScaleCondition:
        expected_observations = 0
        if condition == "global_score":
            expected_observations = 128 * population - 8128
        elif condition.startswith("bounded_score_k"):
            expected_observations = 128 * int(condition.rsplit("k", 1)[1])
        return Gate7HighScaleCondition(
            checkpoint_index=checkpoint,
            population=population,
            condition=condition,
            k=k,
            world_indices=tuple(range(64)),
            runtime_seeds=tuple(range(64)),
            covered_by_world=covered,
            coverage_rate=sum(int(value) for value in covered) / 64,
            score_observations_per_world=(expected_observations,) * 64,
            logical_stage_a_parent_slots=population - 1,
            logical_stage_b_parent_slots=128,
            logical_learned_updates_per_world=(population - 1 + 128) * 16,
            learned_parameter_count=19649,
            parameter_fingerprint=execution.GATE7_HIGH_SCALE_CHECKPOINTS[checkpoint]["fingerprint"],
            wall_seconds=0.0,
            peak_allocated_bytes=0,
            selected_frontier_index_checksum=0,
            terminal_score_checksum=0.0,
        )

    def test_exact_prerequisites_and_checkpoint_family_are_bound(self) -> None:
        self.assertTrue(execution.GATE7_HIGH_SCALE_EXECUTION_ADMITTED)
        self.assertEqual(
            execution.GATE7_HIGH_SCALE_ENGINEERING_RESULT_HEAD,
            "5305475ea1e295c84fadbce3533f13489b10d60d",
        )
        self.assertEqual(
            execution.GATE7_HIGH_SCALE_ENGINEERING_SUMMARY_SHA256,
            "e40823e3e2787151f2a63607aa3d396f18e03428b715b8864af4f549631e2953",
        )
        self.assertEqual(
            {index: row["sha256"] for index, row in execution.GATE7_HIGH_SCALE_CHECKPOINTS.items()},
            {
                0: "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
                1: "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
                2: "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
            },
        )

    def test_checkpoint_loader_rejects_unbound_bytes_before_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            path.write_bytes(b"not one of the bound Gate-7 transition checkpoints")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                execution.load_verified_gate7_high_scale_checkpoint(
                    checkpoint_index=0,
                    checkpoint_path=path,
                    device="cpu",
                )

    def test_namespace_is_population_specific_without_generating_hidden_worlds(self) -> None:
        first = execution.gate7_high_scale_runtime_seed(population=1024, world_index=0)
        repeated = execution.gate7_high_scale_runtime_seed(population=1024, world_index=0)
        next_population = execution.gate7_high_scale_runtime_seed(population=2048, world_index=0)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_population)
        source = inspect.getsource(execution.generate_gate7_high_scale_world)
        self.assertIn("gate7-high-scale-routing-bandwidth-hidden-v0", source)
        self.assertIn("gate7-high-scale-routing-bandwidth-hints-v0", source)
        self.assertNotIn("transition-bridge-hidden", source)

    def test_paired_and_stratified_bootstraps_are_deterministic(self) -> None:
        treatment_vector = (True,) * 40 + (False,) * 24
        reference_vector = (True,) * 20 + (False,) * 44
        treatments = {}
        references = {}
        for checkpoint in (0, 1, 2):
            treatments[checkpoint] = self._condition(
                checkpoint=checkpoint,
                population=1024,
                condition="global_score",
                covered=treatment_vector,
                k=None,
            )
            references[checkpoint] = self._condition(
                checkpoint=checkpoint,
                population=1024,
                condition="global_hash",
                covered=reference_vector,
                k=None,
            )
        first = paired_gate7_high_scale_summary(
            comparison="c0_global_score_vs_global_hash",
            treatment=treatments[0],
            reference=references[0],
        )
        second = paired_gate7_high_scale_summary(
            comparison="c0_global_score_vs_global_hash",
            treatment=treatments[0],
            reference=references[0],
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.coverage_delta, 20 / 64)
        pooled_first = stratified_gate7_high_scale_global_summary(
            population=1024,
            treatment_by_checkpoint=treatments,
            reference_by_checkpoint=references,
        )
        pooled_second = stratified_gate7_high_scale_global_summary(
            population=1024,
            treatment_by_checkpoint=treatments,
            reference_by_checkpoint=references,
        )
        self.assertEqual(pooled_first, pooled_second)
        self.assertGreater(pooled_first.bootstrap_ci_low, 0.0)

    def test_condition_names_bind_k_identity(self) -> None:
        for k in execution.GATE7_HIGH_SCALE_K_LADDER:
            self.assertEqual(execution._condition_mode(f"bounded_score_k{k}"), ("bounded_score", k))
            self.assertEqual(execution._condition_mode(f"bounded_hash_k{k}"), ("bounded_hash", k))
        self.assertEqual(execution._condition_mode("global_score"), ("global_score", None))
        self.assertEqual(execution._condition_mode("global_hash"), ("global_hash", None))

    def test_runner_source_enforces_global_gate_then_ascending_first_pass(self) -> None:
        source = inspect.getsource(runner.run_gate7_high_scale_routing_bandwidth)
        global_position = source.index("opening global reference pair")
        viability_position = source.index("reference_is_viable")
        k_loop_position = source.index("for k_position, k in enumerate(GATE7_HIGH_SCALE_K_LADDER")
        break_position = source.index('if tier_outcome.startswith("G7_K_REQUIRED_")')
        self.assertLess(global_position, viability_position)
        self.assertLess(viability_position, k_loop_position)
        self.assertLess(k_loop_position, break_position)
        self.assertIn("GATE7_HIGH_SCALE_RESOURCE_FRONTIER_REACHED", source)
        self.assertIn("for k_position, k in enumerate(GATE7_HIGH_SCALE_K_LADDER", source)
        self.assertIn("classify_completed_tier", source)
        self.assertNotIn("train_gate7", source)
        self.assertNotIn("confirmation_opened\": True", source)

    def test_independent_auditor_duplicates_classifier_and_does_not_import_execution(self) -> None:
        self.assertEqual(audit.POPULATIONS, execution.GATE7_HIGH_SCALE_POPULATIONS)
        self.assertEqual(audit.K_LADDER, execution.GATE7_HIGH_SCALE_K_LADDER)
        self.assertEqual(
            {index: {"sha256": row["sha256"], "fingerprint": row["fingerprint"]} for index, row in audit.CHECKPOINTS.items()},
            execution.GATE7_HIGH_SCALE_CHECKPOINTS,
        )
        passing = {
            16: {
                **{f"c{checkpoint}_k16_score_vs_hash": 0.01 for checkpoint in (0, 1, 2)},
                **{f"c{checkpoint}_k16_score_vs_global": -0.049 for checkpoint in (0, 1, 2)},
            }
        }
        self.assertEqual(audit._classify_k(passing), 16)
        source = inspect.getsource(audit)
        self.assertNotIn("from .gate7_high_scale_routing_bandwidth import", source)
        self.assertNotIn("generate_gate7_high_scale_world", source)
        self.assertNotIn("import torch", source)

    def test_tests_do_not_generate_or_execute_high_scale_worlds(self) -> None:
        tree = ast.parse(inspect.getsource(type(self)))
        forbidden = {"generate_gate7_high_scale_world", "run_gate7_high_scale_routing_bandwidth"}
        called = set()
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
