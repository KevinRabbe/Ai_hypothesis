from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_distributed_transformation_worlds.py"
)
ARCH_PATH = ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_factorized_organism_architecture.py"
)
QUALIFIED_RUNTIME_PATH = ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_factorized_organism_runtime.py"
)
SCIENCE_RUNTIME_PATH = ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_v1_scientific_population_runtime.py"
)
RUNNER_PATH = ROOT / "scripts/run_gate8_v1_population_science.py"
WRAPPER_PATH = ROOT / "scripts/run_gate8_v1_population_science.ps1"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worlds = _load(WORLD_PATH, "gate8_v1_population_science_test_worlds")
architecture = _load(ARCH_PATH, "gate8_v1_population_science_test_architecture")
qualified = _load(
    QUALIFIED_RUNTIME_PATH, "gate8_v1_population_science_test_qualified_runtime"
)
science = _load(SCIENCE_RUNTIME_PATH, "gate8_v1_population_science_test_runtime")
runner = _load(RUNNER_PATH, "gate8_v1_population_science_test_runner")


class Gate8V1ScientificPopulationExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(47)
        cls.model = architecture.Gate8V1SharedWorkerCore().eval()
        cls.table = science.compile_gate8_v1_transition_table(
            model=cls.model,
            checkpoint_seed=0,
            checkpoint_sha256="a" * 64,
        )

    def test_transition_table_is_complete_deterministic_and_hash_bound(self) -> None:
        self.table.validate()
        self.assertEqual(len(self.table.message_codes), 256)
        self.assertTrue(all(len(row) == 8 for row in self.table.message_codes))
        again = science.compile_gate8_v1_transition_table(
            model=self.model,
            checkpoint_seed=0,
            checkpoint_sha256="a" * 64,
        )
        self.assertEqual(again.message_codes, self.table.message_codes)
        self.assertEqual(again.table_sha256, self.table.table_sha256)

    def test_compiled_modes_match_qualified_runtime(self) -> None:
        for population, depth in ((32, 4), (64, 8), (128, 16)):
            generated = worlds.generate_gate8_world(
                split="contract",
                seed=0,
                world_index=17,
                population=population,
                depth=depth,
            )
            for mode in (
                qualified.GATE8_V1_RUNTIME_FULL,
                qualified.GATE8_V1_RUNTIME_NO_COMMUNICATION,
                qualified.GATE8_V1_RUNTIME_SHUFFLED_WORKER,
            ):
                expected = qualified.run_gate8_v1_contract_runtime(
                    model=self.model,
                    world=generated.public,
                    mode=mode,
                )
                observed = science.run_gate8_v1_scientific_population_contract_probe(
                    table=self.table,
                    world=generated.public,
                    mode=mode,
                )
                self.assertEqual(observed.target_reached, expected.target_reached)
                self.assertEqual(observed.predicted_symbol, expected.predicted_symbol)
                self.assertEqual(observed.rounds, expected.rounds_executed)
                self.assertEqual(observed.active_workers, expected.recurrent_updates)
                self.assertEqual(observed.recurrent_updates, expected.recurrent_updates)
                self.assertEqual(
                    observed.delivered_messages, expected.delivered_messages
                )
                self.assertEqual(observed.communicated_bits, expected.communicated_bits)

    def test_planned_execution_reuses_exact_validated_topology(self) -> None:
        generated = worlds.generate_gate8_world(
            split="contract",
            seed=0,
            world_index=23,
            population=64,
            depth=8,
        )
        plan = science.compile_gate8_v1_scientific_world_plan(
            generated.public, admitted_split="contract"
        )
        plan.validate()
        first = science.run_gate8_v1_scientific_population_plan(
            table=self.table,
            plan=plan,
            mode=science.GATE8_V1_SHUFFLED_MESSAGE_MODE,
        )
        second = science.run_gate8_v1_scientific_population_plan(
            table=self.table,
            plan=plan,
            mode=science.GATE8_V1_SHUFFLED_MESSAGE_MODE,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.delivered_messages * 8, first.communicated_bits)

    def test_target_worker_only_is_one_update_without_communication(self) -> None:
        generated = worlds.generate_gate8_world(
            split="contract",
            seed=0,
            world_index=29,
            population=128,
            depth=16,
        )
        result = science.run_gate8_v1_scientific_population_contract_probe(
            table=self.table,
            world=generated.public,
            mode=science.GATE8_V1_TARGET_WORKER_ONLY_MODE,
        )
        self.assertTrue(result.target_reached)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.active_workers, 1)
        self.assertEqual(result.recurrent_updates, 1)
        self.assertEqual(result.delivered_messages, 0)
        self.assertEqual(result.communicated_bits, 0)

    def test_production_runtime_rejects_non_test_before_validation(self) -> None:
        class Rejected:
            split = "contract"

            def validate(self):
                raise AssertionError("validation must not run after split rejection")

        with self.assertRaisesRegex(ValueError, "test worlds only"):
            science.run_gate8_v1_scientific_population_runtime(
                table=self.table,
                world=Rejected(),
                mode=science.GATE8_V1_FULL_MODE,
            )

    def test_random_control_is_deterministic_and_bounded(self) -> None:
        values = [
            science.gate8_v1_deterministic_random_answer(f"g8_{index:024x}")
            for index in range(64)
        ]
        repeated = [
            science.gate8_v1_deterministic_random_answer(f"g8_{index:024x}")
            for index in range(64)
        ]
        self.assertEqual(values, repeated)
        self.assertTrue(all(0 <= value < 16 for value in values))
        self.assertGreater(len(set(values)), 1)

    def test_bootstrap_is_deterministic_equal_seed_and_paired(self) -> None:
        matrix = np.zeros((3, 512), dtype=np.uint8)
        matrix[0, :384] = 1
        matrix[1, :400] = 1
        matrix[2, :416] = 1
        first = runner._bootstrap_ci(
            matrix, namespace="contract-ci", samples=20_000
        )
        second = runner._bootstrap_ci(
            matrix, namespace="contract-ci", samples=20_000
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], float(matrix.mean()))
        self.assertGreaterEqual(first[1], float(matrix.mean()))
        zeros = np.zeros((3, 512), dtype=np.uint8)
        delta = runner._bootstrap_delta_ci(
            matrix, zeros, namespace="contract-delta", samples=20_000
        )
        self.assertGreater(delta[0], 0.0)

    def test_controls_do_not_borrow_organism_parameters(self) -> None:
        correctness = np.ones((3, 512), dtype=np.uint8)
        oracle = runner._metric_row(
            population=32,
            depth=4,
            mode=runner.ORACLE_MODE,
            correctness=correctness,
            resource={},
            bootstrap_samples=128,
        )
        self.assertEqual(oracle["learned_parameter_count"], 0)
        self.assertIsNone(oracle["capability_per_learned_parameter"])
        self.assertEqual(oracle["capability_per_normalized_compute"], 0.0)

        organism = runner._metric_row(
            population=32,
            depth=4,
            mode="full",
            correctness=correctness,
            resource={"active_workers": 1_536, "recurrent_updates": 1_536},
            bootstrap_samples=128,
        )
        self.assertEqual(organism["learned_parameter_count"], 19_649)
        self.assertGreater(organism["capability_per_learned_parameter"], 0.0)

    def test_execution_plan_and_sources_freeze_exact_boundaries(self) -> None:
        plan = science.gate8_v1_scientific_population_runtime_plan()
        self.assertEqual(
            plan["scientific_protocol_head"],
            "6bb89111a47713bea0a23bb1cae662ed5ec56b42",
        )
        self.assertEqual(
            plan["gemma_binding_result_head"],
            "8237732aecbec083c66668de9fae132e0cc4c1f9",
        )
        self.assertEqual(plan["transition_table_entries_per_checkpoint"], 2_048)
        self.assertEqual(plan["test_seed"], 0)
        self.assertEqual(plan["test_world_indices"], [0, 511])
        self.assertFalse(plan["reference_model_loaded"])
        self.assertFalse(plan["reference_inference_performed"])
        self.assertFalse(plan["training_performed"])

        runner_text = RUNNER_PATH.read_text(encoding="utf-8")
        for digest in (
            "3005369a4830c12baee8ffa7fedc1bed0f1888784e1043bd88f4afd2b7cddde9",
            "cbcae487dd7f4c695e1d6a83a61926cd43f5ccf6add1a7469c16a15697d22d07",
            "e1e35b3864354e8f3398497a897b6a759dfa3454a33d866de63784a323f461e4",
        ):
            self.assertIn(digest, runner_text)
        self.assertIn('split="test"', runner_text)
        self.assertIn('seed=0', runner_text)
        self.assertIn('weights_only=True', runner_text)
        for forbidden in (
            "AutoModel",
            "AutoTokenizer",
            "from_pretrained(",
            "model.safetensors",
            "optimizer.step",
            ".backward(",
        ):
            self.assertNotIn(forbidden, runner_text)

        wrapper_text = WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertIn("GATE8_V1_POPULATION_SCIENCE_WRAPPER_SMOKE", wrapper_text)
        self.assertIn("Reference model: CLOSED", wrapper_text)


if __name__ == "__main__":
    unittest.main()
