from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute.analyze_gate3_v1_robustness import (
    audit_gate3_v1_robustness,
)
from ai_hypothesis.population_compute.gate3_v1_development import (
    Gate3V1ConditionEvaluation,
    Gate3V1DevelopmentResult,
    Gate3V1TrainingConfig,
    Gate3V1TrainingSummary,
    build_gate3_v1_paired_summaries,
)
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import (
    GATE3_V1_DEPTHS,
    GATE3_V1_DEVELOPMENT_WORLD_START,
    GATE3_V1_RESERVE_CAPACITIES,
    GATE3_V1_SEARCH_ROUNDS,
    Gate3V1ControlMode,
)
from ai_hypothesis.population_compute.run_gate3_v1_robustness import (
    GATE3_V1_ROBUSTNESS_SEEDS,
    run_gate3_v1_robustness_seed,
)


PARAMS = 19_649
FINGERPRINT = "a" * 64
WORLD_COUNT = 256


def _synthetic_payload(training_seed: int) -> dict[str, object]:
    worlds = tuple(range(GATE3_V1_DEVELOPMENT_WORLD_START, GATE3_V1_DEVELOPMENT_WORLD_START + WORLD_COUNT))
    conditions = []
    for depth in GATE3_V1_DEPTHS:
        rounds = GATE3_V1_SEARCH_ROUNDS[depth]
        total_updates = {6: 256, 8: 1024, 10: 4096}[depth]
        for capacity in GATE3_V1_RESERVE_CAPACITIES[depth]:
            for mode in Gate3V1ControlMode:
                conditions.append(
                    Gate3V1ConditionEvaluation(
                        depth=depth,
                        reserve_capacity=capacity,
                        mode=mode,
                        world_count=WORLD_COUNT,
                        coverage_rate=0.0,
                        world_seeds=worlds,
                        covered_by_world=(False,) * WORLD_COUNT,
                        generated_terminal_count_by_world=(0,) * WORLD_COUNT,
                        unique_generated_terminal_count_by_world=(0,) * WORLD_COUNT,
                        productive_rounds_by_world=(0,) * WORLD_COUNT,
                        sink_rounds_by_world=(rounds,) * WORLD_COUNT,
                        productive_work_fraction_by_world=(0.0,) * WORLD_COUNT,
                        total_learned_updates_per_world=total_updates,
                        learned_parameter_count=PARAMS,
                        parameter_fingerprint=FINGERPRINT,
                    )
                )
    paired = build_gate3_v1_paired_summaries(tuple(conditions), bootstrap_samples=2_000)
    result = Gate3V1DevelopmentResult(
        experiment_version="gate3-v1-sparse-active-reserve-development-v0",
        evaluation_split="development",
        confirmation_opened=False,
        training=Gate3V1TrainingSummary(
            training_seed=training_seed,
            steps=1_200,
            examples_seen=1_200 * 256,
            initial_loss=1.0,
            final_loss=0.1,
            mean_last_50_loss=0.1,
            learned_parameter_count=PARAMS,
            parameter_fingerprint=FINGERPRINT,
        ),
        training_config=Gate3V1TrainingConfig(),
        evaluation_world_count=WORLD_COUNT,
        evaluation_batch_size=64,
        bootstrap_samples=2_000,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    return result.to_dict()


class Gate3V1RobustnessTests(unittest.TestCase):
    def test_only_precommitted_robustness_seeds_exist(self) -> None:
        self.assertEqual(GATE3_V1_ROBUSTNESS_SEEDS, (1, 2))

    def test_robustness_runner_rejects_non_precommitted_seed_before_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "output"
            with self.assertRaises(ValueError):
                run_gate3_v1_robustness_seed(training_seed=0, output_root=root)
            with self.assertRaises(ValueError):
                run_gate3_v1_robustness_seed(training_seed=3, output_root=root)

    def test_independent_auditor_accepts_seed1_and_seed2_full_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for seed in (1, 2):
                path = Path(temp) / f"seed-{seed}.json"
                path.write_text(json.dumps(_synthetic_payload(seed)), encoding="utf-8")
                audit = audit_gate3_v1_robustness(path)
                self.assertTrue(audit.artifact_valid, audit.errors)
                self.assertEqual(audit.training_seed, seed)
                self.assertEqual(audit.directional_outcome, "A_NO_LATENT_RESERVE_BENEFIT")
                self.assertEqual(audit.scientific_status, "ROBUSTNESS_DEVELOPMENT_ONLY_NO_GATE_VERDICT")

    def test_independent_auditor_rejects_seed0_and_seed3(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for seed in (0, 3):
                path = Path(temp) / f"seed-{seed}.json"
                path.write_text(json.dumps(_synthetic_payload(seed)), encoding="utf-8")
                audit = audit_gate3_v1_robustness(path)
                self.assertFalse(audit.artifact_valid)
                self.assertIn("robustness training seed must be exactly 1 or 2", audit.errors)

    def test_seed_entrypoints_expose_output_path_only(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        for name in ("run_gate3_v1_robustness_seed1.py", "run_gate3_v1_robustness_seed2.py"):
            source = (repo / "ai_hypothesis" / "population_compute" / name).read_text(encoding="utf-8")
            self.assertIn('parser.add_argument("--output-root"', source)
            self.assertNotIn("--training-seed", source)
            self.assertNotIn("--steps", source)
            self.assertNotIn("--learning-rate", source)
            self.assertNotIn("--batch-size", source)

    def test_seed_wrappers_bind_their_seed(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        seed1 = (repo / "scripts" / "run_gate3_v1_robustness_seed1.ps1").read_text(encoding="utf-8")
        seed2 = (repo / "scripts" / "run_gate3_v1_robustness_seed2.ps1").read_text(encoding="utf-8")
        self.assertIn("-TrainingSeed 1", seed1)
        self.assertNotIn("-TrainingSeed 2", seed1)
        self.assertIn("-TrainingSeed 2", seed2)
        self.assertNotIn("-TrainingSeed 1", seed2)


if __name__ == "__main__":
    unittest.main()
