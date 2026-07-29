from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import ai_hypothesis.population_compute.analyze_gate3_development as audit_module
from ai_hypothesis.population_compute.analyze_gate3_development import audit_gate3_development
from ai_hypothesis.population_compute.gate3_development import (
    Gate3ConditionEvaluation,
    Gate3DevelopmentResult,
    Gate3TrainingConfig,
    Gate3TrainingSummary,
    build_gate3_paired_summaries,
)
from ai_hypothesis.population_compute.gate3_hypothesis_population import Gate3ControlMode


FROZEN_BOOTSTRAP_SAMPLES = audit_module.BOOTSTRAP_SAMPLES
TEST_BOOTSTRAP_SAMPLES = 32
audit_module.BOOTSTRAP_SAMPLES = TEST_BOOTSTRAP_SAMPLES


def _solved(count: int) -> tuple[bool, ...]:
    return tuple(index < count for index in range(256))


def _condition(
    *,
    depth: int,
    width: int,
    mode: Gate3ControlMode,
    solved_count: int,
    fingerprint: str,
) -> Gate3ConditionEvaluation:
    solved = _solved(solved_count)
    bit_by_world = tuple(1.0 if value else 0.5 for value in solved)
    phase_count = 2 * depth
    max_width = 1 << depth

    if width == 1:
        unique = (1.0,) * phase_count
        survival = (0.5,) * phase_count
    elif mode is Gate3ControlMode.COLLAPSED_DIVERSITY:
        unique = (1.0,) * phase_count
        survival = (0.4,) * phase_count
    else:
        branch_unique = tuple(float(min(width, 1 << (phase + 1))) for phase in range(depth))
        unique = branch_unique + (float(width),) * depth
        survival = (0.7,) * phase_count

    if width == max_width and mode is Gate3ControlMode.STABLE_DIVERSE:
        survival_list = list(survival)
        survival_list[depth - 1] = 1.0
        survival = tuple(survival_list)

    return Gate3ConditionEvaluation(
        depth=depth,
        width=width,
        mode=mode,
        world_count=256,
        exact_solve_rate=sum(solved) / 256,
        bit_accuracy=sum(bit_by_world) / 256,
        learned_updates_per_world={4: 128, 6: 768, 8: 4096}[depth],
        unique_world_observations_per_world=2 * depth,
        learned_parameter_count=19_873,
        parameter_fingerprint=fingerprint,
        world_seeds=tuple((1 << 30) + index for index in range(256)),
        solved_by_world=solved,
        bit_accuracy_by_world=bit_by_world,
        correct_candidate_survival_rate_by_phase=survival,
        mean_unique_candidates_by_phase=unique,
    )


def _payload() -> dict[str, object]:
    fingerprint = "f" * 64
    stable_counts = {
        4: {1: 20, 4: 30, 16: 45},
        6: {1: 20, 4: 30, 16: 45, 64: 80},
        8: {1: 10, 4: 20, 16: 40, 64: 80, 256: 150},
    }
    conditions: list[Gate3ConditionEvaluation] = []
    for depth, widths in stable_counts.items():
        for width, stable_count in widths.items():
            for mode in Gate3ControlMode:
                if width == 1:
                    count = stable_count
                elif mode is Gate3ControlMode.STABLE_DIVERSE:
                    count = stable_count
                elif mode is Gate3ControlMode.COLLAPSED_DIVERSITY:
                    count = max(5, stable_count - 60)
                else:
                    count = max(7, stable_count - 50)
                conditions.append(
                    _condition(
                        depth=depth,
                        width=width,
                        mode=mode,
                        solved_count=count,
                        fingerprint=fingerprint,
                    )
                )

    paired = build_gate3_paired_summaries(
        conditions,
        bootstrap_samples=TEST_BOOTSTRAP_SAMPLES,
    )
    result = Gate3DevelopmentResult(
        experiment_version="gate3-hypothesis-population-development-v0",
        evaluation_split="development",
        confirmation_opened=False,
        training=Gate3TrainingSummary(
            training_seed=0,
            steps=1200,
            examples_seen=1200 * 128,
            initial_loss=1.0,
            final_loss=0.1,
            mean_last_50_loss=0.1,
            learned_parameter_count=19_873,
            parameter_fingerprint=fingerprint,
            stable_training_condition_count=12,
        ),
        training_config=Gate3TrainingConfig(),
        evaluation_world_count=256,
        evaluation_batch_size=64,
        bootstrap_samples=TEST_BOOTSTRAP_SAMPLES,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    return result.to_dict()


class Gate3DevelopmentAuditTests(unittest.TestCase):
    def test_production_bootstrap_count_is_frozen_at_2000(self) -> None:
        self.assertEqual(FROZEN_BOOTSTRAP_SAMPLES, 2000)

    def test_clean_directional_fixture_is_valid_development_only_outcome_d(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate3-development.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            audit = audit_gate3_development(path)
            self.assertTrue(audit.artifact_valid, audit.errors)
            self.assertEqual(audit.scientific_status, "DEVELOPMENT_ONLY_NO_GATE_VERDICT")
            self.assertEqual(audit.directional_outcome, "D_CLEAN_DIRECTIONAL_PATTERN")
            self.assertTrue(all(value > 0.0 for value in audit.primary_deltas.values()))

    def test_work_accounting_tamper_invalidates_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate3-development.json"
            payload = _payload()
            payload["conditions"][0]["learned_updates_per_world"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            audit = audit_gate3_development(path)
            self.assertFalse(audit.artifact_valid)
            self.assertTrue(any("learned-work identity failed" in error for error in audit.errors))

    def test_confirmation_opened_tamper_invalidates_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate3-development.json"
            payload = _payload()
            payload["confirmation_opened"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            audit = audit_gate3_development(path)
            self.assertFalse(audit.artifact_valid)
            self.assertTrue(any("confirmation must remain closed" in error for error in audit.errors))


if __name__ == "__main__":
    unittest.main()
