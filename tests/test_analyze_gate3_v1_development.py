from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import ai_hypothesis.population_compute.analyze_gate3_v1_development as audit_module
from ai_hypothesis.population_compute.analyze_gate3_v1_development import audit_gate3_v1_development
from ai_hypothesis.population_compute.gate3_v1_development import (
    Gate3V1ConditionEvaluation,
    Gate3V1DevelopmentResult,
    Gate3V1TrainingConfig,
    Gate3V1TrainingSummary,
    build_gate3_v1_paired_summaries,
)
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import (
    GATE3_V1_SEARCH_ROUNDS,
    Gate3V1ControlMode,
)


PRODUCTION_BOOTSTRAP = audit_module.BOOTSTRAP_SAMPLES
TEST_BOOTSTRAP = 32
audit_module.BOOTSTRAP_SAMPLES = TEST_BOOTSTRAP


def _covered(count: int) -> tuple[bool, ...]:
    return tuple(index < count for index in range(256))


def _condition(
    *,
    depth: int,
    capacity: int,
    mode: Gate3V1ControlMode,
    count: int,
    fingerprint: str,
) -> Gate3V1ConditionEvaluation:
    covered = _covered(count)
    rounds = GATE3_V1_SEARCH_ROUNDS[depth]
    if mode is Gate3V1ControlMode.COLLAPSED_DIVERSITY or capacity == 1:
        productive = depth
        terminal_count = 2
    else:
        productive = rounds
        terminal_count = max(2, min(rounds, capacity * 2))
    sink = rounds - productive
    return Gate3V1ConditionEvaluation(
        depth=depth,
        reserve_capacity=capacity,
        mode=mode,
        world_count=256,
        coverage_rate=sum(covered) / 256,
        world_seeds=tuple((1 << 30) + index for index in range(256)),
        covered_by_world=covered,
        generated_terminal_count_by_world=(terminal_count,) * 256,
        unique_generated_terminal_count_by_world=(terminal_count,) * 256,
        productive_rounds_by_world=(productive,) * 256,
        sink_rounds_by_world=(sink,) * 256,
        productive_work_fraction_by_world=(productive / rounds,) * 256,
        total_learned_updates_per_world={6: 256, 8: 1024, 10: 4096}[depth],
        learned_parameter_count=19_649,
        parameter_fingerprint=fingerprint,
    )


def _payload() -> dict[str, object]:
    fingerprint = "a" * 64
    capacities = {6: (1, 4, 16), 8: (1, 4, 16, 64), 10: (1, 4, 16, 64, 256)}
    stable_counts = {
        6: {1: 12, 4: 28, 16: 50},
        8: {1: 8, 4: 20, 16: 45, 64: 90},
        10: {1: 4, 4: 12, 16: 35, 64: 80, 256: 150},
    }
    conditions: list[Gate3V1ConditionEvaluation] = []
    for depth, widths in capacities.items():
        l1 = stable_counts[depth][1]
        for capacity in widths:
            stable = stable_counts[depth][capacity]
            conditions.append(
                _condition(
                    depth=depth,
                    capacity=capacity,
                    mode=Gate3V1ControlMode.STABLE_RESERVE,
                    count=stable,
                    fingerprint=fingerprint,
                )
            )
            conditions.append(
                _condition(
                    depth=depth,
                    capacity=capacity,
                    mode=Gate3V1ControlMode.COLLAPSED_DIVERSITY,
                    count=l1,
                    fingerprint=fingerprint,
                )
            )
            reshuffled = l1 if capacity == 1 else max(l1, stable - 35)
            conditions.append(
                _condition(
                    depth=depth,
                    capacity=capacity,
                    mode=Gate3V1ControlMode.RESHUFFLED_CONTINUITY,
                    count=reshuffled,
                    fingerprint=fingerprint,
                )
            )

    paired = build_gate3_v1_paired_summaries(conditions, bootstrap_samples=TEST_BOOTSTRAP)
    result = Gate3V1DevelopmentResult(
        experiment_version="gate3-v1-sparse-active-reserve-development-v0",
        evaluation_split="development",
        confirmation_opened=False,
        training=Gate3V1TrainingSummary(
            training_seed=0,
            steps=1200,
            examples_seen=1200 * 256,
            initial_loss=1.0,
            final_loss=0.1,
            mean_last_50_loss=0.1,
            learned_parameter_count=19_649,
            parameter_fingerprint=fingerprint,
        ),
        training_config=Gate3V1TrainingConfig(),
        evaluation_world_count=256,
        evaluation_batch_size=64,
        bootstrap_samples=TEST_BOOTSTRAP,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    return result.to_dict()


class Gate3V1DevelopmentAuditTests(unittest.TestCase):
    def test_production_bootstrap_count_is_frozen_at_2000(self) -> None:
        self.assertEqual(PRODUCTION_BOOTSTRAP, 2000)

    def test_clean_fixture_is_valid_development_only_outcome_d(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            audit = audit_gate3_v1_development(path)
            self.assertTrue(audit.artifact_valid, audit.errors)
            self.assertEqual(audit.scientific_status, "DEVELOPMENT_ONLY_NO_GATE_VERDICT")
            self.assertEqual(audit.directional_outcome, "D_CLEAN_SPARSE_ACTIVE_POPULATION_PATTERN")
            self.assertTrue(all(value > 0.0 for value in audit.primary_deltas.values()))

    def test_work_tamper_invalidates_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = _payload()
            payload["conditions"][0]["total_learned_updates_per_world"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            audit = audit_gate3_v1_development(path)
            self.assertFalse(audit.artifact_valid)
            self.assertTrue(any("learned-work total" in error for error in audit.errors))

    def test_collapsed_identity_tamper_invalidates_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = _payload()
            for row in payload["conditions"]:
                if row["depth"] == 10 and row["reserve_capacity"] == 256 and row["mode"] == "collapsed_diversity":
                    row["covered_by_world"][255] = True
                    row["coverage_rate"] = sum(row["covered_by_world"]) / 256
                    break
            path.write_text(json.dumps(payload), encoding="utf-8")
            audit = audit_gate3_v1_development(path)
            self.assertFalse(audit.artifact_valid)
            self.assertTrue(any("collapsed L256 logical-one identity" in error for error in audit.errors))


if __name__ == "__main__":
    unittest.main()
