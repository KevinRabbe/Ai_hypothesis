from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.analyze_gate2_development import (
    EXPECTED_ENTITY_WIDTHS,
    EXPECTED_MODES,
    render_gate2_development_markdown,
    validate_gate2_development_result,
)


class Gate2DevelopmentAnalyzerTests(unittest.TestCase):
    def test_valid_full_matrix_is_accepted_and_rendered(self) -> None:
        payload = _valid_payload()
        validate_gate2_development_result(payload)
        report = render_gate2_development_markdown(payload)
        self.assertIn("DEVELOPMENT ONLY — NO GATE VERDICT", report)
        self.assertIn("stable_width_vs_width1", report)
        self.assertIn("stable_vs_reshuffled", report)
        self.assertIn("stable_vs_reset", report)
        self.assertIn("| 256 | 256 | 1 |", report)

    def test_confirmation_opened_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["confirmation_opened"] = True
        with self.assertRaisesRegex(ValueError, "confirmation closed"):
            validate_gate2_development_result(payload)

    def test_checkpoint_drift_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["conditions"][-1]["parameter_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "immutable checkpoint"):
            validate_gate2_development_result(payload)

    def test_width_one_identity_failure_is_rejected(self) -> None:
        payload = _valid_payload()
        row = next(
            row
            for row in payload["paired_summaries"]
            if row["comparison"] == "stable_vs_reshuffled"
            and row["entity_count"] == 64
            and row["treatment_width"] == 1
        )
        row["exact_solve_delta"] = 0.01
        with self.assertRaisesRegex(ValueError, "identity control failed"):
            validate_gate2_development_result(payload)


def _valid_payload() -> dict[str, object]:
    fingerprint = "f" * 64
    parameter_count = 12345
    world_count = 4
    conditions: list[dict[str, object]] = []
    for entity_count, widths in EXPECTED_ENTITY_WIDTHS.items():
        seeds = list(range(1000 + entity_count, 1000 + entity_count + world_count))
        for width in widths:
            for mode in EXPECTED_MODES:
                conditions.append(
                    {
                        "entity_count": entity_count,
                        "width": width,
                        "mode": mode,
                        "world_count": world_count,
                        "exact_solve_rate": 0.25,
                        "bit_accuracy": 0.5,
                        "collision_load": entity_count // width,
                        "learned_updates_per_world": 8 * entity_count,
                        "inspected_entities_per_world": entity_count,
                        "inspected_observations_per_world": 8 * entity_count,
                        "learned_parameter_count": parameter_count,
                        "parameter_fingerprint": fingerprint,
                        "world_seeds": seeds,
                        "solved_by_world": [False, False, True, False],
                    }
                )

    paired: list[dict[str, object]] = []
    for entity_count, widths in EXPECTED_ENTITY_WIDTHS.items():
        for width in widths[1:]:
            paired.append(_pair("stable_width_vs_width1", entity_count, width, 1))
        for width in widths:
            paired.append(_pair("stable_vs_reshuffled", entity_count, width, width))
            paired.append(_pair("stable_vs_reset", entity_count, width, width))

    return {
        "experiment_version": "gate2-persistent-state-development-v0",
        "evaluation_split": "development",
        "confirmation_opened": False,
        "training": {
            "training_seed": 0,
            "steps": 1000,
            "examples_seen": 32000,
            "initial_loss": 0.7,
            "final_loss": 0.4,
            "mean_last_50_loss": 0.42,
            "learned_parameter_count": parameter_count,
            "parameter_fingerprint": fingerprint,
            "stable_training_condition_count": 12,
        },
        "training_config": {},
        "evaluation_world_count": world_count,
        "evaluation_batch_size": 4,
        "conditions": conditions,
        "paired_summaries": paired,
        "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
    }


def _pair(comparison: str, entity_count: int, treatment_width: int, reference_width: int) -> dict[str, object]:
    return {
        "comparison": comparison,
        "entity_count": entity_count,
        "treatment_width": treatment_width,
        "reference_width": reference_width,
        "treatment_mode": "stable_persistent",
        "reference_mode": "stable_persistent",
        "world_count": 4,
        "treatment_only": 0,
        "reference_only": 0,
        "both_solved": 1,
        "neither_solved": 3,
        "exact_solve_delta": 0.0,
        "bootstrap_ci_low": 0.0,
        "bootstrap_ci_high": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
