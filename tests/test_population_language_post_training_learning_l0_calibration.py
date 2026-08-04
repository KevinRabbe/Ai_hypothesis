from __future__ import annotations

import copy
import hashlib
import unittest

from ai_hypothesis.population_language import (
    post_training_learning_l0_calibration as calibration,
)
from ai_hypothesis.population_language import (
    post_training_learning_l0_protocol as protocol,
)
from ai_hypothesis.population_language import (
    post_training_learning_l0_world as world,
)


def _artifact_hash(candidate_id: str, model_seed: int) -> str:
    return hashlib.sha256(f"{candidate_id}|{model_seed}".encode("utf-8")).hexdigest()


def _result_row(
    candidate: calibration.CalibrationCandidate,
    model_seed: int,
    calibration_world_seed: int,
    *,
    direct_gain: float = 0.10,
    composition_gain: float = 0.02,
) -> dict[str, object]:
    baseline_direct = 0.25
    baseline_composition = 0.10
    return {
        "candidate_id": candidate.identifier,
        "rank": candidate.rank,
        "learning_rate": candidate.learning_rate,
        "adaptation_updates": candidate.updates,
        "model_seed": model_seed,
        "calibration_world_seed": calibration_world_seed,
        "source_adapter_head": calibration.SOURCE_ADAPTER_HEAD,
        "candidate_grid_sha256": calibration.GRID_SHA256,
        "adapter_initialization_seed":
            protocol.adapter_initialization_seed(model_seed),
        "optimizer": calibration.OPTIMIZER,
        "adamw_betas": list(calibration.ADAMW_BETAS),
        "adamw_epsilon": calibration.ADAMW_EPSILON,
        "weight_decay": calibration.WEIGHT_DECAY,
        "max_gradient_norm": calibration.MAX_GRADIENT_NORM,
        "microbatch_size": calibration.MICROBATCH_SIZE,
        "learning_rate_schedule": calibration.LEARNING_RATE_SCHEDULE,
        "adaptation_order": calibration.ADAPTATION_ORDER,
        "autocast_mode": calibration.AUTOCAST_MODE,
        "worker_count": calibration.WORKER_COUNT,
        "early_stopping_used": False,
        "adaptation_examples": world.ADAPTATION_EXAMPLES,
        "adaptation_example_presentations": candidate.example_presentations,
        "direct_holdout_episodes": world.DIRECT_HOLDOUT_EXAMPLES,
        "composition_episodes": world.CALIBRATION_EXAMPLES,
        "composition_depth": world.SPLIT_DEPTH["calibration"],
        "trainable_adaptation_parameters": candidate.trainable_parameters,
        "persisted_adaptation_bytes": candidate.persisted_fp32_bytes,
        "base_checkpoint_sha256": "a" * 64,
        "base_checkpoint_after_sha256": "a" * 64,
        "adaptation_artifact_sha256":
            _artifact_hash(candidate.identifier, model_seed),
        "adaptation_artifact_payload_kind":
            protocol.ARTIFACT_PAYLOAD_KIND,
        "adaptation_artifact_contains_raw_examples": False,
        "adaptation_uses_only_declared_examples_and_gradients": True,
        "raw_adaptation_examples_available_at_evaluation": False,
        "external_retrieval_enabled_at_evaluation": False,
        "world_seed_available_to_adaptation": False,
        "world_rule_parameters_available_to_adaptation": False,
        "world_generator_imported_by_model_runtime": False,
        "symbolic_rule_fitting_used": False,
        "symbolic_execution_used_at_evaluation": False,
        "model_logits_are_authoritative": True,
        "final_worlds_loaded_during_calibration": False,
        "final_world_labels_used_during_calibration": False,
        "calibration_world_fingerprints":
            world.calibration_world_fingerprints(),
        "original_l0_path_bitwise_identical": True,
        "transient_state_cleared_before_restart": True,
        "base_checkpoint_loaded_fresh_after_restart": True,
        "adaptation_artifact_loaded_after_restart": True,
        "fresh_process_restart": True,
        "baseline_direct_accuracy": baseline_direct,
        "immediate_direct_accuracy": baseline_direct + direct_gain,
        "post_restart_direct_accuracy": baseline_direct + direct_gain,
        "baseline_composition_accuracy": baseline_composition,
        "immediate_composition_accuracy":
            baseline_composition + composition_gain,
        "post_restart_composition_accuracy":
            baseline_composition + composition_gain,
    }


def _complete_rows(
    *,
    direct_gain: float = 0.10,
    composition_gain: float = 0.02,
) -> list[dict[str, object]]:
    return [
        _result_row(
            candidate,
            model_seed,
            calibration_world_seed,
            direct_gain=direct_gain,
            composition_gain=composition_gain,
        )
        for candidate in calibration.candidate_grid()
        for model_seed, calibration_world_seed in calibration.CALIBRATION_PAIRS
    ]


class PopulationLanguagePostTrainingLearningL0CalibrationTests(unittest.TestCase):
    def test_locked_grid_and_optimizer_contract(self) -> None:
        contract = calibration.validate_calibration_contract()
        self.assertTrue(contract["valid"], contract["checks"])
        grid = calibration.candidate_grid()
        self.assertEqual(len(grid), 48)
        self.assertEqual(calibration.EXPECTED_RESULT_ROWS, 144)
        self.assertEqual(grid[0].identifier, "r1-lr0p001-u32")
        self.assertEqual(grid[-1].identifier, "r6-lr0p01-u256")
        self.assertEqual(grid[0].trainable_parameters, 33_456)
        self.assertEqual(grid[-1].trainable_parameters, 180_176)
        self.assertEqual(grid[-1].persisted_fp32_bytes, 720_704)
        self.assertEqual(
            calibration.GRID_SHA256,
            "84264fbb475259ca224c01cee81700a62"
            "c6baf7e73017a0510fc5cbc6c036874",
        )
        self.assertFalse(calibration.EARLY_STOPPING_ALLOWED)
        self.assertEqual(calibration.MICROBATCH_SIZE, 8)
        self.assertEqual(calibration.WORKER_COUNT, 32)

    def test_complete_qualified_grid_uses_exact_resource_tie_breaks(self) -> None:
        result = calibration.evaluate_calibration(_complete_rows())
        self.assertEqual(
            result["run_classification"],
            calibration.CALIBRATION_RUN_VALID,
        )
        self.assertEqual(
            result["conclusion"],
            calibration.CALIBRATION_SELECTS_CANDIDATE,
        )
        self.assertEqual(result["qualified_candidate_count"], 48)
        self.assertEqual(result["selected_candidate_id"], "r1-lr0p001-u32")
        self.assertEqual(
            result["final_execution_eligibility"],
            calibration.FINAL_EXECUTION_ELIGIBLE_NOT_AUTHORIZED,
        )
        self.assertTrue(
            result["separate_explicit_authorization_still_required"]
        )

    def test_nonpositive_direct_gain_rejects_every_candidate(self) -> None:
        result = calibration.evaluate_calibration(
            _complete_rows(direct_gain=0.0)
        )
        self.assertEqual(
            result["run_classification"],
            calibration.CALIBRATION_RUN_VALID,
        )
        self.assertEqual(
            result["conclusion"],
            calibration.CALIBRATION_REJECTS_CANDIDATE,
        )
        self.assertIsNone(result["selected_candidate_id"])
        self.assertEqual(
            result["final_execution_eligibility"],
            calibration.FINAL_EXECUTION_NOT_ELIGIBLE,
        )

    def test_mean_composition_gain_must_reach_one_point(self) -> None:
        rows = _complete_rows(composition_gain=0.009)
        result = calibration.evaluate_calibration(rows)
        self.assertEqual(
            result["conclusion"],
            calibration.CALIBRATION_REJECTS_CANDIDATE,
        )

    def test_final_world_access_or_restart_drift_invalidates_run(self) -> None:
        rows = _complete_rows()
        rows[0] = copy.deepcopy(rows[0])
        rows[0]["final_world_labels_used_during_calibration"] = True
        result = calibration.evaluate_calibration(rows)
        self.assertEqual(
            result["run_classification"],
            calibration.CALIBRATION_RUN_INVALID,
        )
        self.assertEqual(
            result["conclusion"],
            calibration.INVALID_CALIBRATION_NO_SELECTION,
        )

        rows = _complete_rows()
        rows[0] = copy.deepcopy(rows[0])
        rows[0]["post_restart_composition_accuracy"] = (
            float(rows[0]["immediate_composition_accuracy"]) + 0.0011
        )
        result = calibration.evaluate_calibration(rows)
        self.assertEqual(
            result["run_classification"],
            calibration.CALIBRATION_RUN_INVALID,
        )

    def test_selection_prefers_worst_seed_then_mean_then_cost(self) -> None:
        first, second, third = calibration.candidate_grid()[:3]
        evaluations = (
            calibration.CandidateEvaluation(
                first,
                direct_gains=(0.10, 0.10, 0.10),
                composition_gains=(0.02, 0.02, 0.02),
                qualified=True,
            ),
            calibration.CandidateEvaluation(
                second,
                direct_gains=(0.20, 0.20, 0.20),
                composition_gains=(0.019, 0.10, 0.10),
                qualified=True,
            ),
            calibration.CandidateEvaluation(
                third,
                direct_gains=(0.10, 0.10, 0.10),
                composition_gains=(0.02, 0.02, 0.02),
                qualified=True,
            ),
        )
        selected = calibration.select_qualified_candidate(evaluations)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.candidate.identifier, first.identifier)

    def test_row_order_and_full_144_row_manifest_are_mandatory(self) -> None:
        rows = _complete_rows()
        rows[0], rows[1] = rows[1], rows[0]
        result = calibration.evaluate_calibration(rows)
        self.assertEqual(
            result["run_classification"],
            calibration.CALIBRATION_RUN_INVALID,
        )
        result = calibration.evaluate_calibration(_complete_rows()[:-1])
        self.assertEqual(
            result["run_classification"],
            calibration.CALIBRATION_RUN_INVALID,
        )


if __name__ == "__main__":
    unittest.main()
