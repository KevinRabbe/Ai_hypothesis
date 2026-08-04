from __future__ import annotations

import copy
import hashlib
import pathlib
import tempfile
import unittest

from ai_hypothesis.population_language import l0_reference_manifest
from ai_hypothesis.population_language import l0_reference_training
from ai_hypothesis.population_language import post_training_learning_l0_calibration as calibration
from ai_hypothesis.population_language import post_training_learning_l0_calibration_orchestrator as orchestrator
from ai_hypothesis.population_language import post_training_learning_l0_calibration_plan as calibration_plan
from ai_hypothesis.population_language import post_training_learning_l0_protocol as protocol
from ai_hypothesis.population_language import post_training_learning_l0_world as world


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(orchestrator._canonical_json_bytes(value)).hexdigest()


def _reference_manifest(root: pathlib.Path) -> l0_reference_manifest.ReferenceOutputManifest:
    checkpoints = tuple(
        l0_reference_manifest.PopulationCheckpointManifest(
            seed=seed,
            path=str(root / "checkpoints" / f"population-seed-{seed}.pt"),
            file_bytes=1_000_000 + index,
            file_sha256=_sha256_text(f"file-{seed}"),
            canonical_state_sha256=_sha256_text(f"state-{seed}"),
        )
        for index, seed in enumerate(world.MODEL_INITIALIZATION_SEEDS)
    )
    return l0_reference_manifest.ReferenceOutputManifest(
        root=str(root),
        summary_sha256=_sha256_text("summary"),
        execution_head="a" * 40,
        diagnosis=l0_reference_training.PASS,
        population_scaling_conclusion=l0_reference_training.SCALING_SUPPORTS,
        post_training_base_eligible=True,
        population_checkpoints=checkpoints,
    )


def _authorization(
    plan: dict[str, object],
    plan_sha256: str,
) -> dict[str, object]:
    return {
        "version": orchestrator.AUTHORIZATION_VERSION,
        "scope": orchestrator.AUTHORIZATION_SCOPE,
        "source_calibration_plan_head": orchestrator.SOURCE_CALIBRATION_PLAN_HEAD,
        "calibration_plan_sha256": plan_sha256,
        "reference_summary_sha256": plan["reference_summary_sha256"],
        "result_root": plan["result_root"],
        "authorization_id": _sha256_text("explicit-user-authorization"),
        "operator_acknowledgement": orchestrator.AUTHORIZATION_ACKNOWLEDGEMENT,
        "calibration_authorized": True,
        "final_execution_authorized": False,
    }


def _result_row(
    candidate: calibration.CalibrationCandidate,
    model_seed: int,
    calibration_world_seed: int,
    base_checkpoint_sha256: str,
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
        "adapter_initialization_seed": protocol.adapter_initialization_seed(model_seed),
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
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_checkpoint_after_sha256": base_checkpoint_sha256,
        "adaptation_artifact_sha256": _sha256_text(
            f"{candidate.identifier}|{model_seed}|{calibration_world_seed}"
        ),
        "adaptation_artifact_payload_kind": protocol.ARTIFACT_PAYLOAD_KIND,
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
        "calibration_world_fingerprints": world.calibration_world_fingerprints(),
        "original_l0_path_bitwise_identical": True,
        "transient_state_cleared_before_restart": True,
        "base_checkpoint_loaded_fresh_after_restart": True,
        "adaptation_artifact_loaded_after_restart": True,
        "fresh_process_restart": True,
        "baseline_direct_accuracy": baseline_direct,
        "immediate_direct_accuracy": baseline_direct + direct_gain,
        "post_restart_direct_accuracy": baseline_direct + direct_gain,
        "baseline_composition_accuracy": baseline_composition,
        "immediate_composition_accuracy": baseline_composition + composition_gain,
        "post_restart_composition_accuracy": baseline_composition + composition_gain,
    }


class CalibrationOrchestratorContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = pathlib.Path(self.temporary.name).absolute()
        self.reference_root = root / "reference"
        self.result_root = root / "calibration"
        self.plan = calibration_plan.build_calibration_plan(
            _reference_manifest(self.reference_root),
            result_root=self.result_root,
        )
        self.plan_sha256 = _canonical_sha256(self.plan)
        self.authorization = _authorization(self.plan, self.plan_sha256)
        self.authorization_sha256 = _canonical_sha256(self.authorization)
        self.run_manifest = orchestrator.build_run_manifest(
            self.plan,
            plan_sha256=self.plan_sha256,
            authorization=self.authorization,
            authorization_sha256=self.authorization_sha256,
        )
        self.run_manifest_sha256 = _canonical_sha256(self.run_manifest)

    def _rows(
        self,
        *,
        direct_gain: float = 0.10,
        composition_gain: float = 0.02,
    ) -> list[dict[str, object]]:
        checkpoint_by_seed = {
            row["model_seed"]: row for row in self.plan["population_checkpoints"]
        }
        return [
            _result_row(
                candidate,
                model_seed,
                calibration_world_seed,
                checkpoint_by_seed[model_seed]["canonical_state_sha256"],
                direct_gain=direct_gain,
                composition_gain=composition_gain,
            )
            for candidate in calibration.candidate_grid()
            for model_seed, calibration_world_seed in calibration.CALIBRATION_PAIRS
        ]

    def test_contract_is_locked_and_non_authorizing_for_final_execution(self) -> None:
        report = orchestrator.validate_calibration_orchestrator_contract()
        self.assertTrue(report["valid"], report["checks"])
        self.assertEqual(report["expected_result_rows"], 144)
        self.assertEqual(
            report["schedule_sha256_by_updates"],
            orchestrator.SCHEDULE_SHA256_BY_UPDATES,
        )
        self.assertEqual(len(orchestrator.AUTHORIZATION_KEYS), 10)
        self.assertEqual(len(orchestrator.RUN_MANIFEST_KEYS), 12)
        self.assertEqual(len(orchestrator.WORK_ITEM_KEYS), 18)
        self.assertEqual(len(orchestrator.VERIFICATION_KEYS), 24)

    def test_authorization_is_exact_plan_bound_and_calibration_only(self) -> None:
        validated = orchestrator.validate_calibration_authorization(
            self.authorization,
            plan=self.plan,
            plan_sha256=self.plan_sha256,
        )
        self.assertTrue(validated["calibration_authorized"])
        self.assertFalse(validated["final_execution_authorized"])

        changed = copy.deepcopy(self.authorization)
        changed["calibration_authorized"] = False
        with self.assertRaises(ValueError):
            orchestrator.validate_calibration_authorization(
                changed,
                plan=self.plan,
                plan_sha256=self.plan_sha256,
            )

        changed = copy.deepcopy(self.authorization)
        changed["final_execution_authorized"] = True
        with self.assertRaises(ValueError):
            orchestrator.validate_calibration_authorization(
                changed,
                plan=self.plan,
                plan_sha256=self.plan_sha256,
            )

        changed = copy.deepcopy(self.authorization)
        changed["calibration_plan_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            orchestrator.validate_calibration_authorization(
                changed,
                plan=self.plan,
                plan_sha256=self.plan_sha256,
            )

    def test_run_manifest_binds_all_144_work_items_and_create_once_persistence(self) -> None:
        work_items = self.run_manifest["work_items"]
        self.assertEqual(len(work_items), 144)
        self.assertEqual(
            [item["expected_result_key"] for item in work_items],
            [list(key) for key in calibration.expected_result_keys()],
        )
        self.assertEqual(work_items[0]["ordinal"], 0)
        self.assertEqual(work_items[-1]["ordinal"], 143)
        self.assertEqual(
            work_items[0]["adaptation_schedule_sha256"],
            orchestrator.SCHEDULE_SHA256_BY_UPDATES[32],
        )
        self.assertTrue(
            all(
                pathlib.Path(item["result_row_path"]).is_absolute()
                and pathlib.Path(item["adapter_artifact_path"]).is_absolute()
                for item in work_items
            )
        )
        self.assertTrue(self.run_manifest["calibration_authorized"])
        self.assertFalse(self.run_manifest["final_execution_authorized"])

        path = pathlib.Path(self.temporary.name) / "run-manifest.json"
        artifact = orchestrator.save_run_manifest_create_once(
            path,
            self.run_manifest,
            plan=self.plan,
            plan_sha256=self.plan_sha256,
            authorization_sha256=self.authorization_sha256,
        )
        loaded = orchestrator.load_run_manifest(
            path,
            expected_sha256=artifact.sha256,
            plan=self.plan,
            plan_sha256=self.plan_sha256,
            authorization_sha256=self.authorization_sha256,
        )
        self.assertEqual(loaded, self.run_manifest)
        with self.assertRaises(FileExistsError):
            orchestrator.save_run_manifest_create_once(
                path,
                self.run_manifest,
                plan=self.plan,
                plan_sha256=self.plan_sha256,
                authorization_sha256=self.authorization_sha256,
            )

    def test_independent_verifier_selects_candidate_but_never_authorizes_final(self) -> None:
        bundle = orchestrator.build_result_bundle(
            self.plan,
            plan_sha256=self.plan_sha256,
            run_manifest=self.run_manifest,
            run_manifest_sha256=self.run_manifest_sha256,
            rows=self._rows(),
        )
        bundle_sha256 = _canonical_sha256(bundle)
        verification = orchestrator.verify_result_bundle(
            self.plan,
            plan_sha256=self.plan_sha256,
            run_manifest=self.run_manifest,
            run_manifest_sha256=self.run_manifest_sha256,
            result_bundle=bundle,
            result_bundle_sha256=bundle_sha256,
        )
        self.assertEqual(
            verification.run_classification,
            calibration.CALIBRATION_RUN_VALID,
        )
        self.assertEqual(
            verification.conclusion,
            calibration.CALIBRATION_SELECTS_CANDIDATE,
        )
        self.assertEqual(
            verification.selected_candidate_id,
            "r1-lr0p001-u32",
        )
        self.assertEqual(
            verification.final_execution_eligibility,
            calibration.FINAL_EXECUTION_ELIGIBLE_NOT_AUTHORIZED,
        )
        self.assertTrue(
            verification.report["separate_explicit_authorization_still_required"]
        )
        self.assertFalse(verification.report["final_execution_authorized"])

        path = pathlib.Path(self.temporary.name) / "verification.json"
        artifact = orchestrator.save_verification_create_once(path, verification)
        self.assertEqual(artifact.sha256, _sha256_text(path.read_text(encoding="utf-8")))

    def test_valid_calibration_rejection_is_preserved(self) -> None:
        bundle = orchestrator.build_result_bundle(
            self.plan,
            plan_sha256=self.plan_sha256,
            run_manifest=self.run_manifest,
            run_manifest_sha256=self.run_manifest_sha256,
            rows=self._rows(direct_gain=0.0),
        )
        verification = orchestrator.verify_result_bundle(
            self.plan,
            plan_sha256=self.plan_sha256,
            run_manifest=self.run_manifest,
            run_manifest_sha256=self.run_manifest_sha256,
            result_bundle=bundle,
            result_bundle_sha256=_canonical_sha256(bundle),
        )
        self.assertEqual(
            verification.run_classification,
            calibration.CALIBRATION_RUN_VALID,
        )
        self.assertEqual(
            verification.conclusion,
            calibration.CALIBRATION_REJECTS_CANDIDATE,
        )
        self.assertIsNone(verification.selected_candidate_id)
        self.assertEqual(
            verification.final_execution_eligibility,
            calibration.FINAL_EXECUTION_NOT_ELIGIBLE,
        )
        self.assertFalse(
            verification.report["separate_explicit_authorization_still_required"]
        )

    def test_checkpoint_binding_failure_becomes_invalid_run(self) -> None:
        rows = self._rows()
        rows[0] = copy.deepcopy(rows[0])
        rows[0]["base_checkpoint_sha256"] = "f" * 64
        rows[0]["base_checkpoint_after_sha256"] = "f" * 64
        bundle = orchestrator.build_result_bundle(
            self.plan,
            plan_sha256=self.plan_sha256,
            run_manifest=self.run_manifest,
            run_manifest_sha256=self.run_manifest_sha256,
            rows=rows,
        )
        verification = orchestrator.verify_result_bundle(
            self.plan,
            plan_sha256=self.plan_sha256,
            run_manifest=self.run_manifest,
            run_manifest_sha256=self.run_manifest_sha256,
            result_bundle=bundle,
            result_bundle_sha256=_canonical_sha256(bundle),
        )
        self.assertEqual(
            verification.run_classification,
            calibration.CALIBRATION_RUN_INVALID,
        )
        self.assertEqual(
            verification.conclusion,
            calibration.INVALID_CALIBRATION_NO_SELECTION,
        )
        self.assertFalse(verification.report["final_execution_authorized"])

    def test_result_order_and_work_item_linkage_fail_closed(self) -> None:
        rows = self._rows()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(ValueError):
            orchestrator.build_result_bundle(
                self.plan,
                plan_sha256=self.plan_sha256,
                run_manifest=self.run_manifest,
                run_manifest_sha256=self.run_manifest_sha256,
                rows=rows,
            )

        bundle = orchestrator.build_result_bundle(
            self.plan,
            plan_sha256=self.plan_sha256,
            run_manifest=self.run_manifest,
            run_manifest_sha256=self.run_manifest_sha256,
            rows=self._rows(),
        )
        changed = copy.deepcopy(bundle)
        changed["result_records"][0]["work_item_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            orchestrator.validate_result_bundle(
                changed,
                plan=self.plan,
                plan_sha256=self.plan_sha256,
                run_manifest=self.run_manifest,
                run_manifest_sha256=self.run_manifest_sha256,
            )

    def test_canonical_provenance_hashes_are_mandatory(self) -> None:
        with self.assertRaises(ValueError):
            orchestrator.build_run_manifest(
                self.plan,
                plan_sha256="0" * 64,
                authorization=self.authorization,
                authorization_sha256=self.authorization_sha256,
            )
        with self.assertRaises(ValueError):
            orchestrator.build_result_bundle(
                self.plan,
                plan_sha256=self.plan_sha256,
                run_manifest=self.run_manifest,
                run_manifest_sha256="0" * 64,
                rows=self._rows(),
            )


if __name__ == "__main__":
    unittest.main()
