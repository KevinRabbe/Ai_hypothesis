from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest

from ai_hypothesis.population_language import l0_reference_manifest as manifest
from ai_hypothesis.population_language import l0_reference_training as training
from ai_hypothesis.population_language import post_training_learning_l0_calibration as calibration
from ai_hypothesis.population_language import post_training_learning_l0_calibration_plan as plan
from ai_hypothesis.population_language import post_training_learning_l0_world as world


class CalibrationPlanContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve()
        checkpoints = tuple(
            manifest.PopulationCheckpointManifest(
                seed=seed,
                path=str(self.root / "reference" / f"population-seed-{seed}.pt"),
                file_bytes=75_900_000 + index,
                file_sha256=f"{index + 1:064x}",
                canonical_state_sha256=f"{index + 101:064x}",
            )
            for index, seed in enumerate(world.MODEL_INITIALIZATION_SEEDS)
        )
        self.reference = manifest.ReferenceOutputManifest(
            root=str(self.root / "reference"),
            summary_sha256="a" * 64,
            execution_head="b" * 40,
            diagnosis=training.PASS,
            population_scaling_conclusion=training.SCALING_SUPPORTS,
            post_training_base_eligible=True,
            population_checkpoints=checkpoints,
        )
        self.result_root = self.root / "calibration-output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_static_contract_and_exact_grid_are_locked(self) -> None:
        report = plan.validate_calibration_plan_contract()
        self.assertTrue(report["valid"])
        self.assertEqual(plan.PLAN_MAX_BYTES, 256 * 1024)
        self.assertEqual(len(plan.PLAN_KEYS), 18)
        self.assertEqual(len(calibration.candidate_grid()), 48)
        self.assertEqual(len(calibration.expected_result_keys()), 144)

    def test_eligible_manifest_builds_create_once_hash_pinned_plan(self) -> None:
        value = plan.build_calibration_plan(
            self.reference,
            result_root=self.result_root,
        )
        report = plan.validate_calibration_plan(value)
        self.assertTrue(report["valid"])
        self.assertEqual(tuple(value), plan.PLAN_KEYS)
        self.assertEqual(value["reference_summary_sha256"], "a" * 64)
        self.assertEqual(value["reference_execution_head"], "b" * 40)
        self.assertEqual(len(value["population_checkpoints"]), 3)
        self.assertEqual(len(value["calibration_candidates"]), 48)
        self.assertEqual(len(value["calibration_pairs"]), 3)
        self.assertEqual(len(value["expected_result_keys"]), 144)
        self.assertFalse(value["calibration_authorized"])
        self.assertFalse(value["final_execution_authorized"])
        self.assertNotIn("final_world_seeds", value)
        self.assertNotIn("final_world_fingerprints", value)
        self.assertNotIn("final_labels", value)

        path = self.root / "calibration-plan.json"
        artifact = plan.save_calibration_plan_create_once(path, value)
        self.assertGreater(artifact.bytes, 0)
        self.assertLessEqual(artifact.bytes, plan.PLAN_MAX_BYTES)
        self.assertEqual(len(artifact.sha256), 64)
        loaded = plan.load_calibration_plan(path, expected_sha256=artifact.sha256)
        self.assertEqual(loaded, value)
        with self.assertRaises(FileExistsError):
            plan.save_calibration_plan_create_once(path, value)

    def test_ineligible_or_reordered_reference_manifest_is_rejected(self) -> None:
        ineligible = manifest.ReferenceOutputManifest(
            root=self.reference.root,
            summary_sha256=self.reference.summary_sha256,
            execution_head=self.reference.execution_head,
            diagnosis=training.INVALID,
            population_scaling_conclusion=self.reference.population_scaling_conclusion,
            post_training_base_eligible=False,
            population_checkpoints=self.reference.population_checkpoints,
        )
        with self.assertRaises(ValueError):
            plan.build_calibration_plan(ineligible, result_root=self.result_root)

        reordered = manifest.ReferenceOutputManifest(
            root=self.reference.root,
            summary_sha256=self.reference.summary_sha256,
            execution_head=self.reference.execution_head,
            diagnosis=self.reference.diagnosis,
            population_scaling_conclusion=self.reference.population_scaling_conclusion,
            post_training_base_eligible=True,
            population_checkpoints=tuple(reversed(self.reference.population_checkpoints)),
        )
        with self.assertRaises(ValueError):
            plan.build_calibration_plan(reordered, result_root=self.result_root)

        with self.assertRaises(ValueError):
            plan.build_calibration_plan(
                self.reference,
                result_root=pathlib.Path("relative-output"),
            )

    def test_authorization_final_world_and_checkpoint_tampering_fail_closed(self) -> None:
        value = plan.build_calibration_plan(
            self.reference,
            result_root=self.result_root,
        )

        authorized = copy.deepcopy(value)
        authorized["calibration_authorized"] = True
        with self.assertRaises(ValueError):
            plan.validate_calibration_plan(authorized)

        final_authorized = copy.deepcopy(value)
        final_authorized["final_execution_authorized"] = True
        with self.assertRaises(ValueError):
            plan.validate_calibration_plan(final_authorized)

        final_injected = copy.deepcopy(value)
        final_injected["final_world_seeds"] = [220100, 220101, 220102]
        with self.assertRaises(ValueError):
            plan.validate_calibration_plan(final_injected)

        duplicate_path = copy.deepcopy(value)
        duplicate_path["population_checkpoints"][1]["path"] = (
            duplicate_path["population_checkpoints"][0]["path"]
        )
        with self.assertRaises(ValueError):
            plan.validate_calibration_plan(duplicate_path)

        wrong_grid = copy.deepcopy(value)
        wrong_grid["calibration_grid_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            plan.validate_calibration_plan(wrong_grid)

    def test_file_hash_and_schema_tampering_fail_closed(self) -> None:
        value = plan.build_calibration_plan(
            self.reference,
            result_root=self.result_root,
        )
        path = self.root / "calibration-plan-tamper.json"
        artifact = plan.save_calibration_plan_create_once(path, value)

        with self.assertRaises(ValueError):
            plan.load_calibration_plan(path, expected_sha256="0" * 64)

        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            plan.load_calibration_plan(path, expected_sha256=artifact.sha256)


if __name__ == "__main__":
    unittest.main()
