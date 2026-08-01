from __future__ import annotations

import hashlib
import json
import pathlib
import unittest

from ai_hypothesis.population_compute import (
    CommunicationMode,
    PopulationCondition,
    PopulationRunMetrics,
    assess_scaling_curve,
    validate_fixed_parameter_identity,
)


class PopulationComputeContractTests(unittest.TestCase):
    def test_fixed_parameter_identity_rejects_parameter_count_change(self) -> None:
        runs = [
            self._run(1, CommunicationMode.SPARSE_SHARED_V0, solved=10),
            self._run(
                4,
                CommunicationMode.SPARSE_SHARED_V0,
                solved=20,
                learned_parameter_count=50_001,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "learned_parameter_count"):
            validate_fixed_parameter_identity(runs)

    def test_fixed_parameter_identity_rejects_checkpoint_change(self) -> None:
        runs = [
            self._run(1, CommunicationMode.SPARSE_SHARED_V0, solved=10),
            self._run(
                4,
                CommunicationMode.SPARSE_SHARED_V0,
                solved=20,
                parameter_fingerprint="different",
            ),
        ]

        with self.assertRaisesRegex(ValueError, "parameter_fingerprint"):
            validate_fixed_parameter_identity(runs)

    def test_no_communication_cannot_report_messages(self) -> None:
        run = self._run(
            16,
            CommunicationMode.NO_COMMUNICATION,
            solved=20,
            messages_emitted=1,
            communicated_scalar_count=4,
        )

        with self.assertRaisesRegex(ValueError, "cannot report inter-worker messages"):
            run.validate()

    def test_serial_control_can_match_population_axis_with_one_active_state(self) -> None:
        condition = PopulationCondition(
            nominal_population_size=256,
            active_state_count=1,
            recurrent_rounds=256,
            communication_mode=CommunicationMode.SERIAL_CONTROL,
        )
        condition.validate()

        self.assertEqual(condition.worker_updates, 256)

    def test_scope_decomposition_reports_available_and_conditional_capability(self) -> None:
        run = self._run(
            64,
            CommunicationMode.SPARSE_SHARED_V0,
            solved=50,
            information_complete=60,
            solved_information_complete=45,
        )
        run.validate()

        self.assertAlmostEqual(run.solve_rate, 0.5)
        self.assertAlmostEqual(run.information_complete_rate, 0.6)
        self.assertAlmostEqual(run.solve_rate_given_information_complete, 0.75)
        self.assertEqual(run.solved_information_incomplete_count, 5)
        self.assertAlmostEqual(run.solve_rate_given_information_incomplete, 0.125)

    def test_scope_decomposition_rejects_impossible_solved_partition(self) -> None:
        run = self._run(
            64,
            CommunicationMode.SPARSE_SHARED_V0,
            solved=95,
            information_complete=20,
            solved_information_complete=10,
        )

        with self.assertRaisesRegex(ValueError, "information-complete decomposition"):
            run.validate()

    def test_positive_curve_passes_preregistered_per_curve_rules(self) -> None:
        communicating = self._curve(
            CommunicationMode.SPARSE_SHARED_V0,
            solved=(30, 34, 38, 43, 48),
        )
        no_communication = self._curve(
            CommunicationMode.NO_COMMUNICATION,
            solved=(30, 31, 32, 33, 34),
        )

        assessment = assess_scaling_curve(communicating, no_communication)

        self.assertTrue(assessment.passes_scaling_signal)
        self.assertAlmostEqual(assessment.endpoint_gain, 0.18)
        self.assertAlmostEqual(assessment.communication_endpoint_advantage, 0.14)
        self.assertEqual(assessment.nondecreasing_steps, 4)
        self.assertEqual(assessment.reasons, ())
        self.assertEqual(assessment.information_complete_rates, (1.0,) * 5)
        self.assertEqual(
            assessment.solve_rates_given_information_complete,
            assessment.solve_rates,
        )

    def test_flat_curve_is_negative_result(self) -> None:
        communicating = self._curve(
            CommunicationMode.SPARSE_SHARED_V0,
            solved=(40, 40, 41, 40, 41),
        )
        no_communication = self._curve(
            CommunicationMode.NO_COMMUNICATION,
            solved=(40, 40, 40, 40, 40),
        )

        assessment = assess_scaling_curve(communicating, no_communication)

        self.assertFalse(assessment.passes_scaling_signal)
        self.assertIn(
            "endpoint gain below preregistered minimum",
            assessment.reasons,
        )
        self.assertIn(
            "communication advantage below preregistered minimum",
            assessment.reasons,
        )

    def test_curve_rejects_different_training_scope(self) -> None:
        communicating = list(
            self._curve(
                CommunicationMode.SPARSE_SHARED_V0,
                solved=(30, 34, 38, 43, 48),
            )
        )
        communicating[-1] = self._run(
            256,
            CommunicationMode.SPARSE_SHARED_V0,
            solved=48,
            training_seed=7,
        )
        no_communication = self._curve(
            CommunicationMode.NO_COMMUNICATION,
            solved=(30, 31, 32, 33, 34),
        )

        with self.assertRaisesRegex(ValueError, "share training/benchmark scope"):
            assess_scaling_curve(communicating, no_communication)

    def test_curve_rejects_mismatched_information_scope_between_modes(self) -> None:
        communicating = list(
            self._curve(
                CommunicationMode.SPARSE_SHARED_V0,
                solved=(30, 34, 38, 43, 48),
            )
        )
        communicating[2] = self._run(
            16,
            CommunicationMode.SPARSE_SHARED_V0,
            solved=38,
            information_complete=99,
            solved_information_complete=38,
        )
        no_communication = self._curve(
            CommunicationMode.NO_COMMUNICATION,
            solved=(30, 31, 32, 33, 34),
        )

        with self.assertRaisesRegex(ValueError, "do not share information scope"):
            assess_scaling_curve(communicating, no_communication)

    def test_gate9d_stage1_seed0_result_record_when_present(self) -> None:
        result_root = (
            pathlib.Path(__file__).resolve().parents[1]
            / "experiments"
            / "population_compute_scaling_v0"
        )
        result_path = result_root / (
            "gate9_contextual_failure_decomposition_stage1_seed0_result_v0.json"
        )
        manifest_path = result_root / (
            "gate9_contextual_failure_decomposition_stage1_seed0_source_manifest_v0.sha256"
        )
        if not result_path.exists() and not manifest_path.exists():
            return

        self.assertTrue(result_path.is_file())
        self.assertTrue(manifest_path.is_file())
        result_bytes = result_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(result_bytes).hexdigest(),
            "20188b8d70637b1599f5f603d700512c3d22d50fc5c3d0d1d0a5fa72843c0a80",
        )
        result = json.loads(result_bytes)
        self.assertEqual(
            result["experiment_version"],
            "gate9-contextual-failure-decomposition-stage1-seed0-result-v0",
        )
        self.assertEqual(
            result["scientific_status"],
            "G9D_STAGE1_SEED0_FAILURE_RECORDED",
        )
        self.assertEqual(result["seed_index"], 0)
        self.assertEqual(result["seed_outcome"], "G9D_STAGE1_SEED_FAILED")
        self.assertEqual(result["role"], "first_ordered_seed_result_only")
        self.assertEqual(
            result["stack_identity"],
            {
                "architecture_head": "c689cc3f38f6f6f642916ee1a702d7de7bd0e43b",
                "execution_head": "2e1b91d578e7bf9b4c54aa2ee1c120a9ec01b21c",
                "operator_contract_head": "be6451e1af82b18749bd0313a9c02ca62c4eee5c",
                "protocol_head": "8deca15aef78d8636b07570aff044f9b7ae31928",
            },
        )
        self.assertEqual(
            result["operator"],
            {
                "counter": 72_057_594_037_927_936,
                "dataset_sha256": "37b4aafb3e184eaa2e8096b649457134bd5025f297912c26ed34100b76a3ff0f",
                "key": 14_550_454_351_299_327_585,
                "non_support_query_count": 247,
            },
        )
        self.assertEqual(
            result["training"],
            {
                "batch_size": 247,
                "examples_seen": 252_928,
                "final_loss": 0.36635908484458923,
                "minimum_loss": 0.36635908484458923,
                "minimum_loss_step": 1_024,
                "rows": 1_024,
                "steps": 1_024,
                "unique_examples": 247,
            },
        )
        self.assertEqual(
            result["evaluation"],
            {
                "bit_accuracy": 0.8456477732793523,
                "bit_correct": 1_671,
                "bit_total": 1_976,
                "exact_accuracy": 0.21052631578947367,
                "full_correct": 52,
                "oracle_accuracy": 1.0,
                "oracle_correct": 247,
                "query_only_accuracy": 0.0,
                "query_only_correct": 0,
                "rows": 247,
                "stage_passes": False,
            },
        )
        self.assertEqual(
            result["diagnostic_consequence"],
            {
                "all_three_initialization_seeds_required_to_advance": True,
                "remaining_seed_role": (
                    "replication_and_mixed-outcome_resolution_only"
                ),
                "stage1_seed0_independently_failed": True,
                "stage2_advancement_allowed": False,
                "terminal_outcomes_still_possible": [
                    "G9D_BASIC_QUERY_MAPPING_FAILED",
                    "G9D_DIAGNOSTIC_INCONCLUSIVE",
                ],
            },
        )
        self.assertEqual(
            result["closed_boundaries"],
            {
                "checkpoint_selection_performed": False,
                "diagnostic_classification_performed": False,
                "gate9_v0_result_mutation_performed": False,
                "later_diagnostic_stage_execution_performed": False,
                "population_execution_performed": False,
                "retraining_performed": False,
                "scientific_execution_performed": False,
                "scientific_test_generation_performed": False,
                "training_performed_in_result_slice": False,
            },
        )
        self.assertEqual(
            result["independent_audit"],
            {
                "checkpoint_all_finite_float32": True,
                "checkpoint_loaded_weights_only": True,
                "checkpoint_parameter_count_verified": True,
                "checkpoint_schema_verified": True,
                "evaluation_ledger_reconstructed": True,
                "git_head_verified": True,
                "git_status_empty_hash_verified_from_terminal": True,
                "manifest_paths_sorted_and_unique": True,
                "manifest_sha256_verified": True,
                "run_config_verified": True,
                "source_artifact_modified": False,
                "summary_evidence_reconciled": True,
                "training_ledger_reconstructed": True,
                "uploaded_artifact_hashes_match_manifest": True,
            },
        )
        expected_source_hashes = {
            "git-head.txt": "39aeac89e593584f77d1365c779f2eb2e5317e07729124026158f64c89abd940",
            "git-status.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "manifest.sha256": "8074a4224c4a51f38f869944f57fd0c350a0f164b50627a2f9173352e170dc0e",
            "run-config.json": "1673b768313b748bd00339247acdab3db37326b7139a36f3a9cdce0ae5824000",
            "seed-0/evaluation-per-episode.jsonl": "98623f9c37f74137722f6d45ef37536064017a429246d45a3820aaee2c9785e1",
            "seed-0/selected-checkpoint.pt": "3c2c2bac4036ccd8bc45c5aca8c28fe3b2e470902489907e224ba602bafaa93f",
            "seed-0/summary.json": "a306fcb06bbd6965553dd8406ffd0aad41259bab45149d39a67c77cab7d143c8",
            "seed-0/train-steps.jsonl": "b74cf27618985dd690c2d50555cf478e6dd06fe5907cbd55e9de7fdf448fcc9c",
        }
        self.assertEqual(result["source_artifact_sha256"], expected_source_hashes)
        expected_manifest = (
            "39aeac89e593584f77d1365c779f2eb2e5317e07729124026158f64c89abd940  git-head.txt\n"
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  git-status.txt\n"
            "1673b768313b748bd00339247acdab3db37326b7139a36f3a9cdce0ae5824000  run-config.json\n"
            "98623f9c37f74137722f6d45ef37536064017a429246d45a3820aaee2c9785e1  seed-0/evaluation-per-episode.jsonl\n"
            "3c2c2bac4036ccd8bc45c5aca8c28fe3b2e470902489907e224ba602bafaa93f  seed-0/selected-checkpoint.pt\n"
            "a306fcb06bbd6965553dd8406ffd0aad41259bab45149d39a67c77cab7d143c8  seed-0/summary.json\n"
            "b74cf27618985dd690c2d50555cf478e6dd06fe5907cbd55e9de7fdf448fcc9c  seed-0/train-steps.jsonl\n"
        )
        self.assertEqual(manifest_path.read_text(encoding="ascii"), expected_manifest)
        self.assertEqual(
            hashlib.sha256(expected_manifest.encode("ascii")).hexdigest(),
            "8074a4224c4a51f38f869944f57fd0c350a0f164b50627a2f9173352e170dc0e",
        )

    def _curve(
        self,
        mode: CommunicationMode,
        *,
        solved: tuple[int, int, int, int, int],
    ) -> tuple[PopulationRunMetrics, ...]:
        populations = (1, 4, 16, 64, 256)
        return tuple(
            self._run(population, mode, solved=solved_count)
            for population, solved_count in zip(populations, solved, strict=True)
        )

    def _run(
        self,
        population: int,
        mode: CommunicationMode,
        *,
        solved: int,
        training_seed: int = 0,
        learned_parameter_count: int = 50_000,
        parameter_fingerprint: str = "frozen-model",
        information_complete: int = 100,
        solved_information_complete: int | None = None,
        messages_emitted: int | None = None,
        communicated_scalar_count: int | None = None,
    ) -> PopulationRunMetrics:
        if solved_information_complete is None:
            solved_information_complete = solved
        if messages_emitted is None:
            messages_emitted = 0 if mode is CommunicationMode.NO_COMMUNICATION else population
        if communicated_scalar_count is None:
            communicated_scalar_count = (
                0 if mode is CommunicationMode.NO_COMMUNICATION else population * 8
            )
        return PopulationRunMetrics(
            training_seed=training_seed,
            benchmark_seed=100,
            difficulty="hard",
            learned_parameter_count=learned_parameter_count,
            parameter_fingerprint=parameter_fingerprint,
            condition=PopulationCondition(
                nominal_population_size=population,
                active_state_count=population,
                recurrent_rounds=1,
                communication_mode=mode,
            ),
            task_count=100,
            solved_count=solved,
            information_complete_count=information_complete,
            solved_information_complete_count=solved_information_complete,
            messages_emitted=messages_emitted,
            communicated_scalar_count=communicated_scalar_count,
            peak_worker_state_bytes=population * 64,
            elapsed_seconds=0.1,
        )


if __name__ == "__main__":
    unittest.main()
