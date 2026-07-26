from __future__ import annotations

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
            solved=50,
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
