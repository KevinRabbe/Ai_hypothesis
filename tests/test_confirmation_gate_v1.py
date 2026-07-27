from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.collective_relay import (
    COLLECTIVE_RELAY_VERSION,
    RELAY_DIFFICULTIES,
    relay_scope_thresholds,
)
from ai_hypothesis.population_compute.confirmation_gate_v1 import (
    assess_confirmation_gate_v1,
)
from ai_hypothesis.population_compute.contract import (
    DEVELOPMENT_POPULATION_SIZES,
    CommunicationMode,
    PopulationCondition,
    PopulationRunMetrics,
)
from ai_hypothesis.population_compute.relay_experiment import (
    RelayEvaluationResult,
    RelayScopeCohortResult,
)
from ai_hypothesis.population_compute.relay_experiment_v1 import (
    RELAY_EXPERIMENT_V1,
    RelayDevelopmentResultV1,
    RelayTrainingConfigV1,
    RelayTrainingSummaryV1,
)
from ai_hypothesis.population_compute.relay_protocol_v1 import RELAY_PROTOCOL_VERSION


class ConfirmationGateV1Tests(unittest.TestCase):
    def test_three_independently_passing_seeds_pass_gate(self) -> None:
        result = assess_confirmation_gate_v1(
            tuple(_confirmation_result(seed, passes=True) for seed in (0, 1, 2))
        )
        self.assertTrue(result.passes_gate)
        self.assertEqual(len(result.seed_assessments), 3)
        self.assertTrue(all(seed.passes_seed_gate for seed in result.seed_assessments))

    def test_one_failing_seed_fails_cross_seed_gate(self) -> None:
        result = assess_confirmation_gate_v1(
            (
                _confirmation_result(0, passes=True),
                _confirmation_result(1, passes=False),
                _confirmation_result(2, passes=True),
            )
        )
        self.assertFalse(result.passes_gate)
        self.assertEqual(
            tuple(seed.training_seed for seed in result.seed_assessments if not seed.passes_seed_gate),
            (1,),
        )

    def test_two_seeds_are_insufficient_even_when_both_pass(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            assess_confirmation_gate_v1(
                (_confirmation_result(0, passes=True), _confirmation_result(1, passes=True))
            )

    def test_duplicate_training_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            assess_confirmation_gate_v1(
                (
                    _confirmation_result(0, passes=True),
                    _confirmation_result(0, passes=True),
                    _confirmation_result(2, passes=True),
                )
            )


def _confirmation_result(seed: int, *, passes: bool) -> RelayDevelopmentResultV1:
    fingerprint = f"seed-{seed}-fingerprint"
    evaluations: list[RelayEvaluationResult] = []
    for difficulty in RELAY_DIFFICULTIES:
        cohort_counts = _cohort_counts(difficulty)
        for mode in (CommunicationMode.SPARSE_SHARED_V1, CommunicationMode.NO_COMMUNICATION):
            for population_index, population in enumerate(DEVELOPMENT_POPULATION_SIZES):
                complete = sum(
                    count for threshold, count in cohort_counts if threshold <= population
                )
                if mode is CommunicationMode.NO_COMMUNICATION or not passes:
                    solved = 0
                else:
                    # A deliberately clean monotonic curve with a strong endpoint.
                    solved = int(round(1000 * (population_index / 4.0) * 0.96))
                    solved = min(solved, complete)
                solved_complete = solved
                metrics = PopulationRunMetrics(
                    training_seed=seed,
                    benchmark_seed=1_500_000_000,
                    difficulty=difficulty.name,
                    learned_parameter_count=26_669,
                    parameter_fingerprint=fingerprint,
                    condition=PopulationCondition(
                        nominal_population_size=population,
                        active_state_count=population,
                        recurrent_rounds=difficulty.hop_count,
                        communication_mode=mode,
                    ),
                    task_count=1000,
                    solved_count=solved,
                    information_complete_count=complete,
                    solved_information_complete_count=solved_complete,
                    messages_emitted=(
                        0 if mode is CommunicationMode.NO_COMMUNICATION else population * difficulty.hop_count * 1000
                    ),
                    communicated_scalar_count=(
                        0 if mode is CommunicationMode.NO_COMMUNICATION else population * difficulty.hop_count * 24 * 2 * 1000
                    ),
                    peak_worker_state_bytes=population * 64 * 4,
                    elapsed_seconds=1.0,
                )
                evaluations.append(
                    RelayEvaluationResult(
                        metrics=metrics,
                        bit_accuracy=0.99 if solved else 0.5,
                        scope_cohorts=tuple(
                            RelayScopeCohortResult(
                                scope_threshold=threshold,
                                task_count=count,
                                solved_count=0,
                            )
                            for threshold, count in cohort_counts
                        ),
                    )
                )

    config = RelayTrainingConfigV1()
    return RelayDevelopmentResultV1(
        experiment_version=RELAY_EXPERIMENT_V1,
        protocol_version=RELAY_PROTOCOL_VERSION,
        benchmark_version=COLLECTIVE_RELAY_VERSION,
        evaluation_split="confirmation",
        confirmation_opened=True,
        training=RelayTrainingSummaryV1(
            training_seed=seed,
            steps=config.steps,
            examples_seen=config.steps * config.batch_size,
            initial_total_loss=2.0,
            final_total_loss=0.1,
            mean_last_50_total_loss=0.1,
            final_relay_loss=0.05,
            final_gate_loss=0.05,
            learned_parameter_count=26_669,
            parameter_fingerprint=fingerprint,
        ),
        training_config=config,
        evaluation_world_count=1000,
        evaluation_batch_size=64,
        evaluations=tuple(evaluations),
        assessments=(),
    )


def _cohort_counts(difficulty) -> tuple[tuple[int, int], ...]:
    thresholds = relay_scope_thresholds(difficulty)
    base, remainder = divmod(1000, len(thresholds))
    return tuple(
        (threshold, base + (1 if index < remainder else 0))
        for index, threshold in enumerate(thresholds)
    )


if __name__ == "__main__":
    unittest.main()
