"""Frozen cross-seed confirmation assessment for population-compute Gate v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .collective_relay import COLLECTIVE_RELAY_VERSION, RELAY_DIFFICULTIES
from .contract import GateCriteria, assess_scaling_curve
from .relay_experiment_v1 import (
    RELAY_EXPERIMENT_V1,
    RelayDevelopmentResultV1,
    RelayTrainingConfigV1,
)
from .relay_protocol_v1 import RELAY_PROTOCOL_VERSION


MINIMUM_CONFIRMATION_SEEDS = 3
FROZEN_CONFIRMATION_WORLD_COUNT = 1_000
FROZEN_CONFIRMATION_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class DifficultyConfirmationAssessment:
    difficulty: str
    endpoint_gain: float
    nondecreasing_steps: int
    communication_endpoint_advantage: float
    endpoint_pass: bool
    shape_pass: bool
    communication_pass: bool

    @property
    def useful_curve_pass(self) -> bool:
        return self.endpoint_pass and self.shape_pass


@dataclass(frozen=True, slots=True)
class SeedConfirmationAssessment:
    training_seed: int
    difficulty_assessments: tuple[DifficultyConfirmationAssessment, ...]
    useful_curve_tier_count: int
    communication_advantage_tier_count: int
    passes_seed_gate: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConfirmationGateAssessment:
    seed_assessments: tuple[SeedConfirmationAssessment, ...]
    passes_gate: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_confirmation_seeds": MINIMUM_CONFIRMATION_SEEDS,
            "passes_gate": self.passes_gate,
            "reasons": list(self.reasons),
            "seeds": [
                {
                    "training_seed": seed.training_seed,
                    "useful_curve_tier_count": seed.useful_curve_tier_count,
                    "communication_advantage_tier_count": (
                        seed.communication_advantage_tier_count
                    ),
                    "passes_seed_gate": seed.passes_seed_gate,
                    "reasons": list(seed.reasons),
                    "difficulties": [
                        {
                            "difficulty": row.difficulty,
                            "endpoint_gain": row.endpoint_gain,
                            "nondecreasing_steps": row.nondecreasing_steps,
                            "communication_endpoint_advantage": (
                                row.communication_endpoint_advantage
                            ),
                            "endpoint_pass": row.endpoint_pass,
                            "shape_pass": row.shape_pass,
                            "communication_pass": row.communication_pass,
                            "useful_curve_pass": row.useful_curve_pass,
                        }
                        for row in seed.difficulty_assessments
                    ],
                }
                for seed in self.seed_assessments
            ],
        }


def assess_confirmation_gate_v1(
    results: Sequence[RelayDevelopmentResultV1],
    *,
    criteria: GateCriteria = GateCriteria(),
) -> ConfirmationGateAssessment:
    """Apply the preregistered Gate-v0 criteria without post-hoc seed aggregation."""

    criteria.validate()
    materialized = tuple(results)
    if len(materialized) < MINIMUM_CONFIRMATION_SEEDS:
        raise ValueError("confirmation requires at least three independent training seeds")

    seeds = tuple(result.training.training_seed for result in materialized)
    if len(set(seeds)) != len(seeds):
        raise ValueError("confirmation training seeds must be unique")

    parameter_counts = {result.training.learned_parameter_count for result in materialized}
    if len(parameter_counts) != 1:
        raise ValueError("confirmation seeds changed learned parameter count")

    seed_assessments = tuple(
        _assess_confirmation_seed(result, criteria=criteria)
        for result in sorted(materialized, key=lambda item: item.training.training_seed)
    )
    reasons: list[str] = []
    failing = tuple(
        seed.training_seed for seed in seed_assessments if not seed.passes_seed_gate
    )
    if failing:
        reasons.append(
            "not every independent confirmation seed passed the frozen seed-level gate: "
            + ", ".join(str(seed) for seed in failing)
        )

    return ConfirmationGateAssessment(
        seed_assessments=seed_assessments,
        passes_gate=not reasons,
        reasons=tuple(reasons),
    )


def _assess_confirmation_seed(
    result: RelayDevelopmentResultV1,
    *,
    criteria: GateCriteria,
) -> SeedConfirmationAssessment:
    _validate_confirmation_result_contract(result)
    rows = tuple(result.evaluations)
    difficulty_assessments: list[DifficultyConfirmationAssessment] = []

    for difficulty in (item.name for item in RELAY_DIFFICULTIES):
        communicating = tuple(
            row.metrics
            for row in rows
            if row.metrics.difficulty == difficulty
            and row.metrics.condition.communication_mode.value == "sparse_shared_v1"
        )
        controls = tuple(
            row.metrics
            for row in rows
            if row.metrics.difficulty == difficulty
            and row.metrics.condition.communication_mode.value == "no_communication"
        )
        curve = assess_scaling_curve(
            communicating,
            controls,
            criteria=criteria,
        )
        difficulty_assessments.append(
            DifficultyConfirmationAssessment(
                difficulty=difficulty,
                endpoint_gain=curve.endpoint_gain,
                nondecreasing_steps=curve.nondecreasing_steps,
                communication_endpoint_advantage=curve.communication_endpoint_advantage,
                endpoint_pass=curve.endpoint_gain >= criteria.endpoint_gain,
                shape_pass=(
                    curve.nondecreasing_steps >= criteria.minimum_nondecreasing_steps
                ),
                communication_pass=(
                    curve.communication_endpoint_advantage
                    >= criteria.communication_advantage
                ),
            )
        )

    useful_count = sum(row.useful_curve_pass for row in difficulty_assessments)
    communication_count = sum(row.communication_pass for row in difficulty_assessments)
    reasons: list[str] = []
    if useful_count < 2:
        reasons.append(
            "fewer than two relay tiers satisfy both endpoint gain and curve-shape criteria"
        )
    if communication_count < 1:
        reasons.append("no relay tier satisfies the communication-advantage criterion")

    return SeedConfirmationAssessment(
        training_seed=result.training.training_seed,
        difficulty_assessments=tuple(difficulty_assessments),
        useful_curve_tier_count=useful_count,
        communication_advantage_tier_count=communication_count,
        passes_seed_gate=not reasons,
        reasons=tuple(reasons),
    )


def _validate_confirmation_result_contract(result: RelayDevelopmentResultV1) -> None:
    if result.experiment_version != RELAY_EXPERIMENT_V1:
        raise ValueError("confirmation result uses the wrong experiment version")
    if result.protocol_version != RELAY_PROTOCOL_VERSION:
        raise ValueError("confirmation result uses the wrong protocol version")
    if result.benchmark_version != COLLECTIVE_RELAY_VERSION:
        raise ValueError("confirmation result uses the wrong benchmark version")
    if result.evaluation_split != "confirmation" or not result.confirmation_opened:
        raise ValueError("confirmation gate requires explicitly opened confirmation results")
    if result.training_config != RelayTrainingConfigV1():
        raise ValueError("confirmation result changed the frozen training configuration")
    if result.evaluation_world_count != FROZEN_CONFIRMATION_WORLD_COUNT:
        raise ValueError("confirmation result changed the frozen world count")
    if result.evaluation_batch_size != FROZEN_CONFIRMATION_BATCH_SIZE:
        raise ValueError("confirmation result changed the frozen evaluation batch size")

    expected_difficulties = {item.name for item in RELAY_DIFFICULTIES}
    seen_difficulties = {row.metrics.difficulty for row in result.evaluations}
    if seen_difficulties != expected_difficulties:
        raise ValueError("confirmation result does not contain every relay difficulty")

    fingerprints_by_difficulty: dict[str, set[str]] = {}
    for row in result.evaluations:
        row.validate()
        fingerprints_by_difficulty.setdefault(row.metrics.difficulty, set()).add(
            row.metrics.parameter_fingerprint
        )
    if any(len(values) != 1 for values in fingerprints_by_difficulty.values()):
        raise ValueError("confirmation curve changed parameter fingerprint within a seed")
    all_fingerprints = {
        row.metrics.parameter_fingerprint for row in result.evaluations
    }
    if all_fingerprints != {result.training.parameter_fingerprint}:
        raise ValueError("confirmation evaluations do not match the trained seed checkpoint")
