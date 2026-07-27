"""Scientific contract for fixed-parameter population-compute scaling experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable


DEVELOPMENT_POPULATION_SIZES: tuple[int, ...] = (1, 4, 16, 64, 256)


class CommunicationMode(str, Enum):
    """Population information-flow condition."""

    NO_COMMUNICATION = "no_communication"
    SPARSE_SHARED_V0 = "sparse_shared_v0"
    SPARSE_SHARED_V1 = "sparse_shared_v1"
    HIERARCHICAL_SUMMARY_V0 = "hierarchical_summary_v0"
    SERIAL_CONTROL = "serial_control"


@dataclass(frozen=True, slots=True)
class PopulationCondition:
    """One runtime-compute condition under one frozen learned model.

    ``nominal_population_size`` is the x-axis population point being matched.
    ``active_state_count`` is the number of worker states active in parallel.
    The distinction lets a serial control spend the matched worker-update budget
    with a small number of recurrent states.
    """

    nominal_population_size: int
    active_state_count: int
    recurrent_rounds: int
    communication_mode: CommunicationMode

    def validate(self) -> None:
        if self.nominal_population_size <= 0:
            raise ValueError("nominal_population_size must be positive")
        if self.active_state_count <= 0:
            raise ValueError("active_state_count must be positive")
        if self.active_state_count > self.nominal_population_size:
            raise ValueError(
                "active_state_count cannot exceed nominal_population_size"
            )
        if self.recurrent_rounds <= 0:
            raise ValueError("recurrent_rounds must be positive")
        if (
            self.communication_mode is CommunicationMode.NO_COMMUNICATION
            and self.active_state_count == 1
            and self.nominal_population_size > 1
        ):
            raise ValueError(
                "no-communication population controls must actually activate the "
                "matched population states"
            )

    @property
    def worker_updates(self) -> int:
        return self.active_state_count * self.recurrent_rounds


@dataclass(frozen=True, slots=True)
class PopulationRunMetrics:
    """Comparable measurements for one trained seed, difficulty and condition.

    The scope decomposition is mandatory for Gate v0. ``information_complete_count``
    records how many benchmark worlds expose every required fact inside the active
    worker prefix. ``solved_information_complete_count`` then separates neural/system
    utilization of available information from the trivial effect of exposing more scope.
    """

    training_seed: int
    benchmark_seed: int
    difficulty: str
    learned_parameter_count: int
    parameter_fingerprint: str
    condition: PopulationCondition
    task_count: int
    solved_count: int
    information_complete_count: int
    solved_information_complete_count: int
    messages_emitted: int
    communicated_scalar_count: int
    peak_worker_state_bytes: int
    elapsed_seconds: float

    def validate(self) -> None:
        self.condition.validate()
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")
        if self.benchmark_seed < 0:
            raise ValueError("benchmark_seed must be non-negative")
        if not self.difficulty.strip():
            raise ValueError("difficulty must be non-empty")
        if self.learned_parameter_count <= 0:
            raise ValueError("learned_parameter_count must be positive")
        if not self.parameter_fingerprint.strip():
            raise ValueError("parameter_fingerprint must be non-empty")
        if self.task_count <= 0:
            raise ValueError("task_count must be positive")
        if not 0 <= self.solved_count <= self.task_count:
            raise ValueError("solved_count must be within [0, task_count]")
        if not 0 <= self.information_complete_count <= self.task_count:
            raise ValueError(
                "information_complete_count must be within [0, task_count]"
            )
        if not 0 <= self.solved_information_complete_count <= self.information_complete_count:
            raise ValueError(
                "solved_information_complete_count must be within information-complete tasks"
            )
        solved_incomplete = self.solved_information_incomplete_count
        incomplete_count = self.task_count - self.information_complete_count
        if not 0 <= solved_incomplete <= incomplete_count:
            raise ValueError(
                "solved_count is inconsistent with the information-complete decomposition"
            )
        if self.messages_emitted < 0:
            raise ValueError("messages_emitted must be non-negative")
        if self.communicated_scalar_count < 0:
            raise ValueError("communicated_scalar_count must be non-negative")
        if self.peak_worker_state_bytes < 0:
            raise ValueError("peak_worker_state_bytes must be non-negative")
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if (
            self.condition.communication_mode is CommunicationMode.NO_COMMUNICATION
            and (
                self.messages_emitted != 0
                or self.communicated_scalar_count != 0
            )
        ):
            raise ValueError(
                "no-communication controls cannot report inter-worker messages"
            )

    @property
    def solve_rate(self) -> float:
        return self.solved_count / self.task_count

    @property
    def information_complete_rate(self) -> float:
        return self.information_complete_count / self.task_count

    @property
    def solve_rate_given_information_complete(self) -> float | None:
        if self.information_complete_count == 0:
            return None
        return self.solved_information_complete_count / self.information_complete_count

    @property
    def solved_information_incomplete_count(self) -> int:
        return self.solved_count - self.solved_information_complete_count

    @property
    def solve_rate_given_information_incomplete(self) -> float | None:
        incomplete_count = self.task_count - self.information_complete_count
        if incomplete_count == 0:
            return None
        return self.solved_information_incomplete_count / incomplete_count

    @property
    def worker_updates(self) -> int:
        return self.condition.worker_updates


@dataclass(frozen=True, slots=True)
class GateCriteria:
    """Frozen preregistration for one per-seed/per-difficulty curve."""

    endpoint_gain: float = 0.05
    communication_advantage: float = 0.05
    adjacent_drop_tolerance: float = 0.01
    minimum_nondecreasing_steps: int = 3

    def validate(self) -> None:
        if not 0.0 <= self.endpoint_gain <= 1.0:
            raise ValueError("endpoint_gain must be in [0, 1]")
        if not 0.0 <= self.communication_advantage <= 1.0:
            raise ValueError("communication_advantage must be in [0, 1]")
        if not 0.0 <= self.adjacent_drop_tolerance <= 1.0:
            raise ValueError("adjacent_drop_tolerance must be in [0, 1]")
        if not 0 <= self.minimum_nondecreasing_steps <= (
            len(DEVELOPMENT_POPULATION_SIZES) - 1
        ):
            raise ValueError("minimum_nondecreasing_steps is out of range")


@dataclass(frozen=True, slots=True)
class CurveAssessment:
    """Threshold-level result for one frozen seed/difficulty comparison."""

    population_sizes: tuple[int, ...]
    solve_rates: tuple[float, ...]
    information_complete_rates: tuple[float, ...]
    solve_rates_given_information_complete: tuple[float | None, ...]
    endpoint_gain: float
    nondecreasing_steps: int
    communication_endpoint_advantage: float
    passes_scaling_signal: bool
    reasons: tuple[str, ...]


def validate_fixed_parameter_identity(
    runs: Iterable[PopulationRunMetrics],
) -> tuple[int, str]:
    """Require one exact learned model identity across compared conditions."""

    materialized = tuple(runs)
    if not materialized:
        raise ValueError("at least one run is required")
    for run in materialized:
        run.validate()

    parameter_counts = {run.learned_parameter_count for run in materialized}
    fingerprints = {run.parameter_fingerprint for run in materialized}
    if len(parameter_counts) != 1:
        raise ValueError("compared runs changed learned_parameter_count")
    if len(fingerprints) != 1:
        raise ValueError("compared runs changed parameter_fingerprint")
    return next(iter(parameter_counts)), next(iter(fingerprints))


def assess_scaling_curve(
    communicating_runs: Iterable[PopulationRunMetrics],
    no_communication_runs: Iterable[PopulationRunMetrics],
    *,
    criteria: GateCriteria = GateCriteria(),
) -> CurveAssessment:
    """Assess one seed/difficulty curve against the preregistered v0 thresholds.

    Cross-seed Gate-v0 acceptance is deliberately a higher-level aggregation.
    This function prevents a runner from silently changing the per-curve rules and
    requires the communication/control curves to expose identical benchmark scope.
    """

    criteria.validate()
    communicating = _ordered_curve(communicating_runs)
    no_communication = _ordered_curve(no_communication_runs)
    validate_fixed_parameter_identity((*communicating, *no_communication))

    if tuple(run.condition.nominal_population_size for run in communicating) != (
        DEVELOPMENT_POPULATION_SIZES
    ):
        raise ValueError(
            "communicating curve must use the frozen development population sizes"
        )
    if tuple(run.condition.nominal_population_size for run in no_communication) != (
        DEVELOPMENT_POPULATION_SIZES
    ):
        raise ValueError(
            "no-communication curve must use the frozen development population sizes"
        )

    communicating_modes = {
        run.condition.communication_mode for run in communicating
    }
    if len(communicating_modes) != 1:
        raise ValueError("communicating curve must use one communication mode")
    communicating_mode = next(iter(communicating_modes))
    if communicating_mode not in {
        CommunicationMode.SPARSE_SHARED_V0,
        CommunicationMode.SPARSE_SHARED_V1,
        CommunicationMode.HIERARCHICAL_SUMMARY_V0,
    }:
        raise ValueError("communicating curve uses an inadmissible mode")
    if {
        run.condition.communication_mode for run in no_communication
    } != {CommunicationMode.NO_COMMUNICATION}:
        raise ValueError("control curve must use no_communication mode")

    _require_same_run_scope(communicating)
    _require_same_run_scope(no_communication)
    if _run_scope(communicating[0]) != _run_scope(no_communication[0]):
        raise ValueError("communication and control curves do not share run scope")

    for communicating_run, control_run in zip(
        communicating, no_communication, strict=True
    ):
        if (
            communicating_run.information_complete_count
            != control_run.information_complete_count
        ):
            raise ValueError(
                "communication and control conditions do not share information scope"
            )

    information_complete_rates = tuple(
        run.information_complete_rate for run in communicating
    )
    if any(
        later + 1e-12 < earlier
        for earlier, later in zip(
            information_complete_rates,
            information_complete_rates[1:],
        )
    ):
        raise ValueError("nested population scope must not lose complete information")

    rates = tuple(run.solve_rate for run in communicating)
    endpoint_gain = rates[-1] - rates[0]
    nondecreasing_steps = sum(
        later >= earlier - criteria.adjacent_drop_tolerance
        for earlier, later in zip(rates, rates[1:])
    )
    communication_endpoint_advantage = (
        communicating[-1].solve_rate - no_communication[-1].solve_rate
    )

    reasons: list[str] = []
    if endpoint_gain < criteria.endpoint_gain:
        reasons.append("endpoint gain below preregistered minimum")
    if nondecreasing_steps < criteria.minimum_nondecreasing_steps:
        reasons.append("too many adjacent population steps decline")
    if communication_endpoint_advantage < criteria.communication_advantage:
        reasons.append("communication advantage below preregistered minimum")

    return CurveAssessment(
        population_sizes=DEVELOPMENT_POPULATION_SIZES,
        solve_rates=rates,
        information_complete_rates=information_complete_rates,
        solve_rates_given_information_complete=tuple(
            run.solve_rate_given_information_complete for run in communicating
        ),
        endpoint_gain=endpoint_gain,
        nondecreasing_steps=nondecreasing_steps,
        communication_endpoint_advantage=communication_endpoint_advantage,
        passes_scaling_signal=not reasons,
        reasons=tuple(reasons),
    )


def _ordered_curve(
    runs: Iterable[PopulationRunMetrics],
) -> tuple[PopulationRunMetrics, ...]:
    materialized = tuple(runs)
    if len(materialized) != len(DEVELOPMENT_POPULATION_SIZES):
        raise ValueError(
            "curve must contain exactly one run per frozen development population size"
        )
    ordered = tuple(
        sorted(materialized, key=lambda run: run.condition.nominal_population_size)
    )
    populations = tuple(run.condition.nominal_population_size for run in ordered)
    if populations != DEVELOPMENT_POPULATION_SIZES:
        raise ValueError("curve population sizes are missing, duplicated, or unexpected")
    return ordered


def _run_scope(run: PopulationRunMetrics) -> tuple[int, int, str, int]:
    return (
        run.training_seed,
        run.benchmark_seed,
        run.difficulty,
        run.task_count,
    )


def _require_same_run_scope(runs: tuple[PopulationRunMetrics, ...]) -> None:
    scopes = {_run_scope(run) for run in runs}
    if len(scopes) != 1:
        raise ValueError("all points in a curve must share training/benchmark scope")
