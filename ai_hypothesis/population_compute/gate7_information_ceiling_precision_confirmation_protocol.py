"""Data-frozen Gate-7 information-ceiling precision-confirmation protocol.

This module contains only immutable design constants and the preregistered
classifier.  It opens no execution path and reads no prior or future artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GATE7_PRECISION_VERSION = "gate7-information-ceiling-precision-confirmation-v0"
GATE7_PRECISION_SCIENTIFIC_STATUS = (
    "DATA_FROZEN_GATE7_INFORMATION_CEILING_PRECISION_PROTOCOL_EXECUTION_CLOSED"
)

GATE7_PRECISION_BASE_RESULT_HEAD = "4eb3e50a3ca7898ff81aebebddb7b049ff855df3"
GATE7_PRECISION_DECOMPOSITION_PROTOCOL_HEAD = (
    "3640699f1727886c9ad2e954269fad660dc34370"
)
GATE7_PRECISION_DECOMPOSITION_EXECUTION_HEAD = (
    "161142c1e5552cb9464216c774397def6a4100be"
)
GATE7_PRECISION_DECOMPOSITION_RECOVERY_HEAD = (
    "b06959ce9d6c7d83bac953ce17a3c3008ad0f306"
)
GATE7_PRECISION_DECOMPOSITION_RESULT_SHA256 = (
    "71a383ced44419f84022738448c460d79a3fb21746f436649e5f14399704f731"
)
GATE7_PRECISION_DECOMPOSITION_RECOVERED_AUDIT_SHA256 = (
    "86a7dbb774119cca9bcd697978081e0872b41e4e61a3f8b08538e0cc89c8397d"
)
GATE7_PRECISION_DECOMPOSITION_RECOVERY_RECORD_SHA256 = (
    "ccd4bbd353aba09b8a2d38d155bb9f883b862123bf196693889b515d5452324b"
)
GATE7_PRECISION_DECOMPOSITION_MANIFEST_SHA256 = (
    "026f75a76888efe020c57da9d719140169eedd5e024555db20da9590cfea2b45"
)
GATE7_PRECISION_DECOMPOSITION_OUTCOME = "G7_INFORMATION_CEILING_INCONCLUSIVE"

GATE7_PRECISION_POPULATIONS = (16_384, 32_768, 65_536, 131_072)
GATE7_PRECISION_CHECKPOINT_INDICES = (0, 1, 2)
GATE7_PRECISION_WORLD_COUNT = 2_048
GATE7_PRECISION_EVALUATION_BATCH_SIZE = 64
GATE7_PRECISION_PHYSICAL_BATCH_COUNT = 32
GATE7_PRECISION_BOOTSTRAP_SAMPLES = 20_000
GATE7_PRECISION_PRIMARY_ATTEMPTS = 128
GATE7_PRECISION_ATTEMPT_LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024)
GATE7_PRECISION_HINT_RELIABILITY = 0.70
GATE7_PRECISION_NEAR_CEILING_MARGIN = 0.02
GATE7_PRECISION_LEARNED_PARAMETER_COUNT = 19_649
GATE7_PRECISION_RANKERS = (
    "learned_score_rank",
    "bayes_hint_likelihood_rank",
    "public_hash_rank",
)

GATE7_PRECISION_HIDDEN_NAMESPACE = (
    "gate7-information-ceiling-precision-confirmation-hidden-v0"
)
GATE7_PRECISION_HINT_NAMESPACE = (
    "gate7-information-ceiling-precision-confirmation-hints-v0"
)
GATE7_PRECISION_RUNTIME_NAMESPACE = (
    "gate7-information-ceiling-precision-confirmation-runtime-v0"
)
GATE7_PRECISION_TIE_NAMESPACE = (
    "gate7-information-ceiling-precision-confirmation-public-tie-v0"
)
GATE7_PRECISION_HASH_NAMESPACE = (
    "gate7-information-ceiling-precision-confirmation-public-hash-v0"
)
GATE7_PRECISION_BOOTSTRAP_NAMESPACE = (
    "gate7-information-ceiling-precision-confirmation-clustered-bootstrap-v0"
)

GATE7_PRECISION_INFORMATION_CEILING_DOMINANT = (
    "G7_PRECISION_INFORMATION_CEILING_DOMINANT"
)
GATE7_PRECISION_SCORER_REPRESENTATION_GAP = (
    "G7_PRECISION_SCORER_REPRESENTATION_GAP"
)
GATE7_PRECISION_INFORMATION_AND_SCORER_GAP_MIXED = (
    "G7_PRECISION_INFORMATION_AND_SCORER_GAP_MIXED"
)
GATE7_PRECISION_INCONCLUSIVE = "G7_PRECISION_INCONCLUSIVE"

GATE7_PRECISION_VALID_OUTCOMES = (
    GATE7_PRECISION_INFORMATION_CEILING_DOMINANT,
    GATE7_PRECISION_SCORER_REPRESENTATION_GAP,
    GATE7_PRECISION_INFORMATION_AND_SCORER_GAP_MIXED,
    GATE7_PRECISION_INCONCLUSIVE,
)


@dataclass(frozen=True, slots=True)
class Gate7PrecisionCellComparison:
    population: int
    checkpoint_index: int
    learned_minus_bayes_delta: float
    learned_minus_bayes_ci_low: float
    learned_minus_bayes_ci_high: float
    learned_minus_hash_ci_low: float
    bayes_minus_hash_ci_low: float

    def validate(self) -> None:
        if self.population not in GATE7_PRECISION_POPULATIONS:
            raise ValueError("precision cell population is outside the frozen ladder")
        if self.checkpoint_index not in GATE7_PRECISION_CHECKPOINT_INDICES:
            raise ValueError("precision cell checkpoint is outside T0/T1/T2")
        if self.learned_minus_bayes_ci_low > self.learned_minus_bayes_ci_high:
            raise ValueError("precision cell learned-vs-Bayes interval is reversed")

    def clear_scorer_gap(self) -> bool:
        self.validate()
        return (
            self.learned_minus_bayes_ci_high
            < -GATE7_PRECISION_NEAR_CEILING_MARGIN
            and self.bayes_minus_hash_ci_low > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7PrecisionPopulationComparison:
    population: int
    learned_minus_bayes_delta: float
    learned_minus_bayes_ci_low: float
    learned_minus_bayes_ci_high: float
    learned_minus_hash_ci_low: float
    bayes_minus_hash_ci_low: float

    def validate(self) -> None:
        if self.population not in GATE7_PRECISION_POPULATIONS:
            raise ValueError("precision population summary is outside the frozen ladder")
        if self.learned_minus_bayes_ci_low > self.learned_minus_bayes_ci_high:
            raise ValueError("precision population interval is reversed")

    def point_within_margin(self) -> bool:
        self.validate()
        return self.learned_minus_bayes_delta > -GATE7_PRECISION_NEAR_CEILING_MARGIN

    def clear_scorer_gap(self) -> bool:
        self.validate()
        return (
            self.learned_minus_bayes_ci_high
            < -GATE7_PRECISION_NEAR_CEILING_MARGIN
            and self.bayes_minus_hash_ci_low > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7PrecisionPooledComparison:
    learned_minus_bayes_delta: float
    learned_minus_bayes_ci_low: float
    learned_minus_bayes_ci_high: float
    learned_minus_hash_ci_low: float
    bayes_minus_hash_ci_low: float

    def validate(self) -> None:
        if self.learned_minus_bayes_ci_low > self.learned_minus_bayes_ci_high:
            raise ValueError("precision pooled interval is reversed")

    def controls_positive(self) -> bool:
        self.validate()
        return self.learned_minus_hash_ci_low > 0.0 and self.bayes_minus_hash_ci_low > 0.0

    def near_ceiling(self) -> bool:
        self.validate()
        return (
            self.learned_minus_bayes_ci_low
            > -GATE7_PRECISION_NEAR_CEILING_MARGIN
            and self.controls_positive()
        )

    def clear_scorer_gap(self) -> bool:
        self.validate()
        return (
            self.learned_minus_bayes_ci_high
            < -GATE7_PRECISION_NEAR_CEILING_MARGIN
            and self.bayes_minus_hash_ci_low > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _validate_complete_matrix(
    cells: tuple[Gate7PrecisionCellComparison, ...],
    populations: tuple[Gate7PrecisionPopulationComparison, ...],
) -> None:
    expected_cells = tuple(
        (population, checkpoint)
        for population in GATE7_PRECISION_POPULATIONS
        for checkpoint in GATE7_PRECISION_CHECKPOINT_INDICES
    )
    observed_cells = tuple((row.population, row.checkpoint_index) for row in cells)
    if observed_cells != expected_cells:
        raise ValueError("precision cells must be population-major T0/T1/T2")
    for row in cells:
        row.validate()

    observed_populations = tuple(row.population for row in populations)
    if observed_populations != GATE7_PRECISION_POPULATIONS:
        raise ValueError("precision population summaries must cover the exact ladder")
    for row in populations:
        row.validate()


def classify_gate7_precision_confirmation(
    *,
    cells: tuple[Gate7PrecisionCellComparison, ...],
    populations: tuple[Gate7PrecisionPopulationComparison, ...],
    pooled: Gate7PrecisionPooledComparison,
) -> str:
    """Apply the frozen precision-confirmation decision hierarchy."""

    _validate_complete_matrix(cells, populations)
    pooled.validate()

    any_local_clear_gap = any(row.clear_scorer_gap() for row in cells) or any(
        row.clear_scorer_gap() for row in populations
    )

    if pooled.clear_scorer_gap():
        return GATE7_PRECISION_SCORER_REPRESENTATION_GAP

    if (
        pooled.near_ceiling()
        and all(row.point_within_margin() for row in populations)
        and not any_local_clear_gap
    ):
        return GATE7_PRECISION_INFORMATION_CEILING_DOMINANT

    if any_local_clear_gap:
        return GATE7_PRECISION_INFORMATION_AND_SCORER_GAP_MIXED

    return GATE7_PRECISION_INCONCLUSIVE


def gate7_precision_confirmation_plan() -> dict[str, Any]:
    """Return the complete immutable protocol plan without opening execution."""

    return {
        "version": GATE7_PRECISION_VERSION,
        "scientific_status": GATE7_PRECISION_SCIENTIFIC_STATUS,
        "execution_admitted": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "communication_intervention_performed": False,
        "adaptive_attempt_exposure_performed": False,
        "prior_worlds_reused": False,
        "populations": list(GATE7_PRECISION_POPULATIONS),
        "checkpoint_indices": list(GATE7_PRECISION_CHECKPOINT_INDICES),
        "world_count_per_population": GATE7_PRECISION_WORLD_COUNT,
        "world_count_per_checkpoint_population": GATE7_PRECISION_WORLD_COUNT,
        "evaluation_batch_size": GATE7_PRECISION_EVALUATION_BATCH_SIZE,
        "physical_batch_count": GATE7_PRECISION_PHYSICAL_BATCH_COUNT,
        "bootstrap_samples": GATE7_PRECISION_BOOTSTRAP_SAMPLES,
        "primary_attempts": GATE7_PRECISION_PRIMARY_ATTEMPTS,
        "attempt_ladder": list(GATE7_PRECISION_ATTEMPT_LADDER),
        "rankers": list(GATE7_PRECISION_RANKERS),
        "hint_reliability": GATE7_PRECISION_HINT_RELIABILITY,
        "near_ceiling_margin": GATE7_PRECISION_NEAR_CEILING_MARGIN,
        "learned_parameter_count": GATE7_PRECISION_LEARNED_PARAMETER_COUNT,
        "bootstrap_unit": "world_index_clustered_within_population_across_T0_T1_T2",
        "pooled_weighting": "equal_population_then_equal_checkpoint",
        "primary_analysis": "pooled_M128_learned_vs_Bayes_noninferiority",
        "local_gap_guards": "population_pooled_and_cellwise_M128_clear_gap",
        "valid_outcomes": list(GATE7_PRECISION_VALID_OUTCOMES),
    }
