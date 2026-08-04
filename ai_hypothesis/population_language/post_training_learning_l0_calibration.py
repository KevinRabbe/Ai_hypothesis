"""Locked calibration and selection contract for Post-Training Learning L0."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from . import post_training_learning_l0_adapter as adapter
from . import post_training_learning_l0_protocol as protocol
from . import post_training_learning_l0_world as world

VERSION = "population-language-post-training-learning-l0-calibration-v0"
BRANCH = "agent/population-language-post-training-learning-l0-calibration-v0"
STATUS = "CALIBRATION_CONTRACT_ONLY_NO_CALIBRATION_OR_FINAL_RESULT"
SOURCE_ADAPTER_HEAD = "508a1021f3724a39023d4a4f7c6918d98f379f5c"

RANKS = (1, 2, 4, 6)
LEARNING_RATES = (0.001, 0.003, 0.01)
UPDATE_COUNTS = (32, 64, 128, 256)
CALIBRATION_PAIR_COUNT = 3
CANDIDATE_COUNT = 48
EXPECTED_RESULT_ROWS = 144
GRID_SHA256 = "84264fbb475259ca224c01cee81700a62c6baf7e73017a0510fc5cbc6c036874"

OPTIMIZER = "AdamW"
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPSILON = 1e-8
WEIGHT_DECAY = 0.0
MAX_GRADIENT_NORM = 1.0
MICROBATCH_SIZE = 8
LEARNING_RATE_SCHEDULE = "CONSTANT"
ADAPTATION_ORDER = "DETERMINISTIC_REPEAT_FROM_ORDINAL_ZERO_V0"
AUTOCAST_MODE = "CUDA_BF16_AUTOCAST_WITH_FP32_WEIGHTS"
WORKER_COUNT = 32
EARLY_STOPPING_ALLOWED = False
MIN_MEAN_CALIBRATION_COMPOSITION_GAIN = 0.01

CALIBRATION_RUN_VALID = "POST_TRAINING_LEARNING_L0_CALIBRATION_RUN_VALID"
CALIBRATION_RUN_INVALID = "POST_TRAINING_LEARNING_L0_CALIBRATION_RUN_INVALID"
CALIBRATION_SELECTS_CANDIDATE = "POST_TRAINING_LEARNING_L0_CALIBRATION_SELECTS_ADAPTER_CANDIDATE"
CALIBRATION_REJECTS_CANDIDATE = (
    "POST_TRAINING_LEARNING_L0_CALIBRATION_REJECTS_ADAPTER_CANDIDATE"
)
FINAL_EXECUTION_ELIGIBLE_NOT_AUTHORIZED = (
    "FINAL_EXECUTION_ELIGIBLE_AFTER_SEPARATE_EXPLICIT_AUTHORIZATION"
)
FINAL_EXECUTION_NOT_ELIGIBLE = "FINAL_EXECUTION_NOT_ELIGIBLE"
INVALID_CALIBRATION_NO_SELECTION = "INVALID_CALIBRATION_NO_CANDIDATE_SELECTION"

CALIBRATION_PAIRS = tuple(
    zip(
        world.MODEL_INITIALIZATION_SEEDS,
        world.CALIBRATION_WORLD_SEEDS,
        strict=True,
    )
)


def _learning_rate_label(value: float) -> str:
    labels = {0.001: "0p001", 0.003: "0p003", 0.01: "0p01"}
    try:
        return labels[value]
    except KeyError as error:
        raise ValueError("learning rate lies outside the locked grid") from error


@dataclass(frozen=True)
class CalibrationCandidate:
    rank: int
    learning_rate: float
    updates: int

    def validate(self) -> "CalibrationCandidate":
        if self.rank not in RANKS:
            raise ValueError("rank lies outside the locked grid")
        if self.learning_rate not in LEARNING_RATES:
            raise ValueError("learning rate lies outside the locked grid")
        if self.updates not in UPDATE_COUNTS:
            raise ValueError("update count lies outside the locked grid")
        return self

    @property
    def identifier(self) -> str:
        self.validate()
        return (
            f"r{self.rank}-lr{_learning_rate_label(self.learning_rate)}"
            f"-u{self.updates}"
        )

    @property
    def trainable_parameters(self) -> int:
        return adapter.parameter_count(self.rank)

    @property
    def persisted_fp32_bytes(self) -> int:
        return adapter.raw_fp32_bytes(self.rank)

    @property
    def example_presentations(self) -> int:
        return self.updates * MICROBATCH_SIZE


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: CalibrationCandidate
    direct_gains: tuple[float, ...]
    composition_gains: tuple[float, ...]
    qualified: bool

    @property
    def minimum_direct_gain(self) -> float:
        return min(self.direct_gains)

    @property
    def mean_direct_gain(self) -> float:
        return sum(self.direct_gains) / len(self.direct_gains)

    @property
    def minimum_composition_gain(self) -> float:
        return min(self.composition_gains)

    @property
    def mean_composition_gain(self) -> float:
        return sum(self.composition_gains) / len(self.composition_gains)


def candidate_grid() -> tuple[CalibrationCandidate, ...]:
    grid = tuple(
        CalibrationCandidate(rank, learning_rate, updates)
        for rank in RANKS
        for learning_rate in LEARNING_RATES
        for updates in UPDATE_COUNTS
    )
    if len(grid) != CANDIDATE_COUNT:
        raise RuntimeError("locked calibration grid size drifted")
    if len({candidate.identifier for candidate in grid}) != len(grid):
        raise RuntimeError("locked calibration candidate identifiers are not unique")
    return grid


def expected_result_keys() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (candidate.identifier, model_seed, calibration_world_seed)
        for candidate in candidate_grid()
        for model_seed, calibration_world_seed in CALIBRATION_PAIRS
    )


def _is_hex_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value: object) -> bool:
    return _is_hex_digest(value, 64)


def _is_git_commit_sha(value: object) -> bool:
    return _is_hex_digest(value, 40)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _probability(value: object) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _exact_float(value: object, expected: float) -> bool:
    return _finite_number(value) and math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-15
    )


def _sequence_matches(value: object, expected: Sequence[float]) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == len(expected)
        and all(_exact_float(actual, target) for actual, target in zip(value, expected))
    )


def _row_metrics_valid(row: Mapping[str, object]) -> bool:
    return all(
        _probability(row.get(field))
        for field in (
            "baseline_direct_accuracy",
            "immediate_direct_accuracy",
            "post_restart_direct_accuracy",
            "baseline_composition_accuracy",
            "immediate_composition_accuracy",
            "post_restart_composition_accuracy",
        )
    )


def validate_result_row(
    row: Mapping[str, object],
    candidate: CalibrationCandidate,
    model_seed: int,
    calibration_world_seed: int,
) -> bool:
    candidate.validate()
    if not _row_metrics_valid(row):
        return False
    base_hash = row.get("base_checkpoint_sha256")
    immediate_direct = float(row["immediate_direct_accuracy"])
    restarted_direct = float(row["post_restart_direct_accuracy"])
    immediate_composition = float(row["immediate_composition_accuracy"])
    restarted_composition = float(row["post_restart_composition_accuracy"])
    restart_drift = max(
        abs(immediate_direct - restarted_direct),
        abs(immediate_composition - restarted_composition),
    )
    return bool(
        row.get("candidate_id") == candidate.identifier
        and row.get("rank") == candidate.rank
        and _exact_float(row.get("learning_rate"), candidate.learning_rate)
        and row.get("adaptation_updates") == candidate.updates
        and row.get("model_seed") == model_seed
        and row.get("calibration_world_seed") == calibration_world_seed
        and row.get("source_adapter_head") == SOURCE_ADAPTER_HEAD
        and row.get("candidate_grid_sha256") == GRID_SHA256
        and row.get("adapter_initialization_seed")
        == protocol.adapter_initialization_seed(model_seed)
        and row.get("optimizer") == OPTIMIZER
        and _sequence_matches(row.get("adamw_betas"), ADAMW_BETAS)
        and _exact_float(row.get("adamw_epsilon"), ADAMW_EPSILON)
        and _exact_float(row.get("weight_decay"), WEIGHT_DECAY)
        and _exact_float(row.get("max_gradient_norm"), MAX_GRADIENT_NORM)
        and row.get("microbatch_size") == MICROBATCH_SIZE
        and row.get("learning_rate_schedule") == LEARNING_RATE_SCHEDULE
        and row.get("adaptation_order") == ADAPTATION_ORDER
        and row.get("autocast_mode") == AUTOCAST_MODE
        and row.get("worker_count") == WORKER_COUNT
        and row.get("early_stopping_used") is False
        and row.get("adaptation_examples") == world.ADAPTATION_EXAMPLES
        and row.get("adaptation_example_presentations")
        == candidate.example_presentations
        and candidate.example_presentations
        <= protocol.MAX_ADAPTATION_EXAMPLE_PRESENTATIONS
        and row.get("direct_holdout_episodes")
        == world.DIRECT_HOLDOUT_EXAMPLES
        and row.get("composition_episodes") == world.CALIBRATION_EXAMPLES
        and row.get("composition_depth") == world.SPLIT_DEPTH["calibration"]
        and row.get("trainable_adaptation_parameters")
        == candidate.trainable_parameters
        and candidate.trainable_parameters
        <= protocol.MAX_TRAINABLE_ADAPTATION_PARAMETERS
        and row.get("persisted_adaptation_bytes")
        == candidate.persisted_fp32_bytes
        and candidate.persisted_fp32_bytes
        <= protocol.MAX_PERSISTED_ADAPTATION_BYTES
        and _is_sha256(base_hash)
        and row.get("base_checkpoint_after_sha256") == base_hash
        and _is_sha256(row.get("adaptation_artifact_sha256"))
        and row.get("adaptation_artifact_payload_kind")
        == protocol.ARTIFACT_PAYLOAD_KIND
        and row.get("adaptation_artifact_contains_raw_examples") is False
        and row.get("adaptation_uses_only_declared_examples_and_gradients")
        is True
        and row.get("raw_adaptation_examples_available_at_evaluation") is False
        and row.get("external_retrieval_enabled_at_evaluation") is False
        and row.get("world_seed_available_to_adaptation") is False
        and row.get("world_rule_parameters_available_to_adaptation") is False
        and row.get("world_generator_imported_by_model_runtime") is False
        and row.get("symbolic_rule_fitting_used") is False
        and row.get("symbolic_execution_used_at_evaluation") is False
        and row.get("model_logits_are_authoritative") is True
        and row.get("final_worlds_loaded_during_calibration") is False
        and row.get("final_world_labels_used_during_calibration") is False
        and row.get("calibration_world_fingerprints")
        == world.calibration_world_fingerprints()
        and row.get("original_l0_path_bitwise_identical") is True
        and row.get("transient_state_cleared_before_restart") is True
        and row.get("base_checkpoint_loaded_fresh_after_restart") is True
        and row.get("adaptation_artifact_loaded_after_restart") is True
        and row.get("fresh_process_restart") is True
        and restart_drift <= protocol.MAX_RESTART_ACCURACY_DRIFT + 1e-12
    )


def evaluate_candidate(
    candidate: CalibrationCandidate,
    rows: Sequence[Mapping[str, object]],
) -> CandidateEvaluation:
    if len(rows) != CALIBRATION_PAIR_COUNT:
        raise ValueError("candidate evaluation requires exactly three seed-pair rows")
    direct_gains: list[float] = []
    composition_gains: list[float] = []
    for row, (model_seed, calibration_world_seed) in zip(
        rows, CALIBRATION_PAIRS, strict=True
    ):
        if not validate_result_row(
            row, candidate, model_seed, calibration_world_seed
        ):
            raise ValueError("candidate row violates the calibration contract")
        direct_gains.append(
            float(row["post_restart_direct_accuracy"])
            - float(row["baseline_direct_accuracy"])
        )
        composition_gains.append(
            float(row["post_restart_composition_accuracy"])
            - float(row["baseline_composition_accuracy"])
        )
    direct = tuple(direct_gains)
    composition = tuple(composition_gains)
    qualified = bool(
        all(gain > 0.0 for gain in direct)
        and all(gain > 0.0 for gain in composition)
        and sum(composition) / len(composition)
        >= MIN_MEAN_CALIBRATION_COMPOSITION_GAIN
    )
    return CandidateEvaluation(candidate, direct, composition, qualified)


def selection_key(evaluation: CandidateEvaluation) -> tuple[object, ...]:
    if not evaluation.qualified:
        raise ValueError("only qualified candidates can be ranked")
    return (
        -evaluation.minimum_composition_gain,
        -evaluation.mean_composition_gain,
        -evaluation.minimum_direct_gain,
        -evaluation.mean_direct_gain,
        evaluation.candidate.trainable_parameters,
        evaluation.candidate.updates,
        evaluation.candidate.learning_rate,
        evaluation.candidate.identifier,
    )


def select_qualified_candidate(
    evaluations: Sequence[CandidateEvaluation],
) -> CandidateEvaluation | None:
    qualified = [evaluation for evaluation in evaluations if evaluation.qualified]
    if not qualified:
        return None
    if len({evaluation.candidate.identifier for evaluation in qualified}) != len(
        qualified
    ):
        raise ValueError("candidate evaluations contain duplicate identifiers")
    return min(qualified, key=selection_key)


def evaluate_calibration(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    expected_keys = expected_result_keys()
    observed_keys = tuple(
        (
            row.get("candidate_id"),
            row.get("model_seed"),
            row.get("calibration_world_seed"),
        )
        for row in rows
    )
    if len(rows) != EXPECTED_RESULT_ROWS or observed_keys != expected_keys:
        return {
            "run_classification": CALIBRATION_RUN_INVALID,
            "conclusion": INVALID_CALIBRATION_NO_SELECTION,
            "final_execution_eligibility": FINAL_EXECUTION_NOT_ELIGIBLE,
        }

    evaluations: list[CandidateEvaluation] = []
    offset = 0
    try:
        for candidate in candidate_grid():
            candidate_rows = rows[offset : offset + CALIBRATION_PAIR_COUNT]
            evaluations.append(evaluate_candidate(candidate, candidate_rows))
            offset += CALIBRATION_PAIR_COUNT
    except (KeyError, TypeError, ValueError):
        return {
            "run_classification": CALIBRATION_RUN_INVALID,
            "conclusion": INVALID_CALIBRATION_NO_SELECTION,
            "final_execution_eligibility": FINAL_EXECUTION_NOT_ELIGIBLE,
        }

    selected = select_qualified_candidate(evaluations)
    if selected is None:
        return {
            "run_classification": CALIBRATION_RUN_VALID,
            "conclusion": CALIBRATION_REJECTS_CANDIDATE,
            "qualified_candidate_count": 0,
            "selected_candidate_id": None,
            "final_execution_eligibility": FINAL_EXECUTION_NOT_ELIGIBLE,
        }
    return {
        "run_classification": CALIBRATION_RUN_VALID,
        "conclusion": CALIBRATION_SELECTS_CANDIDATE,
        "qualified_candidate_count": sum(
            evaluation.qualified for evaluation in evaluations
        ),
        "selected_candidate_id": selected.candidate.identifier,
        "selected_rank": selected.candidate.rank,
        "selected_learning_rate": selected.candidate.learning_rate,
        "selected_updates": selected.candidate.updates,
        "selected_trainable_adaptation_parameters":
            selected.candidate.trainable_parameters,
        "selected_persisted_adaptation_bytes":
            selected.candidate.persisted_fp32_bytes,
        "minimum_direct_gain": selected.minimum_direct_gain,
        "mean_direct_gain": selected.mean_direct_gain,
        "minimum_composition_gain": selected.minimum_composition_gain,
        "mean_composition_gain": selected.mean_composition_gain,
        "final_execution_eligibility":
            FINAL_EXECUTION_ELIGIBLE_NOT_AUTHORIZED,
        "separate_explicit_authorization_still_required": True,
    }


def validate_calibration_contract() -> dict[str, Any]:
    grid = candidate_grid()
    checks = {
        "source_adapter_head_is_locked": _is_git_commit_sha(SOURCE_ADAPTER_HEAD),
        "adapter_ranks_match_grid": tuple(adapter.SUPPORTED_RANKS) == RANKS,
        "candidate_count_is_48": len(grid) == CANDIDATE_COUNT == 48,
        "result_rows_are_144":
            len(expected_result_keys()) == EXPECTED_RESULT_ROWS == 144,
        "three_model_world_pairs":
            len(CALIBRATION_PAIRS) == CALIBRATION_PAIR_COUNT == 3,
        "calibration_and_final_worlds_disjoint":
            set(world.CALIBRATION_WORLD_SEEDS).isdisjoint(
                world.FINAL_WORLD_SEEDS
            ),
        "grid_hash_is_locked":
            GRID_SHA256
            == "84264fbb475259ca224c01cee81700a62c6baf7e73017a0510fc5cbc6c036874",
        "all_parameter_budgets_hold": all(
            candidate.trainable_parameters
            <= protocol.MAX_TRAINABLE_ADAPTATION_PARAMETERS
            for candidate in grid
        ),
        "all_artifact_budgets_hold": all(
            candidate.persisted_fp32_bytes
            <= protocol.MAX_PERSISTED_ADAPTATION_BYTES
            for candidate in grid
        ),
        "all_presentation_budgets_hold": all(
            candidate.example_presentations
            <= protocol.MAX_ADAPTATION_EXAMPLE_PRESENTATIONS
            for candidate in grid
        ),
        "no_early_stopping": EARLY_STOPPING_ALLOWED is False,
        "constant_learning_rate": LEARNING_RATE_SCHEDULE == "CONSTANT",
        "worker_count_is_32": WORKER_COUNT == 32,
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "source_adapter_head": SOURCE_ADAPTER_HEAD,
        "grid_sha256": GRID_SHA256,
        "candidate_count": len(grid),
        "expected_result_rows": len(expected_result_keys()),
        "calibration_pairs": [list(pair) for pair in CALIBRATION_PAIRS],
        "optimizer": {
            "name": OPTIMIZER,
            "betas": list(ADAMW_BETAS),
            "epsilon": ADAMW_EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "max_gradient_norm": MAX_GRADIENT_NORM,
            "microbatch_size": MICROBATCH_SIZE,
            "schedule": LEARNING_RATE_SCHEDULE,
            "autocast_mode": AUTOCAST_MODE,
        },
        "checks": checks,
        "valid": all(checks.values()),
    }
