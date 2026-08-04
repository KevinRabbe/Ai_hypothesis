"""Preregistered validity and success contract for Post-Training Learning L0."""
from __future__ import annotations

import math
from typing import Any

from . import post_training_learning_l0_world as world

VERSION = world.VERSION
BRANCH = "agent/population-language-post-training-learning-l0-protocol-v0"
STATUS = "PROTOCOL_ONLY_NO_ADAPTATION_RESULT"
SOURCE_REFERENCE_IMPLEMENTATION_HEAD = "f7d997e828e2a8791592c060973080e3fe3c43bd"
PROTOCOL_BASE_HEAD = "bfd2111b65f805e6379ad45ecda6f5fe09d2a282"
BASE_MODEL = "population"
BASE_PARAMETER_COUNT = 18_967_968

MAX_TRAINABLE_ADAPTATION_PARAMETERS = math.ceil(BASE_PARAMETER_COUNT * 0.01)
MAX_PERSISTED_ADAPTATION_BYTES = 1_048_576
MAX_ADAPTATION_UPDATES = 256
MAX_ADAPTATION_EXAMPLE_PRESENTATIONS = 4_096
MIN_MEAN_COMPOSITION_TEST_GAIN = 0.02
MAX_MEAN_RETENTION_DROP = 0.005
MAX_PER_SEED_RETENTION_DROP = 0.01
MAX_RESTART_ACCURACY_DRIFT = 0.001

ARTIFACT_PAYLOAD_KIND = "DECLARED_TRAINABLE_TENSORS_ONLY"
PAIRED_PROCEDURE = "DETERMINISTIC_PAIRED_BOOTSTRAP_PERCENTILE_V0"
PAIRED_RNG = "NUMPY_PCG64"
PAIRED_QUANTILE_METHOD = "linear"
PAIRED_BOOTSTRAP_RESAMPLES = 20_000
PAIRED_BOOTSTRAP_LOWER_PERCENTILE = 0.025
PAIRED_BOOTSTRAP_SEED_OFFSET = 900_000
ADAPTER_INITIALIZATION_SEED_OFFSET = 700_000

RUN_VALID = "POST_TRAINING_LEARNING_L0_RUN_VALID"
RUN_INVALID = "POST_TRAINING_LEARNING_L0_RUN_INVALID"
SUPPORTS = "SUPPORTS_PERSISTENT_POST_TRAINING_LEARNING"
DOES_NOT_SUPPORT = "DOES_NOT_SUPPORT_PERSISTENT_POST_TRAINING_LEARNING"
INVALID_CONCLUSION = "INVALID_RUN_NO_POST_TRAINING_LEARNING_CONCLUSION"


def adapter_initialization_seed(model_seed: int) -> int:
    if model_seed not in world.MODEL_INITIALIZATION_SEEDS:
        raise ValueError("adapter initialization requires a preregistered model seed")
    return ADAPTER_INITIALIZATION_SEED_OFFSET + model_seed


def paired_bootstrap_seed(world_seed: int) -> int:
    if world_seed not in world.FINAL_WORLD_SEEDS:
        raise ValueError("paired bootstrap requires a preregistered final world seed")
    return PAIRED_BOOTSTRAP_SEED_OFFSET + world_seed


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _probability(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def validate_protocol() -> dict[str, Any]:
    checks: dict[str, bool] = {
        "operator_tokens_are_unique": len(set(world.OPERATOR_TOKENS)) == len(world.OPERATOR_TOKENS),
        "value_tokens_are_unique": len(set(world.VALUE_TOKENS)) == world.DOMAIN_SIZE,
        "model_and_final_world_seed_counts_match": len(world.MODEL_INITIALIZATION_SEEDS) == len(world.FINAL_WORLD_SEEDS),
        "calibration_and_final_worlds_are_disjoint": set(world.CALIBRATION_WORLD_SEEDS).isdisjoint(world.FINAL_WORLD_SEEDS),
        "adapter_parameter_budget_is_at_most_one_percent": MAX_TRAINABLE_ADAPTATION_PARAMETERS <= math.ceil(BASE_PARAMETER_COUNT * 0.01),
        "artifact_budget_is_one_mib": MAX_PERSISTED_ADAPTATION_BYTES == 1_048_576,
        "paired_bootstrap_is_locked": PAIRED_BOOTSTRAP_RESAMPLES == 20_000 and PAIRED_BOOTSTRAP_LOWER_PERCENTILE == 0.025,
        "adapter_seed_ignores_world_identity": all(
            adapter_initialization_seed(seed) == ADAPTER_INITIALIZATION_SEED_OFFSET + seed
            for seed in world.MODEL_INITIALIZATION_SEEDS
        ),
    }
    for seed in (*world.CALIBRATION_WORLD_SEEDS, *world.FINAL_WORLD_SEEDS):
        generated = world.make_world(seed)
        checks[f"world_{seed}_rules_unique"] = len({(r.multiplier, r.offset) for r in generated.operators}) == len(world.OPERATOR_TOKENS)
        checks[f"world_{seed}_rules_permute_domain"] = all(
            len({rule.apply(value) for value in range(world.DOMAIN_SIZE)}) == world.DOMAIN_SIZE
            for rule in generated.operators
        )
        adaptation = {
            (world.make_example("adaptation", ordinal, seed).operators, world.make_example("adaptation", ordinal, seed).input_value)
            for ordinal in range(world.ADAPTATION_EXAMPLES)
        }
        holdout = {
            (world.make_example("direct_holdout", ordinal, seed).operators, world.make_example("direct_holdout", ordinal, seed).input_value)
            for ordinal in range(world.DIRECT_HOLDOUT_EXAMPLES)
        }
        checks[f"world_{seed}_direct_splits_disjoint"] = adaptation.isdisjoint(holdout)
    return {
        "status": STATUS,
        "version": VERSION,
        "source_reference_implementation_head": SOURCE_REFERENCE_IMPLEMENTATION_HEAD,
        "protocol_base_head": PROTOCOL_BASE_HEAD,
        "base_model": BASE_MODEL,
        "base_parameter_count": BASE_PARAMETER_COUNT,
        "max_trainable_adaptation_parameters": MAX_TRAINABLE_ADAPTATION_PARAMETERS,
        "max_persisted_adaptation_bytes": MAX_PERSISTED_ADAPTATION_BYTES,
        "calibration_world_seeds": list(world.CALIBRATION_WORLD_SEEDS),
        "final_world_seeds": list(world.FINAL_WORLD_SEEDS),
        "split_counts": dict(world.SPLIT_COUNTS),
        "checks": checks,
        "valid": all(checks.values()),
    }


def _metric_row(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("direct_episodes") != world.DIRECT_HOLDOUT_EXAMPLES:
        return False
    if value.get("composition_episodes") != world.TEST_EXAMPLES:
        return False
    if value.get("retention_episodes") != world.RETENTION_TEST_EPISODES:
        return False
    return all(_probability(value.get(field)) for field in (
        "direct_accuracy", "composition_accuracy", "retention_accuracy"
    ))


def _calibration_evidence_valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("calibration_world_seeds") == list(world.CALIBRATION_WORLD_SEEDS)
        and row.get("calibration_world_fingerprints") == world.calibration_world_fingerprints()
        and row.get("final_world_labels_used_during_calibration") is False
    )


def _neural_boundary_valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("adaptation_artifact_payload_kind") == ARTIFACT_PAYLOAD_KIND
        and row.get("adaptation_artifact_contains_raw_examples") is False
        and row.get("adaptation_uses_only_declared_examples_and_gradients") is True
        and row.get("world_seed_available_to_adaptation") is False
        and row.get("world_rule_parameters_available_to_adaptation") is False
        and row.get("world_generator_imported_by_model_runtime") is False
        and row.get("symbolic_rule_fitting_used") is False
        and row.get("symbolic_execution_used_at_evaluation") is False
        and row.get("model_logits_are_authoritative") is True
    )


def _boundaries_valid(row: dict[str, Any]) -> bool:
    base = row.get("base_checkpoint_sha256")
    model_seed = row.get("model_seed")
    final_seed = row.get("world_seed")
    return bool(
        _is_sha256(base)
        and row.get("base_checkpoint_after_sha256") == base
        and _is_sha256(row.get("adaptation_artifact_sha256"))
        and row.get("source_reference_implementation_head") == SOURCE_REFERENCE_IMPLEMENTATION_HEAD
        and row.get("base_model") == BASE_MODEL
        and row.get("base_parameter_count") == BASE_PARAMETER_COUNT
        and row.get("adapter_initialization_seed") == adapter_initialization_seed(int(model_seed))
        and row.get("adaptation_examples") == world.ADAPTATION_EXAMPLES
        and type(row.get("trainable_adaptation_parameters")) is int
        and 0 < int(row["trainable_adaptation_parameters"]) <= MAX_TRAINABLE_ADAPTATION_PARAMETERS
        and type(row.get("persisted_adaptation_bytes")) is int
        and 0 < int(row["persisted_adaptation_bytes"]) <= MAX_PERSISTED_ADAPTATION_BYTES
        and type(row.get("adaptation_updates")) is int
        and 0 < int(row["adaptation_updates"]) <= MAX_ADAPTATION_UPDATES
        and type(row.get("adaptation_example_presentations")) is int
        and world.ADAPTATION_EXAMPLES <= int(row["adaptation_example_presentations"]) <= MAX_ADAPTATION_EXAMPLE_PRESENTATIONS
        and row.get("direct_holdout_used_for_selection") is False
        and row.get("validation_used_for_selection") is False
        and row.get("test_used_for_selection") is False
        and row.get("raw_adaptation_examples_available_at_evaluation") is False
        and row.get("external_retrieval_enabled_at_evaluation") is False
        and row.get("transient_state_cleared_before_restart") is True
        and row.get("base_checkpoint_loaded_fresh_after_restart") is True
        and row.get("adaptation_artifact_loaded_after_restart") is True
        and row.get("fresh_process_restart") is True
        and row.get("paired_procedure") == PAIRED_PROCEDURE
        and row.get("paired_rng") == PAIRED_RNG
        and row.get("paired_quantile_method") == PAIRED_QUANTILE_METHOD
        and row.get("paired_bootstrap_resamples") == PAIRED_BOOTSTRAP_RESAMPLES
        and row.get("paired_bootstrap_lower_percentile") == PAIRED_BOOTSTRAP_LOWER_PERCENTILE
        and row.get("paired_bootstrap_seed") == paired_bootstrap_seed(int(final_seed))
        and _calibration_evidence_valid(row)
        and _neural_boundary_valid(row)
    )


def classify_run(seed_rows: list[dict[str, Any]]) -> str:
    expected = list(zip(world.MODEL_INITIALIZATION_SEEDS, world.FINAL_WORLD_SEEDS))
    observed = [(row.get("model_seed"), row.get("world_seed")) for row in seed_rows]
    if observed != expected:
        return RUN_INVALID
    valid = True
    for row in seed_rows:
        if not _boundaries_valid(row):
            valid = False
        final_seed = int(row["world_seed"])
        if row.get("final_world_fingerprints") != world.final_world_fingerprints(final_seed):
            valid = False
        baseline = row.get("baseline")
        immediate = row.get("post_adaptation_immediate")
        restarted = row.get("post_restart")
        if not all(_metric_row(value) for value in (baseline, immediate, restarted)):
            valid = False
            continue
        assert isinstance(immediate, dict) and isinstance(restarted, dict)
        if any(
            abs(float(immediate[field]) - float(restarted[field])) > MAX_RESTART_ACCURACY_DRIFT
            for field in ("direct_accuracy", "composition_accuracy", "retention_accuracy")
        ):
            valid = False
        ci_lower = row.get("paired_composition_gain_ci95_lower")
        if not isinstance(ci_lower, (int, float)) or not math.isfinite(float(ci_lower)):
            valid = False
    return RUN_VALID if valid else RUN_INVALID


def learning_summary(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if classify_run(seed_rows) != RUN_VALID:
        return {"run_classification": RUN_INVALID, "conclusion": INVALID_CONCLUSION}
    direct = [
        float(row["post_restart"]["direct_accuracy"]) - float(row["baseline"]["direct_accuracy"])
        for row in seed_rows
    ]
    composition = [
        float(row["post_restart"]["composition_accuracy"]) - float(row["baseline"]["composition_accuracy"])
        for row in seed_rows
    ]
    retention = [
        float(row["baseline"]["retention_accuracy"]) - float(row["post_restart"]["retention_accuracy"])
        for row in seed_rows
    ]
    acquisition = all(gain > 0.0 for gain in direct)
    generalization = (
        sum(composition) / len(composition) >= MIN_MEAN_COMPOSITION_TEST_GAIN
        and all(gain > 0.0 for gain in composition)
        and all(float(row["paired_composition_gain_ci95_lower"]) > 0.0 for row in seed_rows)
    )
    retention_pass = (
        sum(retention) / len(retention) <= MAX_MEAN_RETENTION_DROP + 1e-12
        and all(drop <= MAX_PER_SEED_RETENTION_DROP + 1e-12 for drop in retention)
    )
    return {
        "run_classification": RUN_VALID,
        "conclusion": SUPPORTS if acquisition and generalization and retention_pass else DOES_NOT_SUPPORT,
        "mean_direct_holdout_gain": sum(direct) / len(direct),
        "direct_holdout_gains_by_seed": direct,
        "mean_composition_test_gain": sum(composition) / len(composition),
        "composition_test_gains_by_seed": composition,
        "mean_retention_drop": sum(retention) / len(retention),
        "retention_drops_by_seed": retention,
        "acquisition_criterion_passed": acquisition,
        "generalization_criterion_passed": generalization,
        "retention_criterion_passed": retention_pass,
        "persistence_criterion_passed": True,
    }
