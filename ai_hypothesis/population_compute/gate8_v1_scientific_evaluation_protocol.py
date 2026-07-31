"""Frozen Gate-8 v1 three-seed scientific-evaluation protocol.

This module binds the three admitted training checkpoints and freezes the
scientific-test estimands.  It contains no checkpoint loader, world generator,
tokenizer/model loader, inference path, or result artifact reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai_hypothesis.population_compute import (
    gate8_distributed_transformation_capability_protocol as capability,
)


GATE8_V1_SCIENTIFIC_EVALUATION_VERSION = (
    "gate8-v1-three-seed-scientific-evaluation-protocol-v0"
)
GATE8_V1_SCIENTIFIC_STATUS = (
    "DATA_FROZEN_GATE8_V1_THREE_SEED_SCIENTIFIC_EVALUATION_EXECUTION_CLOSED"
)

GATE8_V1_BASE_RESULT_HEAD = "4f3159e5ba7abde6045543fd85e691f8f75ef7c4"
GATE8_V1_ORIGINAL_PROTOCOL_HEAD = "e73541115e8ddd122f336463dc1a9ffdbf82df46"
GATE8_V1_PROTOCOL_CORRECTION_HEAD = "124065691d257d483a37be4200452f1f7ca50063"
GATE8_V1_WORLD_CONTRACT_HEAD = "722c646eacfd05c51fb9d1e8887fe1620d53672c"
GATE8_V1_ENCODER_CONTRACT_HEAD = "9882256ae0152bc266dc4d96cab3bbeb0c4ef95b"
GATE8_V1_TOKENIZER_RESULT_HEAD = "c7f5260189ef9ac1a1beb73596446316631090c7"
GATE8_V1_ARCHITECTURE_HEAD = "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8"
GATE8_V1_RUNTIME_HEAD = "333d88ac4fc52f1651741fba224e0b4605feedd3"
GATE8_V1_TRAINING_PROTOCOL_HEAD = "a33dc123d090268a531d112251ea3ab53cb50062"
GATE8_V1_SEED0_EXECUTION_HEAD = "1b449f0ed4998e9246c86803d4473d0ac9ebdac3"
GATE8_V1_REPLICATION_EXECUTION_HEAD = "31a8d115eb14d876997fb361b02258fbe3a30506"

GATE8_V1_REFERENCE_MODEL_ID = "google/gemma-3-1b-it"
GATE8_V1_REFERENCE_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
GATE8_V1_TOKENIZER_RESULT_SHA256 = (
    "c8d6adb733cadbbd251d91d35f9d224e255705dac49ba144655717f9f4ab7b8d"
)
GATE8_V1_TOKENIZER_MANIFEST_SHA256 = (
    "21de192eb57c0759fbf2236fae2252e5319696b71689ada1471b74a9f1315a88"
)

GATE8_V1_TEST_SPLIT = "test"
GATE8_V1_TEST_SEED = 0
GATE8_V1_TEST_WORLD_START = 0
GATE8_V1_TEST_WORLD_END_INCLUSIVE = 511
GATE8_V1_TEST_WORLDS_PER_CONDITION = 512
GATE8_V1_CHECKPOINT_SEEDS = (0, 1, 2)
GATE8_V1_VALID_CONDITIONS = capability.GATE8_VALID_CONDITIONS
GATE8_V1_BOOTSTRAP_SAMPLES = capability.GATE8_BOOTSTRAP_SAMPLES
GATE8_V1_BOOTSTRAP_CONFIDENCE = 0.95
GATE8_V1_BOOTSTRAP_NAMESPACE = (
    "gate8-v1-three-seed-scientific-evaluation-bootstrap-v0"
)
GATE8_V1_BOOTSTRAP_UNIT = (
    "world_index_shared_across_all_three_checkpoint_seeds"
)
GATE8_V1_CONDITION_AGGREGATION = (
    "equal_checkpoint_seed_weight_then_equal_world_weight"
)
GATE8_V1_REFERENCE_POOLING = "equal_weight_across_21_conditions"

GATE8_V1_FULL_MODE = "full"
GATE8_V1_NO_COMMUNICATION_MODE = "no_communication"
GATE8_V1_SHUFFLED_WORKER_MODE = "shuffled_worker"
GATE8_V1_SHUFFLED_MESSAGE_MODE = "shuffled_message"
GATE8_V1_TARGET_WORKER_ONLY_MODE = "target_worker_only"
GATE8_V1_ORGANISM_MODES = (
    GATE8_V1_FULL_MODE,
    GATE8_V1_NO_COMMUNICATION_MODE,
    GATE8_V1_SHUFFLED_WORKER_MODE,
    GATE8_V1_SHUFFLED_MESSAGE_MODE,
    GATE8_V1_TARGET_WORKER_ONLY_MODE,
)
GATE8_V1_CAUSAL_MODES = (
    GATE8_V1_FULL_MODE,
    GATE8_V1_NO_COMMUNICATION_MODE,
    GATE8_V1_SHUFFLED_WORKER_MODE,
)
GATE8_V1_CAUSAL_ABLATION_CONDITIONS = capability.GATE8_CAUSAL_ABLATION_CONDITIONS

GATE8_V1_RANDOM_ANSWER_NAMESPACE = (
    "gate8-v1-scientific-random-answer-control-v0"
)
GATE8_V1_SHUFFLED_MESSAGE_NAMESPACE = (
    "gate8-v1-scientific-shuffled-message-control-v0"
)

GATE8_V1_REQUIRED_PER_WORLD_FIELDS = (
    "checkpoint_seed",
    "population",
    "depth",
    "world_index",
    "world_id",
    "mode",
    "predicted_symbol",
    "answer_symbol",
    "correct",
    "rounds",
    "active_workers",
    "recurrent_updates",
    "delivered_messages",
    "communicated_bits",
    "wall_seconds",
    "peak_device_bytes",
)
GATE8_V1_REQUIRED_CONDITION_METRICS = (
    "accuracy",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "mean_active_workers",
    "mean_communicated_bits",
    "mean_recurrent_updates",
    "mean_wall_seconds",
    "peak_device_bytes",
    "capability_per_learned_parameter",
    "capability_per_active_worker",
    "capability_per_communicated_bit",
    "capability_per_recurrent_update",
    "capability_per_normalized_compute",
)


@dataclass(frozen=True, slots=True)
class Gate8V1CheckpointBinding:
    seed: int
    result_head: str
    result_sha256: str
    selected_checkpoint_sha256: str
    source_manifest_sha256: str
    selected_step: int = 1_024
    learned_parameter_count: int = 19_649
    tensor_count: int = 12

    def validate(self) -> None:
        if self.seed not in GATE8_V1_CHECKPOINT_SEEDS:
            raise ValueError("Gate8 v1 checkpoint seed is outside 0..2")
        if len(self.result_head) != 40 or any(
            character not in "0123456789abcdef" for character in self.result_head
        ):
            raise ValueError("Gate8 v1 checkpoint binding contains a malformed Git head")
        for value in (
            self.result_sha256,
            self.selected_checkpoint_sha256,
            self.source_manifest_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("Gate8 v1 checkpoint binding contains a malformed SHA-256")
        if self.selected_step != 1_024:
            raise ValueError("Gate8 v1 selected checkpoint step changed")
        if self.learned_parameter_count != 19_649:
            raise ValueError("Gate8 v1 learned-parameter budget changed")
        if self.tensor_count != 12:
            raise ValueError("Gate8 v1 checkpoint tensor count changed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


GATE8_V1_CHECKPOINT_BINDINGS = (
    Gate8V1CheckpointBinding(
        seed=0,
        result_head="f259620f7d3beab2f886c76271c753e9ebf96dc9",
        result_sha256="1e42eb53f6446e4eeb66bbb2090c8dad7551e2098b76f289b43cf0c05975e829",
        selected_checkpoint_sha256=(
            "3005369a4830c12baee8ffa7fedc1bed0f1888784e1043bd88f4afd2b7cddde9"
        ),
        source_manifest_sha256=(
            "3db3284b37d4ddd7dfec03ab9fd6c0aa6193d59c0cb887fcb773927eaa13e3ac"
        ),
    ),
    Gate8V1CheckpointBinding(
        seed=1,
        result_head="66532cb72c2bb0703e7af395ef51bbbef31d9b3b",
        result_sha256="873cacdb5965b29c59a14d74fc0df7a32c036f35aeeda2cdd4cb5ac3640a7e8e",
        selected_checkpoint_sha256=(
            "cbcae487dd7f4c695e1d6a83a61926cd43f5ccf6add1a7469c16a15697d22d07"
        ),
        source_manifest_sha256=(
            "22a22993ebe3aff46997fd83605aed25170db6abca631e5c109d8bcc33446133"
        ),
    ),
    Gate8V1CheckpointBinding(
        seed=2,
        result_head="4f3159e5ba7abde6045543fd85e691f8f75ef7c4",
        result_sha256="cc9dad3bd05982ff5390a8f23bff3bfe8227c5a4c4c457e6578426b186bb6df2",
        selected_checkpoint_sha256=(
            "e1e35b3864354e8f3398497a897b6a759dfa3454a33d866de63784a323f461e4"
        ),
        source_manifest_sha256=(
            "2df3483f63e6c31e06a51fde57e57eb773bb183d3cf71a405407748645c89ef0"
        ),
    ),
)


def validate_gate8_v1_checkpoint_bindings(
    bindings: tuple[Gate8V1CheckpointBinding, ...] = GATE8_V1_CHECKPOINT_BINDINGS,
) -> None:
    if tuple(binding.seed for binding in bindings) != GATE8_V1_CHECKPOINT_SEEDS:
        raise ValueError("Gate8 v1 checkpoint bindings must be ordered seeds 0, 1, 2")
    if len({binding.selected_checkpoint_sha256 for binding in bindings}) != 3:
        raise ValueError("Gate8 v1 selected checkpoint identities must be distinct")
    for binding in bindings:
        binding.validate()


@dataclass(frozen=True, slots=True)
class Gate8V1ConditionEvidence:
    population: int
    depth: int
    accuracy: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    seed_accuracies: tuple[float, float, float]
    mean_active_workers: float
    mean_communicated_bits: float
    mean_recurrent_updates: float

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_V1_VALID_CONDITIONS:
            raise ValueError("Gate8 v1 condition evidence is outside the frozen matrix")
        for value in (
            self.accuracy,
            self.bootstrap_ci_low,
            self.bootstrap_ci_high,
            *self.seed_accuracies,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Gate8 v1 accuracy or interval is outside 0..1")
        if self.bootstrap_ci_low > self.bootstrap_ci_high:
            raise ValueError("Gate8 v1 condition interval is reversed")
        if abs(self.accuracy - sum(self.seed_accuracies) / 3.0) > 1e-12:
            raise ValueError("Gate8 v1 pooled accuracy is not equal-seed weighted")
        if not 0.0 <= self.mean_active_workers <= self.population:
            raise ValueError("Gate8 v1 mean active-worker count is invalid")
        if self.mean_communicated_bits < 0.0 or self.mean_recurrent_updates < 0.0:
            raise ValueError("Gate8 v1 runtime accounting cannot be negative")

    def solved(self) -> bool:
        self.validate()
        return (
            self.accuracy >= capability.GATE8_FRONTIER_POINT_ACCURACY
            and self.bootstrap_ci_low >= capability.GATE8_FRONTIER_CI_LOW
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _validate_condition_matrix(
    rows: tuple[Gate8V1ConditionEvidence, ...],
) -> None:
    observed = tuple((row.population, row.depth) for row in rows)
    if observed != GATE8_V1_VALID_CONDITIONS:
        raise ValueError("Gate8 v1 condition rows must cover the exact 21-condition matrix")
    for row in rows:
        row.validate()


def build_gate8_v1_population_frontiers(
    rows: tuple[Gate8V1ConditionEvidence, ...],
) -> tuple[capability.Gate8CapabilityFrontierRow, ...]:
    """Build the original Gate-8 frontier rows from equal-seed pooled evidence."""

    _validate_condition_matrix(rows)
    frontiers: list[capability.Gate8CapabilityFrontierRow] = []
    for population in capability.GATE8_POPULATIONS:
        candidates = tuple(row for row in rows if row.population == population)
        solved = tuple(row for row in candidates if row.solved())
        selected = solved[-1] if solved else candidates[0]
        frontiers.append(
            capability.Gate8CapabilityFrontierRow(
                population=population,
                max_solved_depth=selected.depth,
                frontier_accuracy=selected.accuracy,
                frontier_ci_low=selected.bootstrap_ci_low,
                frontier_ci_high=selected.bootstrap_ci_high,
                active_workers=selected.mean_active_workers,
                communicated_bits=selected.mean_communicated_bits,
                recurrent_updates=selected.mean_recurrent_updates,
            )
        )
    return tuple(frontiers)


@dataclass(frozen=True, slots=True)
class Gate8V1AblationEvidence:
    population: int
    depth: int
    full_accuracy: float
    no_communication_accuracy: float
    shuffled_worker_accuracy: float
    full_seed_accuracies: tuple[float, float, float]
    no_communication_seed_accuracies: tuple[float, float, float]
    shuffled_worker_seed_accuracies: tuple[float, float, float]
    full_minus_no_communication_ci_low: float
    full_minus_shuffled_worker_ci_low: float

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_V1_CAUSAL_ABLATION_CONDITIONS:
            raise ValueError("Gate8 v1 ablation evidence is outside the frozen conditions")
        groups = (
            (self.full_accuracy, self.full_seed_accuracies),
            (
                self.no_communication_accuracy,
                self.no_communication_seed_accuracies,
            ),
            (self.shuffled_worker_accuracy, self.shuffled_worker_seed_accuracies),
        )
        for pooled, seeds in groups:
            if not 0.0 <= pooled <= 1.0 or any(
                not 0.0 <= value <= 1.0 for value in seeds
            ):
                raise ValueError("Gate8 v1 ablation accuracy is outside 0..1")
            if abs(pooled - sum(seeds) / 3.0) > 1e-12:
                raise ValueError("Gate8 v1 ablation accuracy is not equal-seed weighted")

    def to_base_row(self) -> capability.Gate8AblationRow:
        self.validate()
        return capability.Gate8AblationRow(
            population=self.population,
            depth=self.depth,
            full_accuracy=self.full_accuracy,
            no_communication_accuracy=self.no_communication_accuracy,
            shuffled_worker_accuracy=self.shuffled_worker_accuracy,
            full_minus_no_communication_ci_low=(
                self.full_minus_no_communication_ci_low
            ),
            full_minus_shuffled_worker_ci_low=(
                self.full_minus_shuffled_worker_ci_low
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def build_gate8_v1_ablation_rows(
    rows: tuple[Gate8V1AblationEvidence, ...],
) -> tuple[capability.Gate8AblationRow, ...]:
    observed = tuple((row.population, row.depth) for row in rows)
    if observed != GATE8_V1_CAUSAL_ABLATION_CONDITIONS:
        raise ValueError("Gate8 v1 ablations must cover the exact causal conditions")
    return tuple(row.to_base_row() for row in rows)


def classify_gate8_v1_population_scaling(
    *,
    conditions: tuple[Gate8V1ConditionEvidence, ...],
    ablations: tuple[Gate8V1AblationEvidence, ...],
) -> str:
    """Apply the original frozen classifier to pooled three-seed evidence."""

    return capability.classify_gate8_population_scaling(
        frontiers=build_gate8_v1_population_frontiers(conditions),
        ablations=build_gate8_v1_ablation_rows(ablations),
    )


@dataclass(frozen=True, slots=True)
class Gate8V1ReferenceEvidence:
    population: int
    depth: int
    population_accuracy: float
    reference_accuracy: float
    population_seed_accuracies: tuple[float, float, float]
    population_minus_reference_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    maximum_reference_input_tokens: int

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_V1_VALID_CONDITIONS:
            raise ValueError("Gate8 v1 reference evidence is outside the frozen matrix")
        for value in (
            self.population_accuracy,
            self.reference_accuracy,
            *self.population_seed_accuracies,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Gate8 v1 reference accuracy is outside 0..1")
        if abs(
            self.population_accuracy
            - sum(self.population_seed_accuracies) / 3.0
        ) > 1e-12:
            raise ValueError("Gate8 v1 population accuracy is not equal-seed weighted")
        expected_delta = self.population_accuracy - self.reference_accuracy
        if abs(self.population_minus_reference_delta - expected_delta) > 1e-12:
            raise ValueError("Gate8 v1 reference delta disagrees with its accuracies")
        if self.bootstrap_ci_low > self.bootstrap_ci_high:
            raise ValueError("Gate8 v1 reference interval is reversed")
        if not (
            0
            < self.maximum_reference_input_tokens
            <= capability.GATE8_REFERENCE_MAX_INPUT_TOKENS
        ):
            raise ValueError("Gate8 v1 reference prompt exceeds the frozen input budget")

    def to_base_row(self) -> capability.Gate8ReferenceConditionRow:
        self.validate()
        return capability.Gate8ReferenceConditionRow(
            population=self.population,
            depth=self.depth,
            population_accuracy=self.population_accuracy,
            reference_accuracy=self.reference_accuracy,
            population_minus_reference_delta=(
                self.population_minus_reference_delta
            ),
            bootstrap_ci_low=self.bootstrap_ci_low,
            bootstrap_ci_high=self.bootstrap_ci_high,
            reference_input_tokens=self.maximum_reference_input_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def build_gate8_v1_reference_rows(
    rows: tuple[Gate8V1ReferenceEvidence, ...],
) -> tuple[capability.Gate8ReferenceConditionRow, ...]:
    observed = tuple((row.population, row.depth) for row in rows)
    if observed != GATE8_V1_VALID_CONDITIONS:
        raise ValueError("Gate8 v1 reference rows must cover the exact 21-condition matrix")
    return tuple(row.to_base_row() for row in rows)


def classify_gate8_v1_reference_comparison(
    *,
    conditions: tuple[Gate8V1ReferenceEvidence, ...],
    pooled: capability.Gate8ReferencePooledRow,
) -> str:
    """Apply the original frozen 1B comparison classifier unchanged."""

    return capability.classify_gate8_reference_comparison(
        conditions=build_gate8_v1_reference_rows(conditions),
        pooled=pooled,
    )


def gate8_v1_scientific_evaluation_plan() -> dict[str, Any]:
    """Return the immutable plan without opening any execution surface."""

    validate_gate8_v1_checkpoint_bindings()
    return {
        "version": GATE8_V1_SCIENTIFIC_EVALUATION_VERSION,
        "scientific_status": GATE8_V1_SCIENTIFIC_STATUS,
        "base_result_head": GATE8_V1_BASE_RESULT_HEAD,
        "execution_admitted": False,
        "checkpoint_loading_admitted": False,
        "scientific_test_generation_admitted": False,
        "reference_weight_binding_admitted": False,
        "reference_inference_admitted": False,
        "training_admitted": False,
        "checkpoint_bindings": [
            binding.to_dict() for binding in GATE8_V1_CHECKPOINT_BINDINGS
        ],
        "learned_parameter_count": capability.GATE8_LEARNED_PARAMETER_COUNT,
        "checkpoint_seeds": list(GATE8_V1_CHECKPOINT_SEEDS),
        "test_split": GATE8_V1_TEST_SPLIT,
        "test_seed": GATE8_V1_TEST_SEED,
        "test_world_index_start": GATE8_V1_TEST_WORLD_START,
        "test_world_index_end_inclusive": GATE8_V1_TEST_WORLD_END_INCLUSIVE,
        "test_worlds_per_condition": GATE8_V1_TEST_WORLDS_PER_CONDITION,
        "conditions": [list(row) for row in GATE8_V1_VALID_CONDITIONS],
        "condition_count": len(GATE8_V1_VALID_CONDITIONS),
        "organism_modes": list(GATE8_V1_ORGANISM_MODES),
        "causal_modes": list(GATE8_V1_CAUSAL_MODES),
        "causal_ablation_conditions": [
            list(row) for row in GATE8_V1_CAUSAL_ABLATION_CONDITIONS
        ],
        "condition_aggregation": GATE8_V1_CONDITION_AGGREGATION,
        "bootstrap_samples": GATE8_V1_BOOTSTRAP_SAMPLES,
        "bootstrap_confidence": GATE8_V1_BOOTSTRAP_CONFIDENCE,
        "bootstrap_namespace": GATE8_V1_BOOTSTRAP_NAMESPACE,
        "bootstrap_unit": GATE8_V1_BOOTSTRAP_UNIT,
        "reference_pooling": GATE8_V1_REFERENCE_POOLING,
        "reference_model_id": GATE8_V1_REFERENCE_MODEL_ID,
        "reference_revision": GATE8_V1_REFERENCE_REVISION,
        "reference_tokenizer_result_sha256": GATE8_V1_TOKENIZER_RESULT_SHA256,
        "reference_tokenizer_manifest_sha256": (
            GATE8_V1_TOKENIZER_MANIFEST_SHA256
        ),
        "reference_max_input_tokens": capability.GATE8_REFERENCE_MAX_INPUT_TOKENS,
        "reference_max_new_tokens": capability.GATE8_REFERENCE_MAX_NEW_TOKENS,
        "reference_demonstrations": capability.GATE8_REFERENCE_DEMONSTRATIONS,
        "required_per_world_fields": list(GATE8_V1_REQUIRED_PER_WORLD_FIELDS),
        "required_condition_metrics": list(GATE8_V1_REQUIRED_CONDITION_METRICS),
        "population_scaling_classifier": "original_gate8_classifier_unchanged",
        "reference_comparison_classifier": "original_gate8_classifier_unchanged",
        "test_answers_exposed": False,
    }
