"""Data-frozen Gate-8 distributed-transformation capability protocol.

This module contains only immutable benchmark constants and preregistered
classifiers. It opens no generator, training, model, baseline, or execution path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GATE8_VERSION = "gate8-distributed-transformation-capability-v0"
GATE8_SCIENTIFIC_STATUS = (
    "DATA_FROZEN_GATE8_DISTRIBUTED_TRANSFORMATION_CAPABILITY_PROTOCOL_EXECUTION_CLOSED"
)

GATE8_BASE_RESULT_HEAD = "7cd29aa02d8ad0d4819978cd04f4a39b94a9bb0c"
GATE8_GATE7_EXECUTION_HEAD = "bdc5f2c03cbc77a79a419a16460387ac0d226a27"
GATE8_GATE7_RESULT_SHA256 = (
    "89f0a69d530355031f02666e403977f4ad3b622bc6468627c4cbfa9a7d1ea489"
)
GATE8_GATE7_AUDIT_SHA256 = (
    "857256f4ae71f0fbf6744a531ece0120fb9bb3088e36e30471e3614ffd602a79"
)
GATE8_GATE7_MANIFEST_SHA256 = (
    "2e212ab56dd251ab527f8aeb95e40f0e06bcacad9b7e1900f49a1b8d96efffa3"
)
GATE8_GATE7_OUTCOME = "G7_PRECISION_INFORMATION_CEILING_DOMINANT"

GATE8_LEARNED_PARAMETER_COUNT = 19_649
GATE8_SYMBOL_ALPHABET_SIZE = 16
GATE8_PRIMITIVE_TRANSFORM_COUNT = 8
GATE8_MESSAGE_CODEBOOK_SIZE = 256
GATE8_MESSAGE_BITS = 8
GATE8_MESSAGES_PER_ACTIVE_WORKER_PER_ROUND = 1
GATE8_RELEVANT_EDGE_FRACTION_NUMERATOR = 1
GATE8_RELEVANT_EDGE_FRACTION_DENOMINATOR = 8

GATE8_POPULATIONS = (32, 64, 128, 256, 512, 1_024)
GATE8_DEPTHS = (4, 8, 16, 32, 64, 128)
GATE8_TEST_WORLDS_PER_CONDITION = 512
GATE8_TRAINING_SEEDS = (0, 1, 2)
GATE8_TRAINING_WORLDS_PER_SEED = 262_144
GATE8_BOOTSTRAP_SAMPLES = 20_000
GATE8_FRONTIER_POINT_ACCURACY = 0.90
GATE8_FRONTIER_CI_LOW = 0.85
GATE8_ABLATION_MIN_DELTA = 0.20
GATE8_REFERENCE_NONINFERIORITY_MARGIN = 0.05

GATE8_TRAIN_POPULATIONS = (32, 64, 128)
GATE8_TRAIN_DEPTHS = (4, 8, 16)
GATE8_EXTRAPOLATION_POPULATIONS = (256, 512, 1_024)
GATE8_EXTRAPOLATION_DEPTHS = (32, 64, 128)
GATE8_CAUSAL_ABLATION_CONDITIONS = ((512, 64), (1_024, 128))

GATE8_WORLD_NAMESPACE = "gate8-distributed-transformation-world-v0"
GATE8_TRANSFORM_NAMESPACE = "gate8-distributed-transformation-library-v0"
GATE8_NODE_LABEL_NAMESPACE = "gate8-distributed-transformation-node-label-v0"
GATE8_TRAIN_NAMESPACE = "gate8-distributed-transformation-train-v0"
GATE8_TEST_NAMESPACE = "gate8-distributed-transformation-test-v0"
GATE8_BOOTSTRAP_NAMESPACE = "gate8-distributed-transformation-bootstrap-v0"
GATE8_BASELINE_DEMONSTRATION_NAMESPACE = (
    "gate8-distributed-transformation-gemma3-1b-demonstrations-v0"
)

GATE8_REFERENCE_MODEL_ID = "google/gemma-3-1b-it"
GATE8_REFERENCE_PARAMETER_CLASS = "1.0B"
GATE8_REFERENCE_CONTEXT_TOKENS = 32_768
GATE8_REFERENCE_MAX_INPUT_TOKENS = 24_576
GATE8_REFERENCE_MAX_NEW_TOKENS = 64
GATE8_REFERENCE_DTYPE = "bfloat16"
GATE8_REFERENCE_DECODING = "greedy_temperature_0"
GATE8_REFERENCE_DEMONSTRATIONS = 8
GATE8_REFERENCE_TASK_SPECIFIC_WEIGHT_UPDATES = False
GATE8_REFERENCE_EXACT_REVISION_REQUIRED_BEFORE_EXECUTION = True
GATE8_REFERENCE_ROLE = "pretrained_1B_reference_not_matched_training_baseline"

GATE8_SCALING_POSITIVE = "G8_POSITIVE_CAPABILITY_SCALING"
GATE8_SCALING_FLAT = "G8_CAPABILITY_PRESENT_NO_SCALING"
GATE8_SCALING_NEGATIVE = "G8_NEGATIVE_CAPABILITY_SCALING"
GATE8_SCALING_INCONCLUSIVE = "G8_CAPABILITY_SCALING_INCONCLUSIVE"
GATE8_SCALING_OUTCOMES = (
    GATE8_SCALING_POSITIVE,
    GATE8_SCALING_FLAT,
    GATE8_SCALING_NEGATIVE,
    GATE8_SCALING_INCONCLUSIVE,
)

GATE8_REFERENCE_EXCEEDS = "G8_POPULATION_EXCEEDS_1B_REFERENCE"
GATE8_REFERENCE_NONINFERIOR = "G8_POPULATION_NONINFERIOR_TO_1B_REFERENCE"
GATE8_REFERENCE_SUPERIOR = "G8_1B_REFERENCE_SUPERIOR"
GATE8_REFERENCE_MIXED = "G8_1B_REFERENCE_MIXED"
GATE8_REFERENCE_INCONCLUSIVE = "G8_1B_REFERENCE_COMPARISON_INCONCLUSIVE"
GATE8_REFERENCE_OUTCOMES = (
    GATE8_REFERENCE_EXCEEDS,
    GATE8_REFERENCE_NONINFERIOR,
    GATE8_REFERENCE_SUPERIOR,
    GATE8_REFERENCE_MIXED,
    GATE8_REFERENCE_INCONCLUSIVE,
)


def gate8_valid_conditions() -> tuple[tuple[int, int], ...]:
    return tuple(
        (population, depth)
        for population in GATE8_POPULATIONS
        for depth in GATE8_DEPTHS
        if depth * GATE8_RELEVANT_EDGE_FRACTION_DENOMINATOR <= population
    )


GATE8_VALID_CONDITIONS = gate8_valid_conditions()


@dataclass(frozen=True, slots=True)
class Gate8CapabilityFrontierRow:
    population: int
    max_solved_depth: int
    frontier_accuracy: float
    frontier_ci_low: float
    frontier_ci_high: float
    active_workers: int
    communicated_bits: int
    recurrent_updates: int

    def validate(self) -> None:
        if self.population not in GATE8_POPULATIONS:
            raise ValueError("Gate8 frontier population is outside the frozen ladder")
        if self.max_solved_depth not in GATE8_DEPTHS:
            raise ValueError("Gate8 frontier depth is outside the frozen ladder")
        if (self.population, self.max_solved_depth) not in GATE8_VALID_CONDITIONS:
            raise ValueError("Gate8 frontier depth is invalid for its population")
        if not 0.0 <= self.frontier_accuracy <= 1.0:
            raise ValueError("Gate8 frontier accuracy is outside 0..1")
        if not 0.0 <= self.frontier_ci_low <= self.frontier_ci_high <= 1.0:
            raise ValueError("Gate8 frontier interval is invalid")
        if not 0 <= self.active_workers <= self.population:
            raise ValueError("Gate8 active-worker count is invalid")
        if self.communicated_bits < 0 or self.recurrent_updates < 0:
            raise ValueError("Gate8 runtime accounting cannot be negative")

    def solved(self) -> bool:
        self.validate()
        return (
            self.frontier_accuracy >= GATE8_FRONTIER_POINT_ACCURACY
            and self.frontier_ci_low >= GATE8_FRONTIER_CI_LOW
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate8AblationRow:
    population: int
    depth: int
    full_accuracy: float
    no_communication_accuracy: float
    shuffled_worker_accuracy: float
    full_minus_no_communication_ci_low: float
    full_minus_shuffled_worker_ci_low: float

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_CAUSAL_ABLATION_CONDITIONS:
            raise ValueError("Gate8 ablation row is outside the frozen conditions")
        for value in (
            self.full_accuracy,
            self.no_communication_accuracy,
            self.shuffled_worker_accuracy,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Gate8 ablation accuracy is outside 0..1")

    def causal_guard_passes(self) -> bool:
        self.validate()
        return (
            self.full_minus_no_communication_ci_low > GATE8_ABLATION_MIN_DELTA
            and self.full_minus_shuffled_worker_ci_low > GATE8_ABLATION_MIN_DELTA
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate8ReferenceConditionRow:
    population: int
    depth: int
    population_accuracy: float
    reference_accuracy: float
    population_minus_reference_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    reference_input_tokens: int

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_VALID_CONDITIONS:
            raise ValueError("Gate8 reference row is outside the frozen condition matrix")
        for value in (self.population_accuracy, self.reference_accuracy):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Gate8 reference accuracy is outside 0..1")
        if self.bootstrap_ci_low > self.bootstrap_ci_high:
            raise ValueError("Gate8 reference interval is reversed")
        if not 0 < self.reference_input_tokens <= GATE8_REFERENCE_MAX_INPUT_TOKENS:
            raise ValueError("Gate8 reference prompt exceeds the frozen input budget")

    def clear_population_win(self) -> bool:
        self.validate()
        return self.bootstrap_ci_low > 0.0

    def clear_reference_win(self) -> bool:
        self.validate()
        return self.bootstrap_ci_high < -GATE8_REFERENCE_NONINFERIORITY_MARGIN

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate8ReferencePooledRow:
    population_minus_reference_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def validate(self) -> None:
        if self.bootstrap_ci_low > self.bootstrap_ci_high:
            raise ValueError("Gate8 pooled reference interval is reversed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _validate_frontier_rows(
    rows: tuple[Gate8CapabilityFrontierRow, ...],
) -> None:
    observed = tuple(row.population for row in rows)
    if observed != GATE8_POPULATIONS:
        raise ValueError("Gate8 frontier rows must cover the exact population ladder")
    for row in rows:
        row.validate()


def _validate_ablation_rows(rows: tuple[Gate8AblationRow, ...]) -> None:
    observed = tuple((row.population, row.depth) for row in rows)
    if observed != GATE8_CAUSAL_ABLATION_CONDITIONS:
        raise ValueError("Gate8 ablations must cover the exact causal conditions")
    for row in rows:
        row.validate()


def classify_gate8_population_scaling(
    *,
    frontiers: tuple[Gate8CapabilityFrontierRow, ...],
    ablations: tuple[Gate8AblationRow, ...],
) -> str:
    """Apply the frozen capability-scaling classifier."""

    _validate_frontier_rows(frontiers)
    _validate_ablation_rows(ablations)

    depths = tuple(row.max_solved_depth for row in frontiers)
    solved = tuple(row.solved() for row in frontiers)
    nondecreasing = all(left <= right for left, right in zip(depths, depths[1:]))
    strict_increases = sum(left < right for left, right in zip(depths, depths[1:]))
    causal = all(row.causal_guard_passes() for row in ablations)

    if (
        all(solved)
        and nondecreasing
        and strict_increases >= 3
        and depths[-1] >= 4 * depths[0]
        and causal
    ):
        return GATE8_SCALING_POSITIVE

    if any(left > right for left, right in zip(depths, depths[1:])) or depths[-1] < depths[0]:
        return GATE8_SCALING_NEGATIVE

    if all(solved) and strict_increases < 2 and depths[-1] <= 2 * depths[0]:
        return GATE8_SCALING_FLAT

    return GATE8_SCALING_INCONCLUSIVE


def classify_gate8_reference_comparison(
    *,
    conditions: tuple[Gate8ReferenceConditionRow, ...],
    pooled: Gate8ReferencePooledRow,
) -> str:
    """Apply the frozen population-versus-1B reference classifier."""

    observed = tuple((row.population, row.depth) for row in conditions)
    if observed != GATE8_VALID_CONDITIONS:
        raise ValueError("Gate8 reference rows must cover the exact condition matrix")
    for row in conditions:
        row.validate()
    pooled.validate()

    any_population_win = any(row.clear_population_win() for row in conditions)
    any_reference_win = any(row.clear_reference_win() for row in conditions)

    if pooled.bootstrap_ci_low > 0.0 and not any_reference_win:
        return GATE8_REFERENCE_EXCEEDS

    if (
        pooled.bootstrap_ci_low > -GATE8_REFERENCE_NONINFERIORITY_MARGIN
        and not any_reference_win
    ):
        return GATE8_REFERENCE_NONINFERIOR

    if pooled.bootstrap_ci_high < -GATE8_REFERENCE_NONINFERIORITY_MARGIN:
        return GATE8_REFERENCE_SUPERIOR

    if any_population_win and any_reference_win:
        return GATE8_REFERENCE_MIXED

    return GATE8_REFERENCE_INCONCLUSIVE


def gate8_protocol_plan() -> dict[str, Any]:
    """Return the immutable protocol plan without opening execution."""

    return {
        "version": GATE8_VERSION,
        "scientific_status": GATE8_SCIENTIFIC_STATUS,
        "execution_admitted": False,
        "generator_admitted": False,
        "training_admitted": False,
        "baseline_execution_admitted": False,
        "gate7_outcome": GATE8_GATE7_OUTCOME,
        "learned_parameter_count": GATE8_LEARNED_PARAMETER_COUNT,
        "symbol_alphabet_size": GATE8_SYMBOL_ALPHABET_SIZE,
        "primitive_transform_count": GATE8_PRIMITIVE_TRANSFORM_COUNT,
        "message_codebook_size": GATE8_MESSAGE_CODEBOOK_SIZE,
        "message_bits": GATE8_MESSAGE_BITS,
        "messages_per_active_worker_per_round": (
            GATE8_MESSAGES_PER_ACTIVE_WORKER_PER_ROUND
        ),
        "relevant_edge_fraction": (
            GATE8_RELEVANT_EDGE_FRACTION_NUMERATOR
            / GATE8_RELEVANT_EDGE_FRACTION_DENOMINATOR
        ),
        "populations": list(GATE8_POPULATIONS),
        "depths": list(GATE8_DEPTHS),
        "valid_conditions": [list(row) for row in GATE8_VALID_CONDITIONS],
        "test_worlds_per_condition": GATE8_TEST_WORLDS_PER_CONDITION,
        "training_seeds": list(GATE8_TRAINING_SEEDS),
        "training_worlds_per_seed": GATE8_TRAINING_WORLDS_PER_SEED,
        "bootstrap_samples": GATE8_BOOTSTRAP_SAMPLES,
        "training_populations": list(GATE8_TRAIN_POPULATIONS),
        "training_depths": list(GATE8_TRAIN_DEPTHS),
        "extrapolation_populations": list(GATE8_EXTRAPOLATION_POPULATIONS),
        "extrapolation_depths": list(GATE8_EXTRAPOLATION_DEPTHS),
        "causal_ablation_conditions": [
            list(row) for row in GATE8_CAUSAL_ABLATION_CONDITIONS
        ],
        "task": "ordered_noncommuting_transform_composition_on_a_rooted_graph",
        "worker_observation": "one_edge_shard_plus_public_query",
        "correctness_oracle": "exact_symbolic_path_composition",
        "brute_force_candidate_search": False,
        "reference_model_id": GATE8_REFERENCE_MODEL_ID,
        "reference_parameter_class": GATE8_REFERENCE_PARAMETER_CLASS,
        "reference_context_tokens": GATE8_REFERENCE_CONTEXT_TOKENS,
        "reference_max_input_tokens": GATE8_REFERENCE_MAX_INPUT_TOKENS,
        "reference_max_new_tokens": GATE8_REFERENCE_MAX_NEW_TOKENS,
        "reference_dtype": GATE8_REFERENCE_DTYPE,
        "reference_decoding": GATE8_REFERENCE_DECODING,
        "reference_demonstrations": GATE8_REFERENCE_DEMONSTRATIONS,
        "reference_task_specific_weight_updates": (
            GATE8_REFERENCE_TASK_SPECIFIC_WEIGHT_UPDATES
        ),
        "reference_exact_revision_required_before_execution": (
            GATE8_REFERENCE_EXACT_REVISION_REQUIRED_BEFORE_EXECUTION
        ),
        "reference_role": GATE8_REFERENCE_ROLE,
        "scaling_outcomes": list(GATE8_SCALING_OUTCOMES),
        "reference_outcomes": list(GATE8_REFERENCE_OUTCOMES),
    }
