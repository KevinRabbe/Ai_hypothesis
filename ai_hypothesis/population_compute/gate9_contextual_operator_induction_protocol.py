"""Data-frozen Gate-9 contextual operator-induction protocol.

Gate-9 asks whether fixed learned machinery can scale through population
computation when every worker must infer a previously unseen local operator
from public support examples. This module contains only immutable constants,
evidence schemas, and preregistered classifiers. It opens no operator
generator, world generator, model, optimizer, checkpoint, execution, or result
artifact surface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GATE9_VERSION = "gate9-contextual-affine-operator-induction-v0"
GATE9_SCIENTIFIC_STATUS = (
    "DATA_FROZEN_GATE9_CONTEXTUAL_OPERATOR_INDUCTION_PROTOCOL_EXECUTION_CLOSED"
)

GATE9_GATE8_FINAL_RESULT_HEAD = "a063be4bf04979b026370f87cdd0aa05712cdd05"
GATE9_GATE8_FINAL_EXECUTION_HEAD = "474b2590e5e138134bcb993e1d8114c473f0455b"
GATE9_GATE8_SUMMARY_SHA256 = (
    "a63f1c6c7cb7facdc71a48e5df05297cc823017ea342dc052310d36c97394462"
)
GATE9_GATE8_CONDITION_LEDGER_SHA256 = (
    "276969b304beee1edbeb3979c44a12db4b256b436e6d82c86ff92da7ce64f44d"
)
GATE9_GATE8_MANIFEST_SHA256 = (
    "bc7ff2ca604c914a2bb610d0089450454c47fbb36eb76954db28ea898c3ced59"
)
GATE9_GATE8_SCALING_OUTCOME = "G8_POSITIVE_CAPABILITY_SCALING"
GATE9_GATE8_REFERENCE_OUTCOME = "G8_POPULATION_EXCEEDS_1B_REFERENCE"

GATE9_LEARNED_PARAMETER_BUDGET = 19_649
GATE9_CHECKPOINT_SEEDS = (0, 1, 2)
GATE9_SYMBOL_BITS = 8
GATE9_SYMBOL_COUNT = 1 << GATE9_SYMBOL_BITS
GATE9_MESSAGE_BITS = 8
GATE9_SUPPORT_EXAMPLES = GATE9_SYMBOL_BITS + 1
GATE9_OPERATOR_FAMILY = "affine_bijections_x_to_LU_x_xor_b_over_GF2_8"
GATE9_OPERATOR_KEY_BITS = 64
GATE9_OPERATOR_FAMILY_SIZE = 1 << GATE9_OPERATOR_KEY_BITS
GATE9_OPERATOR_COUNTER_PERMUTATION = "splitmix64_bijection_v0"
GATE9_OPERATOR_ID_VISIBLE_TO_MODEL = False
GATE9_SUPPORT_INPUTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
GATE9_QUERY_EXCLUDES_SUPPORT_INPUTS = True

GATE9_TRAIN_OPERATOR_COUNTER_START = 0
GATE9_TRAIN_OPERATOR_COUNT = 262_144
GATE9_VALIDATION_OPERATOR_COUNTER_START = 1 << 32
GATE9_VALIDATION_OPERATOR_COUNT = 32_768
GATE9_LOCAL_TEST_OPERATOR_COUNTER_START = 1 << 40
GATE9_LOCAL_TEST_OPERATOR_COUNT = 4_096
GATE9_GRAPH_TEST_OPERATOR_COUNTER_START = 1 << 48

GATE9_POPULATIONS = (32, 64, 128, 256, 512, 1_024)
GATE9_DEPTHS = (4, 8, 16, 32, 64, 128)
GATE9_RELEVANT_EDGE_FRACTION_DENOMINATOR = 8
GATE9_WORLDS_PER_CONDITION = 256
GATE9_TEST_SPLIT = "test"
GATE9_TEST_SEED = 0
GATE9_TEST_WORLD_START = 0
GATE9_TEST_WORLD_END_INCLUSIVE = GATE9_WORLDS_PER_CONDITION - 1

GATE9_VALID_CONDITIONS = tuple(
    (population, depth)
    for population in GATE9_POPULATIONS
    for depth in GATE9_DEPTHS
    if depth * GATE9_RELEVANT_EDGE_FRACTION_DENOMINATOR <= population
)
GATE9_GRAPH_TEST_OPERATOR_COUNT = (
    sum(population for population, _ in GATE9_VALID_CONDITIONS)
    * GATE9_WORLDS_PER_CONDITION
)
GATE9_SCIENTIFIC_TEST_OPERATOR_COUNT = (
    GATE9_LOCAL_TEST_OPERATOR_COUNT + GATE9_GRAPH_TEST_OPERATOR_COUNT
)

GATE9_BOOTSTRAP_SAMPLES = 20_000
GATE9_BOOTSTRAP_CONFIDENCE = 0.95
GATE9_BOOTSTRAP_NAMESPACE = "gate9-contextual-operator-induction-bootstrap-v0"
GATE9_BOOTSTRAP_UNIT = (
    "world_index_shared_across_three_checkpoint_seeds_within_condition"
)

GATE9_LOCAL_ACCURACY_MIN = 0.995
GATE9_LOCAL_CI_LOW_MIN = 0.990
GATE9_LOCAL_CONTEXT_DELTA_MIN = 0.50
GATE9_FRONTIER_POINT_ACCURACY = 0.90
GATE9_FRONTIER_CI_LOW = 0.85
GATE9_CAUSAL_DELTA_MIN = 0.20
GATE9_CAUSAL_CONDITIONS = ((512, 64), (1_024, 128))

GATE9_POSITIVE = "G9_CONTEXTUAL_POPULATION_SCALING"
GATE9_PRESENT_NO_SCALING = "G9_CONTEXTUAL_CAPABILITY_PRESENT_NO_SCALING"
GATE9_NEGATIVE = "G9_CONTEXTUAL_CAPABILITY_NEGATIVE_SCALING"
GATE9_CONTEXT_NOT_CAUSAL = "G9_CONTEXT_NOT_CAUSAL"
GATE9_LOCAL_INDUCTION_FAILED = "G9_NOVEL_OPERATOR_INDUCTION_FAILED"
GATE9_INCONCLUSIVE = "G9_CONTEXTUAL_CAPABILITY_INCONCLUSIVE"
GATE9_OUTCOMES = (
    GATE9_POSITIVE,
    GATE9_PRESENT_NO_SCALING,
    GATE9_NEGATIVE,
    GATE9_CONTEXT_NOT_CAUSAL,
    GATE9_LOCAL_INDUCTION_FAILED,
    GATE9_INCONCLUSIVE,
)


def _probability(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be within 0..1")


def _ordered_condition_identity(rows: tuple[Any, ...], label: str) -> None:
    observed = tuple((row.population, row.depth) for row in rows)
    if observed != GATE9_VALID_CONDITIONS:
        raise ValueError(f"{label} must cover the exact 21-condition matrix")


@dataclass(frozen=True, slots=True)
class Gate9OperatorSplitEvidence:
    train_operators: int
    validation_operators: int
    local_test_operators: int
    graph_test_operators: int
    train_validation_intersection: int
    train_scientific_intersection: int
    validation_scientific_intersection: int
    local_graph_test_intersection: int
    injective_counter_to_operator_mapping_proven: bool
    operator_keys_exposed_to_model: bool

    def validate(self) -> None:
        expected = (
            GATE9_TRAIN_OPERATOR_COUNT,
            GATE9_VALIDATION_OPERATOR_COUNT,
            GATE9_LOCAL_TEST_OPERATOR_COUNT,
            GATE9_GRAPH_TEST_OPERATOR_COUNT,
        )
        observed = (
            self.train_operators,
            self.validation_operators,
            self.local_test_operators,
            self.graph_test_operators,
        )
        if observed != expected:
            raise ValueError("Gate9 operator split counts drifted")
        intersections = (
            self.train_validation_intersection,
            self.train_scientific_intersection,
            self.validation_scientific_intersection,
            self.local_graph_test_intersection,
        )
        if any(value != 0 for value in intersections):
            raise ValueError("Gate9 operator identities overlap across frozen splits")
        if not self.injective_counter_to_operator_mapping_proven:
            raise ValueError("Gate9 counter-to-operator injectivity was not proven")
        if self.operator_keys_exposed_to_model:
            raise ValueError("Gate9 model input exposed an operator key")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate9LocalInductionEvidence:
    checkpoint_seed: int
    accuracy: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    shuffled_context_accuracy: float
    query_only_accuracy: float
    full_minus_shuffled_context_ci_low: float
    full_minus_query_only_ci_low: float

    def validate(self) -> None:
        if self.checkpoint_seed not in GATE9_CHECKPOINT_SEEDS:
            raise ValueError("Gate9 local evidence seed is outside 0..2")
        for label, value in (
            ("accuracy", self.accuracy),
            ("bootstrap_ci_low", self.bootstrap_ci_low),
            ("bootstrap_ci_high", self.bootstrap_ci_high),
            ("shuffled_context_accuracy", self.shuffled_context_accuracy),
            ("query_only_accuracy", self.query_only_accuracy),
        ):
            _probability(value, f"Gate9 local {label}")
        if self.bootstrap_ci_low > self.bootstrap_ci_high:
            raise ValueError("Gate9 local confidence interval is reversed")

    def induction_passes(self) -> bool:
        self.validate()
        return (
            self.accuracy >= GATE9_LOCAL_ACCURACY_MIN
            and self.bootstrap_ci_low >= GATE9_LOCAL_CI_LOW_MIN
            and self.accuracy - self.shuffled_context_accuracy
            >= GATE9_LOCAL_CONTEXT_DELTA_MIN
            and self.accuracy - self.query_only_accuracy
            >= GATE9_LOCAL_CONTEXT_DELTA_MIN
            and self.full_minus_shuffled_context_ci_low
            > GATE9_LOCAL_CONTEXT_DELTA_MIN
            and self.full_minus_query_only_ci_low
            > GATE9_LOCAL_CONTEXT_DELTA_MIN
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate9ConditionEvidence:
    population: int
    depth: int
    accuracy: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    seed_accuracies: tuple[float, float, float]
    mean_active_workers: float
    mean_communicated_bits: float
    mean_context_examples_read: float
    mean_worker_updates: float

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE9_VALID_CONDITIONS:
            raise ValueError("Gate9 condition lies outside the frozen matrix")
        for label, value in (
            ("accuracy", self.accuracy),
            ("bootstrap_ci_low", self.bootstrap_ci_low),
            ("bootstrap_ci_high", self.bootstrap_ci_high),
            *tuple(
                (f"seed_{index}_accuracy", value)
                for index, value in enumerate(self.seed_accuracies)
            ),
        ):
            _probability(value, f"Gate9 condition {label}")
        if self.bootstrap_ci_low > self.bootstrap_ci_high:
            raise ValueError("Gate9 condition confidence interval is reversed")
        if abs(self.accuracy - sum(self.seed_accuracies) / 3.0) > 1e-12:
            raise ValueError("Gate9 condition accuracy is not equal-seed weighted")
        if not 0.0 <= self.mean_active_workers <= self.population:
            raise ValueError("Gate9 active-worker mean is invalid")
        if (
            self.mean_communicated_bits < 0.0
            or self.mean_context_examples_read < 0.0
            or self.mean_worker_updates < 0.0
        ):
            raise ValueError("Gate9 resource accounting cannot be negative")

    def solved(self) -> bool:
        self.validate()
        return (
            self.accuracy >= GATE9_FRONTIER_POINT_ACCURACY
            and self.bootstrap_ci_low >= GATE9_FRONTIER_CI_LOW
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate9FrontierEvidence:
    population: int
    max_solved_depth: int
    accuracy: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    solved: bool

    def validate(self) -> None:
        if self.population not in GATE9_POPULATIONS:
            raise ValueError("Gate9 frontier population is outside the ladder")
        if (self.population, self.max_solved_depth) not in GATE9_VALID_CONDITIONS:
            raise ValueError("Gate9 frontier depth is invalid for its population")
        _probability(self.accuracy, "Gate9 frontier accuracy")
        _probability(self.bootstrap_ci_low, "Gate9 frontier CI low")
        _probability(self.bootstrap_ci_high, "Gate9 frontier CI high")
        if self.bootstrap_ci_low > self.bootstrap_ci_high:
            raise ValueError("Gate9 frontier confidence interval is reversed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate9CausalEvidence:
    population: int
    depth: int
    full_accuracy: float
    no_communication_accuracy: float
    shuffled_context_accuracy: float
    query_only_accuracy: float
    full_minus_no_communication_ci_low: float
    full_minus_shuffled_context_ci_low: float
    full_minus_query_only_ci_low: float

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE9_CAUSAL_CONDITIONS:
            raise ValueError("Gate9 causal row is outside the frozen conditions")
        for label, value in (
            ("full_accuracy", self.full_accuracy),
            ("no_communication_accuracy", self.no_communication_accuracy),
            ("shuffled_context_accuracy", self.shuffled_context_accuracy),
            ("query_only_accuracy", self.query_only_accuracy),
        ):
            _probability(value, f"Gate9 causal {label}")

    def context_guard_passes(self) -> bool:
        self.validate()
        return (
            self.full_minus_shuffled_context_ci_low > GATE9_CAUSAL_DELTA_MIN
            and self.full_minus_query_only_ci_low > GATE9_CAUSAL_DELTA_MIN
        )

    def communication_guard_passes(self) -> bool:
        self.validate()
        return self.full_minus_no_communication_ci_low > GATE9_CAUSAL_DELTA_MIN

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def validate_gate9_local_rows(
    rows: tuple[Gate9LocalInductionEvidence, ...],
) -> None:
    if tuple(row.checkpoint_seed for row in rows) != GATE9_CHECKPOINT_SEEDS:
        raise ValueError("Gate9 local rows must be ordered seeds 0, 1, 2")
    for row in rows:
        row.validate()


def validate_gate9_causal_rows(rows: tuple[Gate9CausalEvidence, ...]) -> None:
    observed = tuple((row.population, row.depth) for row in rows)
    if observed != GATE9_CAUSAL_CONDITIONS:
        raise ValueError("Gate9 causal rows must cover the exact two conditions")
    for row in rows:
        row.validate()


def build_gate9_frontiers(
    rows: tuple[Gate9ConditionEvidence, ...],
) -> tuple[Gate9FrontierEvidence, ...]:
    _ordered_condition_identity(rows, "Gate9 condition rows")
    for row in rows:
        row.validate()
    frontiers: list[Gate9FrontierEvidence] = []
    for population in GATE9_POPULATIONS:
        candidates = tuple(row for row in rows if row.population == population)
        solved = tuple(row for row in candidates if row.solved())
        selected = solved[-1] if solved else candidates[0]
        frontier = Gate9FrontierEvidence(
            population=population,
            max_solved_depth=selected.depth,
            accuracy=selected.accuracy,
            bootstrap_ci_low=selected.bootstrap_ci_low,
            bootstrap_ci_high=selected.bootstrap_ci_high,
            solved=bool(solved),
        )
        frontier.validate()
        frontiers.append(frontier)
    return tuple(frontiers)


def classify_gate9(
    *,
    operator_splits: Gate9OperatorSplitEvidence,
    local: tuple[Gate9LocalInductionEvidence, ...],
    conditions: tuple[Gate9ConditionEvidence, ...],
    causal: tuple[Gate9CausalEvidence, ...],
) -> str:
    operator_splits.validate()
    validate_gate9_local_rows(local)
    validate_gate9_causal_rows(causal)
    frontiers = build_gate9_frontiers(conditions)

    if not all(row.induction_passes() for row in local):
        return GATE9_LOCAL_INDUCTION_FAILED

    if not all(row.context_guard_passes() for row in causal):
        return GATE9_CONTEXT_NOT_CAUSAL

    depths = tuple(row.max_solved_depth for row in frontiers)
    all_solved = all(row.solved for row in frontiers)
    nondecreasing = all(left <= right for left, right in zip(depths, depths[1:]))
    strict_increases = sum(left < right for left, right in zip(depths, depths[1:]))
    communication_causal = all(
        row.communication_guard_passes() for row in causal
    )

    if (
        all_solved
        and nondecreasing
        and strict_increases >= 3
        and depths[-1] >= 4 * depths[0]
        and communication_causal
    ):
        return GATE9_POSITIVE

    if any(left > right for left, right in zip(depths, depths[1:])):
        return GATE9_NEGATIVE

    if (
        all_solved
        and strict_increases < 2
        and depths[-1] <= 2 * depths[0]
        and communication_causal
    ):
        return GATE9_PRESENT_NO_SCALING

    return GATE9_INCONCLUSIVE


def gate9_protocol_plan() -> dict[str, Any]:
    return {
        "version": GATE9_VERSION,
        "scientific_status": GATE9_SCIENTIFIC_STATUS,
        "gate8_final_result_head": GATE9_GATE8_FINAL_RESULT_HEAD,
        "gate8_final_execution_head": GATE9_GATE8_FINAL_EXECUTION_HEAD,
        "gate8_summary_sha256": GATE9_GATE8_SUMMARY_SHA256,
        "gate8_condition_ledger_sha256": GATE9_GATE8_CONDITION_LEDGER_SHA256,
        "gate8_manifest_sha256": GATE9_GATE8_MANIFEST_SHA256,
        "gate8_scaling_outcome": GATE9_GATE8_SCALING_OUTCOME,
        "gate8_reference_outcome": GATE9_GATE8_REFERENCE_OUTCOME,
        "scientific_question": (
            "does fixed learned machinery retain population capability scaling "
            "when each worker must infer a previously unseen local operator "
            "from public support examples"
        ),
        "learned_parameter_budget": GATE9_LEARNED_PARAMETER_BUDGET,
        "checkpoint_seeds": list(GATE9_CHECKPOINT_SEEDS),
        "symbol_bits": GATE9_SYMBOL_BITS,
        "symbol_count": GATE9_SYMBOL_COUNT,
        "message_bits": GATE9_MESSAGE_BITS,
        "support_examples": GATE9_SUPPORT_EXAMPLES,
        "support_inputs": list(GATE9_SUPPORT_INPUTS),
        "query_excludes_support_inputs": GATE9_QUERY_EXCLUDES_SUPPORT_INPUTS,
        "operator_family": GATE9_OPERATOR_FAMILY,
        "operator_key_bits": GATE9_OPERATOR_KEY_BITS,
        "operator_family_size": GATE9_OPERATOR_FAMILY_SIZE,
        "operator_counter_permutation": GATE9_OPERATOR_COUNTER_PERMUTATION,
        "operator_id_visible_to_model": GATE9_OPERATOR_ID_VISIBLE_TO_MODEL,
        "operator_counts": {
            "train": GATE9_TRAIN_OPERATOR_COUNT,
            "validation": GATE9_VALIDATION_OPERATOR_COUNT,
            "local_test": GATE9_LOCAL_TEST_OPERATOR_COUNT,
            "graph_test": GATE9_GRAPH_TEST_OPERATOR_COUNT,
        },
        "conditions": [list(row) for row in GATE9_VALID_CONDITIONS],
        "condition_count": len(GATE9_VALID_CONDITIONS),
        "worlds_per_condition": GATE9_WORLDS_PER_CONDITION,
        "test_split": GATE9_TEST_SPLIT,
        "test_seed": GATE9_TEST_SEED,
        "test_world_start": GATE9_TEST_WORLD_START,
        "test_world_end_inclusive": GATE9_TEST_WORLD_END_INCLUSIVE,
        "bootstrap_samples": GATE9_BOOTSTRAP_SAMPLES,
        "bootstrap_confidence": GATE9_BOOTSTRAP_CONFIDENCE,
        "bootstrap_namespace": GATE9_BOOTSTRAP_NAMESPACE,
        "bootstrap_unit": GATE9_BOOTSTRAP_UNIT,
        "causal_conditions": [list(row) for row in GATE9_CAUSAL_CONDITIONS],
        "outcomes": list(GATE9_OUTCOMES),
        "execution_admitted": False,
        "operator_generation_admitted": False,
        "world_generation_admitted": False,
        "architecture_admitted": False,
        "training_admitted": False,
        "checkpoint_loading_admitted": False,
        "scientific_test_admitted": False,
        "reference_inference_admitted": False,
        "result_classification_admitted": False,
    }
