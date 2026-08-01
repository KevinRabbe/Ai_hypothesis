"""Frozen, development-only Gate-9 v0 failure-decomposition protocol.

No model, Torch, optimizer execution, checkpoint loading, scientific world, or
population runtime is admitted by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

GATE9D_PROTOCOL_VERSION = "gate9-contextual-failure-decomposition-protocol-v0"
GATE9D_PROTOCOL_STATUS = "G9D_PROTOCOL_FROZEN_EXECUTION_CLOSED"
GATE9D_FINAL_RESULT_HEAD = "33f2860795a1b70e5fbe20998f4fe8a2a6fc8452"
GATE9D_OPERATOR_CONTRACT_HEAD = "be6451e1af82b18749bd0313a9c02ca62c4eee5c"
GATE9D_ARCHITECTURE_HEAD = (
    "c689cc3f38" + "f6" + "f6f642916ee1a702d7de7bd0e43b"
)
GATE9D_TRAINING_PROTOCOL_HEAD = "1228c19cbf85da4ab738c3355c58f946cd6a965c"
GATE9D_LEARNED_PARAMETER_COUNT = 19_649

for _label, _head in (
    ("final result", GATE9D_FINAL_RESULT_HEAD),
    ("operator contract", GATE9D_OPERATOR_CONTRACT_HEAD),
    ("architecture", GATE9D_ARCHITECTURE_HEAD),
    ("training protocol", GATE9D_TRAINING_PROTOCOL_HEAD),
):
    if len(_head) != 40 or any(character not in "0123456789abcdef" for character in _head):
        raise RuntimeError(f"Gate9D {_label} head is not a full lowercase Git SHA")

GATE9D_SUPPORT_INPUTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
GATE9D_QUERY_VALUES = tuple(
    value for value in range(256) if value not in GATE9D_SUPPORT_INPUTS
)
GATE9D_QUERY_COUNT = 247
if len(GATE9D_QUERY_VALUES) != GATE9D_QUERY_COUNT:
    raise RuntimeError("Gate9D query-domain arithmetic drifted")

GATE9D_INITIALIZATION_SEEDS = (910_900, 910_901, 910_902)
GATE9D_EXACT_ACCURACY_MIN = 0.995
GATE9D_BIT_ACCURACY_MIN = 0.999
GATE9D_CONTEXT_DELTA_MIN_STRICT = 0.50
GATE9D_ORACLE_ACCURACY_REQUIRED = 1.0

GATE9D_OPTIMIZER = "AdamW"
GATE9D_BASE_LEARNING_RATE = 1.0e-3
GATE9D_MIN_LEARNING_RATE = 1.0e-4
GATE9D_WARMUP_STEPS = 16
GATE9D_ADAM_BETAS = (0.9, 0.95)
GATE9D_ADAM_EPSILON = 1.0e-8
GATE9D_WEIGHT_DECAY = 1.0e-4
GATE9D_GRADIENT_CLIP_NORM = 1.0
GATE9D_DIAGNOSTIC_COUNTER_BASE = 1 << 56


@dataclass(frozen=True, slots=True)
class CounterRange:
    name: str
    start: int
    count: int

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Gate9D counter range requires a name")
        if type(self.start) is not int or type(self.count) is not int:
            raise ValueError("Gate9D counter range values must be exact integers")
        if not 0 <= self.start < (1 << 64):
            raise ValueError("Gate9D counter range start lies outside uint64")
        if self.count <= 0 or self.start + self.count > (1 << 64):
            raise ValueError("Gate9D counter range count lies outside uint64")

    @property
    def stop(self) -> int:
        self.validate()
        return self.start + self.count

    def intersects(self, other: "CounterRange") -> bool:
        self.validate()
        other.validate()
        return max(self.start, other.start) < min(self.stop, other.stop)

    def counters(self) -> tuple[int, ...]:
        self.validate()
        return tuple(range(self.start, self.stop))


GATE9D_FROZEN_V0_RANGES = (
    CounterRange("v0-training", 0, 262_144),
    CounterRange("v0-validation", 1 << 32, 32_768),
    CounterRange("v0-local-science", 1 << 40, 4_096),
    CounterRange("v0-graph-science", 1 << 48, 2_629_632),
)
GATE9D_SINGLE_OPERATOR_RANGE = CounterRange(
    "single-operator-query-fit", GATE9D_DIAGNOSTIC_COUNTER_BASE, 1
)

# Exact affine keys with one identity linear map and opposite byte bias. Their
# inverse SplitMix64 counters are fixed and lie outside all Gate-9 v0 ranges.
GATE9D_COLLISION_OPERATOR_KEYS = (0x0000000000000000, 0xFF00000000000000)
GATE9D_COLLISION_OPERATOR_COUNTERS = (
    0x61C8864680B583EB,
    0xE8CF068191D03BBC,
)
GATE9D_HELD_IN_OPERATOR_RANGE = CounterRange(
    "held-in-multi-operator-fit", GATE9D_DIAGNOSTIC_COUNTER_BASE + 0x100, 16
)
GATE9D_UNSEEN_TRAIN_OPERATOR_RANGE = CounterRange(
    "unseen-generalization-train",
    GATE9D_DIAGNOSTIC_COUNTER_BASE + 0x1000,
    256,
)
GATE9D_UNSEEN_EVAL_OPERATOR_RANGE = CounterRange(
    "unseen-generalization-eval",
    GATE9D_DIAGNOSTIC_COUNTER_BASE + 0x2000,
    64,
)


@dataclass(frozen=True, slots=True)
class DiagnosticStage:
    order: int
    name: str
    purpose: str
    train_operator_counters: tuple[int, ...]
    evaluation_operator_counters: tuple[int, ...]
    steps: int
    batch_size: int
    requires_context_causality: bool
    unseen_operator_evaluation: bool
    shuffle_namespace: str

    def validate(self) -> None:
        if type(self.order) is not int or self.order < 1 or not self.name or not self.purpose:
            raise ValueError("Gate9D stage identity is invalid")
        for label, counters in (
            ("training", self.train_operator_counters),
            ("evaluation", self.evaluation_operator_counters),
        ):
            if not counters or len(counters) != len(set(counters)):
                raise ValueError(f"Gate9D {label} operator identities are invalid")
            if any(type(counter) is not int or not 0 <= counter < (1 << 64) for counter in counters):
                raise ValueError(f"Gate9D {label} counter lies outside uint64")
        if type(self.steps) is not int or type(self.batch_size) is not int:
            raise ValueError("Gate9D stage schedule must use exact integers")
        if self.steps <= 0 or self.batch_size <= 0 or not self.shuffle_namespace:
            raise ValueError("Gate9D stage schedule is invalid")
        train = set(self.train_operator_counters)
        evaluation = set(self.evaluation_operator_counters)
        if self.unseen_operator_evaluation:
            if train & evaluation:
                raise ValueError("Gate9D unseen evaluation overlaps training")
        elif train != evaluation:
            raise ValueError("Gate9D held-in evaluation must reuse training operators")

    @property
    def train_examples(self) -> int:
        self.validate()
        return len(self.train_operator_counters) * GATE9D_QUERY_COUNT

    @property
    def evaluation_examples(self) -> int:
        self.validate()
        return len(self.evaluation_operator_counters) * GATE9D_QUERY_COUNT


GATE9D_STAGES = (
    DiagnosticStage(
        1,
        "single_operator_query_fit",
        "Fit one complete non-support byte mapping with constant context.",
        GATE9D_SINGLE_OPERATOR_RANGE.counters(),
        GATE9D_SINGLE_OPERATOR_RANGE.counters(),
        1_024,
        247,
        False,
        False,
        "gate9d-single-operator-query-fit-v0",
    ),
    DiagnosticStage(
        2,
        "paired_operator_context_collision",
        "Force support context to select opposite answers for every query.",
        GATE9D_COLLISION_OPERATOR_COUNTERS,
        GATE9D_COLLISION_OPERATOR_COUNTERS,
        2_048,
        494,
        True,
        False,
        "gate9d-paired-context-collision-v0",
    ),
    DiagnosticStage(
        3,
        "held_in_multi_operator_fit",
        "Fit and condition on sixteen held-in operators.",
        GATE9D_HELD_IN_OPERATOR_RANGE.counters(),
        GATE9D_HELD_IN_OPERATOR_RANGE.counters(),
        4_096,
        512,
        True,
        False,
        "gate9d-held-in-multi-operator-fit-v0",
    ),
    DiagnosticStage(
        4,
        "unseen_operator_generalization",
        "Generalize from 256 diagnostic operators to 64 disjoint operators.",
        GATE9D_UNSEEN_TRAIN_OPERATOR_RANGE.counters(),
        GATE9D_UNSEEN_EVAL_OPERATOR_RANGE.counters(),
        8_192,
        512,
        True,
        True,
        "gate9d-unseen-operator-generalization-v0",
    ),
)

GATE9D_STAGE_FAILURE_OUTCOMES = {
    "single_operator_query_fit": "G9D_BASIC_QUERY_MAPPING_FAILED",
    "paired_operator_context_collision": "G9D_CONTEXTUAL_CONTROL_FAILED",
    "held_in_multi_operator_fit": "G9D_HELD_IN_OPERATOR_FIT_FAILED",
    "unseen_operator_generalization": "G9D_UNSEEN_OPERATOR_GENERALIZATION_FAILED",
}
GATE9D_MIXED_OUTCOME = "G9D_DIAGNOSTIC_INCONCLUSIVE"
GATE9D_INCOMPLETE_OUTCOME = "G9D_DIAGNOSTIC_INCOMPLETE"
GATE9D_ALL_PASS_OUTCOME = "G9D_V0_FAILURE_NOT_LOCALIZED"


@dataclass(frozen=True, slots=True)
class SeedStageEvidence:
    exact_accuracy: float
    bit_accuracy: float
    full_minus_shuffled: float
    full_minus_query_only: float
    oracle_accuracy: float

    def validate(self) -> None:
        for label, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Gate9D {label} must be numeric")
            lower = -1.0 if label.startswith("full_minus_") else 0.0
            if not lower <= float(value) <= 1.0:
                raise ValueError(f"Gate9D {label} lies outside its valid range")

    def passes(self, stage: DiagnosticStage) -> bool:
        self.validate()
        stage.validate()
        if not (
            self.exact_accuracy >= GATE9D_EXACT_ACCURACY_MIN
            and self.bit_accuracy >= GATE9D_BIT_ACCURACY_MIN
            and self.oracle_accuracy == GATE9D_ORACLE_ACCURACY_REQUIRED
        ):
            return False
        return not stage.requires_context_causality or (
            self.full_minus_shuffled > GATE9D_CONTEXT_DELTA_MIN_STRICT
            and self.full_minus_query_only > GATE9D_CONTEXT_DELTA_MIN_STRICT
        )


def classify_diagnostic(stage_seed_passes: Mapping[str, Sequence[bool]]) -> str:
    known = {stage.name for stage in GATE9D_STAGES}
    extra = set(stage_seed_passes) - known
    if extra:
        raise ValueError(f"unknown Gate9D stage results: {sorted(extra)!r}")
    for stage in GATE9D_STAGES:
        values = stage_seed_passes.get(stage.name)
        if values is None:
            return GATE9D_INCOMPLETE_OUTCOME
        passes = tuple(values)
        if len(passes) != 3 or any(type(value) is not bool for value in passes):
            raise ValueError("Gate9D stage requires three exact Boolean results")
        if all(passes):
            continue
        if not any(passes):
            return GATE9D_STAGE_FAILURE_OUTCOMES[stage.name]
        return GATE9D_MIXED_OUTCOME
    return GATE9D_ALL_PASS_OUTCOME


def validate_protocol() -> None:
    if tuple(stage.order for stage in GATE9D_STAGES) != (1, 2, 3, 4):
        raise RuntimeError("Gate9D stage order drifted")
    if len({stage.name for stage in GATE9D_STAGES}) != 4:
        raise RuntimeError("Gate9D stage names repeat")
    for stage in GATE9D_STAGES:
        stage.validate()

    groups = (
        set(GATE9D_SINGLE_OPERATOR_RANGE.counters()),
        set(GATE9D_COLLISION_OPERATOR_COUNTERS),
        set(GATE9D_HELD_IN_OPERATOR_RANGE.counters()),
        set(GATE9D_UNSEEN_TRAIN_OPERATOR_RANGE.counters()),
        set(GATE9D_UNSEEN_EVAL_OPERATOR_RANGE.counters()),
    )
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            if left & right:
                raise RuntimeError("Gate9D diagnostic operator identities overlap")
    for frozen in GATE9D_FROZEN_V0_RANGES:
        frozen.validate()
        for group in groups:
            if any(frozen.start <= counter < frozen.stop for counter in group):
                raise RuntimeError("Gate9D operator identity overlaps Gate9 v0")


validate_protocol()


def gate9d_protocol_plan() -> dict[str, Any]:
    return {
        "version": GATE9D_PROTOCOL_VERSION,
        "status": GATE9D_PROTOCOL_STATUS,
        "source_bindings": {
            "gate9_final_result_head": GATE9D_FINAL_RESULT_HEAD,
            "operator_contract_head": GATE9D_OPERATOR_CONTRACT_HEAD,
            "architecture_head": GATE9D_ARCHITECTURE_HEAD,
            "training_protocol_head": GATE9D_TRAINING_PROTOCOL_HEAD,
        },
        "learned_parameter_count": GATE9D_LEARNED_PARAMETER_COUNT,
        "initialization_seeds": list(GATE9D_INITIALIZATION_SEEDS),
        "query_count": GATE9D_QUERY_COUNT,
        "thresholds": {
            "exact_accuracy_min": GATE9D_EXACT_ACCURACY_MIN,
            "bit_accuracy_min": GATE9D_BIT_ACCURACY_MIN,
            "context_delta_min_strict": GATE9D_CONTEXT_DELTA_MIN_STRICT,
            "oracle_accuracy_required": GATE9D_ORACLE_ACCURACY_REQUIRED,
        },
        "optimizer": {
            "name": GATE9D_OPTIMIZER,
            "base_learning_rate": GATE9D_BASE_LEARNING_RATE,
            "min_learning_rate": GATE9D_MIN_LEARNING_RATE,
            "warmup_steps": GATE9D_WARMUP_STEPS,
            "betas": list(GATE9D_ADAM_BETAS),
            "epsilon": GATE9D_ADAM_EPSILON,
            "weight_decay": GATE9D_WEIGHT_DECAY,
            "gradient_clip_norm": GATE9D_GRADIENT_CLIP_NORM,
            "schedule": "linear_warmup_then_cosine",
        },
        "stages": [
            {
                **asdict(stage),
                "train_operator_count": len(stage.train_operator_counters),
                "evaluation_operator_count": len(stage.evaluation_operator_counters),
                "train_examples": stage.train_examples,
                "evaluation_examples": stage.evaluation_examples,
            }
            for stage in GATE9D_STAGES
        ],
        "decision_rule": {
            "advance_only_when": "all_three_initialization_seeds_pass",
            "all_three_fail": "return_stage_specific_failure",
            "mixed_seed_result": GATE9D_MIXED_OUTCOME,
            "missing_stage": GATE9D_INCOMPLETE_OUTCOME,
            "all_stages_pass": GATE9D_ALL_PASS_OUTCOME,
            "stop_after_first_nonpassing_stage": True,
        },
        "determinism": {
            "float32": True,
            "deterministic_algorithms": True,
            "amp": False,
            "tf32": False,
            "compile": False,
            "fixed_final_checkpoint_only": True,
            "early_stopping": False,
            "checkpoint_selection": False,
        },
        "boundaries": {
            "protocol_only": True,
            "torch_imported": False,
            "execution_admitted": False,
            "checkpoint_loading_admitted": False,
            "gate9_v0_result_mutation_admitted": False,
            "local_scientific_test_generation_admitted": False,
            "graph_scientific_test_generation_admitted": False,
            "population_execution_admitted": False,
        },
    }
