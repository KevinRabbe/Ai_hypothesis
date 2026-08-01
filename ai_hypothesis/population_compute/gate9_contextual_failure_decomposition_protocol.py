"""Pre-execution Gate-9 v0 failure-decomposition protocol.

This standard-library module freezes a development-only diagnostic ladder after
the immutable three-seed result ``G9_NOVEL_OPERATOR_INDUCTION_FAILED``.  It
contains no model import, optimizer execution, checkpoint load, operator
materialization, scientific world generation, or population runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

GATE9D_PROTOCOL_VERSION = "gate9-contextual-failure-decomposition-protocol-v0"
GATE9D_PROTOCOL_STATUS = "G9D_PROTOCOL_FROZEN_EXECUTION_CLOSED"
GATE9D_FINAL_RESULT_HEAD = "33f2860795a1b70e5fbe20998f4fe8a2a6fc8452"
GATE9D_OPERATOR_CONTRACT_HEAD = "be6451e1af82b18749bd0313a9c02ca62c4eee5c"
GATE9D_ARCHITECTURE_HEAD = "c689cc3f38f6f642916ee1a702d7de7bd0e43b"
GATE9D_TRAINING_PROTOCOL_HEAD = "1228c19cbf85da4ab738c3355c58f946cd6a965c"
GATE9D_LEARNED_PARAMETER_COUNT = 19_649

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
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise ValueError("Gate9D counter range start must be an integer")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise ValueError("Gate9D counter range count must be an integer")
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
    "single-operator-query-fit",
    GATE9D_DIAGNOSTIC_COUNTER_BASE,
    1,
)

# These two explicit keys have the same identity linear map and opposite bias.
# Their answers differ for every byte, which makes query-only prediction unable
# to satisfy both operators simultaneously.  The counters are the exact inverse
# SplitMix64 identities and lie outside every frozen Gate-9 v0 range.
GATE9D_COLLISION_OPERATOR_KEYS = (0x0000000000000000, 0xFF00000000000000)
GATE9D_COLLISION_OPERATOR_COUNTERS = (
    0x61C8864680B583EB,
    0xE8CF068191D03BBC,
)

GATE9D_HELD_IN_OPERATOR_RANGE = CounterRange(
    "held-in-multi-operator-fit",
    GATE9D_DIAGNOSTIC_COUNTER_BASE + 0x100,
    16,
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
        if self.order < 1 or not self.name or not self.purpose:
            raise ValueError("Gate9D stage identity is invalid")
        if not self.train_operator_counters or not self.evaluation_operator_counters:
            raise ValueError("Gate9D stage requires train and evaluation operators")
        for label, counters in (
            ("training", self.train_operator_counters),
            ("evaluation", self.evaluation_operator_counters),
        ):
            if len(counters) != len(set(counters)):
                raise ValueError(f"Gate9D {label} operator counters repeat")
            if any(
                isinstance(counter, bool)
                or not isinstance(counter, int)
                or not 0 <= counter < (1 << 64)
                for counter in counters
            ):
                raise ValueError(f"Gate9D {label} counter lies outside uint64")
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("Gate9D stage schedule must be positive")
        if not self.shuffle_namespace:
            raise ValueError("Gate9D stage requires a shuffle namespace")
        overlap = set(self.train_operator_counters) & set(
            self.evaluation_operator_counters
        )
        if self.unseen_operator_evaluation and overlap:
            raise ValueError("Gate9D unseen evaluation overlaps training operators")
        if not self.unseen_operator_evaluation and (
            set(self.train_operator_counters)
            != set(self.evaluation_operator_counters)
        ):
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
        order=1,
        name="single_operator_query_fit",
        purpose=(
            "Test whether the frozen worker and optimizer can fit one complete "
            "non-support byte mapping when operator context is constant."
        ),
        train_operator_counters=GATE9D_SINGLE_OPERATOR_RANGE.counters(),
        evaluation_operator_counters=GATE9D_SINGLE_OPERATOR_RANGE.counters(),
        steps=1_024,
        batch_size=247,
        requires_context_causality=False,
        unseen_operator_evaluation=False,
        shuffle_namespace="gate9d-single-operator-query-fit-v0",
    ),
    DiagnosticStage(
        order=2,
        name="paired_operator_context_collision",
        purpose=(
            "Force causal support use with two operators that share every query "
            "but require opposite output bytes."
        ),
        train_operator_counters=GATE9D_COLLISION_OPERATOR_COUNTERS,
        evaluation_operator_counters=GATE9D_COLLISION_OPERATOR_COUNTERS,
        steps=2_048,
        batch_size=494,
        requires_context_causality=True,
        unseen_operator_evaluation=False,
        shuffle_namespace="gate9d-paired-context-collision-v0",
    ),
    DiagnosticStage(
        order=3,
        name="held_in_multi_operator_fit",
        purpose=(
            "Test contextual conditioning and memorization across sixteen fixed "
            "operators before asking for unseen-operator induction."
        ),
        train_operator_counters=GATE9D_HELD_IN_OPERATOR_RANGE.counters(),
        evaluation_operator_counters=GATE9D_HELD_IN_OPERATOR_RANGE.counters(),
        steps=4_096,
        batch_size=512,
        requires_context_causality=True,
        unseen_operator_evaluation=False,
        shuffle_namespace="gate9d-held-in-multi-operator-fit-v0",
    ),
    DiagnosticStage(
        order=4,
        name="unseen_operator_generalization",
        purpose=(
            "Test induction on sixty-four operators disjoint from the 256 "
            "diagnostic training operators only after all prior stages pass."
        ),
        train_operator_counters=GATE9D_UNSEEN_TRAIN_OPERATOR_RANGE.counters(),
        evaluation_operator_counters=GATE9D_UNSEEN_EVAL_OPERATOR_RANGE.counters(),
        steps=8_192,
        batch_size=512,
        requires_context_causality=True,
        unseen_operator_evaluation=True,
        shuffle_namespace="gate9d-unseen-operator-generalization-v0",
    ),
)

GATE9D_STAGE_FAILURE_OUTCOMES = {
    "single_operator_query_fit": "G9D_BASIC_QUERY_MAPPING_FAILED",
    "paired_operator_context_collision": "G9D_CONTEXTUAL_CONTROL_FAILED",
    "held_in_multi_operator_fit": "G9D_HELD_IN_OPERATOR_FIT_FAILED",
    "unseen_operator_generalization": (
        "G9D_UNSEEN_OPERATOR_GENERALIZATION_FAILED"
    ),
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
            if label.startswith("full_minus_"):
                if not -1.0 <= float(value) <= 1.0:
                    raise ValueError(f"Gate9D {label} lies outside -1..1")
            elif not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Gate9D {label} lies outside 0..1")

    def passes(self, stage: DiagnosticStage) -> bool:
        self.validate()
        stage.validate()
        base = (
            self.exact_accuracy >= GATE9D_EXACT_ACCURACY_MIN
            and self.bit_accuracy >= GATE9D_BIT_ACCURACY_MIN
            and self.oracle_accuracy == GATE9D_ORACLE_ACCURACY_REQUIRED
        )
        if not base:
            return False
        if not stage.requires_context_causality:
            return True
        return (
            self.full_minus_shuffled > GATE9D_CONTEXT_DELTA_MIN_STRICT
            and self.full_minus_query_only > GATE9D_CONTEXT_DELTA_MIN_STRICT
        )


def classify_diagnostic(
    stage_seed_passes: Mapping[str, Sequence[bool]],
) -> str:
    """Return the first preregistered deficiency without post-hoc rescue."""

    known = {stage.name for stage in GATE9D_STAGES}
    extra = set(stage_seed_passes) - known
    if extra:
        raise ValueError(f"unknown Gate9D stage results: {sorted(extra)!r}")
    for stage in GATE9D_STAGES:
        values = stage_seed_passes.get(stage.name)
        if values is None:
            return GATE9D_INCOMPLETE_OUTCOME
        passes = tuple(values)
        if len(passes) != len(GATE9D_INITIALIZATION_SEEDS) or any(
            type(value) is not bool for value in passes
        ):
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
    if len({stage.name for stage in GATE9D_STAGES}) != len(GATE9D_STAGES):
        raise RuntimeError("Gate9D stage names repeat")
    for stage in GATE9D_STAGES:
        stage.validate()

    diagnostic_groups = (
        set(GATE9D_SINGLE_OPERATOR_RANGE.counters()),
        set(GATE9D_COLLISION_OPERATOR_COUNTERS),
        set(GATE9D_HELD_IN_OPERATOR_RANGE.counters()),
        set(GATE9D_UNSEEN_TRAIN_OPERATOR_RANGE.counters()),
        set(GATE9D_UNSEEN_EVAL_OPERATOR_RANGE.counters()),
    )
    for index, left in enumerate(diagnostic_groups):
        for right in diagnostic_groups[index + 1 :]:
            if left & right:
                raise RuntimeError("Gate9D diagnostic operator identities overlap")
    for frozen in GATE9D_FROZEN_V0_RANGES:
        frozen.validate()
        frozen_values = range(frozen.start, frozen.stop)
        for group in diagnostic_groups:
            if any(counter in frozen_values for counter in group):
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
                "evaluation_operator_count": len(
                    stage.evaluation_operator_counters
                ),
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
