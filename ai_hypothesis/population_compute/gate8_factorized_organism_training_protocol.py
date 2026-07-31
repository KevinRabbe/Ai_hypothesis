"""Data-frozen Gate-8 v1 factorized-message training protocol.

This module freezes local transition semantics, world scheduling, optimizer
settings, checkpoint candidates, development validation and admission
thresholds. It opens no world generation, optimizer execution, checkpoint
write, scientific-test world or reference-model path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

GATE8_V1_TRAINING_PROTOCOL_VERSION = (
    "gate8-factorized-message-training-protocol-v1"
)
GATE8_V1_TRAINING_PROTOCOL_STATUS = (
    "DATA_FROZEN_GATE8_V1_FACTORIZED_TRAINING_PROTOCOL_EXECUTION_CLOSED"
)
GATE8_V1_TRAINING_PROTOCOL_ARCHITECTURE_HEAD = (
    "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8"
)
GATE8_V1_TRAINING_PROTOCOL_RUNTIME_HEAD = (
    "333d88ac4fc52f1651741fba224e0b4605feedd3"
)

GATE8_V1_LEARNED_PARAMETER_COUNT = 19_649
GATE8_V1_MESSAGE_CODEBOOK_SIZE = 256
GATE8_V1_MESSAGE_BITS = 8
GATE8_V1_SYMBOL_COUNT = 16
GATE8_V1_CARRIER_COUNT = 16
GATE8_V1_TRANSFORM_COUNT = 8

GATE8_V1_TRAINING_SEEDS = (0, 1, 2)
GATE8_V1_TRAINING_WORLDS_PER_SEED = 262_144
GATE8_V1_TRAINING_WORLD_BATCH_SIZE = 256
GATE8_V1_OPTIMIZER_STEPS = 1_024
GATE8_V1_TRAINING_CONDITIONS = (
    (32, 4),
    (64, 4),
    (64, 8),
    (128, 4),
    (128, 8),
    (128, 16),
)
GATE8_V1_VALIDATION_WORLD_INDEX_START = 512
GATE8_V1_VALIDATION_WORLDS_PER_CONDITION = 512
GATE8_V1_VALIDATION_WORLD_INDEX_STOP = (
    GATE8_V1_VALIDATION_WORLD_INDEX_START
    + GATE8_V1_VALIDATION_WORLDS_PER_CONDITION
)
GATE8_V1_CHECKPOINT_STEPS = (256, 512, 768, 1_024)

GATE8_V1_OPTIMIZER = "adamw"
GATE8_V1_LEARNING_RATE = 3.0e-3
GATE8_V1_MINIMUM_LEARNING_RATE = 3.0e-5
GATE8_V1_WARMUP_STEPS = 64
GATE8_V1_ADAM_BETAS = (0.9, 0.95)
GATE8_V1_ADAM_EPSILON = 1.0e-8
GATE8_V1_WEIGHT_DECAY = 1.0e-4
GATE8_V1_GRADIENT_CLIP_NORM = 1.0
GATE8_V1_PARAMETER_DTYPE = "float32"
GATE8_V1_AUTOCAST_ENABLED = False
GATE8_V1_TF32_ENABLED = False
GATE8_V1_DETERMINISTIC_ALGORITHMS = True

GATE8_V1_CARRIER_LOSS_WEIGHT = 1.0
GATE8_V1_SYMBOL_LOSS_WEIGHT = 1.0
GATE8_V1_LOCAL_SUPERVISION = "all_edges_in_every_training_world"
GATE8_V1_VALIDATION_RUNTIME = "qualified_v1_deterministic_full_runtime"

GATE8_V1_MIN_CONDITION_TARGET_ACCURACY = 0.99
GATE8_V1_MIN_EXACT_MESSAGE_ACCURACY = 0.995
GATE8_V1_MIN_CARRIER_ACCURACY = 0.995
GATE8_V1_MIN_SYMBOL_ACCURACY = 0.995
GATE8_V1_REQUIRED_INBOX_CODE_COVERAGE = 256
GATE8_V1_REQUIRED_TARGET_CODE_COVERAGE = 256
GATE8_V1_REQUIRED_TARGET_CARRIER_COVERAGE = 16
GATE8_V1_REQUIRED_TARGET_SYMBOL_COVERAGE = 16

GATE8_V1_TRAINING_ADMITTED = "G8_V1_TRAINING_CHECKPOINT_ADMITTED"
GATE8_V1_TRAINING_NOT_ADMITTED = "G8_V1_TRAINING_CHECKPOINT_NOT_ADMITTED"
GATE8_V1_TRAINING_OUTCOMES = (
    GATE8_V1_TRAINING_ADMITTED,
    GATE8_V1_TRAINING_NOT_ADMITTED,
)

if (
    GATE8_V1_TRAINING_WORLDS_PER_SEED
    % GATE8_V1_TRAINING_WORLD_BATCH_SIZE
    != 0
):
    raise RuntimeError(
        "Gate8 v1 training worlds must divide exactly into optimizer batches"
    )
if (
    GATE8_V1_TRAINING_WORLDS_PER_SEED
    // GATE8_V1_TRAINING_WORLD_BATCH_SIZE
    != GATE8_V1_OPTIMIZER_STEPS
):
    raise RuntimeError("Gate8 v1 optimizer-step count drifted")
if GATE8_V1_VALIDATION_WORLD_INDEX_START < 512:
    raise RuntimeError("Gate8 v1 validation reuses v0 development worlds")
if (
    GATE8_V1_VALIDATION_WORLD_INDEX_STOP
    - GATE8_V1_VALIDATION_WORLD_INDEX_START
    != GATE8_V1_VALIDATION_WORLDS_PER_CONDITION
):
    raise RuntimeError("Gate8 v1 validation range drifted")


@dataclass(frozen=True, slots=True)
class Gate8V1TrainingWorldAddress:
    global_world_index: int
    population: int
    depth: int
    condition_world_index: int

    def validate(self) -> None:
        if not 0 <= self.global_world_index < GATE8_V1_TRAINING_WORLDS_PER_SEED:
            raise ValueError(
                "Gate8 v1 global training-world index is outside the frozen run"
            )
        if (self.population, self.depth) not in GATE8_V1_TRAINING_CONDITIONS:
            raise ValueError(
                "Gate8 v1 training-world condition is outside the frozen schedule"
            )
        if self.condition_world_index < 0:
            raise ValueError(
                "Gate8 v1 condition-local world index cannot be negative"
            )


@dataclass(frozen=True, slots=True)
class Gate8V1ValidationConditionRow:
    population: int
    depth: int
    target_accuracy: float

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_V1_TRAINING_CONDITIONS:
            raise ValueError(
                "Gate8 v1 validation row is outside the frozen training regime"
            )
        if not 0.0 <= self.target_accuracy <= 1.0:
            raise ValueError(
                "Gate8 v1 validation target accuracy is outside 0..1"
            )


@dataclass(frozen=True, slots=True)
class Gate8V1CheckpointCandidate:
    step: int
    conditions: tuple[Gate8V1ValidationConditionRow, ...]
    exact_message_accuracy: float
    carrier_accuracy: float
    symbol_accuracy: float
    validation_loss: float
    inbox_code_coverage: int
    target_code_coverage: int
    target_carrier_coverage: int
    target_symbol_coverage: int

    def validate(self) -> None:
        if self.step not in GATE8_V1_CHECKPOINT_STEPS:
            raise ValueError(
                "Gate8 v1 checkpoint step is outside the frozen candidate set"
            )
        observed = tuple(
            (row.population, row.depth) for row in self.conditions
        )
        if observed != GATE8_V1_TRAINING_CONDITIONS:
            raise ValueError(
                "Gate8 v1 checkpoint validation rows are incomplete or reordered"
            )
        for row in self.conditions:
            row.validate()
        for name, value in (
            ("exact_message_accuracy", self.exact_message_accuracy),
            ("carrier_accuracy", self.carrier_accuracy),
            ("symbol_accuracy", self.symbol_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Gate8 v1 {name} is outside 0..1")
        if not math.isfinite(self.validation_loss) or self.validation_loss < 0.0:
            raise ValueError(
                "Gate8 v1 validation loss must be finite and non-negative"
            )
        for name, value, upper_bound in (
            (
                "inbox_code_coverage",
                self.inbox_code_coverage,
                GATE8_V1_MESSAGE_CODEBOOK_SIZE,
            ),
            (
                "target_code_coverage",
                self.target_code_coverage,
                GATE8_V1_MESSAGE_CODEBOOK_SIZE,
            ),
            (
                "target_carrier_coverage",
                self.target_carrier_coverage,
                GATE8_V1_CARRIER_COUNT,
            ),
            (
                "target_symbol_coverage",
                self.target_symbol_coverage,
                GATE8_V1_SYMBOL_COUNT,
            ),
        ):
            if not 0 <= value <= upper_bound:
                raise ValueError(
                    f"Gate8 v1 {name} is outside 0..{upper_bound}"
                )

    @property
    def mean_target_accuracy(self) -> float:
        self.validate()
        return sum(row.target_accuracy for row in self.conditions) / len(
            self.conditions
        )

    @property
    def minimum_target_accuracy(self) -> float:
        self.validate()
        return min(row.target_accuracy for row in self.conditions)

    def admitted(self) -> bool:
        self.validate()
        return (
            self.minimum_target_accuracy
            >= GATE8_V1_MIN_CONDITION_TARGET_ACCURACY
            and self.exact_message_accuracy
            >= GATE8_V1_MIN_EXACT_MESSAGE_ACCURACY
            and self.carrier_accuracy >= GATE8_V1_MIN_CARRIER_ACCURACY
            and self.symbol_accuracy >= GATE8_V1_MIN_SYMBOL_ACCURACY
            and self.inbox_code_coverage
            == GATE8_V1_REQUIRED_INBOX_CODE_COVERAGE
            and self.target_code_coverage
            == GATE8_V1_REQUIRED_TARGET_CODE_COVERAGE
            and self.target_carrier_coverage
            == GATE8_V1_REQUIRED_TARGET_CARRIER_COVERAGE
            and self.target_symbol_coverage
            == GATE8_V1_REQUIRED_TARGET_SYMBOL_COVERAGE
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["mean_target_accuracy"] = self.mean_target_accuracy
        payload["minimum_target_accuracy"] = self.minimum_target_accuracy
        payload["admitted"] = self.admitted()
        return payload


def gate8_v1_encode_message_code(*, carrier: int, symbol: int) -> int:
    if not 0 <= carrier < GATE8_V1_CARRIER_COUNT:
        raise ValueError("Gate8 v1 carrier is outside 0..15")
    if not 0 <= symbol < GATE8_V1_SYMBOL_COUNT:
        raise ValueError("Gate8 v1 symbol is outside 0..15")
    return carrier * GATE8_V1_SYMBOL_COUNT + symbol


def gate8_v1_decode_message_code(code: int) -> tuple[int, int]:
    if not 0 <= code < GATE8_V1_MESSAGE_CODEBOOK_SIZE:
        raise ValueError("Gate8 v1 message code is outside 0..255")
    return divmod(code, GATE8_V1_SYMBOL_COUNT)


def _validate_transform(transform: Sequence[int]) -> tuple[int, ...]:
    frozen = tuple(transform)
    if len(frozen) != GATE8_V1_SYMBOL_COUNT:
        raise ValueError("Gate8 v1 transform must contain exactly 16 outputs")
    if tuple(sorted(frozen)) != tuple(range(GATE8_V1_SYMBOL_COUNT)):
        raise ValueError("Gate8 v1 transform must be a permutation of 0..15")
    return frozen


def gate8_v1_target_transition(
    *,
    inbox_code: int,
    transform: Sequence[int],
) -> tuple[int, int, int]:
    input_carrier, input_symbol = gate8_v1_decode_message_code(inbox_code)
    frozen_transform = _validate_transform(transform)
    target_carrier = (input_carrier + 1) % GATE8_V1_CARRIER_COUNT
    target_symbol = frozen_transform[input_symbol]
    target_code = gate8_v1_encode_message_code(
        carrier=target_carrier,
        symbol=target_symbol,
    )
    return target_carrier, target_symbol, target_code


def gate8_v1_training_world_address(
    global_world_index: int,
) -> Gate8V1TrainingWorldAddress:
    if not 0 <= global_world_index < GATE8_V1_TRAINING_WORLDS_PER_SEED:
        raise ValueError(
            "Gate8 v1 global training-world index is outside the frozen run"
        )
    condition_index = global_world_index % len(GATE8_V1_TRAINING_CONDITIONS)
    population, depth = GATE8_V1_TRAINING_CONDITIONS[condition_index]
    address = Gate8V1TrainingWorldAddress(
        global_world_index=global_world_index,
        population=population,
        depth=depth,
        condition_world_index=global_world_index
        // len(GATE8_V1_TRAINING_CONDITIONS),
    )
    address.validate()
    return address


def gate8_v1_condition_world_counts() -> dict[tuple[int, int], int]:
    counts = {
        condition: 0 for condition in GATE8_V1_TRAINING_CONDITIONS
    }
    for global_world_index in range(GATE8_V1_TRAINING_WORLDS_PER_SEED):
        address = gate8_v1_training_world_address(global_world_index)
        counts[(address.population, address.depth)] += 1
    return counts


def gate8_v1_validation_world_indices() -> tuple[int, ...]:
    return tuple(
        range(
            GATE8_V1_VALIDATION_WORLD_INDEX_START,
            GATE8_V1_VALIDATION_WORLD_INDEX_STOP,
        )
    )


def gate8_v1_learning_rate(step: int) -> float:
    if not 1 <= step <= GATE8_V1_OPTIMIZER_STEPS:
        raise ValueError("Gate8 v1 optimizer step is outside 1..1024")
    if step <= GATE8_V1_WARMUP_STEPS:
        return (
            GATE8_V1_LEARNING_RATE
            * step
            / GATE8_V1_WARMUP_STEPS
        )
    progress = (step - GATE8_V1_WARMUP_STEPS) / (
        GATE8_V1_OPTIMIZER_STEPS - GATE8_V1_WARMUP_STEPS
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return GATE8_V1_MINIMUM_LEARNING_RATE + (
        GATE8_V1_LEARNING_RATE - GATE8_V1_MINIMUM_LEARNING_RATE
    ) * cosine


def select_gate8_v1_checkpoint(
    candidates: tuple[Gate8V1CheckpointCandidate, ...],
) -> Gate8V1CheckpointCandidate:
    if tuple(candidate.step for candidate in candidates) != (
        GATE8_V1_CHECKPOINT_STEPS
    ):
        raise ValueError(
            "Gate8 v1 checkpoint candidates must cover exact frozen steps"
        )
    for candidate in candidates:
        candidate.validate()
    return max(
        candidates,
        key=lambda candidate: (
            candidate.mean_target_accuracy,
            candidate.minimum_target_accuracy,
            candidate.exact_message_accuracy,
            candidate.symbol_accuracy,
            candidate.carrier_accuracy,
            -candidate.validation_loss,
            -candidate.step,
        ),
    )


def classify_gate8_v1_training(
    candidates: tuple[Gate8V1CheckpointCandidate, ...],
) -> str:
    selected = select_gate8_v1_checkpoint(candidates)
    return (
        GATE8_V1_TRAINING_ADMITTED
        if selected.admitted()
        else GATE8_V1_TRAINING_NOT_ADMITTED
    )


def gate8_v1_training_protocol_plan() -> dict[str, Any]:
    counts = gate8_v1_condition_world_counts()
    return {
        "version": GATE8_V1_TRAINING_PROTOCOL_VERSION,
        "scientific_status": GATE8_V1_TRAINING_PROTOCOL_STATUS,
        "architecture_head": GATE8_V1_TRAINING_PROTOCOL_ARCHITECTURE_HEAD,
        "runtime_head": GATE8_V1_TRAINING_PROTOCOL_RUNTIME_HEAD,
        "execution_admitted": False,
        "training_admitted": False,
        "checkpoint_write_admitted": False,
        "scientific_test_worlds_admitted": False,
        "reference_model_admitted": False,
        "learned_parameter_count": GATE8_V1_LEARNED_PARAMETER_COUNT,
        "training_seeds": list(GATE8_V1_TRAINING_SEEDS),
        "seed0_first": True,
        "seeds_1_and_2_blocked_until_seed0_admission": True,
        "training_worlds_reused_for_controlled_architecture_comparison": True,
        "training_worlds_per_seed": GATE8_V1_TRAINING_WORLDS_PER_SEED,
        "world_batch_size": GATE8_V1_TRAINING_WORLD_BATCH_SIZE,
        "optimizer_steps": GATE8_V1_OPTIMIZER_STEPS,
        "training_conditions": [
            list(condition)
            for condition in GATE8_V1_TRAINING_CONDITIONS
        ],
        "condition_world_counts": {
            f"{population}x{depth}": count
            for (population, depth), count in counts.items()
        },
        "validation_namespace": "validation",
        "validation_world_indices": [
            GATE8_V1_VALIDATION_WORLD_INDEX_START,
            GATE8_V1_VALIDATION_WORLD_INDEX_STOP - 1,
        ],
        "validation_worlds_per_condition": (
            GATE8_V1_VALIDATION_WORLDS_PER_CONDITION
        ),
        "validation_disjoint_from_v0_indices_0_through_511": True,
        "checkpoint_steps": list(GATE8_V1_CHECKPOINT_STEPS),
        "message_code_format": "high4_carrier_low4_symbol",
        "root_input_message": "carrier_zero_plus_public_root_symbol",
        "carrier_transition": "plus_one_mod16_per_edge",
        "symbol_transition": "canonical_primitive_permutation",
        "local_supervision": GATE8_V1_LOCAL_SUPERVISION,
        "validation_runtime": GATE8_V1_VALIDATION_RUNTIME,
        "optimizer": GATE8_V1_OPTIMIZER,
        "learning_rate": GATE8_V1_LEARNING_RATE,
        "minimum_learning_rate": GATE8_V1_MINIMUM_LEARNING_RATE,
        "warmup_steps": GATE8_V1_WARMUP_STEPS,
        "adam_betas": list(GATE8_V1_ADAM_BETAS),
        "adam_epsilon": GATE8_V1_ADAM_EPSILON,
        "weight_decay": GATE8_V1_WEIGHT_DECAY,
        "gradient_clip_norm": GATE8_V1_GRADIENT_CLIP_NORM,
        "parameter_dtype": GATE8_V1_PARAMETER_DTYPE,
        "autocast_enabled": GATE8_V1_AUTOCAST_ENABLED,
        "tf32_enabled": GATE8_V1_TF32_ENABLED,
        "deterministic_algorithms": GATE8_V1_DETERMINISTIC_ALGORITHMS,
        "loss_weights": {
            "carrier_cross_entropy": GATE8_V1_CARRIER_LOSS_WEIGHT,
            "symbol_cross_entropy": GATE8_V1_SYMBOL_LOSS_WEIGHT,
        },
        "removed_losses": [
            "joint_256_way_message_cross_entropy",
            "answer_cross_entropy",
            "activity_binary_cross_entropy",
        ],
        "checkpoint_selection": (
            "max_mean_target_then_min_target_then_exact_message_then_symbol_"
            "then_carrier_then_min_loss_then_earliest_step"
        ),
        "admission_thresholds": {
            "minimum_condition_target_accuracy": (
                GATE8_V1_MIN_CONDITION_TARGET_ACCURACY
            ),
            "minimum_exact_message_accuracy": (
                GATE8_V1_MIN_EXACT_MESSAGE_ACCURACY
            ),
            "minimum_carrier_accuracy": (
                GATE8_V1_MIN_CARRIER_ACCURACY
            ),
            "minimum_symbol_accuracy": GATE8_V1_MIN_SYMBOL_ACCURACY,
            "inbox_code_coverage": (
                GATE8_V1_REQUIRED_INBOX_CODE_COVERAGE
            ),
            "target_code_coverage": (
                GATE8_V1_REQUIRED_TARGET_CODE_COVERAGE
            ),
            "target_carrier_coverage": (
                GATE8_V1_REQUIRED_TARGET_CARRIER_COVERAGE
            ),
            "target_symbol_coverage": (
                GATE8_V1_REQUIRED_TARGET_SYMBOL_COVERAGE
            ),
        },
    }
