"""Data-frozen Gate-8 organism training protocol.

This module freezes message semantics, world scheduling, optimizer settings,
checkpoint candidates, development validation and admission thresholds. It opens
no world generation, optimizer execution, checkpoint write, scientific-test
world or reference-model path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

GATE8_ORGANISM_TRAINING_PROTOCOL_VERSION = "gate8-organism-training-protocol-v0"
GATE8_ORGANISM_TRAINING_PROTOCOL_STATUS = (
    "DATA_FROZEN_GATE8_ORGANISM_TRAINING_PROTOCOL_EXECUTION_CLOSED"
)
GATE8_ORGANISM_TRAINING_PROTOCOL_RUNTIME_HEAD = (
    "1a2be148411bc71ba35fda12b035b724f06ec166"
)

GATE8_LEARNED_PARAMETER_COUNT = 19_649
GATE8_MESSAGE_CODEBOOK_SIZE = 256
GATE8_SYMBOL_COUNT = 16
GATE8_CARRIER_COUNT = 16
GATE8_ROOT_SEED_CODE = 0
GATE8_TRAINING_SEEDS = (0, 1, 2)
GATE8_TRAINING_WORLDS_PER_SEED = 262_144
GATE8_TRAINING_WORLD_BATCH_SIZE = 256
GATE8_OPTIMIZER_STEPS = 1_024
GATE8_TRAINING_CONDITIONS = (
    (32, 4),
    (64, 4),
    (64, 8),
    (128, 4),
    (128, 8),
    (128, 16),
)
GATE8_VALIDATION_WORLDS_PER_CONDITION = 512
GATE8_CHECKPOINT_STEPS = (256, 512, 768, 1_024)

GATE8_OPTIMIZER = "adamw"
GATE8_LEARNING_RATE = 3.0e-3
GATE8_MINIMUM_LEARNING_RATE = 3.0e-5
GATE8_WARMUP_STEPS = 64
GATE8_ADAM_BETAS = (0.9, 0.95)
GATE8_ADAM_EPSILON = 1.0e-8
GATE8_WEIGHT_DECAY = 1.0e-4
GATE8_GRADIENT_CLIP_NORM = 1.0
GATE8_PARAMETER_DTYPE = "float32"
GATE8_AUTOCAST_ENABLED = False
GATE8_TF32_ENABLED = False
GATE8_DETERMINISTIC_ALGORITHMS = True

GATE8_MESSAGE_LOSS_WEIGHT = 1.0
GATE8_ANSWER_LOSS_WEIGHT = 1.0
GATE8_ACTIVITY_LOSS_WEIGHT = 0.1
GATE8_ACTIVITY_TARGET = 1.0
GATE8_LOCAL_SUPERVISION = "all_edges_in_every_training_world"
GATE8_VALIDATION_RUNTIME = "qualified_deterministic_full_runtime"

GATE8_MIN_CONDITION_TARGET_ACCURACY = 0.99
GATE8_MIN_MESSAGE_ACCURACY = 0.995
GATE8_MIN_ACTIVITY_ACCURACY = 0.99
GATE8_REQUIRED_INBOX_CODE_COVERAGE = 256
GATE8_REQUIRED_TARGET_CODE_COVERAGE = 256

GATE8_TRAINING_ADMITTED = "G8_TRAINING_CHECKPOINT_ADMITTED"
GATE8_TRAINING_NOT_ADMITTED = "G8_TRAINING_CHECKPOINT_NOT_ADMITTED"
GATE8_TRAINING_OUTCOMES = (
    GATE8_TRAINING_ADMITTED,
    GATE8_TRAINING_NOT_ADMITTED,
)

if GATE8_TRAINING_WORLDS_PER_SEED % GATE8_TRAINING_WORLD_BATCH_SIZE != 0:
    raise RuntimeError("Gate8 training worlds must divide exactly into optimizer batches")
if GATE8_TRAINING_WORLDS_PER_SEED // GATE8_TRAINING_WORLD_BATCH_SIZE != GATE8_OPTIMIZER_STEPS:
    raise RuntimeError("Gate8 optimizer-step count drifted")


@dataclass(frozen=True, slots=True)
class Gate8TrainingWorldAddress:
    global_world_index: int
    population: int
    depth: int
    condition_world_index: int

    def validate(self) -> None:
        if not 0 <= self.global_world_index < GATE8_TRAINING_WORLDS_PER_SEED:
            raise ValueError("Gate8 global training-world index is outside the frozen run")
        if (self.population, self.depth) not in GATE8_TRAINING_CONDITIONS:
            raise ValueError("Gate8 training-world condition is outside the frozen schedule")
        if self.condition_world_index < 0:
            raise ValueError("Gate8 condition-local world index cannot be negative")


@dataclass(frozen=True, slots=True)
class Gate8ValidationConditionRow:
    population: int
    depth: int
    target_accuracy: float

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_TRAINING_CONDITIONS:
            raise ValueError("Gate8 validation row is outside the frozen training regime")
        if not 0.0 <= self.target_accuracy <= 1.0:
            raise ValueError("Gate8 validation target accuracy is outside 0..1")


@dataclass(frozen=True, slots=True)
class Gate8CheckpointCandidate:
    step: int
    conditions: tuple[Gate8ValidationConditionRow, ...]
    message_accuracy: float
    activity_accuracy: float
    validation_loss: float
    inbox_code_coverage: int
    target_code_coverage: int

    def validate(self) -> None:
        if self.step not in GATE8_CHECKPOINT_STEPS:
            raise ValueError("Gate8 checkpoint step is outside the frozen candidate set")
        observed = tuple((row.population, row.depth) for row in self.conditions)
        if observed != GATE8_TRAINING_CONDITIONS:
            raise ValueError("Gate8 checkpoint validation rows are incomplete or reordered")
        for row in self.conditions:
            row.validate()
        for name, value in (
            ("message_accuracy", self.message_accuracy),
            ("activity_accuracy", self.activity_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Gate8 {name} is outside 0..1")
        if not math.isfinite(self.validation_loss) or self.validation_loss < 0.0:
            raise ValueError("Gate8 validation loss must be finite and non-negative")
        for name, value in (
            ("inbox_code_coverage", self.inbox_code_coverage),
            ("target_code_coverage", self.target_code_coverage),
        ):
            if not 0 <= value <= GATE8_MESSAGE_CODEBOOK_SIZE:
                raise ValueError(f"Gate8 {name} is outside 0..256")

    @property
    def mean_target_accuracy(self) -> float:
        self.validate()
        return sum(row.target_accuracy for row in self.conditions) / len(self.conditions)

    @property
    def minimum_target_accuracy(self) -> float:
        self.validate()
        return min(row.target_accuracy for row in self.conditions)

    def admitted(self) -> bool:
        self.validate()
        return (
            self.minimum_target_accuracy >= GATE8_MIN_CONDITION_TARGET_ACCURACY
            and self.message_accuracy >= GATE8_MIN_MESSAGE_ACCURACY
            and self.activity_accuracy >= GATE8_MIN_ACTIVITY_ACCURACY
            and self.inbox_code_coverage == GATE8_REQUIRED_INBOX_CODE_COVERAGE
            and self.target_code_coverage == GATE8_REQUIRED_TARGET_CODE_COVERAGE
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["mean_target_accuracy"] = self.mean_target_accuracy
        payload["minimum_target_accuracy"] = self.minimum_target_accuracy
        payload["admitted"] = self.admitted()
        return payload


def gate8_encode_message_code(*, carrier: int, symbol: int) -> int:
    if not 0 <= carrier < GATE8_CARRIER_COUNT:
        raise ValueError("Gate8 carrier is outside 0..15")
    if not 0 <= symbol < GATE8_SYMBOL_COUNT:
        raise ValueError("Gate8 symbol is outside 0..15")
    return carrier * GATE8_SYMBOL_COUNT + symbol


def gate8_decode_message_code(code: int) -> tuple[int, int]:
    if not 0 <= code < GATE8_MESSAGE_CODEBOOK_SIZE:
        raise ValueError("Gate8 message code is outside 0..255")
    return divmod(code, GATE8_SYMBOL_COUNT)


def gate8_target_message_code(
    *,
    inbox_code: int,
    output_symbol: int,
    source_is_root: bool,
    root_symbol: int,
) -> int:
    if source_is_root:
        if inbox_code != GATE8_ROOT_SEED_CODE:
            raise ValueError("Gate8 root worker must receive the frozen seed code")
        if not 0 <= root_symbol < GATE8_SYMBOL_COUNT:
            raise ValueError("Gate8 root symbol is outside 0..15")
        input_carrier = root_symbol
    else:
        input_carrier, _input_symbol = gate8_decode_message_code(inbox_code)
    next_carrier = (input_carrier + 1) % GATE8_CARRIER_COUNT
    return gate8_encode_message_code(
        carrier=next_carrier,
        symbol=output_symbol,
    )


def gate8_training_world_address(global_world_index: int) -> Gate8TrainingWorldAddress:
    if not 0 <= global_world_index < GATE8_TRAINING_WORLDS_PER_SEED:
        raise ValueError("Gate8 global training-world index is outside the frozen run")
    condition_index = global_world_index % len(GATE8_TRAINING_CONDITIONS)
    population, depth = GATE8_TRAINING_CONDITIONS[condition_index]
    address = Gate8TrainingWorldAddress(
        global_world_index=global_world_index,
        population=population,
        depth=depth,
        condition_world_index=global_world_index // len(GATE8_TRAINING_CONDITIONS),
    )
    address.validate()
    return address


def gate8_condition_world_counts() -> dict[tuple[int, int], int]:
    counts = {condition: 0 for condition in GATE8_TRAINING_CONDITIONS}
    for global_world_index in range(GATE8_TRAINING_WORLDS_PER_SEED):
        address = gate8_training_world_address(global_world_index)
        counts[(address.population, address.depth)] += 1
    return counts


def gate8_learning_rate(step: int) -> float:
    if not 1 <= step <= GATE8_OPTIMIZER_STEPS:
        raise ValueError("Gate8 optimizer step is outside 1..1024")
    if step <= GATE8_WARMUP_STEPS:
        return GATE8_LEARNING_RATE * step / GATE8_WARMUP_STEPS
    progress = (step - GATE8_WARMUP_STEPS) / (
        GATE8_OPTIMIZER_STEPS - GATE8_WARMUP_STEPS
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return GATE8_MINIMUM_LEARNING_RATE + (
        GATE8_LEARNING_RATE - GATE8_MINIMUM_LEARNING_RATE
    ) * cosine


def select_gate8_checkpoint(
    candidates: tuple[Gate8CheckpointCandidate, ...],
) -> Gate8CheckpointCandidate:
    if tuple(candidate.step for candidate in candidates) != GATE8_CHECKPOINT_STEPS:
        raise ValueError("Gate8 checkpoint candidates must cover exact frozen steps")
    for candidate in candidates:
        candidate.validate()
    return max(
        candidates,
        key=lambda candidate: (
            candidate.mean_target_accuracy,
            candidate.minimum_target_accuracy,
            candidate.message_accuracy,
            candidate.activity_accuracy,
            -candidate.validation_loss,
            -candidate.step,
        ),
    )


def classify_gate8_training(
    candidates: tuple[Gate8CheckpointCandidate, ...],
) -> str:
    selected = select_gate8_checkpoint(candidates)
    return GATE8_TRAINING_ADMITTED if selected.admitted() else GATE8_TRAINING_NOT_ADMITTED


def gate8_organism_training_protocol_plan() -> dict[str, Any]:
    counts = gate8_condition_world_counts()
    return {
        "version": GATE8_ORGANISM_TRAINING_PROTOCOL_VERSION,
        "scientific_status": GATE8_ORGANISM_TRAINING_PROTOCOL_STATUS,
        "runtime_head": GATE8_ORGANISM_TRAINING_PROTOCOL_RUNTIME_HEAD,
        "execution_admitted": False,
        "training_admitted": False,
        "checkpoint_write_admitted": False,
        "scientific_test_worlds_admitted": False,
        "reference_model_admitted": False,
        "learned_parameter_count": GATE8_LEARNED_PARAMETER_COUNT,
        "training_seeds": list(GATE8_TRAINING_SEEDS),
        "training_worlds_per_seed": GATE8_TRAINING_WORLDS_PER_SEED,
        "world_batch_size": GATE8_TRAINING_WORLD_BATCH_SIZE,
        "optimizer_steps": GATE8_OPTIMIZER_STEPS,
        "training_conditions": [list(condition) for condition in GATE8_TRAINING_CONDITIONS],
        "condition_world_counts": {
            f"{population}x{depth}": count
            for (population, depth), count in counts.items()
        },
        "validation_namespace": "validation",
        "validation_world_indices": [0, GATE8_VALIDATION_WORLDS_PER_CONDITION - 1],
        "validation_worlds_per_condition": GATE8_VALIDATION_WORLDS_PER_CONDITION,
        "checkpoint_steps": list(GATE8_CHECKPOINT_STEPS),
        "message_code_format": "high4_carrier_low4_symbol",
        "carrier_transition": "plus_one_mod16_per_edge",
        "root_input_carrier": "public_root_symbol",
        "local_supervision": GATE8_LOCAL_SUPERVISION,
        "optimizer": GATE8_OPTIMIZER,
        "learning_rate": GATE8_LEARNING_RATE,
        "minimum_learning_rate": GATE8_MINIMUM_LEARNING_RATE,
        "warmup_steps": GATE8_WARMUP_STEPS,
        "adam_betas": list(GATE8_ADAM_BETAS),
        "adam_epsilon": GATE8_ADAM_EPSILON,
        "weight_decay": GATE8_WEIGHT_DECAY,
        "gradient_clip_norm": GATE8_GRADIENT_CLIP_NORM,
        "parameter_dtype": GATE8_PARAMETER_DTYPE,
        "autocast_enabled": GATE8_AUTOCAST_ENABLED,
        "tf32_enabled": GATE8_TF32_ENABLED,
        "deterministic_algorithms": GATE8_DETERMINISTIC_ALGORITHMS,
        "loss_weights": {
            "message_cross_entropy": GATE8_MESSAGE_LOSS_WEIGHT,
            "answer_cross_entropy": GATE8_ANSWER_LOSS_WEIGHT,
            "activity_binary_cross_entropy": GATE8_ACTIVITY_LOSS_WEIGHT,
        },
        "checkpoint_selection": (
            "max_mean_target_then_min_target_then_message_then_activity_"
            "then_min_loss_then_earliest_step"
        ),
        "admission_thresholds": {
            "minimum_condition_target_accuracy": GATE8_MIN_CONDITION_TARGET_ACCURACY,
            "minimum_message_accuracy": GATE8_MIN_MESSAGE_ACCURACY,
            "minimum_activity_accuracy": GATE8_MIN_ACTIVITY_ACCURACY,
            "inbox_code_coverage": GATE8_REQUIRED_INBOX_CODE_COVERAGE,
            "target_code_coverage": GATE8_REQUIRED_TARGET_CODE_COVERAGE,
        },
    }
