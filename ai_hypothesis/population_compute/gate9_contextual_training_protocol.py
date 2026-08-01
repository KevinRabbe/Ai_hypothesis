"""Data-frozen Gate-9 contextual-worker training and admission protocol.

This standard-library module freezes the exact three-seed episode allocation,
optimizer hyperparameters, learning-rate schedule, validation controls,
checkpoint schema, and checkpoint-admission classifier. It opens no operator
generator, Tensor runtime, optimizer instance, training execution, checkpoint
serialization, scientific test world, or result artifact.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

GATE9_TRAINING_PROTOCOL_VERSION = "gate9-contextual-worker-training-protocol-v0"
GATE9_TRAINING_PROTOCOL_STATUS = (
    "DATA_FROZEN_GATE9_CONTEXTUAL_WORKER_TRAINING_PROTOCOL_EXECUTION_CLOSED"
)
GATE9_ARCHITECTURE_HEAD = "c689cc3f38f6f6f642916ee1a702d7de7bd0e43b"
GATE9_OPERATOR_CONTRACT_HEAD = "be6451e1af82b18749bd0313a9c02ca62c4eee5c"
GATE9_PROTOCOL_HEAD = "e5e20e8de6707d35f1a7a9315a5a9a67deacc9a1"

GATE9_CHECKPOINT_SEEDS = (0, 1, 2)
GATE9_INITIALIZATION_SEEDS = (900_900, 900_901, 900_902)
GATE9_LEARNED_PARAMETER_COUNT = 19_649
GATE9_STATE_TENSOR_COUNT = 17
GATE9_STATE_TENSOR_SHAPES = {
    "support_slot_modulation": (9, 24),
    "output_scale": (),
    "pair_projection.weight": (48, 16),
    "pair_projection.bias": (48,),
    "query_projection.weight": (48, 8),
    "query_projection.bias": (48,),
    "support_attention.in_proj_weight": (144, 48),
    "support_attention.in_proj_bias": (144,),
    "support_attention.out_proj.weight": (48, 48),
    "support_attention.out_proj.bias": (48,),
    "support_ff_in.weight": (64, 48),
    "support_ff_in.bias": (64,),
    "support_ff_out.weight": (48, 64),
    "support_ff_out.bias": (48,),
    "query_support_fusion.weight": (24, 96),
    "query_support_fusion.bias": (24,),
    "output_bit_head.weight": (8, 24),
}

GATE9_SUPPORT_INPUTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
GATE9_NOVEL_QUERY_VALUES = tuple(
    value for value in range(256) if value not in GATE9_SUPPORT_INPUTS
)
GATE9_NOVEL_QUERY_COUNT = len(GATE9_NOVEL_QUERY_VALUES)
if GATE9_NOVEL_QUERY_COUNT != 247:
    raise RuntimeError("Gate9 novel-query domain drifted")

GATE9_TRAIN_OPERATOR_COUNTER_START = 0
GATE9_TRAIN_OPERATOR_COUNT = 262_144
GATE9_VALIDATION_OPERATOR_COUNTER_START = 1 << 32
GATE9_VALIDATION_OPERATOR_COUNT = 32_768
GATE9_LOCAL_TEST_OPERATOR_COUNTER_START = 1 << 40
GATE9_LOCAL_TEST_OPERATOR_COUNT = 4_096
GATE9_GRAPH_TEST_OPERATOR_COUNTER_START = 1 << 48
GATE9_GRAPH_TEST_OPERATOR_COUNT = 2_629_632

GATE9_TRAIN_EPISODES = GATE9_TRAIN_OPERATOR_COUNT
GATE9_TRAIN_BATCH_SIZE = 512
GATE9_TRAIN_STEPS = GATE9_TRAIN_EPISODES // GATE9_TRAIN_BATCH_SIZE
GATE9_VALIDATION_EPISODES = GATE9_VALIDATION_OPERATOR_COUNT
GATE9_VALIDATION_BATCH_SIZE = 512
GATE9_VALIDATION_BATCHES = (
    GATE9_VALIDATION_EPISODES // GATE9_VALIDATION_BATCH_SIZE
)
if GATE9_TRAIN_STEPS != 512 or GATE9_VALIDATION_BATCHES != 64:
    raise RuntimeError("Gate9 training/validation batch arithmetic drifted")

GATE9_TRAIN_ORDER_MULTIPLIERS = (65_537, 131_071, 196_613)
GATE9_TRAIN_ORDER_OFFSETS = (17, 100_003, 200_003)
GATE9_TRAIN_QUERY_MULTIPLIERS = (17, 29, 43)
GATE9_TRAIN_QUERY_OFFSETS = (3, 71, 149)
GATE9_VALIDATION_ORDER_MULTIPLIER = 8_191
GATE9_VALIDATION_ORDER_OFFSET = 12_345
GATE9_VALIDATION_QUERY_MULTIPLIER = 61
GATE9_VALIDATION_QUERY_OFFSET = 211

GATE9_OPTIMIZER = "AdamW"
GATE9_BASE_LEARNING_RATE = 1.0e-3
GATE9_MIN_LEARNING_RATE = 1.0e-4
GATE9_WARMUP_STEPS = 16
GATE9_ADAM_BETAS = (0.9, 0.95)
GATE9_ADAM_EPSILON = 1.0e-8
GATE9_WEIGHT_DECAY = 1.0e-4
GATE9_GRADIENT_CLIP_NORM = 1.0
GATE9_LOSS = "binary_cross_entropy_with_logits_mean_over_batch_and_eight_bits"
GATE9_PRECISION = "float32"
GATE9_AMP_ENABLED = False
GATE9_TF32_ENABLED = False
GATE9_COMPILE_ENABLED = False
GATE9_DETERMINISTIC_ALGORITHMS = True
GATE9_DATALOADER_WORKERS = 0

GATE9_CHECKPOINT_STEP = GATE9_TRAIN_STEPS
GATE9_CHECKPOINT_SELECTION = "fixed_final_step_only_no_best_checkpoint_selection"
GATE9_CHECKPOINT_REQUIRED_FIELDS = (
    "experiment_version",
    "architecture_head",
    "training_protocol_head",
    "seed",
    "initialization_seed",
    "step",
    "train_episodes",
    "learned_parameter_count",
    "tensor_count",
    "state_dict",
)

GATE9_VALIDATION_MODES = ("full", "shuffled_context", "query_only", "oracle")
GATE9_VALIDATION_SHUFFLE_NAMESPACE = (
    "gate9-contextual-validation-shuffled-context-v0"
)
GATE9_VALIDATION_EXACT_ACCURACY_MIN = 0.995
GATE9_VALIDATION_BIT_ACCURACY_MIN = 0.999
GATE9_VALIDATION_CONTEXT_DELTA_MIN = 0.50
GATE9_VALIDATION_ORACLE_ACCURACY_REQUIRED = 1.0

GATE9_CHECKPOINTS_ADMITTED = "G9_CONTEXTUAL_CHECKPOINTS_ADMITTED"
GATE9_CHECKPOINT_ADMISSION_FAILED = "G9_CONTEXTUAL_CHECKPOINT_ADMISSION_FAILED"
GATE9_TRAINING_OUTCOMES = (
    GATE9_CHECKPOINTS_ADMITTED,
    GATE9_CHECKPOINT_ADMISSION_FAILED,
)

GATE9_RUNTIME_REQUIREMENTS = {
    "python": "3.11.9",
    "torch": "2.9.1+cu130",
    "numpy": "2.3.5",
    "device": "CUDA",
}


def _seed_index(seed: int) -> int:
    try:
        return GATE9_CHECKPOINT_SEEDS.index(seed)
    except ValueError as error:
        raise ValueError("Gate9 checkpoint seed is outside 0..2") from error


def _validate_affine_permutation(multiplier: int, modulus: int, label: str) -> None:
    if not 0 < multiplier < modulus or math.gcd(multiplier, modulus) != 1:
        raise ValueError(f"{label} is not invertible modulo its domain")


def training_operator_ordinal(seed: int, episode_index: int) -> int:
    index = _seed_index(seed)
    if not 0 <= episode_index < GATE9_TRAIN_EPISODES:
        raise ValueError("Gate9 training episode index lies outside frozen range")
    multiplier = GATE9_TRAIN_ORDER_MULTIPLIERS[index]
    offset = GATE9_TRAIN_ORDER_OFFSETS[index]
    _validate_affine_permutation(multiplier, GATE9_TRAIN_OPERATOR_COUNT, "Gate9 training order")
    return (multiplier * episode_index + offset) % GATE9_TRAIN_OPERATOR_COUNT


def inverse_training_episode_index(seed: int, operator_ordinal: int) -> int:
    index = _seed_index(seed)
    if not 0 <= operator_ordinal < GATE9_TRAIN_OPERATOR_COUNT:
        raise ValueError("Gate9 training operator ordinal lies outside frozen range")
    multiplier = GATE9_TRAIN_ORDER_MULTIPLIERS[index]
    offset = GATE9_TRAIN_ORDER_OFFSETS[index]
    inverse = pow(multiplier, -1, GATE9_TRAIN_OPERATOR_COUNT)
    return (inverse * (operator_ordinal - offset)) % GATE9_TRAIN_OPERATOR_COUNT


def training_query(seed: int, operator_ordinal: int) -> int:
    index = _seed_index(seed)
    if not 0 <= operator_ordinal < GATE9_TRAIN_OPERATOR_COUNT:
        raise ValueError("Gate9 training operator ordinal lies outside frozen range")
    query_index = (
        GATE9_TRAIN_QUERY_MULTIPLIERS[index] * operator_ordinal
        + GATE9_TRAIN_QUERY_OFFSETS[index]
    ) % GATE9_NOVEL_QUERY_COUNT
    return GATE9_NOVEL_QUERY_VALUES[query_index]


def validation_operator_ordinal(episode_index: int) -> int:
    if not 0 <= episode_index < GATE9_VALIDATION_EPISODES:
        raise ValueError("Gate9 validation episode index lies outside frozen range")
    _validate_affine_permutation(
        GATE9_VALIDATION_ORDER_MULTIPLIER,
        GATE9_VALIDATION_OPERATOR_COUNT,
        "Gate9 validation order",
    )
    return (
        GATE9_VALIDATION_ORDER_MULTIPLIER * episode_index
        + GATE9_VALIDATION_ORDER_OFFSET
    ) % GATE9_VALIDATION_OPERATOR_COUNT


def inverse_validation_episode_index(operator_ordinal: int) -> int:
    if not 0 <= operator_ordinal < GATE9_VALIDATION_OPERATOR_COUNT:
        raise ValueError("Gate9 validation operator ordinal lies outside frozen range")
    inverse = pow(GATE9_VALIDATION_ORDER_MULTIPLIER, -1, GATE9_VALIDATION_OPERATOR_COUNT)
    return (
        inverse * (operator_ordinal - GATE9_VALIDATION_ORDER_OFFSET)
    ) % GATE9_VALIDATION_OPERATOR_COUNT


def validation_query(operator_ordinal: int) -> int:
    if not 0 <= operator_ordinal < GATE9_VALIDATION_OPERATOR_COUNT:
        raise ValueError("Gate9 validation operator ordinal lies outside frozen range")
    query_index = (
        GATE9_VALIDATION_QUERY_MULTIPLIER * operator_ordinal
        + GATE9_VALIDATION_QUERY_OFFSET
    ) % GATE9_NOVEL_QUERY_COUNT
    return GATE9_NOVEL_QUERY_VALUES[query_index]


def learning_rate_at_step(step: int) -> float:
    if not 1 <= step <= GATE9_TRAIN_STEPS:
        raise ValueError("Gate9 learning-rate step lies outside 1..512")
    if step <= GATE9_WARMUP_STEPS:
        return GATE9_BASE_LEARNING_RATE * step / GATE9_WARMUP_STEPS
    progress = (step - GATE9_WARMUP_STEPS) / (
        GATE9_TRAIN_STEPS - GATE9_WARMUP_STEPS
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return GATE9_MIN_LEARNING_RATE + (
        GATE9_BASE_LEARNING_RATE - GATE9_MIN_LEARNING_RATE
    ) * cosine


def _valid_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class Gate9CheckpointValidationEvidence:
    seed: int
    initialization_seed: int
    checkpoint_step: int
    train_episodes: int
    unique_train_operators: int
    validation_episodes: int
    unique_validation_operators: int
    learned_parameter_count: int
    tensor_count: int
    checkpoint_sha256: str
    parameters_finite: bool
    final_train_loss: float
    validation_exact_accuracy: float
    validation_bit_accuracy: float
    shuffled_context_accuracy: float
    query_only_accuracy: float
    oracle_accuracy: float

    def validate(self) -> None:
        index = _seed_index(self.seed)
        if self.initialization_seed != GATE9_INITIALIZATION_SEEDS[index]:
            raise ValueError("Gate9 initialization seed drifted")
        if self.checkpoint_step != GATE9_CHECKPOINT_STEP:
            raise ValueError("Gate9 checkpoint is not the fixed final step")
        if (self.train_episodes, self.unique_train_operators) != (
            GATE9_TRAIN_EPISODES,
            GATE9_TRAIN_OPERATOR_COUNT,
        ):
            raise ValueError("Gate9 training coverage drifted")
        if (self.validation_episodes, self.unique_validation_operators) != (
            GATE9_VALIDATION_EPISODES,
            GATE9_VALIDATION_OPERATOR_COUNT,
        ):
            raise ValueError("Gate9 validation coverage drifted")
        if self.learned_parameter_count != GATE9_LEARNED_PARAMETER_COUNT:
            raise ValueError("Gate9 checkpoint learned-parameter count drifted")
        if self.tensor_count != GATE9_STATE_TENSOR_COUNT:
            raise ValueError("Gate9 checkpoint tensor count drifted")
        _valid_sha256(self.checkpoint_sha256, "Gate9 checkpoint identity")
        if not self.parameters_finite:
            raise ValueError("Gate9 checkpoint contains a non-finite parameter")
        if not math.isfinite(self.final_train_loss) or self.final_train_loss < 0.0:
            raise ValueError("Gate9 final training loss is invalid")
        for label, value in (
            ("validation exact accuracy", self.validation_exact_accuracy),
            ("validation bit accuracy", self.validation_bit_accuracy),
            ("shuffled-context accuracy", self.shuffled_context_accuracy),
            ("query-only accuracy", self.query_only_accuracy),
            ("oracle accuracy", self.oracle_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Gate9 {label} lies outside 0..1")

    def admission_passes(self) -> bool:
        self.validate()
        return (
            self.validation_exact_accuracy >= GATE9_VALIDATION_EXACT_ACCURACY_MIN
            and self.validation_bit_accuracy >= GATE9_VALIDATION_BIT_ACCURACY_MIN
            and self.validation_exact_accuracy - self.shuffled_context_accuracy
            > GATE9_VALIDATION_CONTEXT_DELTA_MIN
            and self.validation_exact_accuracy - self.query_only_accuracy
            > GATE9_VALIDATION_CONTEXT_DELTA_MIN
            and self.oracle_accuracy == GATE9_VALIDATION_ORACLE_ACCURACY_REQUIRED
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["admission_passes"] = self.admission_passes()
        return payload


def classify_gate9_checkpoint_admission(
    rows: tuple[Gate9CheckpointValidationEvidence, ...],
) -> str:
    if tuple(row.seed for row in rows) != GATE9_CHECKPOINT_SEEDS:
        raise ValueError("Gate9 checkpoint evidence must be ordered seeds 0, 1, 2")
    for row in rows:
        row.validate()
    if len({row.checkpoint_sha256 for row in rows}) != 3:
        raise ValueError("Gate9 three checkpoint identities must be distinct")
    return (
        GATE9_CHECKPOINTS_ADMITTED
        if all(row.admission_passes() for row in rows)
        else GATE9_CHECKPOINT_ADMISSION_FAILED
    )


def gate9_training_protocol_plan() -> dict[str, Any]:
    for multiplier in GATE9_TRAIN_ORDER_MULTIPLIERS:
        _validate_affine_permutation(
            multiplier, GATE9_TRAIN_OPERATOR_COUNT, "Gate9 training order"
        )
    _validate_affine_permutation(
        GATE9_VALIDATION_ORDER_MULTIPLIER,
        GATE9_VALIDATION_OPERATOR_COUNT,
        "Gate9 validation order",
    )
    return {
        "version": GATE9_TRAINING_PROTOCOL_VERSION,
        "status": GATE9_TRAINING_PROTOCOL_STATUS,
        "architecture_head": GATE9_ARCHITECTURE_HEAD,
        "operator_contract_head": GATE9_OPERATOR_CONTRACT_HEAD,
        "gate9_protocol_head": GATE9_PROTOCOL_HEAD,
        "checkpoint_seeds": list(GATE9_CHECKPOINT_SEEDS),
        "initialization_seeds": list(GATE9_INITIALIZATION_SEEDS),
        "learned_parameter_count": GATE9_LEARNED_PARAMETER_COUNT,
        "state_tensor_count": GATE9_STATE_TENSOR_COUNT,
        "state_tensor_shapes": {
            name: list(shape) for name, shape in GATE9_STATE_TENSOR_SHAPES.items()
        },
        "train": {
            "operator_counter_start": GATE9_TRAIN_OPERATOR_COUNTER_START,
            "operator_count": GATE9_TRAIN_OPERATOR_COUNT,
            "episodes_per_seed": GATE9_TRAIN_EPISODES,
            "batch_size": GATE9_TRAIN_BATCH_SIZE,
            "steps": GATE9_TRAIN_STEPS,
            "one_query_per_unique_operator": True,
            "query_excludes_support_inputs": True,
            "order_multipliers": list(GATE9_TRAIN_ORDER_MULTIPLIERS),
            "order_offsets": list(GATE9_TRAIN_ORDER_OFFSETS),
            "query_multipliers": list(GATE9_TRAIN_QUERY_MULTIPLIERS),
            "query_offsets": list(GATE9_TRAIN_QUERY_OFFSETS),
        },
        "validation": {
            "operator_counter_start": GATE9_VALIDATION_OPERATOR_COUNTER_START,
            "operator_count": GATE9_VALIDATION_OPERATOR_COUNT,
            "episodes": GATE9_VALIDATION_EPISODES,
            "batch_size": GATE9_VALIDATION_BATCH_SIZE,
            "batches": GATE9_VALIDATION_BATCHES,
            "shared_across_checkpoint_seeds": True,
            "query_excludes_support_inputs": True,
            "modes": list(GATE9_VALIDATION_MODES),
            "shuffled_context_namespace": GATE9_VALIDATION_SHUFFLE_NAMESPACE,
            "exact_accuracy_min": GATE9_VALIDATION_EXACT_ACCURACY_MIN,
            "bit_accuracy_min": GATE9_VALIDATION_BIT_ACCURACY_MIN,
            "context_delta_min_strict": GATE9_VALIDATION_CONTEXT_DELTA_MIN,
            "oracle_accuracy_required": GATE9_VALIDATION_ORACLE_ACCURACY_REQUIRED,
        },
        "novel_query_values": list(GATE9_NOVEL_QUERY_VALUES),
        "optimizer": {
            "name": GATE9_OPTIMIZER,
            "base_learning_rate": GATE9_BASE_LEARNING_RATE,
            "minimum_learning_rate": GATE9_MIN_LEARNING_RATE,
            "warmup_steps": GATE9_WARMUP_STEPS,
            "schedule": "linear_warmup_then_cosine_decay",
            "betas": list(GATE9_ADAM_BETAS),
            "epsilon": GATE9_ADAM_EPSILON,
            "weight_decay": GATE9_WEIGHT_DECAY,
            "gradient_clip_norm": GATE9_GRADIENT_CLIP_NORM,
        },
        "loss": GATE9_LOSS,
        "precision": GATE9_PRECISION,
        "amp_enabled": GATE9_AMP_ENABLED,
        "tf32_enabled": GATE9_TF32_ENABLED,
        "compile_enabled": GATE9_COMPILE_ENABLED,
        "deterministic_algorithms": GATE9_DETERMINISTIC_ALGORITHMS,
        "dataloader_workers": GATE9_DATALOADER_WORKERS,
        "checkpoint": {
            "selected_step": GATE9_CHECKPOINT_STEP,
            "selection": GATE9_CHECKPOINT_SELECTION,
            "required_fields": list(GATE9_CHECKPOINT_REQUIRED_FIELDS),
            "optimizer_state_in_selected_checkpoint": False,
            "retraining_after_failure_allowed": False,
            "seed_selection_allowed": False,
        },
        "runtime_requirements": GATE9_RUNTIME_REQUIREMENTS,
        "local_test_operator_access": False,
        "graph_test_operator_access": False,
        "scientific_assignment_key_access": False,
        "operator_generation_admitted": False,
        "training_execution_admitted": False,
        "optimizer_instantiation_admitted": False,
        "checkpoint_serialization_admitted": False,
        "checkpoint_loading_admitted": False,
        "scientific_test_generation_admitted": False,
        "scientific_execution_admitted": False,
        "result_classification_admitted": False,
    }
