"""Pre-exposure causal diagnostic protocol for the Gate-8 seed-0 checkpoint.

This module freezes checkpoint identity, diagnostic interventions, fresh training
addresses, evaluation surfaces, thresholds, and interpretation. It imports only
the Python standard library and opens no checkpoint load, Torch execution, world
generation, scientific-test access, or reference-model path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GATE8_SEED0_DIAGNOSTIC_PROTOCOL_VERSION = (
    "gate8-seed0-causal-diagnostic-protocol-v0"
)
GATE8_SEED0_DIAGNOSTIC_PROTOCOL_STATUS = (
    "GATE8_SEED0_CAUSAL_DIAGNOSTIC_PROTOCOL_FROZEN_EXECUTION_CLOSED"
)
GATE8_SEED0_RESULT_HEAD = "70e7e40149f9259d36b0e37ab17fc8c30370201e"
GATE8_TRAINING_EXECUTION_HEAD = "6c68b51741a30229b1be23d522d0009507c806d5"
GATE8_ARCHITECTURE_HEAD = "2afdcc9f13f138e97c7b3821cc2a5a77bd87cf0c"
GATE8_RUNTIME_HEAD = "1a2be148411bc71ba35fda12b035b724f06ec166"
GATE8_TRAINING_PROTOCOL_HEAD = "869791e5b44089f9c79447b8ae212ce830f8496a"

GATE8_SEED0_RESULT_SHA256 = (
    "5f477022ac45a80d8b05b112b3485e4112519ddef78e0bb23990a090e0cc92e2"
)
GATE8_SEED0_CHECKPOINT_SHA256 = (
    "4aca6bfde7fa82cd2c1fec3613c4cc59303788616f352e8d419c90662d7b9a1b"
)
GATE8_SEED0_MANIFEST_SHA256 = (
    "bb814b9ebb5116f0a13ff2ce130c5ad8e32ed4bd80453ddc167143b6cbf0bb8d"
)
GATE8_SEED = 0
GATE8_LEARNED_PARAMETER_COUNT = 19_649
GATE8_ORIGINAL_TRAINING_WORLDS = 262_144
GATE8_WORLD_BATCH_SIZE = 256
GATE8_VALIDATION_WORLDS_PER_CONDITION = 512
GATE8_VALIDATION_CONDITIONS = (
    (32, 4),
    (64, 4),
    (64, 8),
    (128, 4),
    (128, 8),
    (128, 16),
)

GATE8_PROBE_BASELINE = "baseline"
GATE8_PROBE_FORCED_ACTIVE = "forced_active"
GATE8_PROBE_MESSAGE_DECODE = "message_low4_decode"
GATE8_PROBE_FORCED_ACTIVE_MESSAGE_DECODE = "forced_active_message_low4_decode"
GATE8_RUNTIME_PROBES = (
    GATE8_PROBE_BASELINE,
    GATE8_PROBE_FORCED_ACTIVE,
    GATE8_PROBE_MESSAGE_DECODE,
    GATE8_PROBE_FORCED_ACTIVE_MESSAGE_DECODE,
)

GATE8_HEAD_ONLY_STEPS = 256
GATE8_HEAD_ONLY_CHECKPOINT_STEPS = (64, 128, 192, 256)
GATE8_HEAD_ONLY_WORLD_START = GATE8_ORIGINAL_TRAINING_WORLDS
GATE8_HEAD_ONLY_WORLD_COUNT = GATE8_HEAD_ONLY_STEPS * GATE8_WORLD_BATCH_SIZE
GATE8_HEAD_ONLY_WORLD_END_EXCLUSIVE = (
    GATE8_HEAD_ONLY_WORLD_START + GATE8_HEAD_ONLY_WORLD_COUNT
)
GATE8_HEAD_ONLY_TRAINABLE_PREFIXES = (
    "message_head.",
    "activity_head.",
    "answer_head.",
)
GATE8_HEAD_ONLY_OPTIMIZER = "adamw"
GATE8_HEAD_ONLY_LEARNING_RATE = 1.0e-3
GATE8_HEAD_ONLY_BETAS = (0.9, 0.95)
GATE8_HEAD_ONLY_EPSILON = 1.0e-8
GATE8_HEAD_ONLY_WEIGHT_DECAY = 0.0
GATE8_HEAD_ONLY_GRADIENT_CLIP_NORM = 1.0

GATE8_FULL_RESUME_STEPS = 512
GATE8_FULL_RESUME_CHECKPOINT_STEPS = (128, 256, 384, 512)
GATE8_FULL_RESUME_WORLD_START = GATE8_HEAD_ONLY_WORLD_END_EXCLUSIVE
GATE8_FULL_RESUME_WORLD_COUNT = GATE8_FULL_RESUME_STEPS * GATE8_WORLD_BATCH_SIZE
GATE8_FULL_RESUME_WORLD_END_EXCLUSIVE = (
    GATE8_FULL_RESUME_WORLD_START + GATE8_FULL_RESUME_WORLD_COUNT
)
GATE8_FULL_RESUME_OPTIMIZER = "adamw"
GATE8_FULL_RESUME_INITIAL_LEARNING_RATE = 3.0e-4
GATE8_FULL_RESUME_MINIMUM_LEARNING_RATE = 3.0e-5
GATE8_FULL_RESUME_BETAS = (0.9, 0.95)
GATE8_FULL_RESUME_EPSILON = 1.0e-8
GATE8_FULL_RESUME_WEIGHT_DECAY = 1.0e-4
GATE8_FULL_RESUME_GRADIENT_CLIP_NORM = 1.0
GATE8_FULL_RESUME_SCHEDULE = "cosine_without_warmup"

GATE8_BASELINE_MESSAGE_ACCURACY = 0.9167085535386029
GATE8_BASELINE_ANSWER_ACCURACY = 0.6418421128216911
GATE8_BASELINE_ACTIVITY_ACCURACY = 0.9941442153033089
GATE8_BASELINE_MEAN_TARGET_ACCURACY = 0.4036458333333333
GATE8_BASELINE_MIN_TARGET_ACCURACY = 0.19921875
GATE8_BASELINE_MESSAGE_ROOT_INVARIANCE = 0.88427734375
GATE8_BASELINE_ANSWER_ROOT_INVARIANCE = 0.77587890625

GATE8_MATERIAL_RUNTIME_DELTA = 0.02
GATE8_HEAD_ONLY_MESSAGE_SUFFICIENCY = 0.995
GATE8_HEAD_ONLY_ANSWER_SUFFICIENCY = 0.99
GATE8_HEAD_ONLY_ACTIVITY_SUFFICIENCY = 0.999
GATE8_HEAD_ONLY_ROOT_INVARIANCE = 0.99
GATE8_RESUME_MESSAGE_GAIN = 0.03
GATE8_RESUME_MEAN_TARGET_GAIN = 0.10
GATE8_PERSISTENT_ROOT_INTERFERENCE = 0.95


@dataclass(frozen=True, slots=True)
class Gate8RuntimeProbeMetrics:
    probe: str
    mean_target_accuracy: float
    minimum_target_accuracy: float
    condition_target_accuracies: tuple[float, ...]

    def validate(self) -> None:
        if self.probe not in GATE8_RUNTIME_PROBES:
            raise ValueError("Gate8 diagnostic runtime probe is unknown")
        if len(self.condition_target_accuracies) != len(GATE8_VALIDATION_CONDITIONS):
            raise ValueError("Gate8 diagnostic condition metrics are incomplete")
        for value in (
            self.mean_target_accuracy,
            self.minimum_target_accuracy,
            *self.condition_target_accuracies,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Gate8 diagnostic accuracy is outside 0..1")
        observed_mean = sum(self.condition_target_accuracies) / len(
            self.condition_target_accuracies
        )
        if abs(observed_mean - self.mean_target_accuracy) > 1.0e-12:
            raise ValueError("Gate8 diagnostic mean target accuracy is inconsistent")
        if abs(min(self.condition_target_accuracies) - self.minimum_target_accuracy) > 1.0e-12:
            raise ValueError("Gate8 diagnostic minimum target accuracy is inconsistent")


@dataclass(frozen=True, slots=True)
class Gate8TrainingProbeMetrics:
    probe: str
    step: int
    message_accuracy: float
    answer_accuracy: float
    activity_accuracy: float
    mean_target_accuracy: float
    minimum_target_accuracy: float
    message_root_invariance: float
    answer_root_invariance: float

    def validate(self) -> None:
        valid_steps = {
            "head_only": GATE8_HEAD_ONLY_CHECKPOINT_STEPS,
            "full_resume": GATE8_FULL_RESUME_CHECKPOINT_STEPS,
        }
        if self.probe not in valid_steps:
            raise ValueError("Gate8 diagnostic training probe is unknown")
        if self.step not in valid_steps[self.probe]:
            raise ValueError("Gate8 diagnostic checkpoint step is not preregistered")
        for value in (
            self.message_accuracy,
            self.answer_accuracy,
            self.activity_accuracy,
            self.mean_target_accuracy,
            self.minimum_target_accuracy,
            self.message_root_invariance,
            self.answer_root_invariance,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Gate8 diagnostic metric is outside 0..1")


@dataclass(frozen=True, slots=True)
class Gate8DiagnosticFindings:
    activity_gate_material: bool
    answer_head_material: bool
    frozen_core_linearly_sufficient: bool
    continued_optimization_effective: bool
    core_interference_persists: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def gate8_classify_diagnostic(
    *,
    runtime_rows: tuple[Gate8RuntimeProbeMetrics, ...],
    head_only: Gate8TrainingProbeMetrics,
    full_resume: Gate8TrainingProbeMetrics,
) -> Gate8DiagnosticFindings:
    if tuple(row.probe for row in runtime_rows) != GATE8_RUNTIME_PROBES:
        raise ValueError("Gate8 diagnostic runtime rows are incomplete or reordered")
    for row in runtime_rows:
        row.validate()
    head_only.validate()
    full_resume.validate()
    if head_only.probe != "head_only" or head_only.step != GATE8_HEAD_ONLY_STEPS:
        raise ValueError("Gate8 diagnostic requires the final head-only checkpoint")
    if full_resume.probe != "full_resume" or full_resume.step != GATE8_FULL_RESUME_STEPS:
        raise ValueError("Gate8 diagnostic requires the final full-resume checkpoint")

    by_probe = {row.probe: row for row in runtime_rows}
    baseline = by_probe[GATE8_PROBE_BASELINE]
    forced_active = by_probe[GATE8_PROBE_FORCED_ACTIVE]
    message_decode = by_probe[GATE8_PROBE_MESSAGE_DECODE]

    return Gate8DiagnosticFindings(
        activity_gate_material=(
            forced_active.mean_target_accuracy - baseline.mean_target_accuracy
            >= GATE8_MATERIAL_RUNTIME_DELTA
        ),
        answer_head_material=(
            message_decode.mean_target_accuracy - baseline.mean_target_accuracy
            >= GATE8_MATERIAL_RUNTIME_DELTA
        ),
        frozen_core_linearly_sufficient=(
            head_only.message_accuracy >= GATE8_HEAD_ONLY_MESSAGE_SUFFICIENCY
            and head_only.answer_accuracy >= GATE8_HEAD_ONLY_ANSWER_SUFFICIENCY
            and head_only.activity_accuracy >= GATE8_HEAD_ONLY_ACTIVITY_SUFFICIENCY
            and head_only.message_root_invariance >= GATE8_HEAD_ONLY_ROOT_INVARIANCE
            and head_only.answer_root_invariance >= GATE8_HEAD_ONLY_ROOT_INVARIANCE
        ),
        continued_optimization_effective=(
            full_resume.message_accuracy - GATE8_BASELINE_MESSAGE_ACCURACY
            >= GATE8_RESUME_MESSAGE_GAIN
            and full_resume.mean_target_accuracy - GATE8_BASELINE_MEAN_TARGET_ACCURACY
            >= GATE8_RESUME_MEAN_TARGET_GAIN
        ),
        core_interference_persists=(
            full_resume.message_root_invariance < GATE8_PERSISTENT_ROOT_INTERFERENCE
            or full_resume.answer_root_invariance < GATE8_PERSISTENT_ROOT_INTERFERENCE
        ),
    )


def gate8_seed0_diagnostic_protocol_plan() -> dict[str, Any]:
    return {
        "version": GATE8_SEED0_DIAGNOSTIC_PROTOCOL_VERSION,
        "status": GATE8_SEED0_DIAGNOSTIC_PROTOCOL_STATUS,
        "result_head": GATE8_SEED0_RESULT_HEAD,
        "training_execution_head": GATE8_TRAINING_EXECUTION_HEAD,
        "checkpoint_sha256": GATE8_SEED0_CHECKPOINT_SHA256,
        "result_sha256": GATE8_SEED0_RESULT_SHA256,
        "manifest_sha256": GATE8_SEED0_MANIFEST_SHA256,
        "seed": GATE8_SEED,
        "learned_parameter_count": GATE8_LEARNED_PARAMETER_COUNT,
        "validation_conditions": [list(value) for value in GATE8_VALIDATION_CONDITIONS],
        "validation_worlds_per_condition": GATE8_VALIDATION_WORLDS_PER_CONDITION,
        "runtime_probes": list(GATE8_RUNTIME_PROBES),
        "head_only": {
            "steps": GATE8_HEAD_ONLY_STEPS,
            "checkpoint_steps": list(GATE8_HEAD_ONLY_CHECKPOINT_STEPS),
            "fresh_world_range": [
                GATE8_HEAD_ONLY_WORLD_START,
                GATE8_HEAD_ONLY_WORLD_END_EXCLUSIVE,
            ],
            "trainable_prefixes": list(GATE8_HEAD_ONLY_TRAINABLE_PREFIXES),
            "optimizer": GATE8_HEAD_ONLY_OPTIMIZER,
            "learning_rate": GATE8_HEAD_ONLY_LEARNING_RATE,
            "weight_decay": GATE8_HEAD_ONLY_WEIGHT_DECAY,
        },
        "full_resume": {
            "steps": GATE8_FULL_RESUME_STEPS,
            "checkpoint_steps": list(GATE8_FULL_RESUME_CHECKPOINT_STEPS),
            "fresh_world_range": [
                GATE8_FULL_RESUME_WORLD_START,
                GATE8_FULL_RESUME_WORLD_END_EXCLUSIVE,
            ],
            "optimizer": GATE8_FULL_RESUME_OPTIMIZER,
            "initial_learning_rate": GATE8_FULL_RESUME_INITIAL_LEARNING_RATE,
            "minimum_learning_rate": GATE8_FULL_RESUME_MINIMUM_LEARNING_RATE,
            "schedule": GATE8_FULL_RESUME_SCHEDULE,
        },
        "finding_names": [
            "activity_gate_material",
            "answer_head_material",
            "frozen_core_linearly_sufficient",
            "continued_optimization_effective",
            "core_interference_persists",
        ],
        "execution_admitted": False,
        "scientific_test_worlds_admitted": False,
        "seeds_1_2_admitted": False,
        "reference_model_admitted": False,
    }
