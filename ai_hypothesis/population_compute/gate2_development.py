"""Development-only training and paired evaluation for Gate 2.

This module deliberately keeps confirmation locked.  It may be used to determine whether the
frozen persistent-state substrate is learnable and to select a coherent training recipe, but no
result produced here is confirmation evidence.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .gate2_persistent_model import (
    Gate2PersistentModelConfig,
    Gate2PersistentStateModel,
    build_gate2_tensor_batch,
    decode_gate2_payload_logits,
    parallel_persistent_forward,
)
from .gate2_persistent_state_capacity import (
    GATE2_ENTITY_COUNTS,
    GATE2_PAYLOAD_BITS,
    Gate2ControlMode,
    Gate2World,
    gate2_population_widths,
    generate_gate2_world,
)

GATE2_DEVELOPMENT_EXPERIMENT_VERSION = "gate2-persistent-state-development-v0"
GATE2_TRAINING_SEED_LIMIT = 1 << 30
GATE2_DEVELOPMENT_SEED_START = 1 << 30
GATE2_DEVELOPMENT_SEED_LIMIT = 2 << 30
GATE2_CONFIRMATION_SEED_START = 2 << 30
GATE2_CONFIRMATION_SEED_LIMIT = 3 << 30


@dataclass(frozen=True, slots=True)
class Gate2TrainingConfig:
    """Development recipe; intentionally not a frozen confirmation recipe yet."""

    steps: int = 2_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    model: Gate2PersistentModelConfig = Gate2PersistentModelConfig()

    def validate(self) -> None:
        self.model.validate()
        if self.steps <= 0:
            raise ValueError("training steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("training batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")


@dataclass(frozen=True, slots=True)
class Gate2TrainingSummary:
    training_seed: int
    steps: int
    examples_seen: int
    initial_loss: float
    final_loss: float
    mean_last_50_loss: float
    learned_parameter_count: int
    parameter_fingerprint: str
    stable_training_condition_count: int


@dataclass(frozen=True, slots=True)
class Gate2ConditionEvaluation:
    entity_count: int
    width: int
    mode: Gate2ControlMode
    world_count: int
    exact_solve_rate: float
    bit_accuracy: float
    collision_load: int
    learned_updates_per_world: int
    inspected_entities_per_world: int
    inspected_observations_per_world: int
    learned_parameter_count: int
    parameter_fingerprint: str
    world_seeds: tuple[int, ...]
    solved_by_world: tuple[bool, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_count": self.entity_count,
            "width": self.width,
            "mode": self.mode.value,
            "world_count": self.world_count,
            "exact_solve_rate": self.exact_solve_rate,
            "bit_accuracy": self.bit_accuracy,
            "collision_load": self.collision_load,
            "learned_updates_per_world": self.learned_updates_per_world,
            "inspected_entities_per_world": self.inspected_entities_per_world,
            "inspected_observations_per_world": self.inspected_observations_per_world,
            "learned_parameter_count": self.learned_parameter_count,
            "parameter_fingerprint": self.parameter_fingerprint,
            "world_seeds": list(self.world_seeds),
            "solved_by_world": list(self.solved_by_world),
        }


@dataclass(frozen=True, slots=True)
class Gate2PairedSummary:
    comparison: str
    entity_count: int
    treatment_width: int
    reference_width: int
    treatment_mode: Gate2ControlMode
    reference_mode: Gate2ControlMode
    world_count: int
    treatment_only: int
    reference_only: int
    both_solved: int
    neither_solved: int
    exact_solve_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["treatment_mode"] = self.treatment_mode.value
        payload["reference_mode"] = self.reference_mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate2DevelopmentResult:
    experiment_version: str
    evaluation_split: str
    confirmation_opened: bool
    training: Gate2TrainingSummary
    training_config: Gate2TrainingConfig
    evaluation_world_count: int
    evaluation_batch_size: int
    conditions: tuple[Gate2ConditionEvaluation, ...]
    paired_summaries: tuple[Gate2PairedSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_version": self.experiment_version,
            "evaluation_split": self.evaluation_split,
            "confirmation_opened": self.confirmation_opened,
            "training": asdict(self.training),
            "training_config": {
                "steps": self.training_config.steps,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "weight_decay": self.training_config.weight_decay,
                "gradient_clip_norm": self.training_config.gradient_clip_norm,
                "model": asdict(self.training_config.model),
            },
            "evaluation_world_count": self.evaluation_world_count,
            "evaluation_batch_size": self.evaluation_batch_size,
            "conditions": [row.to_dict() for row in self.conditions],
            "paired_summaries": [row.to_dict() for row in self.paired_summaries],
            "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
            "interpretation_note": (
                "This artifact is development evidence only. Confirmation worlds and new "
                "confirmation training seeds remain locked until the architecture, optimizer, "
                "evaluation matrix, numerical rule and confirmation decision rule are frozen."
            ),
        }


def gate2_stable_training_conditions() -> tuple[tuple[int, int], ...]:
    return tuple(
        (entity_count, width)
        for entity_count in GATE2_ENTITY_COUNTS
        for width in gate2_population_widths(entity_count)
    )


def train_gate2_development_model(
    *,
    training_seed: int,
    config: Gate2TrainingConfig = Gate2TrainingConfig(),
    device: torch.device | str = "cpu",
) -> tuple[Gate2PersistentStateModel, Gate2TrainingSummary]:
    """Train one shared checkpoint only on the primary stable-persistent mechanism."""

    config.validate()
    if not 0 <= training_seed < GATE2_TRAINING_SEED_LIMIT:
        raise ValueError("development training_seed must stay inside the reserved training domain")

    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = Gate2PersistentStateModel(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    conditions = gate2_stable_training_conditions()
    examples_seen = 0
    model.train()

    for step in range(config.steps):
        entity_count, width = conditions[step % len(conditions)]
        worlds = tuple(
            generate_gate2_world(
                seed=_training_world_seed(
                    training_seed=training_seed,
                    step=step,
                    sample_index=sample_index,
                    entity_count=entity_count,
                    width=width,
                ),
                entity_count=entity_count,
            )
            for sample_index in range(config.batch_size)
        )
        batch = build_gate2_tensor_batch(
            worlds,
            width=width,
            mode=Gate2ControlMode.STABLE_PERSISTENT,
            device=target_device,
        )
        output = parallel_persistent_forward(model, batch)
        targets = _payload_targets(batch.answer_payloads, dtype=output.logits.dtype)
        loss = loss_fn(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Gate-2 development training produced non-finite loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        losses.append(float(loss.detach().item()))
        examples_seen += config.batch_size

    summary = Gate2TrainingSummary(
        training_seed=training_seed,
        steps=config.steps,
        examples_seen=examples_seen,
        initial_loss=losses[0],
        final_loss=losses[-1],
        mean_last_50_loss=sum(losses[-50:]) / len(losses[-50:]),
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
        stable_training_condition_count=len(conditions),
    )
    return model, summary


def evaluate_gate2_split(
    model: Gate2PersistentStateModel,
    *,
    split: str = "development",
    world_count: int = 1_000,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    allow_confirmation: bool = False,
    bootstrap_samples: int = 2_000,
) -> tuple[tuple[Gate2ConditionEvaluation, ...], tuple[Gate2PairedSummary, ...]]:
    """Evaluate one unchanged checkpoint across the full equal-information matrix."""

    if world_count <= 0:
        raise ValueError("evaluation world_count must be positive")
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if split == "confirmation" and not allow_confirmation:
        raise ValueError("Gate-2 confirmation split is locked unless allow_confirmation=True")
    if split not in {"development", "confirmation"}:
        raise ValueError("Gate-2 evaluation split must be development or confirmation")

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    expected_count = model.trainable_parameter_count()
    expected_fingerprint = model.parameter_fingerprint()
    rows: list[Gate2ConditionEvaluation] = []

    for entity_count in GATE2_ENTITY_COUNTS:
        worlds = generate_gate2_split_worlds(
            split=split,
            entity_count=entity_count,
            world_count=world_count,
            allow_confirmation=allow_confirmation,
        )
        for width in gate2_population_widths(entity_count):
            for mode in Gate2ControlMode:
                row = evaluate_gate2_condition(
                    model,
                    worlds,
                    width=width,
                    mode=mode,
                    batch_size=batch_size,
                    device=target_device,
                )
                if row.learned_parameter_count != expected_count:
                    raise RuntimeError("Gate-2 evaluation changed learned parameter count")
                if row.parameter_fingerprint != expected_fingerprint:
                    raise RuntimeError("Gate-2 evaluation changed checkpoint fingerprint")
                rows.append(row)

    if model.parameter_fingerprint() != expected_fingerprint:
        raise RuntimeError("Gate-2 evaluation mutated the checkpoint")

    paired = build_gate2_paired_summaries(rows, bootstrap_samples=bootstrap_samples)
    return tuple(rows), paired


def evaluate_gate2_condition(
    model: Gate2PersistentStateModel,
    worlds: Sequence[Gate2World],
    *,
    width: int,
    mode: Gate2ControlMode,
    batch_size: int,
    device: torch.device | str,
) -> Gate2ConditionEvaluation:
    if not worlds:
        raise ValueError("Gate-2 evaluation requires at least one world")
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    entity_count = worlds[0].entity_count
    if any(world.entity_count != entity_count for world in worlds):
        raise ValueError("one Gate-2 evaluation condition must use one entity count")

    target_device = torch.device(device)
    fingerprint = model.parameter_fingerprint()
    parameter_count = model.trainable_parameter_count()
    solved: list[bool] = []
    correct_bits = 0
    total_bits = 0

    with torch.inference_mode():
        for offset in range(0, len(worlds), batch_size):
            world_batch = worlds[offset : offset + batch_size]
            batch = build_gate2_tensor_batch(
                world_batch,
                width=width,
                mode=mode,
                device=target_device,
            )
            output = parallel_persistent_forward(model, batch)
            predicted = decode_gate2_payload_logits(output.logits)
            target = batch.answer_payloads
            batch_solved = predicted.eq(target)
            solved.extend(bool(value) for value in batch_solved.cpu().tolist())

            xor_values = torch.bitwise_xor(predicted.to(torch.int64), target.to(torch.int64))
            for value in xor_values.cpu().tolist():
                correct_bits += GATE2_PAYLOAD_BITS - int(value).bit_count()
            total_bits += len(world_batch) * GATE2_PAYLOAD_BITS

            if output.telemetry.learned_updates_per_sample != 8 * entity_count:
                raise RuntimeError("Gate-2 evaluator observed invalid learned-update accounting")

    if model.parameter_fingerprint() != fingerprint:
        raise RuntimeError("Gate-2 condition evaluation mutated the checkpoint")

    return Gate2ConditionEvaluation(
        entity_count=entity_count,
        width=width,
        mode=mode,
        world_count=len(worlds),
        exact_solve_rate=sum(solved) / len(solved),
        bit_accuracy=correct_bits / total_bits,
        collision_load=entity_count // width,
        learned_updates_per_world=8 * entity_count,
        inspected_entities_per_world=entity_count,
        inspected_observations_per_world=8 * entity_count,
        learned_parameter_count=parameter_count,
        parameter_fingerprint=fingerprint,
        world_seeds=tuple(world.seed for world in worlds),
        solved_by_world=tuple(solved),
    )


def generate_gate2_split_worlds(
    *,
    split: str,
    entity_count: int,
    world_count: int,
    allow_confirmation: bool = False,
) -> tuple[Gate2World, ...]:
    if world_count <= 0:
        raise ValueError("world_count must be positive")
    if split == "development":
        start, limit = GATE2_DEVELOPMENT_SEED_START, GATE2_DEVELOPMENT_SEED_LIMIT
    elif split == "confirmation":
        if not allow_confirmation:
            raise ValueError("Gate-2 confirmation worlds are locked")
        start, limit = GATE2_CONFIRMATION_SEED_START, GATE2_CONFIRMATION_SEED_LIMIT
    else:
        raise ValueError("unknown Gate-2 split")
    if start + world_count > limit:
        raise ValueError("requested Gate-2 worlds exceed the reserved split domain")
    return tuple(
        generate_gate2_world(seed=start + offset, entity_count=entity_count)
        for offset in range(world_count)
    )


def build_gate2_paired_summaries(
    rows: Iterable[Gate2ConditionEvaluation],
    *,
    bootstrap_samples: int = 2_000,
) -> tuple[Gate2PairedSummary, ...]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    index = {(row.entity_count, row.width, row.mode): row for row in rows}
    summaries: list[Gate2PairedSummary] = []

    for entity_count in GATE2_ENTITY_COUNTS:
        widths = gate2_population_widths(entity_count)
        reference = index[(entity_count, 1, Gate2ControlMode.STABLE_PERSISTENT)]
        for width in widths[1:]:
            treatment = index[(entity_count, width, Gate2ControlMode.STABLE_PERSISTENT)]
            summaries.append(
                _paired_summary(
                    "stable_width_vs_width1",
                    treatment,
                    reference,
                    bootstrap_samples=bootstrap_samples,
                )
            )
        for width in widths:
            stable = index[(entity_count, width, Gate2ControlMode.STABLE_PERSISTENT)]
            reshuffled = index[(entity_count, width, Gate2ControlMode.RESHUFFLED_LOCALITY)]
            reset = index[(entity_count, width, Gate2ControlMode.RESET_STATE)]
            summaries.append(
                _paired_summary(
                    "stable_vs_reshuffled",
                    stable,
                    reshuffled,
                    bootstrap_samples=bootstrap_samples,
                )
            )
            summaries.append(
                _paired_summary(
                    "stable_vs_reset",
                    stable,
                    reset,
                    bootstrap_samples=bootstrap_samples,
                )
            )
    return tuple(summaries)


def run_gate2_development(
    *,
    training_seed: int,
    training_config: Gate2TrainingConfig = Gate2TrainingConfig(),
    evaluation_world_count: int = 1_000,
    evaluation_batch_size: int = 64,
    bootstrap_samples: int = 2_000,
    device: torch.device | str = "cpu",
) -> tuple[Gate2PersistentStateModel, Gate2DevelopmentResult]:
    model, training = train_gate2_development_model(
        training_seed=training_seed,
        config=training_config,
        device=device,
    )
    conditions, paired = evaluate_gate2_split(
        model,
        split="development",
        world_count=evaluation_world_count,
        batch_size=evaluation_batch_size,
        device=device,
        allow_confirmation=False,
        bootstrap_samples=bootstrap_samples,
    )
    result = Gate2DevelopmentResult(
        experiment_version=GATE2_DEVELOPMENT_EXPERIMENT_VERSION,
        evaluation_split="development",
        confirmation_opened=False,
        training=training,
        training_config=training_config,
        evaluation_world_count=evaluation_world_count,
        evaluation_batch_size=evaluation_batch_size,
        conditions=conditions,
        paired_summaries=paired,
    )
    return model, result


def _paired_summary(
    comparison: str,
    treatment: Gate2ConditionEvaluation,
    reference: Gate2ConditionEvaluation,
    *,
    bootstrap_samples: int,
) -> Gate2PairedSummary:
    if treatment.entity_count != reference.entity_count:
        raise ValueError("paired Gate-2 rows must use the same entity count")
    if treatment.world_seeds != reference.world_seeds:
        raise ValueError("paired Gate-2 rows must use identical worlds in identical order")
    if treatment.learned_updates_per_world != reference.learned_updates_per_world:
        raise ValueError("paired Gate-2 rows must use identical learned work")
    if treatment.inspected_observations_per_world != reference.inspected_observations_per_world:
        raise ValueError("paired Gate-2 rows must inspect identical information volume")

    pairs = tuple(zip(treatment.solved_by_world, reference.solved_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = len(pairs) - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    delta = sum(differences) / len(differences)
    ci_low, ci_high = _paired_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        seed=_bootstrap_seed(comparison, treatment, reference),
    )
    return Gate2PairedSummary(
        comparison=comparison,
        entity_count=treatment.entity_count,
        treatment_width=treatment.width,
        reference_width=reference.width,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        world_count=len(pairs),
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_solved=both,
        neither_solved=neither,
        exact_solve_delta=delta,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
    )


def _paired_bootstrap_ci(
    differences: Sequence[int],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("paired bootstrap requires at least one world")
    rng = random.Random(seed)
    count = len(differences)
    estimates = []
    for _ in range(samples):
        total = 0
        for _ in range(count):
            total += differences[rng.randrange(count)]
        estimates.append(total / count)
    estimates.sort()
    low_index = int(math.floor(0.025 * (samples - 1)))
    high_index = int(math.ceil(0.975 * (samples - 1)))
    return estimates[low_index], estimates[high_index]


def _payload_targets(payloads: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    shifts = torch.arange(GATE2_PAYLOAD_BITS, device=payloads.device, dtype=torch.int64)
    return ((payloads.to(torch.int64).unsqueeze(-1) >> shifts) & 1).to(dtype=dtype)


def _training_world_seed(
    *,
    training_seed: int,
    step: int,
    sample_index: int,
    entity_count: int,
    width: int,
) -> int:
    digest = hashlib.sha256(
        f"gate2-training-world-v0:{training_seed}:{step}:{sample_index}:{entity_count}:{width}".encode(
            "ascii"
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") % GATE2_TRAINING_SEED_LIMIT


def _bootstrap_seed(
    comparison: str,
    treatment: Gate2ConditionEvaluation,
    reference: Gate2ConditionEvaluation,
) -> int:
    digest = hashlib.sha256(
        (
            f"gate2-bootstrap-v0:{comparison}:{treatment.entity_count}:"
            f"{treatment.width}:{treatment.mode.value}:{reference.width}:{reference.mode.value}"
        ).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")
