"""Canonical train-once/evaluate-many runner for repaired collective-relay protocol v1."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from .collective_relay import (
    COLLECTIVE_RELAY_VERSION,
    RELAY_DIFFICULTIES,
    RelayWorld,
    generate_relay_dataset,
    relay_scope_thresholds,
)
from .contract import (
    DEVELOPMENT_POPULATION_SIZES,
    CommunicationMode,
    CurveAssessment,
    PopulationCondition,
    PopulationRunMetrics,
    assess_scaling_curve,
)
from .relay_experiment import (
    RelayEvaluationResult,
    RelayScopeCohortResult,
    split_seed_limit,
    split_seed_start,
    training_world_batch,
)
from .relay_model import (
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
)
from .relay_protocol_v1 import (
    DEFAULT_GATE_SUPERVISION_WEIGHT,
    RELAY_PROTOCOL_VERSION,
    forward_relay_v1,
    gate_supervision_loss,
)


RELAY_EXPERIMENT_V1 = "population-compute-relay-training-v1"


@dataclass(frozen=True, slots=True)
class RelayTrainingConfigV1:
    steps: int = 2_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    gate_supervision_weight: float = DEFAULT_GATE_SUPERVISION_WEIGHT
    model: RelayPopulationConfig = RelayPopulationConfig()

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
        if (
            not math.isfinite(self.gate_supervision_weight)
            or self.gate_supervision_weight < 0.0
        ):
            raise ValueError("gate_supervision_weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RelayTrainingSummaryV1:
    training_seed: int
    steps: int
    examples_seen: int
    initial_total_loss: float
    final_total_loss: float
    mean_last_50_total_loss: float
    final_relay_loss: float
    final_gate_loss: float
    learned_parameter_count: int
    parameter_fingerprint: str


@dataclass(frozen=True, slots=True)
class RelayDevelopmentResultV1:
    experiment_version: str
    protocol_version: str
    benchmark_version: str
    evaluation_split: str
    confirmation_opened: bool
    training: RelayTrainingSummaryV1
    training_config: RelayTrainingConfigV1
    evaluation_world_count: int
    evaluation_batch_size: int
    evaluations: tuple[RelayEvaluationResult, ...]
    assessments: tuple[tuple[str, CurveAssessment], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_version": self.experiment_version,
            "protocol_version": self.protocol_version,
            "benchmark_version": self.benchmark_version,
            "evaluation_split": self.evaluation_split,
            "confirmation_opened": self.confirmation_opened,
            "training": asdict(self.training),
            "training_config": {
                "steps": self.training_config.steps,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "weight_decay": self.training_config.weight_decay,
                "gradient_clip_norm": self.training_config.gradient_clip_norm,
                "gate_supervision_weight": self.training_config.gate_supervision_weight,
                "communication_mode": CommunicationMode.SPARSE_SHARED_V1.value,
                "model": asdict(self.training_config.model),
            },
            "evaluation_world_count": self.evaluation_world_count,
            "evaluation_batch_size": self.evaluation_batch_size,
            "evaluations": [row.to_dict() for row in self.evaluations],
            "assessments": [
                {
                    "difficulty": difficulty,
                    "population_sizes": list(assessment.population_sizes),
                    "solve_rates": list(assessment.solve_rates),
                    "information_complete_rates": list(
                        assessment.information_complete_rates
                    ),
                    "solve_rates_given_information_complete": list(
                        assessment.solve_rates_given_information_complete
                    ),
                    "endpoint_gain": assessment.endpoint_gain,
                    "nondecreasing_steps": assessment.nondecreasing_steps,
                    "communication_endpoint_advantage": (
                        assessment.communication_endpoint_advantage
                    ),
                    "passes_per_curve_scaling_signal": assessment.passes_scaling_signal,
                    "reasons": list(assessment.reasons),
                }
                for difficulty, assessment in self.assessments
            ],
            "interpretation_note": (
                "Development assessments are diagnostic only. Frozen Gate-v0 acceptance "
                "requires untouched confirmation worlds across at least three independent "
                "training seeds. Serial schedule equivalence must be reported separately."
            ),
        }


def train_relay_model_v1(
    *,
    training_seed: int,
    config: RelayTrainingConfigV1 = RelayTrainingConfigV1(),
    device: torch.device | str = "cpu",
) -> tuple[RelayPopulationModel, RelayTrainingSummaryV1]:
    """Train one shared checkpoint with normalized transport and selector supervision."""

    config.validate()
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = RelayPopulationModel(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    relay_loss_fn = nn.BCEWithLogitsLoss()
    total_losses: list[float] = []
    final_relay_loss = 0.0
    final_gate_loss = 0.0
    examples_seen = 0
    model.train()

    for step in range(config.steps):
        difficulty = RELAY_DIFFICULTIES[step % len(RELAY_DIFFICULTIES)]
        thresholds = relay_scope_thresholds(difficulty)
        threshold_cycle = step // len(RELAY_DIFFICULTIES)
        active_workers = thresholds[threshold_cycle % len(thresholds)]
        worlds = training_world_batch(
            training_seed=training_seed,
            step=step,
            difficulty=difficulty,
            active_workers=active_workers,
            batch_size=config.batch_size,
        )
        batch = build_relay_tensor_batch(
            worlds,
            active_workers=active_workers,
            device=target_device,
        )
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError("relay v1 training batch contains incomplete worlds")

        output = forward_relay_v1(model, batch)
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        relay_loss = relay_loss_fn(output.logits, targets)
        selector_loss = gate_supervision_loss(model, batch, worlds)
        total_loss = relay_loss + config.gate_supervision_weight * selector_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError("relay v1 training produced non-finite loss")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        total_losses.append(float(total_loss.detach().item()))
        final_relay_loss = float(relay_loss.detach().item())
        final_gate_loss = float(selector_loss.detach().item())
        examples_seen += config.batch_size

    summary = RelayTrainingSummaryV1(
        training_seed=training_seed,
        steps=config.steps,
        examples_seen=examples_seen,
        initial_total_loss=total_losses[0],
        final_total_loss=total_losses[-1],
        mean_last_50_total_loss=(
            sum(total_losses[-50:]) / len(total_losses[-50:])
        ),
        final_relay_loss=final_relay_loss,
        final_gate_loss=final_gate_loss,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )
    return model, summary


def evaluate_relay_split_v1(
    model: RelayPopulationModel,
    *,
    training_seed: int,
    split: str = "development",
    world_count: int = 1_000,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    allow_confirmation: bool = False,
) -> tuple[RelayEvaluationResult, ...]:
    """Evaluate one immutable v1 checkpoint across the frozen population ladder."""

    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    if split == "confirmation" and not allow_confirmation:
        raise ValueError("confirmation split is locked unless allow_confirmation=True")
    if world_count <= 0:
        raise ValueError("evaluation world_count must be positive")
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")

    benchmark_seed = split_seed_start(split)
    if benchmark_seed + world_count > split_seed_limit(split):
        raise ValueError("evaluation worlds exceed the reserved split seed range")

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    expected_count = model.trainable_parameter_count()
    expected_fingerprint = model.parameter_fingerprint()
    results: list[RelayEvaluationResult] = []

    for difficulty in RELAY_DIFFICULTIES:
        worlds = generate_relay_dataset(
            start_seed=benchmark_seed,
            world_count=world_count,
            difficulty=difficulty,
        )
        for mode in (
            CommunicationMode.SPARSE_SHARED_V1,
            CommunicationMode.NO_COMMUNICATION,
        ):
            for active_workers in DEVELOPMENT_POPULATION_SIZES:
                result = evaluate_relay_condition_v1(
                    model,
                    worlds,
                    training_seed=training_seed,
                    benchmark_seed=benchmark_seed,
                    active_workers=active_workers,
                    communication_mode=mode,
                    batch_size=batch_size,
                    device=target_device,
                )
                if result.metrics.learned_parameter_count != expected_count:
                    raise RuntimeError("relay v1 evaluation changed learned parameter count")
                if result.metrics.parameter_fingerprint != expected_fingerprint:
                    raise RuntimeError("relay v1 evaluation changed parameter fingerprint")
                results.append(result)

    if model.parameter_fingerprint() != expected_fingerprint:
        raise RuntimeError("relay v1 evaluation mutated the checkpoint")
    return tuple(results)


def evaluate_relay_condition_v1(
    model: RelayPopulationModel,
    worlds: Sequence[RelayWorld],
    *,
    training_seed: int,
    benchmark_seed: int,
    active_workers: int,
    communication_mode: CommunicationMode,
    batch_size: int,
    device: torch.device | str,
) -> RelayEvaluationResult:
    """Evaluate one repaired-protocol or matched no-communication condition."""

    if not worlds:
        raise ValueError("relay evaluation requires at least one world")
    if batch_size <= 0:
        raise ValueError("relay evaluation batch_size must be positive")
    if communication_mode not in {
        CommunicationMode.SPARSE_SHARED_V1,
        CommunicationMode.NO_COMMUNICATION,
    }:
        raise ValueError("relay v1 evaluation received an unsupported communication mode")

    difficulty = worlds[0].difficulty
    if any(world.difficulty != difficulty for world in worlds):
        raise ValueError("one relay evaluation condition must use one difficulty")

    target_device = torch.device(device)
    parameter_count = model.trainable_parameter_count()
    fingerprint = model.parameter_fingerprint()
    cohort_counts = {
        threshold: [0, 0] for threshold in relay_scope_thresholds(difficulty)
    }
    solved_count = 0
    information_complete_count = 0
    solved_complete_count = 0
    correct_bits = 0
    total_bits = 0
    messages_emitted = 0
    communicated_scalars = 0
    peak_state_bytes = 0

    _synchronize(target_device)
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(worlds), batch_size):
            world_batch = worlds[offset : offset + batch_size]
            batch = build_relay_tensor_batch(
                world_batch,
                active_workers=active_workers,
                device=target_device,
            )

            if communication_mode is CommunicationMode.SPARSE_SHARED_V1:
                output = forward_relay_v1(model, batch)
                logits = output.logits
                messages_emitted += (
                    len(world_batch)
                    * output.telemetry.candidate_evaluations_per_sample
                )
                communicated_scalars += (
                    len(world_batch)
                    * output.telemetry.inter_state_communicated_scalars_per_sample
                )
                element_size = batch.local_inputs.element_size()
            else:
                output = model(
                    batch,
                    communication_mode=CommunicationMode.NO_COMMUNICATION,
                )
                logits = output.logits
                messages_emitted += output.telemetry.messages_emitted
                communicated_scalars += output.telemetry.communicated_scalar_count
                element_size = output.final_states.element_size()

            predicted = decode_node_logits(logits)
            solved = predicted.eq(batch.answer_keys)
            complete = batch.information_complete
            solved_count += int(solved.sum().item())
            information_complete_count += int(complete.sum().item())
            solved_complete_count += int((solved & complete).sum().item())

            solved_rows = tuple(bool(value) for value in solved.detach().cpu().tolist())
            for world, row_solved in zip(world_batch, solved_rows, strict=True):
                cohort = cohort_counts[world.scope_threshold]
                cohort[0] += 1
                cohort[1] += int(row_solved)

            target_bits = batch.target_bits > 0
            predicted_bits = logits >= 0
            correct_bits += int(predicted_bits.eq(target_bits).sum().item())
            total_bits += int(target_bits.numel())
            logical_state_bytes = (
                len(world_batch)
                * active_workers
                * model.config.state_width
                * element_size
            )
            peak_state_bytes = max(peak_state_bytes, logical_state_bytes)

    _synchronize(target_device)
    elapsed = time.perf_counter() - started
    metrics = PopulationRunMetrics(
        training_seed=training_seed,
        benchmark_seed=benchmark_seed,
        difficulty=difficulty.name,
        learned_parameter_count=parameter_count,
        parameter_fingerprint=fingerprint,
        condition=PopulationCondition(
            nominal_population_size=active_workers,
            active_state_count=active_workers,
            recurrent_rounds=difficulty.hop_count,
            communication_mode=communication_mode,
        ),
        task_count=len(worlds),
        solved_count=solved_count,
        information_complete_count=information_complete_count,
        solved_information_complete_count=solved_complete_count,
        messages_emitted=messages_emitted,
        communicated_scalar_count=communicated_scalars,
        peak_worker_state_bytes=peak_state_bytes,
        elapsed_seconds=elapsed,
    )
    metrics.validate()
    result = RelayEvaluationResult(
        metrics=metrics,
        bit_accuracy=correct_bits / total_bits,
        scope_cohorts=tuple(
            RelayScopeCohortResult(
                scope_threshold=threshold,
                task_count=cohort_counts[threshold][0],
                solved_count=cohort_counts[threshold][1],
            )
            for threshold in relay_scope_thresholds(difficulty)
        ),
    )
    result.validate()
    return result


def assess_relay_results_v1(
    evaluations: Iterable[RelayEvaluationResult],
) -> tuple[tuple[str, CurveAssessment], ...]:
    rows = tuple(evaluations)
    assessments: list[tuple[str, CurveAssessment]] = []
    for difficulty in (difficulty.name for difficulty in RELAY_DIFFICULTIES):
        communicating = tuple(
            row.metrics
            for row in rows
            if row.metrics.difficulty == difficulty
            and row.metrics.condition.communication_mode
            is CommunicationMode.SPARSE_SHARED_V1
        )
        controls = tuple(
            row.metrics
            for row in rows
            if row.metrics.difficulty == difficulty
            and row.metrics.condition.communication_mode
            is CommunicationMode.NO_COMMUNICATION
        )
        assessments.append((difficulty, assess_scaling_curve(communicating, controls)))
    return tuple(assessments)


def run_relay_experiment_v1(
    *,
    training_seed: int,
    training_config: RelayTrainingConfigV1 = RelayTrainingConfigV1(),
    evaluation_world_count: int = 1_000,
    evaluation_batch_size: int = 64,
    device: torch.device | str = "cpu",
    evaluation_split: str = "development",
    allow_confirmation: bool = False,
) -> RelayDevelopmentResultV1:
    model, training = train_relay_model_v1(
        training_seed=training_seed,
        config=training_config,
        device=device,
    )
    evaluations = evaluate_relay_split_v1(
        model,
        training_seed=training_seed,
        split=evaluation_split,
        world_count=evaluation_world_count,
        batch_size=evaluation_batch_size,
        device=device,
        allow_confirmation=allow_confirmation,
    )
    return RelayDevelopmentResultV1(
        experiment_version=RELAY_EXPERIMENT_V1,
        protocol_version=RELAY_PROTOCOL_VERSION,
        benchmark_version=COLLECTIVE_RELAY_VERSION,
        evaluation_split=evaluation_split,
        confirmation_opened=evaluation_split == "confirmation",
        training=training,
        training_config=training_config,
        evaluation_world_count=evaluation_world_count,
        evaluation_batch_size=evaluation_batch_size,
        evaluations=evaluations,
        assessments=assess_relay_results_v1(evaluations),
    )


def save_relay_checkpoint_v1(
    model: RelayPopulationModel,
    training: RelayTrainingSummaryV1,
    config: RelayTrainingConfigV1,
    path: Path | str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_version": RELAY_EXPERIMENT_V1,
        "protocol_version": RELAY_PROTOCOL_VERSION,
        "benchmark_version": COLLECTIVE_RELAY_VERSION,
        "training_seed": training.training_seed,
        "training_summary": asdict(training),
        "training_config": {
            "steps": config.steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "gate_supervision_weight": config.gate_supervision_weight,
            "communication_mode": CommunicationMode.SPARSE_SHARED_V1.value,
            "model": asdict(config.model),
        },
        "state_dict": model.state_dict(),
        "parameter_fingerprint": model.parameter_fingerprint(),
    }
    torch.save(payload, output)
    return output


def load_relay_checkpoint_v1(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[RelayPopulationModel, dict[str, object]]:
    payload = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
    if payload.get("experiment_version") != RELAY_EXPERIMENT_V1:
        raise ValueError("unexpected relay v1 checkpoint experiment version")
    if payload.get("protocol_version") != RELAY_PROTOCOL_VERSION:
        raise ValueError("unexpected relay v1 protocol version")
    if payload.get("benchmark_version") != COLLECTIVE_RELAY_VERSION:
        raise ValueError("unexpected relay v1 benchmark version")
    model_config = payload.get("training_config", {}).get("model")
    if not isinstance(model_config, dict):
        raise ValueError("relay v1 checkpoint is missing model configuration")
    model = RelayPopulationModel(RelayPopulationConfig(**model_config)).to(device)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("relay v1 checkpoint is missing state_dict")
    model.load_state_dict(state_dict)
    if payload.get("parameter_fingerprint") != model.parameter_fingerprint():
        raise ValueError("relay v1 checkpoint fingerprint does not match payload")
    return model, payload


def write_relay_result_v1(
    result: RelayDevelopmentResultV1,
    path: Path | str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
