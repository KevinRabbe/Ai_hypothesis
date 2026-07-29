"""Frozen development-only training and paired evaluation for Gate 3.

Confirmation remains closed. The shared scorer is trained on individual candidate trajectories
across the frozen depth/width recurrence schedules, then evaluated with the matched population
runtime. No artifact produced here can assign a Gate-3 verdict.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn

from .gate3_hypothesis_batch import run_gate3_world_batch
from .gate3_hypothesis_model import (
    Gate3HypothesisModelConfig,
    Gate3HypothesisScorer,
    encode_gate3_phase_input,
)
from .gate3_hypothesis_population import (
    GATE3_CONFIRMATION_WORLD_START,
    GATE3_DEPTHS,
    GATE3_DEVELOPMENT_WORLD_START,
    GATE3_HINT_RELIABILITY,
    GATE3_WIDTHS_BY_DEPTH,
    Gate3ControlMode,
    Gate3World,
    build_gate3_condition_plan,
    generate_gate3_world,
)


GATE3_DEVELOPMENT_EXPERIMENT_VERSION = "gate3-hypothesis-population-development-v0"
GATE3_TRAINING_SEED_LIMIT = GATE3_DEVELOPMENT_WORLD_START
GATE3_DEVELOPMENT_WORLD_LIMIT = GATE3_CONFIRMATION_WORLD_START
GATE3_REVEAL_MISMATCH_PENALTY = 16.0
GATE3_DEVELOPMENT_BOOTSTRAP_SAMPLES = 2_000
GATE3_DEVELOPMENT_EVAL_WORLD_COUNT = 256
GATE3_DEVELOPMENT_EVAL_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class Gate3TrainingConfig:
    steps: int = 1_200
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    model: Gate3HypothesisModelConfig = Gate3HypothesisModelConfig()

    def validate(self) -> None:
        self.model.validate()
        if self.steps <= 0:
            raise ValueError("training steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("training batch size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight decay must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be finite and positive")


@dataclass(frozen=True, slots=True)
class Gate3TrainingSummary:
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
class Gate3ConditionEvaluation:
    depth: int
    width: int
    mode: Gate3ControlMode
    world_count: int
    exact_solve_rate: float
    bit_accuracy: float
    learned_updates_per_world: int
    unique_world_observations_per_world: int
    learned_parameter_count: int
    parameter_fingerprint: str
    world_seeds: tuple[int, ...]
    solved_by_world: tuple[bool, ...]
    bit_accuracy_by_world: tuple[float, ...]
    correct_candidate_survival_rate_by_phase: tuple[float, ...]
    mean_unique_candidates_by_phase: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "width": self.width,
            "mode": self.mode.value,
            "world_count": self.world_count,
            "exact_solve_rate": self.exact_solve_rate,
            "bit_accuracy": self.bit_accuracy,
            "learned_updates_per_world": self.learned_updates_per_world,
            "unique_world_observations_per_world": self.unique_world_observations_per_world,
            "learned_parameter_count": self.learned_parameter_count,
            "parameter_fingerprint": self.parameter_fingerprint,
            "world_seeds": list(self.world_seeds),
            "solved_by_world": list(self.solved_by_world),
            "bit_accuracy_by_world": list(self.bit_accuracy_by_world),
            "correct_candidate_survival_rate_by_phase": list(self.correct_candidate_survival_rate_by_phase),
            "mean_unique_candidates_by_phase": list(self.mean_unique_candidates_by_phase),
        }


@dataclass(frozen=True, slots=True)
class Gate3PairedSummary:
    comparison: str
    depth: int
    treatment_width: int
    reference_width: int
    treatment_mode: Gate3ControlMode
    reference_mode: Gate3ControlMode
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
class Gate3DevelopmentResult:
    experiment_version: str
    evaluation_split: str
    confirmation_opened: bool
    training: Gate3TrainingSummary
    training_config: Gate3TrainingConfig
    evaluation_world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    conditions: tuple[Gate3ConditionEvaluation, ...]
    paired_summaries: tuple[Gate3PairedSummary, ...]

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
            "bootstrap_samples": self.bootstrap_samples,
            "hint_reliability": GATE3_HINT_RELIABILITY,
            "reveal_mismatch_penalty": GATE3_REVEAL_MISMATCH_PENALTY,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "paired_summaries": [summary.to_dict() for summary in self.paired_summaries],
            "primary_comparisons": [
                summary.to_dict() for summary in self.paired_summaries if _primary_key(summary) is not None
            ],
            "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
            "interpretation_note": (
                "Gate-3 development evidence cannot assign a gate verdict. Confirmation worlds "
                "remain closed until robustness and confirmation rules are frozen before exposure."
            ),
        }


def gate3_stable_training_conditions() -> tuple[tuple[int, int], ...]:
    return tuple(
        (depth, width)
        for depth in GATE3_DEPTHS
        for width in GATE3_WIDTHS_BY_DEPTH[depth]
    )


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _training_world_seed(
    *,
    training_seed: int,
    step: int,
    sample_index: int,
    depth: int,
    width: int,
) -> int:
    return _seed_from_parts("gate3-train-world", training_seed, step, sample_index, depth, width) % GATE3_TRAINING_SEED_LIMIT


def _candidate_path(
    *,
    training_seed: int,
    step: int,
    sample_index: int,
    depth: int,
    width: int,
) -> tuple[int, ...]:
    rng = random.Random(
        _seed_from_parts("gate3-train-candidate", training_seed, step, sample_index, depth, width)
    )
    return tuple(rng.randrange(2) for _ in range(depth))


def _target_score(
    world: Gate3World,
    candidate_path: tuple[int, ...],
    *,
    phase_index: int,
) -> float:
    depth = world.depth
    if len(candidate_path) != depth:
        raise ValueError("training candidate path length does not match world depth")
    log_match = math.log(GATE3_HINT_RELIABILITY)
    log_mismatch = math.log(1.0 - GATE3_HINT_RELIABILITY)

    if phase_index < depth:
        observed_count = phase_index + 1
        hint_score = sum(
            log_match if candidate_path[index] == world.noisy_hints[index] else log_mismatch
            for index in range(observed_count)
        )
        return hint_score / depth

    hint_score = sum(
        log_match if candidate_path[index] == world.noisy_hints[index] else log_mismatch
        for index in range(depth)
    )
    reveal_count = phase_index - depth + 1
    reveal_mismatches = sum(
        int(candidate_path[index] != world.hidden_path[index])
        for index in range(reveal_count)
    )
    return (hint_score - GATE3_REVEAL_MISMATCH_PENALTY * reveal_mismatches) / depth


def train_gate3_development_model(
    *,
    training_seed: int,
    config: Gate3TrainingConfig = Gate3TrainingConfig(),
    device: torch.device | str = "cpu",
) -> tuple[Gate3HypothesisScorer, Gate3TrainingSummary]:
    config.validate()
    if not 0 <= training_seed < GATE3_TRAINING_SEED_LIMIT:
        raise ValueError("Gate-3 development training seed is outside the reserved training domain")

    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = Gate3HypothesisScorer(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()
    conditions = gate3_stable_training_conditions()
    losses: list[float] = []
    examples_seen = 0
    model.train()

    for step in range(config.steps):
        depth, width = conditions[step % len(conditions)]
        worlds = tuple(
            generate_gate3_world(
                seed=_training_world_seed(
                    training_seed=training_seed,
                    step=step,
                    sample_index=sample_index,
                    depth=depth,
                    width=width,
                ),
                depth=depth,
            )
            for sample_index in range(config.batch_size)
        )
        candidates = tuple(
            _candidate_path(
                training_seed=training_seed,
                step=step,
                sample_index=sample_index,
                depth=depth,
                width=width,
            )
            for sample_index in range(config.batch_size)
        )
        plan = build_gate3_condition_plan(
            worlds[0],
            width=width,
            mode=Gate3ControlMode.STABLE_DIVERSE,
        )
        states = model.initial_state(config.batch_size, device=target_device)
        phase_losses: list[torch.Tensor] = []

        for phase in plan.phases:
            phase_inputs = torch.stack(
                [
                    encode_gate3_phase_input(
                        depth=depth,
                        observation=world.observations[phase.phase_index],
                        branch_action=(candidate[phase.phase_index] if phase.phase_index < depth else None),
                        device=target_device,
                    )
                    for world, candidate in zip(worlds, candidates, strict=True)
                ],
                dim=0,
            )
            states = model.advance(
                states,
                phase_inputs,
                repeats=phase.recurrent_updates_per_evaluated_state,
            )
            predictions = model.score(states)
            targets = torch.tensor(
                [
                    _target_score(world, candidate, phase_index=phase.phase_index)
                    for world, candidate in zip(worlds, candidates, strict=True)
                ],
                dtype=predictions.dtype,
                device=target_device,
            )
            phase_losses.append(loss_fn(predictions, targets))

        loss = torch.stack(phase_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Gate-3 development training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        losses.append(float(loss.detach().item()))
        examples_seen += config.batch_size

    summary = Gate3TrainingSummary(
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


def _development_worlds(*, depth: int, world_count: int) -> tuple[Gate3World, ...]:
    if world_count <= 0:
        raise ValueError("development world count must be positive")
    if GATE3_DEVELOPMENT_WORLD_START + world_count > GATE3_DEVELOPMENT_WORLD_LIMIT:
        raise ValueError("development evaluation would cross into the confirmation world domain")
    return tuple(
        generate_gate3_world(seed=GATE3_DEVELOPMENT_WORLD_START + offset, depth=depth)
        for offset in range(world_count)
    )


def evaluate_gate3_condition(
    model: Gate3HypothesisScorer,
    *,
    depth: int,
    width: int,
    mode: Gate3ControlMode,
    world_count: int = GATE3_DEVELOPMENT_EVAL_WORLD_COUNT,
    evaluation_batch_size: int = GATE3_DEVELOPMENT_EVAL_BATCH_SIZE,
    device: torch.device | str = "cpu",
) -> Gate3ConditionEvaluation:
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    worlds = _development_worlds(depth=depth, world_count=world_count)
    solved: list[bool] = []
    bit_accuracies: list[float] = []
    survival_rows: list[tuple[bool, ...]] = []
    unique_rows: list[tuple[int, ...]] = []
    learned_updates: int | None = None
    unique_observations: int | None = None

    model.eval()
    for start in range(0, world_count, evaluation_batch_size):
        chunk = worlds[start : start + evaluation_batch_size]
        batch_result = run_gate3_world_batch(
            model,
            chunk,
            width=width,
            mode=mode,
            device=device,
        )
        solved.extend(batch_result.exact_solved_by_world)
        bit_accuracies.extend(batch_result.bit_accuracy_by_world)
        survival_rows.extend(batch_result.correct_candidate_present_by_world_by_phase)
        unique_rows.extend(batch_result.unique_candidate_count_by_world_by_phase)
        if learned_updates is None:
            learned_updates = batch_result.learned_updates_per_world
            unique_observations = batch_result.unique_world_observations_per_world
        elif (
            learned_updates != batch_result.learned_updates_per_world
            or unique_observations != batch_result.unique_world_observations_per_world
        ):
            raise RuntimeError("Gate-3 evaluation batches disagree on frozen work/information accounting")

    if learned_updates is None or unique_observations is None:
        raise RuntimeError("Gate-3 evaluation produced no batches")
    phase_count = 2 * depth
    survival_rates = tuple(
        sum(int(row[phase]) for row in survival_rows) / world_count
        for phase in range(phase_count)
    )
    mean_unique = tuple(
        sum(row[phase] for row in unique_rows) / world_count
        for phase in range(phase_count)
    )
    return Gate3ConditionEvaluation(
        depth=depth,
        width=width,
        mode=mode,
        world_count=world_count,
        exact_solve_rate=sum(solved) / world_count,
        bit_accuracy=sum(bit_accuracies) / world_count,
        learned_updates_per_world=learned_updates,
        unique_world_observations_per_world=unique_observations,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
        world_seeds=tuple(world.seed for world in worlds),
        solved_by_world=tuple(solved),
        bit_accuracy_by_world=tuple(bit_accuracies),
        correct_candidate_survival_rate_by_phase=survival_rates,
        mean_unique_candidates_by_phase=mean_unique,
    )


def _bootstrap_seed(summary_key: tuple[object, ...]) -> int:
    digest = hashlib.sha256(
        ("gate3-bootstrap-v0:" + ":".join(str(value) for value in summary_key)).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _paired_summary(
    *,
    comparison: str,
    treatment: Gate3ConditionEvaluation,
    reference: Gate3ConditionEvaluation,
    bootstrap_samples: int,
) -> Gate3PairedSummary:
    if treatment.depth != reference.depth:
        raise ValueError("Gate-3 paired conditions must use the same hidden depth")
    if treatment.world_seeds != reference.world_seeds:
        raise ValueError("Gate-3 paired conditions must use identical worlds")
    if treatment.learned_updates_per_world != reference.learned_updates_per_world:
        raise ValueError("Gate-3 paired conditions must preserve learned-work identity")
    if treatment.unique_world_observations_per_world != reference.unique_world_observations_per_world:
        raise ValueError("Gate-3 paired conditions must preserve information identity")
    if treatment.learned_parameter_count != reference.learned_parameter_count:
        raise ValueError("Gate-3 paired conditions must use the same learned parameter count")
    if treatment.parameter_fingerprint != reference.parameter_fingerprint:
        raise ValueError("Gate-3 paired conditions must use one checkpoint")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    pairs = tuple(zip(treatment.solved_by_world, reference.solved_by_world, strict=True))
    treatment_only = sum(int(bool(a) and not bool(b)) for a, b in pairs)
    reference_only = sum(int(bool(b) and not bool(a)) for a, b in pairs)
    both = sum(int(bool(a) and bool(b)) for a, b in pairs)
    neither = len(pairs) - treatment_only - reference_only - both
    differences = tuple(int(bool(a)) - int(bool(b)) for a, b in pairs)
    delta = sum(differences) / len(differences)

    rng = random.Random(
        _bootstrap_seed(
            (
                comparison,
                treatment.depth,
                treatment.width,
                treatment.mode.value,
                reference.width,
                reference.mode.value,
            )
        )
    )
    estimates: list[float] = []
    count = len(differences)
    for _ in range(bootstrap_samples):
        estimates.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
    estimates.sort()
    low_index = int(math.floor(0.025 * (bootstrap_samples - 1)))
    high_index = int(math.ceil(0.975 * (bootstrap_samples - 1)))
    return Gate3PairedSummary(
        comparison=comparison,
        depth=treatment.depth,
        treatment_width=treatment.width,
        reference_width=reference.width,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        world_count=count,
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_solved=both,
        neither_solved=neither,
        exact_solve_delta=delta,
        bootstrap_ci_low=estimates[low_index],
        bootstrap_ci_high=estimates[high_index],
    )


def _primary_key(summary: Gate3PairedSummary) -> str | None:
    key = (
        summary.comparison,
        summary.depth,
        summary.treatment_width,
        summary.reference_width,
        summary.treatment_mode,
        summary.reference_mode,
    )
    mapping = {
        (
            "stable_width_vs_width1",
            6,
            64,
            1,
            Gate3ControlMode.STABLE_DIVERSE,
            Gate3ControlMode.STABLE_DIVERSE,
        ): "h6_w64_vs_w1",
        (
            "stable_width_vs_width1",
            8,
            256,
            1,
            Gate3ControlMode.STABLE_DIVERSE,
            Gate3ControlMode.STABLE_DIVERSE,
        ): "h8_w256_vs_w1",
        (
            "stable_width_vs_previous",
            8,
            256,
            64,
            Gate3ControlMode.STABLE_DIVERSE,
            Gate3ControlMode.STABLE_DIVERSE,
        ): "h8_w256_vs_w64",
        (
            "stable_vs_collapsed",
            8,
            256,
            256,
            Gate3ControlMode.STABLE_DIVERSE,
            Gate3ControlMode.COLLAPSED_DIVERSITY,
        ): "h8_stable_vs_collapsed",
        (
            "stable_vs_reshuffled",
            8,
            256,
            256,
            Gate3ControlMode.STABLE_DIVERSE,
            Gate3ControlMode.RESHUFFLED_CONTINUITY,
        ): "h8_stable_vs_reshuffled",
    }
    return mapping.get(key)


def build_gate3_paired_summaries(
    conditions: Iterable[Gate3ConditionEvaluation],
    *,
    bootstrap_samples: int = GATE3_DEVELOPMENT_BOOTSTRAP_SAMPLES,
) -> tuple[Gate3PairedSummary, ...]:
    rows = tuple(conditions)
    index = {(row.depth, row.width, row.mode): row for row in rows}
    summaries: list[Gate3PairedSummary] = []

    for depth in GATE3_DEPTHS:
        widths = GATE3_WIDTHS_BY_DEPTH[depth]
        stable_w1 = index[(depth, 1, Gate3ControlMode.STABLE_DIVERSE)]
        for width in widths[1:]:
            summaries.append(
                _paired_summary(
                    comparison="stable_width_vs_width1",
                    treatment=index[(depth, width, Gate3ControlMode.STABLE_DIVERSE)],
                    reference=stable_w1,
                    bootstrap_samples=bootstrap_samples,
                )
            )
        for previous, width in zip(widths[:-1], widths[1:], strict=True):
            summaries.append(
                _paired_summary(
                    comparison="stable_width_vs_previous",
                    treatment=index[(depth, width, Gate3ControlMode.STABLE_DIVERSE)],
                    reference=index[(depth, previous, Gate3ControlMode.STABLE_DIVERSE)],
                    bootstrap_samples=bootstrap_samples,
                )
            )
        for width in widths:
            stable = index[(depth, width, Gate3ControlMode.STABLE_DIVERSE)]
            summaries.append(
                _paired_summary(
                    comparison="stable_vs_collapsed",
                    treatment=stable,
                    reference=index[(depth, width, Gate3ControlMode.COLLAPSED_DIVERSITY)],
                    bootstrap_samples=bootstrap_samples,
                )
            )
            summaries.append(
                _paired_summary(
                    comparison="stable_vs_reshuffled",
                    treatment=stable,
                    reference=index[(depth, width, Gate3ControlMode.RESHUFFLED_CONTINUITY)],
                    bootstrap_samples=bootstrap_samples,
                )
            )
    return tuple(summaries)


def run_gate3_development(
    *,
    training_seed: int,
    training_config: Gate3TrainingConfig = Gate3TrainingConfig(),
    evaluation_world_count: int = GATE3_DEVELOPMENT_EVAL_WORLD_COUNT,
    evaluation_batch_size: int = GATE3_DEVELOPMENT_EVAL_BATCH_SIZE,
    bootstrap_samples: int = GATE3_DEVELOPMENT_BOOTSTRAP_SAMPLES,
    device: torch.device | str = "cpu",
) -> tuple[Gate3HypothesisScorer, Gate3DevelopmentResult]:
    model, training_summary = train_gate3_development_model(
        training_seed=training_seed,
        config=training_config,
        device=device,
    )
    conditions: list[Gate3ConditionEvaluation] = []
    for depth in GATE3_DEPTHS:
        for width in GATE3_WIDTHS_BY_DEPTH[depth]:
            for mode in Gate3ControlMode:
                conditions.append(
                    evaluate_gate3_condition(
                        model,
                        depth=depth,
                        width=width,
                        mode=mode,
                        world_count=evaluation_world_count,
                        evaluation_batch_size=evaluation_batch_size,
                        device=device,
                    )
                )

    paired = build_gate3_paired_summaries(conditions, bootstrap_samples=bootstrap_samples)
    result = Gate3DevelopmentResult(
        experiment_version=GATE3_DEVELOPMENT_EXPERIMENT_VERSION,
        evaluation_split="development",
        confirmation_opened=False,
        training=training_summary,
        training_config=training_config,
        evaluation_world_count=evaluation_world_count,
        evaluation_batch_size=evaluation_batch_size,
        bootstrap_samples=bootstrap_samples,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
    return model, result
