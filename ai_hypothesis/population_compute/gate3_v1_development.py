"""Frozen development-only training/evaluation for Gate-3 v1 sparse-active reserve search."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn

from .gate3_v1_batch import run_gate3_v1_public_world_batch
from .gate3_v1_model import Gate3V1ModelConfig, Gate3V1Scorer, encode_gate3_v1_child_input
from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_CONFIRMATION_WORLD_START,
    GATE3_V1_DEPTHS,
    GATE3_V1_DEVELOPMENT_WORLD_START,
    GATE3_V1_HINT_RELIABILITY,
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    GATE3_V1_RESERVE_CAPACITIES,
    GATE3_V1_SEARCH_ROUNDS,
    Gate3V1ControlMode,
    generate_gate3_v1_world,
    score_generated_solution,
)


GATE3_V1_DEVELOPMENT_EXPERIMENT_VERSION = "gate3-v1-sparse-active-reserve-development-v0"
GATE3_V1_DEVELOPMENT_WORLD_COUNT = 256
GATE3_V1_DEVELOPMENT_EVAL_BATCH_SIZE = 64
GATE3_V1_DEVELOPMENT_BOOTSTRAP_SAMPLES = 2_000
GATE3_V1_TRAINING_SEED_LIMIT = GATE3_V1_DEVELOPMENT_WORLD_START
GATE3_V1_SIGNED_HINT_EVIDENCE = math.log(
    GATE3_V1_HINT_RELIABILITY / (1.0 - GATE3_V1_HINT_RELIABILITY)
)


@dataclass(frozen=True, slots=True)
class Gate3V1TrainingConfig:
    steps: int = 1_200
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    model: Gate3V1ModelConfig = Gate3V1ModelConfig()

    def validate(self) -> None:
        self.model.validate()
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("Gate-3 v1 training steps/batch must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight decay must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be finite and positive")


@dataclass(frozen=True, slots=True)
class Gate3V1TrainingSummary:
    training_seed: int
    steps: int
    examples_seen: int
    initial_loss: float
    final_loss: float
    mean_last_50_loss: float
    learned_parameter_count: int
    parameter_fingerprint: str


@dataclass(frozen=True, slots=True)
class Gate3V1ConditionEvaluation:
    depth: int
    reserve_capacity: int
    mode: Gate3V1ControlMode
    world_count: int
    coverage_rate: float
    world_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    generated_terminal_count_by_world: tuple[int, ...]
    unique_generated_terminal_count_by_world: tuple[int, ...]
    productive_rounds_by_world: tuple[int, ...]
    sink_rounds_by_world: tuple[int, ...]
    productive_work_fraction_by_world: tuple[float, ...]
    total_learned_updates_per_world: int
    learned_parameter_count: int
    parameter_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "reserve_capacity": self.reserve_capacity,
            "mode": self.mode.value,
            "world_count": self.world_count,
            "coverage_rate": self.coverage_rate,
            "world_seeds": list(self.world_seeds),
            "covered_by_world": list(self.covered_by_world),
            "generated_terminal_count_by_world": list(self.generated_terminal_count_by_world),
            "unique_generated_terminal_count_by_world": list(self.unique_generated_terminal_count_by_world),
            "productive_rounds_by_world": list(self.productive_rounds_by_world),
            "sink_rounds_by_world": list(self.sink_rounds_by_world),
            "productive_work_fraction_by_world": list(self.productive_work_fraction_by_world),
            "total_learned_updates_per_world": self.total_learned_updates_per_world,
            "learned_parameter_count": self.learned_parameter_count,
            "parameter_fingerprint": self.parameter_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class Gate3V1PairedSummary:
    comparison: str
    depth: int
    treatment_capacity: int
    reference_capacity: int
    treatment_mode: Gate3V1ControlMode
    reference_mode: Gate3V1ControlMode
    world_count: int
    treatment_only: int
    reference_only: int
    both_covered: int
    neither_covered: int
    coverage_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["treatment_mode"] = self.treatment_mode.value
        payload["reference_mode"] = self.reference_mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate3V1DevelopmentResult:
    experiment_version: str
    evaluation_split: str
    confirmation_opened: bool
    training: Gate3V1TrainingSummary
    training_config: Gate3V1TrainingConfig
    evaluation_world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    conditions: tuple[Gate3V1ConditionEvaluation, ...]
    paired_summaries: tuple[Gate3V1PairedSummary, ...]

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
            "hint_reliability": GATE3_V1_HINT_RELIABILITY,
            "recurrent_updates_per_child": GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "paired_summaries": [summary.to_dict() for summary in self.paired_summaries],
            "primary_comparisons": [
                {"primary_key": key, **summary.to_dict()}
                for summary in self.paired_summaries
                if (key := primary_gate3_v1_key(summary)) is not None
            ],
            "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
        }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _training_world_seed(*, training_seed: int, step: int, sample_index: int, depth: int) -> int:
    return _seed_from_parts("gate3-v1-train-world", training_seed, step, sample_index, depth) % GATE3_V1_TRAINING_SEED_LIMIT


def _training_candidate(*, training_seed: int, step: int, sample_index: int, depth: int) -> tuple[int, ...]:
    rng = random.Random(
        _seed_from_parts("gate3-v1-train-candidate", training_seed, step, sample_index, depth)
    )
    return tuple(rng.randrange(2) for _ in range(depth))


def _prefix_targets(*, noisy_hints: tuple[int, ...], candidate: tuple[int, ...], depth: int) -> tuple[float, ...]:
    if len(noisy_hints) != depth or len(candidate) != depth:
        raise ValueError("Gate-3 v1 training trajectory has invalid depth")
    cumulative = 0.0
    targets: list[float] = []
    for hint, action in zip(noisy_hints, candidate, strict=True):
        cumulative += GATE3_V1_SIGNED_HINT_EVIDENCE if action == hint else -GATE3_V1_SIGNED_HINT_EVIDENCE
        targets.append(cumulative / depth)
    return tuple(targets)


def train_gate3_v1_development_model(
    *,
    training_seed: int,
    config: Gate3V1TrainingConfig = Gate3V1TrainingConfig(),
    device: torch.device | str = "cpu",
) -> tuple[Gate3V1Scorer, Gate3V1TrainingSummary]:
    config.validate()
    if not 0 <= training_seed < GATE3_V1_TRAINING_SEED_LIMIT:
        raise ValueError("training seed is outside the reserved Gate-3 v1 training domain")
    target_device = torch.device(device)
    torch.manual_seed(training_seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = Gate3V1Scorer(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    model.train()

    for step in range(config.steps):
        depth = GATE3_V1_DEPTHS[step % len(GATE3_V1_DEPTHS)]
        worlds = tuple(
            generate_gate3_v1_world(
                seed=_training_world_seed(
                    training_seed=training_seed,
                    step=step,
                    sample_index=sample_index,
                    depth=depth,
                ),
                depth=depth,
            )
            for sample_index in range(config.batch_size)
        )
        candidates = tuple(
            _training_candidate(
                training_seed=training_seed,
                step=step,
                sample_index=sample_index,
                depth=depth,
            )
            for sample_index in range(config.batch_size)
        )
        targets = tuple(
            _prefix_targets(
                noisy_hints=world.public.noisy_hints,
                candidate=candidate,
                depth=depth,
            )
            for world, candidate in zip(worlds, candidates, strict=True)
        )
        states = model.initial_state(config.batch_size, device=target_device)
        phase_losses: list[torch.Tensor] = []

        for prefix_index in range(depth):
            inputs = torch.stack(
                [
                    encode_gate3_v1_child_input(
                        world=world.public,
                        child_depth=prefix_index + 1,
                        observed_hint=world.public.noisy_hints[prefix_index],
                        branch_action=candidate[prefix_index],
                        sink=False,
                        device=target_device,
                    )
                    for world, candidate in zip(worlds, candidates, strict=True)
                ],
                dim=0,
            )
            states = model.advance(
                states,
                inputs,
                repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            )
            predictions = model.score(states)
            target_tensor = torch.tensor(
                [row[prefix_index] for row in targets],
                dtype=predictions.dtype,
                device=target_device,
            )
            phase_losses.append(loss_fn(predictions, target_tensor))

        loss = torch.stack(phase_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Gate-3 v1 training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))

    summary = Gate3V1TrainingSummary(
        training_seed=training_seed,
        steps=config.steps,
        examples_seen=config.steps * config.batch_size,
        initial_loss=losses[0],
        final_loss=losses[-1],
        mean_last_50_loss=sum(losses[-50:]) / len(losses[-50:]),
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )
    return model, summary


def _development_worlds(*, depth: int, world_count: int) -> tuple[object, ...]:
    if world_count <= 0:
        raise ValueError("development world count must be positive")
    if GATE3_V1_DEVELOPMENT_WORLD_START + world_count > GATE3_V1_CONFIRMATION_WORLD_START:
        raise ValueError("development evaluation would cross into confirmation domain")
    return tuple(
        generate_gate3_v1_world(seed=GATE3_V1_DEVELOPMENT_WORLD_START + offset, depth=depth)
        for offset in range(world_count)
    )


def evaluate_gate3_v1_condition(
    model: Gate3V1Scorer,
    *,
    depth: int,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    world_count: int = GATE3_V1_DEVELOPMENT_WORLD_COUNT,
    evaluation_batch_size: int = GATE3_V1_DEVELOPMENT_EVAL_BATCH_SIZE,
    device: torch.device | str = "cpu",
) -> Gate3V1ConditionEvaluation:
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    worlds = _development_worlds(depth=depth, world_count=world_count)
    covered: list[bool] = []
    terminal_counts: list[int] = []
    unique_terminal_counts: list[int] = []
    productive_rounds: list[int] = []
    sink_rounds: list[int] = []
    productive_fractions: list[float] = []
    expected_total: int | None = None

    model.eval()
    for start in range(0, world_count, evaluation_batch_size):
        chunk = worlds[start : start + evaluation_batch_size]
        batch = run_gate3_v1_public_world_batch(
            model,
            (world.public for world in chunk),
            reserve_capacity=reserve_capacity,
            mode=mode,
            device=device,
        )
        for world, runtime in zip(chunk, batch.world_results, strict=True):
            covered.append(
                score_generated_solution(
                    hidden_path=world.hidden_path,
                    generated_terminal_paths=runtime.generated_terminal_paths,
                )
            )
            telemetry = runtime.telemetry
            terminal_counts.append(telemetry.generated_terminal_count)
            unique_terminal_counts.append(telemetry.unique_generated_terminal_count)
            productive_rounds.append(telemetry.productive_rounds)
            sink_rounds.append(telemetry.sink_rounds)
            productive_fractions.append(telemetry.productive_rounds / GATE3_V1_SEARCH_ROUNDS[depth])
            if expected_total is None:
                expected_total = telemetry.total_learned_updates
            elif expected_total != telemetry.total_learned_updates:
                raise RuntimeError("Gate-3 v1 evaluation batches disagree on learned-work total")

    if expected_total is None:
        raise RuntimeError("Gate-3 v1 evaluation produced no worlds")
    return Gate3V1ConditionEvaluation(
        depth=depth,
        reserve_capacity=reserve_capacity,
        mode=mode,
        world_count=world_count,
        coverage_rate=sum(int(value) for value in covered) / world_count,
        world_seeds=tuple(world.public.seed for world in worlds),
        covered_by_world=tuple(covered),
        generated_terminal_count_by_world=tuple(terminal_counts),
        unique_generated_terminal_count_by_world=tuple(unique_terminal_counts),
        productive_rounds_by_world=tuple(productive_rounds),
        sink_rounds_by_world=tuple(sink_rounds),
        productive_work_fraction_by_world=tuple(productive_fractions),
        total_learned_updates_per_world=expected_total,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_seed(spec: tuple[object, ...]) -> int:
    return _seed_from_parts("gate3-v1-bootstrap", *spec)


def _paired_summary(
    *,
    comparison: str,
    treatment: Gate3V1ConditionEvaluation,
    reference: Gate3V1ConditionEvaluation,
    bootstrap_samples: int,
) -> Gate3V1PairedSummary:
    if treatment.depth != reference.depth or treatment.world_seeds != reference.world_seeds:
        raise ValueError("paired Gate-3 v1 conditions must use identical worlds/depth")
    if treatment.total_learned_updates_per_world != reference.total_learned_updates_per_world:
        raise ValueError("paired Gate-3 v1 conditions must preserve learned-work identity")
    if treatment.learned_parameter_count != reference.learned_parameter_count:
        raise ValueError("paired Gate-3 v1 conditions must preserve parameter count")
    if treatment.parameter_fingerprint != reference.parameter_fingerprint:
        raise ValueError("paired Gate-3 v1 conditions must use one checkpoint")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")

    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = len(pairs) - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    delta = sum(differences) / len(differences)
    spec = (
        comparison,
        treatment.depth,
        treatment.reserve_capacity,
        treatment.mode.value,
        reference.reserve_capacity,
        reference.mode.value,
    )
    rng = random.Random(_bootstrap_seed(spec))
    estimates = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(bootstrap_samples)
    )
    low = estimates[int(math.floor(0.025 * (bootstrap_samples - 1)))]
    high = estimates[int(math.ceil(0.975 * (bootstrap_samples - 1)))]
    return Gate3V1PairedSummary(
        comparison=comparison,
        depth=treatment.depth,
        treatment_capacity=treatment.reserve_capacity,
        reference_capacity=reference.reserve_capacity,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        world_count=len(differences),
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=delta,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate3_v1_paired_summaries(
    conditions: Iterable[Gate3V1ConditionEvaluation],
    *,
    bootstrap_samples: int = GATE3_V1_DEVELOPMENT_BOOTSTRAP_SAMPLES,
) -> tuple[Gate3V1PairedSummary, ...]:
    rows = tuple(conditions)
    index = {(row.depth, row.reserve_capacity, row.mode): row for row in rows}
    summaries: list[Gate3V1PairedSummary] = []
    for depth in GATE3_V1_DEPTHS:
        capacities = GATE3_V1_RESERVE_CAPACITIES[depth]
        stable_l1 = index[(depth, 1, Gate3V1ControlMode.STABLE_RESERVE)]
        for capacity in capacities[1:]:
            summaries.append(
                _paired_summary(
                    comparison="stable_capacity_vs_l1",
                    treatment=index[(depth, capacity, Gate3V1ControlMode.STABLE_RESERVE)],
                    reference=stable_l1,
                    bootstrap_samples=bootstrap_samples,
                )
            )
        for previous, capacity in zip(capacities[:-1], capacities[1:], strict=True):
            summaries.append(
                _paired_summary(
                    comparison="stable_capacity_vs_previous",
                    treatment=index[(depth, capacity, Gate3V1ControlMode.STABLE_RESERVE)],
                    reference=index[(depth, previous, Gate3V1ControlMode.STABLE_RESERVE)],
                    bootstrap_samples=bootstrap_samples,
                )
            )
        for capacity in capacities:
            stable = index[(depth, capacity, Gate3V1ControlMode.STABLE_RESERVE)]
            summaries.append(
                _paired_summary(
                    comparison="stable_vs_collapsed",
                    treatment=stable,
                    reference=index[(depth, capacity, Gate3V1ControlMode.COLLAPSED_DIVERSITY)],
                    bootstrap_samples=bootstrap_samples,
                )
            )
            summaries.append(
                _paired_summary(
                    comparison="stable_vs_reshuffled",
                    treatment=stable,
                    reference=index[(depth, capacity, Gate3V1ControlMode.RESHUFFLED_CONTINUITY)],
                    bootstrap_samples=bootstrap_samples,
                )
            )
    return tuple(summaries)


def primary_gate3_v1_key(summary: Gate3V1PairedSummary) -> str | None:
    key = (
        summary.comparison,
        summary.depth,
        summary.treatment_capacity,
        summary.reference_capacity,
        summary.treatment_mode,
        summary.reference_mode,
    )
    mapping = {
        ("stable_capacity_vs_l1", 8, 64, 1, Gate3V1ControlMode.STABLE_RESERVE, Gate3V1ControlMode.STABLE_RESERVE): "s8_l64_vs_l1",
        ("stable_capacity_vs_l1", 10, 256, 1, Gate3V1ControlMode.STABLE_RESERVE, Gate3V1ControlMode.STABLE_RESERVE): "s10_l256_vs_l1",
        ("stable_capacity_vs_previous", 10, 256, 64, Gate3V1ControlMode.STABLE_RESERVE, Gate3V1ControlMode.STABLE_RESERVE): "s10_l256_vs_l64",
        ("stable_vs_collapsed", 10, 256, 256, Gate3V1ControlMode.STABLE_RESERVE, Gate3V1ControlMode.COLLAPSED_DIVERSITY): "s10_stable_vs_collapsed",
        ("stable_vs_reshuffled", 10, 256, 256, Gate3V1ControlMode.STABLE_RESERVE, Gate3V1ControlMode.RESHUFFLED_CONTINUITY): "s10_stable_vs_reshuffled",
    }
    return mapping.get(key)


def run_gate3_v1_development(
    *,
    training_seed: int,
    training_config: Gate3V1TrainingConfig = Gate3V1TrainingConfig(),
    evaluation_world_count: int = GATE3_V1_DEVELOPMENT_WORLD_COUNT,
    evaluation_batch_size: int = GATE3_V1_DEVELOPMENT_EVAL_BATCH_SIZE,
    bootstrap_samples: int = GATE3_V1_DEVELOPMENT_BOOTSTRAP_SAMPLES,
    device: torch.device | str = "cpu",
) -> tuple[Gate3V1Scorer, Gate3V1DevelopmentResult]:
    model, training = train_gate3_v1_development_model(
        training_seed=training_seed,
        config=training_config,
        device=device,
    )
    conditions: list[Gate3V1ConditionEvaluation] = []
    for depth in GATE3_V1_DEPTHS:
        for capacity in GATE3_V1_RESERVE_CAPACITIES[depth]:
            for mode in Gate3V1ControlMode:
                conditions.append(
                    evaluate_gate3_v1_condition(
                        model,
                        depth=depth,
                        reserve_capacity=capacity,
                        mode=mode,
                        world_count=evaluation_world_count,
                        evaluation_batch_size=evaluation_batch_size,
                        device=device,
                    )
                )
    paired = build_gate3_v1_paired_summaries(
        conditions,
        bootstrap_samples=bootstrap_samples,
    )
    return model, Gate3V1DevelopmentResult(
        experiment_version=GATE3_V1_DEVELOPMENT_EXPERIMENT_VERSION,
        evaluation_split="development",
        confirmation_opened=False,
        training=training,
        training_config=training_config,
        evaluation_world_count=evaluation_world_count,
        evaluation_batch_size=evaluation_batch_size,
        bootstrap_samples=bootstrap_samples,
        conditions=tuple(conditions),
        paired_summaries=paired,
    )
