"""Data-blind Gate-7 scale-neutral transition bridge preparation.

No checkpoint hashes are bound here and no admitted bridge runner exists. The module reuses the frozen
Gate-6 scheduler unchanged through a parameter-free input adapter, defines fresh bridge-only worlds and
paired bootstrap summaries, and freezes the already-preregistered bridge classifier.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .gate3_v1_model import Gate3V1Scorer
from .gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from .gate6_fixed_k_population_scaling import (
    GATE6_EVAL_BATCH_SIZE,
    GATE6_STAGE_A_PARENT_SLOTS,
    GATE6_STAGE_B_PARENT_SLOTS,
    GATE6_TOTAL_LEARNED_UPDATES,
    Gate6EvaluationWorld,
    Gate6SchedulerMode,
    run_gate6_world_batch,
)
from .gate7_scale_neutral_model_prep import (
    GATE7_SCALE_NEUTRAL_PARAMETER_COUNT,
    Gate7ScaleNeutralScorer,
    encode_gate7_scale_neutral_child_inputs_batch,
)

GATE7_TRANSITION_BRIDGE_PREPARATION_ONLY = True
GATE7_TRANSITION_BRIDGE_VERSION = "gate7-scale-neutral-transition-bridge-v0"
GATE7_TRANSITION_BRIDGE_WORLD_COUNT = 256
GATE7_TRANSITION_BRIDGE_BATCH_SIZE = 64
GATE7_TRANSITION_BRIDGE_BOOTSTRAP_SAMPLES = 2_000
GATE7_TRANSITION_BRIDGE_DEPTH = 10
GATE7_TRANSITION_BRIDGE_HINT_RELIABILITY = 0.70
GATE7_TRANSITION_BRIDGE_POPULATIONS = (128, 256)
GATE7_TRANSITION_BRIDGE_MODES = (
    Gate6SchedulerMode.GLOBAL_SCORE,
    Gate6SchedulerMode.BOUNDED_SCORE_K16,
    Gate6SchedulerMode.BOUNDED_HASH_K16,
)
GATE7_TRANSITION_BRIDGE_NONINFERIORITY_MARGIN = 0.05
GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class Gate7TransitionBridgeCondition:
    checkpoint_index: int
    checkpoint_family: str
    population_size: int
    mode: Gate6SchedulerMode
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    coverage_rate: float
    total_learned_updates_per_world: int
    stage_a_parent_slots: int
    stage_b_parent_slots: int
    learned_parameter_count: int
    parameter_fingerprint: str


@dataclass(frozen=True, slots=True)
class Gate7TransitionBridgePairedSummary:
    comparison: str
    checkpoint_index: int
    population_size: int
    treatment_family: str
    reference_family: str
    treatment_mode: Gate6SchedulerMode
    reference_mode: Gate6SchedulerMode
    coverage_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def gate7_transition_bridge_runtime_seed(*, world_index: int) -> int:
    if not 0 <= world_index < GATE7_TRANSITION_BRIDGE_WORLD_COUNT:
        raise ValueError("bridge world index is outside 0..255")
    return _seed_from_parts(
        "gate7-scale-neutral-transition-bridge-runtime",
        world_index,
        GATE7_TRANSITION_BRIDGE_DEPTH,
    )


def generate_gate7_transition_bridge_world(*, world_index: int) -> Gate6EvaluationWorld:
    """Fresh depth-10 world under bridge-only hidden/hint/runtime namespaces."""

    if not 0 <= world_index < GATE7_TRANSITION_BRIDGE_WORLD_COUNT:
        raise ValueError("bridge world index is outside 0..255")
    hidden_rng = random.Random(
        _seed_from_parts(
            "gate7-scale-neutral-transition-bridge-hidden",
            world_index,
            GATE7_TRANSITION_BRIDGE_DEPTH,
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE7_TRANSITION_BRIDGE_DEPTH))
    hint_rng = random.Random(
        _seed_from_parts(
            "gate7-scale-neutral-transition-bridge-hints",
            world_index,
            GATE7_TRANSITION_BRIDGE_DEPTH,
        )
    )
    noisy_hints = tuple(
        hidden_bit
        if hint_rng.random() < GATE7_TRANSITION_BRIDGE_HINT_RELIABILITY
        else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    world = Gate6EvaluationWorld(
        world_index=world_index,
        public=Gate3V1PublicWorld(
            seed=gate7_transition_bridge_runtime_seed(world_index=world_index),
            depth=GATE7_TRANSITION_BRIDGE_DEPTH,
            noisy_hints=noisy_hints,
        ),
        hidden_path=hidden_path,
    )
    world.validate()
    return world


class Gate7ScaleNeutralGate6Adapter(nn.Module):
    """Parameter-free compatibility adapter for the exact frozen Gate-6 scheduler.

    Gate-6 constructs its historical 19-input representation before calling `advance`. This adapter
    decodes only the public child-depth/hint/action tokens and immediately re-encodes them with the
    frozen scale-neutral representation. It owns no parameters beyond the wrapped scorer.
    """

    def __init__(self, scorer: Gate7ScaleNeutralScorer) -> None:
        super().__init__()
        self.scorer = scorer

    def initial_state(self, count: int, *, device: torch.device | str) -> torch.Tensor:
        return self.scorer.initial_state(count, device=device)

    def advance(self, state: torch.Tensor, phase_input: torch.Tensor, *, repeats: int) -> torch.Tensor:
        if phase_input.ndim != 2 or phase_input.shape[1] != 19:
            raise ValueError("Gate-7 bridge adapter expects the frozen Gate-6 19-input tensor")
        batch = phase_input.shape[0]
        child_depths = phase_input[:, :10].argmax(dim=1).to(torch.int64) + 1
        world_depths = torch.full(
            (batch,),
            GATE7_TRANSITION_BRIDGE_DEPTH,
            dtype=torch.int64,
            device=phase_input.device,
        )
        observed_hints = phase_input[:, 13:16].argmax(dim=1).to(torch.int64)
        branch_actions = phase_input[:, 16:19].argmax(dim=1).to(torch.int64)
        sink = torch.zeros(batch, dtype=torch.bool, device=phase_input.device)
        neutral_input = encode_gate7_scale_neutral_child_inputs_batch(
            world_depths=world_depths,
            child_depths=child_depths,
            observed_hints=observed_hints,
            branch_actions=branch_actions,
            sink=sink,
        )
        return self.scorer.advance(state, neutral_input, repeats=repeats)

    def score(self, state: torch.Tensor) -> torch.Tensor:
        return self.scorer.score(state)

    def trainable_parameter_count(self) -> int:
        return self.scorer.trainable_parameter_count()

    def parameter_fingerprint(self) -> str:
        return self.scorer.parameter_fingerprint()


def evaluate_gate7_transition_bridge_condition(
    model: Gate3V1Scorer | Gate7ScaleNeutralGate6Adapter,
    *,
    checkpoint_index: int,
    checkpoint_family: str,
    population_size: int,
    mode: Gate6SchedulerMode,
    device: torch.device | str,
) -> Gate7TransitionBridgeCondition:
    """Evaluate one bridge condition on the exact 256 fresh bridge worlds."""

    if checkpoint_index not in GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES:
        raise ValueError("bridge checkpoint index must be 0, 1 or 2")
    if checkpoint_family not in ("transition", "original"):
        raise ValueError("bridge checkpoint family must be transition or original")
    if population_size not in GATE7_TRANSITION_BRIDGE_POPULATIONS:
        raise ValueError("bridge population must be N128 or N256")
    if mode not in GATE7_TRANSITION_BRIDGE_MODES:
        raise ValueError("bridge mode is outside the frozen matrix")
    if model.trainable_parameter_count() != GATE7_SCALE_NEUTRAL_PARAMETER_COUNT:
        raise ValueError("bridge model must contain exactly 19,649 trainable parameters")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    for start in range(0, GATE7_TRANSITION_BRIDGE_WORLD_COUNT, GATE7_TRANSITION_BRIDGE_BATCH_SIZE):
        stop = min(start + GATE7_TRANSITION_BRIDGE_BATCH_SIZE, GATE7_TRANSITION_BRIDGE_WORLD_COUNT)
        worlds = tuple(generate_gate7_transition_bridge_world(world_index=index) for index in range(start, stop))
        results = run_gate6_world_batch(
            model,
            worlds,
            population_size=population_size,
            mode=mode,
            device=device,
        )
        for world, result in zip(worlds, results, strict=True):
            telemetry = result.telemetry
            if telemetry.stage_a_parent_slots != GATE6_STAGE_A_PARENT_SLOTS:
                raise RuntimeError("bridge Stage-A work differs from frozen Gate-6 geometry")
            if telemetry.stage_b_productive_slots != GATE6_STAGE_B_PARENT_SLOTS:
                raise RuntimeError("bridge Stage-B work differs from frozen Gate-6 geometry")
            if telemetry.total_learned_updates != GATE6_TOTAL_LEARNED_UPDATES:
                raise RuntimeError("bridge learned-work identity changed")
            covered.append(world.hidden_path in set(result.generated_terminal_paths))
            runtime_seeds.append(world.public.seed)

    vector = tuple(covered)
    return Gate7TransitionBridgeCondition(
        checkpoint_index=checkpoint_index,
        checkpoint_family=checkpoint_family,
        population_size=population_size,
        mode=mode,
        world_indices=tuple(range(GATE7_TRANSITION_BRIDGE_WORLD_COUNT)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / GATE7_TRANSITION_BRIDGE_WORLD_COUNT,
        total_learned_updates_per_world=GATE6_TOTAL_LEARNED_UPDATES,
        stage_a_parent_slots=GATE6_STAGE_A_PARENT_SLOTS,
        stage_b_parent_slots=GATE6_STAGE_B_PARENT_SLOTS,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_ci(
    differences: tuple[int, ...],
    *,
    checkpoint_index: int,
    population_size: int,
    comparison: str,
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts(
            "gate7-scale-neutral-transition-bridge-bootstrap",
            checkpoint_index,
            population_size,
            comparison,
        )
    )
    count = len(differences)
    if count != GATE7_TRANSITION_BRIDGE_WORLD_COUNT:
        raise ValueError("bridge bootstrap requires exactly 256 paired worlds")
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE7_TRANSITION_BRIDGE_BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (GATE7_TRANSITION_BRIDGE_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE7_TRANSITION_BRIDGE_BOOTSTRAP_SAMPLES - 1)))],
    )


def build_gate7_transition_bridge_pair(
    *,
    comparison: str,
    treatment: Gate7TransitionBridgeCondition,
    reference: Gate7TransitionBridgeCondition,
) -> Gate7TransitionBridgePairedSummary:
    if treatment.checkpoint_index != reference.checkpoint_index:
        raise ValueError("bridge pair checkpoint mismatch")
    if treatment.population_size != reference.population_size:
        raise ValueError("bridge pair population mismatch")
    if treatment.world_indices != reference.world_indices or treatment.runtime_seeds != reference.runtime_seeds:
        raise ValueError("bridge pair does not share the same fresh worlds")
    differences = tuple(
        int(treatment_value) - int(reference_value)
        for treatment_value, reference_value in zip(
            treatment.covered_by_world,
            reference.covered_by_world,
            strict=True,
        )
    )
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=treatment.checkpoint_index,
        population_size=treatment.population_size,
        comparison=comparison,
    )
    return Gate7TransitionBridgePairedSummary(
        comparison=comparison,
        checkpoint_index=treatment.checkpoint_index,
        population_size=treatment.population_size,
        treatment_family=treatment.checkpoint_family,
        reference_family=reference.checkpoint_family,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        coverage_delta=sum(differences) / GATE7_TRANSITION_BRIDGE_WORLD_COUNT,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def classify_gate7_scale_neutral_transition_bridge(lows: dict[str, float]) -> str:
    """Apply only the preregistered twelve primary bridge criteria."""

    required: list[bool] = []
    for checkpoint in GATE7_TRANSITION_BRIDGE_CHECKPOINT_INDICES:
        required.extend(
            (
                lows[f"t{checkpoint}_n128_k16_vs_hash"] > 0.0,
                lows[f"t{checkpoint}_n256_k16_vs_hash"] > 0.0,
                lows[f"t{checkpoint}_n128_k16_vs_global"]
                > -GATE7_TRANSITION_BRIDGE_NONINFERIORITY_MARGIN,
                lows[f"t{checkpoint}_n256_transition_global_vs_original_global"]
                > -GATE7_TRANSITION_BRIDGE_NONINFERIORITY_MARGIN,
            )
        )
    if all(required):
        return "GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED"
    return "GATE7_SCALE_NEUTRAL_TRANSITION_NOT_QUALIFIED"
