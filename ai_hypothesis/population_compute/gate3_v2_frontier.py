"""Gate-3 v2 ambiguity-frontier world generation and development evaluation.

Gate-3 v2 deliberately reuses the frozen Gate-3 v1 scorer/runtime.  There is no training code here.
The only new scientific variable is public-hint ambiguity in a new deterministic data namespace.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import torch

from .gate3_v1_batch import run_gate3_v1_public_world_batch
from .gate3_v1_model import Gate3V1Scorer
from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_UPDATES_PER_ROUND,
    Gate3V1ControlMode,
    Gate3V1PublicWorld,
    score_generated_solution,
)

GATE3_V2_EXPERIMENT_VERSION = "gate3-v2-ambiguity-frontier-v0"
GATE3_V2_DEPTH = 10
GATE3_V2_SEARCH_ROUNDS = 256
GATE3_V2_TOTAL_LEARNED_UPDATES = GATE3_V2_SEARCH_ROUNDS * GATE3_V1_UPDATES_PER_ROUND
GATE3_V2_WORLD_COUNT = 256
GATE3_V2_EVAL_BATCH_SIZE = 64
GATE3_V2_BOOTSTRAP_SAMPLES = 2_000
GATE3_V2_STABLE_CAPACITIES = (1, 16, 64, 256)
GATE3_V2_CONTROL_CAPACITY = 256
GATE3_V2_CHECKPOINT_INDICES = (0, 1, 2)


class Gate3V2AmbiguityTier(str, Enum):
    A60 = "A60"
    A55 = "A55"


GATE3_V2_HINT_RELIABILITY = {
    Gate3V2AmbiguityTier.A60: 0.60,
    Gate3V2AmbiguityTier.A55: 0.55,
}

GATE3_V2_CONDITIONS: tuple[tuple[int, Gate3V1ControlMode], ...] = (
    (1, Gate3V1ControlMode.STABLE_RESERVE),
    (16, Gate3V1ControlMode.STABLE_RESERVE),
    (64, Gate3V1ControlMode.STABLE_RESERVE),
    (256, Gate3V1ControlMode.STABLE_RESERVE),
    (256, Gate3V1ControlMode.COLLAPSED_DIVERSITY),
    (256, Gate3V1ControlMode.RESHUFFLED_CONTINUITY),
)


@dataclass(frozen=True, slots=True)
class Gate3V2EvaluationWorld:
    world_index: int
    tier: Gate3V2AmbiguityTier
    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.world_index < GATE3_V2_WORLD_COUNT:
            raise ValueError("Gate-3 v2 world index is outside the frozen development domain")
        if self.public.depth != GATE3_V2_DEPTH:
            raise ValueError("Gate-3 v2 must remain at depth 10")
        self.public.validate()
        if len(self.hidden_path) != GATE3_V2_DEPTH:
            raise ValueError("hidden path must match depth 10")
        if any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("hidden path must be binary")


@dataclass(frozen=True, slots=True)
class Gate3V2CheckpointIdentity:
    checkpoint_index: int
    checkpoint_sha256: str
    parameter_fingerprint: str
    learned_parameter_count: int


@dataclass(frozen=True, slots=True)
class Gate3V2ConditionEvaluation:
    checkpoint_index: int
    tier: Gate3V2AmbiguityTier
    reserve_capacity: int
    mode: Gate3V1ControlMode
    world_count: int
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    coverage_rate: float
    productive_rounds_by_world: tuple[int, ...]
    sink_rounds_by_world: tuple[int, ...]
    generated_terminal_count_by_world: tuple[int, ...]
    unique_generated_terminal_count_by_world: tuple[int, ...]
    max_reserve_population_by_world: tuple[int, ...]
    mean_reserve_population_by_world: tuple[float, ...]
    fraction_rounds_at_capacity_by_world: tuple[float, ...]
    reached_capacity_by_world: tuple[bool, ...]
    total_learned_updates_per_world: int
    learned_parameter_count: int
    parameter_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tier"] = self.tier.value
        payload["mode"] = self.mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate3V2PairedSummary:
    comparison: str
    checkpoint_index: int
    tier: Gate3V2AmbiguityTier
    treatment_capacity: int
    treatment_mode: Gate3V1ControlMode
    reference_capacity: int
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
        payload["tier"] = self.tier.value
        payload["treatment_mode"] = self.treatment_mode.value
        payload["reference_mode"] = self.reference_mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate3V2DevelopmentResult:
    experiment_version: str
    scientific_status: str
    confirmation_opened: bool
    checkpoints: tuple[Gate3V2CheckpointIdentity, ...]
    world_count_per_tier: int
    evaluation_batch_size: int
    bootstrap_samples: int
    depth: int
    search_rounds: int
    total_learned_updates_per_world: int
    conditions: tuple[Gate3V2ConditionEvaluation, ...]
    paired_summaries: tuple[Gate3V2PairedSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_version": self.experiment_version,
            "scientific_status": self.scientific_status,
            "confirmation_opened": self.confirmation_opened,
            "checkpoints": [asdict(row) for row in self.checkpoints],
            "world_count_per_tier": self.world_count_per_tier,
            "evaluation_batch_size": self.evaluation_batch_size,
            "bootstrap_samples": self.bootstrap_samples,
            "depth": self.depth,
            "search_rounds": self.search_rounds,
            "total_learned_updates_per_world": self.total_learned_updates_per_world,
            "hint_reliability": {tier.value: value for tier, value in GATE3_V2_HINT_RELIABILITY.items()},
            "conditions": [condition.to_dict() for condition in self.conditions],
            "paired_summaries": [summary.to_dict() for summary in self.paired_summaries],
            "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
        }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def gate3_v2_runtime_seed(*, world_index: int, tier: Gate3V2AmbiguityTier) -> int:
    return _seed_from_parts("gate3-v2-frontier-development-runtime", tier.value, world_index, GATE3_V2_DEPTH)


def generate_gate3_v2_development_world(
    *, world_index: int, tier: Gate3V2AmbiguityTier
) -> Gate3V2EvaluationWorld:
    if not 0 <= world_index < GATE3_V2_WORLD_COUNT:
        raise ValueError("Gate-3 v2 world index is outside 0..255")
    reliability = GATE3_V2_HINT_RELIABILITY[tier]

    hidden_rng = random.Random(
        _seed_from_parts("gate3-v2-frontier-development-hidden", world_index, GATE3_V2_DEPTH)
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE3_V2_DEPTH))

    # A60/A55 share the same underlying uniform draws.  Lowering the threshold therefore changes
    # only ambiguity while holding the hidden solution and random corruption propensity paired.
    hint_rng = random.Random(
        _seed_from_parts("gate3-v2-frontier-development-hints", world_index, GATE3_V2_DEPTH)
    )
    uniforms = tuple(hint_rng.random() for _ in range(GATE3_V2_DEPTH))
    noisy_hints = tuple(
        hidden_bit if uniform < reliability else 1 - hidden_bit
        for hidden_bit, uniform in zip(hidden_path, uniforms, strict=True)
    )

    world = Gate3V2EvaluationWorld(
        world_index=world_index,
        tier=tier,
        public=Gate3V1PublicWorld(
            seed=gate3_v2_runtime_seed(world_index=world_index, tier=tier),
            depth=GATE3_V2_DEPTH,
            noisy_hints=noisy_hints,
        ),
        hidden_path=hidden_path,
    )
    world.validate()
    return world


def _mean(values: tuple[int, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_gate3_v2_condition(
    model: Gate3V1Scorer,
    *,
    checkpoint_index: int,
    tier: Gate3V2AmbiguityTier,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    device: torch.device | str,
    world_count: int = GATE3_V2_WORLD_COUNT,
    evaluation_batch_size: int = GATE3_V2_EVAL_BATCH_SIZE,
) -> Gate3V2ConditionEvaluation:
    if checkpoint_index not in GATE3_V2_CHECKPOINT_INDICES:
        raise ValueError("checkpoint index must be 0, 1 or 2")
    if world_count != GATE3_V2_WORLD_COUNT:
        raise ValueError("Gate-3 v2 development must use exactly 256 worlds/tier")
    if evaluation_batch_size != GATE3_V2_EVAL_BATCH_SIZE:
        raise ValueError("Gate-3 v2 development batch size is frozen at 64")
    if (reserve_capacity, mode) not in GATE3_V2_CONDITIONS:
        raise ValueError("condition is outside the frozen Gate-3 v2 matrix")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    productive_rounds: list[int] = []
    sink_rounds: list[int] = []
    generated_terminals: list[int] = []
    unique_terminals: list[int] = []
    max_reserve: list[int] = []
    mean_reserve: list[float] = []
    fraction_at_capacity: list[float] = []
    reached_capacity: list[bool] = []

    for start in range(0, world_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, world_count)
        worlds = tuple(
            generate_gate3_v2_development_world(world_index=index, tier=tier)
            for index in range(start, stop)
        )
        batch = run_gate3_v1_public_world_batch(
            model,
            (world.public for world in worlds),
            reserve_capacity=reserve_capacity,
            mode=mode,
            device=device,
        )
        if batch.world_seeds != tuple(world.public.seed for world in worlds):
            raise RuntimeError("Gate-3 v2 batched runtime changed world ordering")

        for world, runtime_result in zip(worlds, batch.world_results, strict=True):
            telemetry = runtime_result.telemetry
            if telemetry.total_learned_updates != GATE3_V2_TOTAL_LEARNED_UPDATES:
                raise RuntimeError("Gate-3 v2 learned-work identity was violated")
            populations = telemetry.reserve_population_by_round
            if len(populations) != GATE3_V2_SEARCH_ROUNDS:
                raise RuntimeError("Gate-3 v2 reserve telemetry must contain all 256 rounds")

            covered.append(
                score_generated_solution(
                    hidden_path=world.hidden_path,
                    generated_terminal_paths=runtime_result.generated_terminal_paths,
                )
            )
            runtime_seeds.append(world.public.seed)
            productive_rounds.append(telemetry.productive_rounds)
            sink_rounds.append(telemetry.sink_rounds)
            generated_terminals.append(telemetry.generated_terminal_count)
            unique_terminals.append(telemetry.unique_generated_terminal_count)
            max_value = max(populations) if populations else 0
            max_reserve.append(max_value)
            mean_reserve.append(_mean(populations))
            at_capacity = sum(int(value >= reserve_capacity) for value in populations)
            fraction_at_capacity.append(at_capacity / GATE3_V2_SEARCH_ROUNDS)
            reached_capacity.append(max_value >= reserve_capacity)

    vector = tuple(covered)
    return Gate3V2ConditionEvaluation(
        checkpoint_index=checkpoint_index,
        tier=tier,
        reserve_capacity=reserve_capacity,
        mode=mode,
        world_count=world_count,
        world_indices=tuple(range(world_count)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / world_count,
        productive_rounds_by_world=tuple(productive_rounds),
        sink_rounds_by_world=tuple(sink_rounds),
        generated_terminal_count_by_world=tuple(generated_terminals),
        unique_generated_terminal_count_by_world=tuple(unique_terminals),
        max_reserve_population_by_world=tuple(max_reserve),
        mean_reserve_population_by_world=tuple(mean_reserve),
        fraction_rounds_at_capacity_by_world=tuple(fraction_at_capacity),
        reached_capacity_by_world=tuple(reached_capacity),
        total_learned_updates_per_world=GATE3_V2_TOTAL_LEARNED_UPDATES,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint_index: int, tier: Gate3V2AmbiguityTier, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts("gate3-v2-frontier-bootstrap", checkpoint_index, tier.value, comparison)
    )
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE3_V2_BOOTSTRAP_SAMPLES)
    )
    low = estimates[int(math.floor(0.025 * (GATE3_V2_BOOTSTRAP_SAMPLES - 1)))]
    high = estimates[int(math.ceil(0.975 * (GATE3_V2_BOOTSTRAP_SAMPLES - 1)))]
    return low, high


def _paired_summary(
    *,
    comparison: str,
    treatment: Gate3V2ConditionEvaluation,
    reference: Gate3V2ConditionEvaluation,
) -> Gate3V2PairedSummary:
    if treatment.checkpoint_index != reference.checkpoint_index or treatment.tier != reference.tier:
        raise ValueError("paired conditions must share checkpoint and ambiguity tier")
    if treatment.world_indices != reference.world_indices:
        raise ValueError("paired conditions must share exact world indices")
    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    differences = tuple(int(bool(a)) - int(bool(b)) for a, b in pairs)
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = len(pairs) - treatment_only - reference_only - both
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=treatment.checkpoint_index,
        tier=treatment.tier,
        comparison=comparison,
    )
    return Gate3V2PairedSummary(
        comparison=comparison,
        checkpoint_index=treatment.checkpoint_index,
        tier=treatment.tier,
        treatment_capacity=treatment.reserve_capacity,
        treatment_mode=treatment.mode,
        reference_capacity=reference.reserve_capacity,
        reference_mode=reference.mode,
        world_count=len(pairs),
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=sum(differences) / len(differences),
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate3_v2_paired_summaries(
    conditions: Iterable[Gate3V2ConditionEvaluation],
) -> tuple[Gate3V2PairedSummary, ...]:
    rows = tuple(conditions)
    index = {
        (row.checkpoint_index, row.tier, row.reserve_capacity, row.mode): row
        for row in rows
    }
    expected = len(GATE3_V2_CHECKPOINT_INDICES) * len(Gate3V2AmbiguityTier) * len(GATE3_V2_CONDITIONS)
    if len(rows) != expected or len(index) != expected:
        raise ValueError("Gate-3 v2 requires the complete 36-cell development matrix")

    summaries: list[Gate3V2PairedSummary] = []
    for checkpoint_index in GATE3_V2_CHECKPOINT_INDICES:
        for tier in Gate3V2AmbiguityTier:
            stable = Gate3V1ControlMode.STABLE_RESERVE
            collapsed = Gate3V1ControlMode.COLLAPSED_DIVERSITY
            reshuffled = Gate3V1ControlMode.RESHUFFLED_CONTINUITY
            specs = (
                ("stable_l256_vs_l64", (256, stable), (64, stable)),
                ("stable_l64_vs_l16", (64, stable), (16, stable)),
                ("stable_l256_vs_l1", (256, stable), (1, stable)),
                ("stable_l256_vs_collapsed", (256, stable), (256, collapsed)),
                ("stable_l256_vs_reshuffled", (256, stable), (256, reshuffled)),
            )
            for comparison, treatment_key, reference_key in specs:
                summaries.append(
                    _paired_summary(
                        comparison=comparison,
                        treatment=index[(checkpoint_index, tier, *treatment_key)],
                        reference=index[(checkpoint_index, tier, *reference_key)],
                    )
                )
    return tuple(summaries)
