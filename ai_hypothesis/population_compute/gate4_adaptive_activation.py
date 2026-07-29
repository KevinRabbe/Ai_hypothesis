"""Gate-4 v0 adaptive activation under fixed latent population and learned work.

Gate-4 reuses the frozen Gate-3 v1 recurrent scorer/checkpoints.  All conditions keep L256,
two active child lanes, eight recurrent updates/child and exactly 159 scheduled parent slots.
Only the answer-blind parent activation schedule differs.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import torch

from .gate3_v1_model import Gate3V1NeuralCandidate, Gate3V1Scorer, encode_gate3_v1_child_input
from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    GATE3_V1_UPDATES_PER_ROUND,
    Gate3V1PublicWorld,
    deterministic_gate3_v1_tie_break,
    quantize_gate3_v1_score,
)

GATE4_EXPERIMENT_VERSION = "gate4-adaptive-activation-v0"
GATE4_DEPTH = 8
GATE4_HINT_RELIABILITY = 0.70
GATE4_WORLD_COUNT = 256
GATE4_EVAL_BATCH_SIZE = 64
GATE4_BOOTSTRAP_SAMPLES = 2_000
GATE4_RESERVE_CAPACITY = 256
GATE4_SCHEDULED_SLOTS = 159
GATE4_TOTAL_LEARNED_UPDATES = GATE4_SCHEDULED_SLOTS * GATE3_V1_UPDATES_PER_ROUND
GATE4_CHECKPOINT_INDICES = (0, 1, 2)
GATE4_PARAMETER_COUNT = 19_649
GATE4_STATIC_DEPTH7_BUILD_SLOTS = 127
GATE4_STATIC_FINAL_PARENT_SLOTS = 32
GATE4_STATIC_TERMINAL_COUNT = 64


class Gate4SchedulerMode(str, Enum):
    ADAPTIVE_SCORE = "adaptive_score"
    STATIC_GENERATION = "static_generation"
    ADAPTIVE_HASH = "adaptive_hash"


GATE4_CONDITIONS = (
    Gate4SchedulerMode.ADAPTIVE_SCORE,
    Gate4SchedulerMode.STATIC_GENERATION,
    Gate4SchedulerMode.ADAPTIVE_HASH,
)


@dataclass(frozen=True, slots=True)
class Gate4EvaluationWorld:
    world_index: int
    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.world_index < GATE4_WORLD_COUNT:
            raise ValueError("Gate-4 world index is outside the frozen development domain")
        self.public.validate()
        if self.public.depth != GATE4_DEPTH:
            raise ValueError("Gate-4 must remain at depth 8")
        if len(self.hidden_path) != GATE4_DEPTH or any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("Gate-4 hidden path must contain eight binary decisions")


@dataclass(frozen=True, slots=True)
class Gate4CheckpointIdentity:
    checkpoint_index: int
    checkpoint_sha256: str
    parameter_fingerprint: str
    learned_parameter_count: int


@dataclass(frozen=True, slots=True)
class Gate4WorldTelemetry:
    productive_slots: int
    sink_slots: int
    total_learned_updates: int
    live_nonterminal_population_by_slot: tuple[int, ...]
    max_live_nonterminal_population: int
    mean_live_nonterminal_population: float
    distinct_parent_depths_activated: int
    productive_activations_by_parent_depth: tuple[int, ...]
    activated_parent_depth_by_slot: tuple[int, ...]
    terminal_generation_slot_indices: tuple[int, ...]
    generated_terminal_count: int
    unique_generated_terminal_count: int


@dataclass(frozen=True, slots=True)
class Gate4WorldResult:
    generated_terminal_paths: tuple[tuple[int, ...], ...]
    telemetry: Gate4WorldTelemetry


@dataclass(frozen=True, slots=True)
class Gate4ConditionEvaluation:
    checkpoint_index: int
    mode: Gate4SchedulerMode
    world_count: int
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    coverage_rate: float
    productive_slots_by_world: tuple[int, ...]
    sink_slots_by_world: tuple[int, ...]
    live_nonterminal_population_by_slot_by_world: tuple[tuple[int, ...], ...]
    max_live_nonterminal_population_by_world: tuple[int, ...]
    mean_live_nonterminal_population_by_world: tuple[float, ...]
    distinct_parent_depths_activated_by_world: tuple[int, ...]
    productive_activations_by_parent_depth_by_world: tuple[tuple[int, ...], ...]
    activated_parent_depth_by_slot_by_world: tuple[tuple[int, ...], ...]
    terminal_generation_slot_indices_by_world: tuple[tuple[int, ...], ...]
    generated_terminal_count_by_world: tuple[int, ...]
    unique_generated_terminal_count_by_world: tuple[int, ...]
    total_learned_updates_per_world: int
    reserve_capacity: int
    learned_parameter_count: int
    parameter_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate4PairedSummary:
    comparison: str
    checkpoint_index: int
    treatment_mode: Gate4SchedulerMode
    reference_mode: Gate4SchedulerMode
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
class Gate4DevelopmentResult:
    experiment_version: str
    scientific_status: str
    confirmation_opened: bool
    training_performed: bool
    checkpoints: tuple[Gate4CheckpointIdentity, ...]
    world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    depth: int
    hint_reliability: float
    reserve_capacity: int
    scheduled_slots: int
    total_learned_updates_per_world: int
    conditions: tuple[Gate4ConditionEvaluation, ...]
    paired_summaries: tuple[Gate4PairedSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_version": self.experiment_version,
            "scientific_status": self.scientific_status,
            "confirmation_opened": self.confirmation_opened,
            "training_performed": self.training_performed,
            "checkpoints": [asdict(row) for row in self.checkpoints],
            "world_count": self.world_count,
            "evaluation_batch_size": self.evaluation_batch_size,
            "bootstrap_samples": self.bootstrap_samples,
            "depth": self.depth,
            "hint_reliability": self.hint_reliability,
            "reserve_capacity": self.reserve_capacity,
            "scheduled_slots": self.scheduled_slots,
            "active_child_lanes": 2,
            "recurrent_updates_per_child": GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            "total_learned_updates_per_world": self.total_learned_updates_per_world,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "paired_summaries": [summary.to_dict() for summary in self.paired_summaries],
            "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
        }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def gate4_runtime_seed(*, world_index: int) -> int:
    return _seed_from_parts("gate4-adaptive-activation-development-runtime", world_index, GATE4_DEPTH)


def generate_gate4_development_world(*, world_index: int) -> Gate4EvaluationWorld:
    if not 0 <= world_index < GATE4_WORLD_COUNT:
        raise ValueError("Gate-4 world index is outside 0..255")
    hidden_rng = random.Random(
        _seed_from_parts("gate4-adaptive-activation-development-hidden", world_index, GATE4_DEPTH)
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE4_DEPTH))
    hint_rng = random.Random(
        _seed_from_parts("gate4-adaptive-activation-development-hints", world_index, GATE4_DEPTH)
    )
    noisy_hints = tuple(
        hidden_bit if hint_rng.random() < GATE4_HINT_RELIABILITY else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    world = Gate4EvaluationWorld(
        world_index=world_index,
        public=Gate3V1PublicWorld(
            seed=gate4_runtime_seed(world_index=world_index),
            depth=GATE4_DEPTH,
            noisy_hints=noisy_hints,
        ),
        hidden_path=hidden_path,
    )
    world.validate()
    return world


def _score_rank(
    candidates: Iterable[Gate3V1NeuralCandidate], *, world_seed: int, expansion_index: int
) -> tuple[Gate3V1NeuralCandidate, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -quantize_gate3_v1_score(candidate.score),
                deterministic_gate3_v1_tie_break(
                    world_seed=world_seed,
                    expansion_index=expansion_index,
                    candidate_path=candidate.path,
                ),
            ),
        )
    )


def _hash_priority(*, world_seed: int, slot_index: int, candidate_path: tuple[int, ...]) -> int:
    return _seed_from_parts(
        "gate4-adaptive-hash-priority", world_seed, slot_index, "".join(str(bit) for bit in candidate_path)
    )


def _hash_rank(
    candidates: Iterable[Gate3V1NeuralCandidate], *, world_seed: int, slot_index: int
) -> tuple[Gate3V1NeuralCandidate, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                _hash_priority(
                    world_seed=world_seed,
                    slot_index=slot_index,
                    candidate_path=candidate.path,
                ),
                candidate.path,
            ),
        )
    )


def _advance_slot_batch(
    model: Gate3V1Scorer,
    worlds: tuple[Gate4EvaluationWorld, ...],
    parents: tuple[Gate3V1NeuralCandidate | None, ...],
    *,
    device: torch.device,
) -> tuple[tuple[Gate3V1NeuralCandidate, ...], ...]:
    if len(worlds) != len(parents):
        raise ValueError("Gate-4 worlds/parents batch lengths differ")

    states: list[torch.Tensor] = []
    inputs: list[torch.Tensor] = []
    owner: list[tuple[int, tuple[int, ...] | None, int, int | None]] = []

    for world_offset, (world, parent) in enumerate(zip(worlds, parents, strict=True)):
        if parent is None:
            sink_states = model.initial_state(2, device=device)
            for action in (0, 1):
                states.append(sink_states[action])
                inputs.append(
                    encode_gate3_v1_child_input(
                        world=world.public,
                        child_depth=GATE4_DEPTH,
                        observed_hint=None,
                        branch_action=None,
                        sink=True,
                        device=device,
                    )
                )
                owner.append((world_offset, None, action, None))
            continue

        next_depth = parent.depth + 1
        if not 1 <= next_depth <= GATE4_DEPTH:
            raise RuntimeError("Gate-4 attempted to expand a terminal/nonexistent parent")
        hint = world.public.noisy_hints[next_depth - 1]
        for action in (0, 1):
            states.append(parent.state.clone())
            inputs.append(
                encode_gate3_v1_child_input(
                    world=world.public,
                    child_depth=next_depth,
                    observed_hint=hint,
                    branch_action=action,
                    sink=False,
                    device=device,
                )
            )
            owner.append((world_offset, parent.path, action, next_depth))

    state_batch = torch.stack(states, dim=0)
    input_batch = torch.stack(inputs, dim=0)
    advanced = model.advance(
        state_batch,
        input_batch,
        repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    )
    scores = model.score(advanced)

    children: list[list[Gate3V1NeuralCandidate]] = [[] for _ in worlds]
    for row_index, (world_offset, parent_path, action, next_depth) in enumerate(owner):
        if parent_path is None or next_depth is None:
            continue
        children[world_offset].append(
            Gate3V1NeuralCandidate(
                path=parent_path + (action,),
                state=advanced[row_index].clone(),
                score=float(scores[row_index].item()),
            )
        )
    return tuple(tuple(row) for row in children)


def _build_world_result(
    *,
    generated_terminals: list[tuple[int, ...]],
    productive_slots: int,
    live_population_by_slot: list[int],
    productive_by_depth: list[int],
    activated_depth_by_slot: list[int],
    terminal_slot_indices: list[int],
) -> Gate4WorldResult:
    if len(live_population_by_slot) != GATE4_SCHEDULED_SLOTS:
        raise RuntimeError("Gate-4 live-population telemetry is incomplete")
    if len(activated_depth_by_slot) != GATE4_SCHEDULED_SLOTS:
        raise RuntimeError("Gate-4 activation-depth telemetry is incomplete")
    if len(productive_by_depth) != GATE4_DEPTH:
        raise RuntimeError("Gate-4 parent-depth telemetry is malformed")
    sink_slots = GATE4_SCHEDULED_SLOTS - productive_slots
    if sink_slots < 0:
        raise RuntimeError("Gate-4 productive slots exceeded frozen budget")
    if sum(productive_by_depth) != productive_slots:
        raise RuntimeError("Gate-4 productive-by-depth accounting mismatch")
    if activated_depth_by_slot.count(-1) != sink_slots:
        raise RuntimeError("Gate-4 sink activation-depth accounting mismatch")
    total_updates = (productive_slots + sink_slots) * GATE3_V1_UPDATES_PER_ROUND
    if total_updates != GATE4_TOTAL_LEARNED_UPDATES:
        raise RuntimeError("Gate-4 learned-work identity was violated")
    if any(value < 0 or value > GATE4_RESERVE_CAPACITY for value in live_population_by_slot):
        raise RuntimeError("Gate-4 live population violated frozen L256 capacity")
    if len(terminal_slot_indices) != len(generated_terminals):
        raise RuntimeError("Gate-4 terminal slot telemetry does not match generated terminals")

    mean_population = sum(live_population_by_slot) / GATE4_SCHEDULED_SLOTS
    return Gate4WorldResult(
        generated_terminal_paths=tuple(generated_terminals),
        telemetry=Gate4WorldTelemetry(
            productive_slots=productive_slots,
            sink_slots=sink_slots,
            total_learned_updates=total_updates,
            live_nonterminal_population_by_slot=tuple(live_population_by_slot),
            max_live_nonterminal_population=max(live_population_by_slot, default=0),
            mean_live_nonterminal_population=mean_population,
            distinct_parent_depths_activated=sum(int(value > 0) for value in productive_by_depth),
            productive_activations_by_parent_depth=tuple(productive_by_depth),
            activated_parent_depth_by_slot=tuple(activated_depth_by_slot),
            terminal_generation_slot_indices=tuple(terminal_slot_indices),
            generated_terminal_count=len(generated_terminals),
            unique_generated_terminal_count=len(set(generated_terminals)),
        ),
    )


def _run_adaptive_batch(
    model: Gate3V1Scorer,
    worlds: tuple[Gate4EvaluationWorld, ...],
    *,
    mode: Gate4SchedulerMode,
    device: torch.device,
) -> tuple[Gate4WorldResult, ...]:
    if mode not in (Gate4SchedulerMode.ADAPTIVE_SCORE, Gate4SchedulerMode.ADAPTIVE_HASH):
        raise ValueError("adaptive Gate-4 runner received a non-adaptive mode")

    reserves: list[list[Gate3V1NeuralCandidate]] = [
        [
            Gate3V1NeuralCandidate(
                path=(),
                state=model.initial_state(1, device=device)[0],
                score=0.0,
            )
        ]
        for _ in worlds
    ]
    generated_terminals: list[list[tuple[int, ...]]] = [[] for _ in worlds]
    terminal_slots: list[list[int]] = [[] for _ in worlds]
    live_populations: list[list[int]] = [[] for _ in worlds]
    productive_by_depth: list[list[int]] = [[0 for _ in range(GATE4_DEPTH)] for _ in worlds]
    activated_depths: list[list[int]] = [[] for _ in worlds]
    productive = [0 for _ in worlds]

    with torch.inference_mode():
        for slot_index in range(GATE4_SCHEDULED_SLOTS):
            selected: list[Gate3V1NeuralCandidate | None] = []
            for world_offset, world in enumerate(worlds):
                reserve = reserves[world_offset]
                if not reserve:
                    selected.append(None)
                    continue
                if mode is Gate4SchedulerMode.ADAPTIVE_SCORE:
                    ranked = _score_rank(
                        reserve,
                        world_seed=world.public.seed,
                        expansion_index=slot_index,
                    )
                else:
                    ranked = _hash_rank(
                        reserve,
                        world_seed=world.public.seed,
                        slot_index=slot_index,
                    )
                parent = ranked[0]
                selected.append(parent)
                reserves[world_offset] = [candidate for candidate in reserve if candidate.path != parent.path]

            children_by_world = _advance_slot_batch(
                model,
                worlds,
                tuple(selected),
                device=device,
            )

            for world_offset, (world, parent, children) in enumerate(
                zip(worlds, selected, children_by_world, strict=True)
            ):
                if parent is None:
                    activated_depths[world_offset].append(-1)
                    live_populations[world_offset].append(len(reserves[world_offset]))
                    continue

                productive[world_offset] += 1
                productive_by_depth[world_offset][parent.depth] += 1
                activated_depths[world_offset].append(parent.depth)
                next_depth = parent.depth + 1
                if next_depth == GATE4_DEPTH:
                    for child in children:
                        generated_terminals[world_offset].append(child.path)
                        terminal_slots[world_offset].append(slot_index)
                else:
                    reserves[world_offset].extend(children)
                    if len(reserves[world_offset]) > GATE4_RESERVE_CAPACITY:
                        if mode is Gate4SchedulerMode.ADAPTIVE_SCORE:
                            reserves[world_offset] = list(
                                _score_rank(
                                    reserves[world_offset],
                                    world_seed=world.public.seed,
                                    expansion_index=slot_index,
                                )[:GATE4_RESERVE_CAPACITY]
                            )
                        else:
                            reserves[world_offset] = list(
                                _hash_rank(
                                    reserves[world_offset],
                                    world_seed=world.public.seed,
                                    slot_index=slot_index,
                                )[:GATE4_RESERVE_CAPACITY]
                            )
                live_populations[world_offset].append(len(reserves[world_offset]))

    return tuple(
        _build_world_result(
            generated_terminals=generated_terminals[index],
            productive_slots=productive[index],
            live_population_by_slot=live_populations[index],
            productive_by_depth=productive_by_depth[index],
            activated_depth_by_slot=activated_depths[index],
            terminal_slot_indices=terminal_slots[index],
        )
        for index in range(len(worlds))
    )


def _static_generation_rank_index(parent_depth: int) -> int:
    # Match Gate-3 v3's final depth-7 parent ranking, which uses expansion index == depth (8).
    return GATE4_DEPTH if parent_depth == GATE4_DEPTH - 1 else parent_depth


def _run_static_generation_batch(
    model: Gate3V1Scorer,
    worlds: tuple[Gate4EvaluationWorld, ...],
    *,
    device: torch.device,
) -> tuple[Gate4WorldResult, ...]:
    current: list[list[Gate3V1NeuralCandidate]] = [
        [
            Gate3V1NeuralCandidate(
                path=(),
                state=model.initial_state(1, device=device)[0],
                score=0.0,
            )
        ]
        for _ in worlds
    ]
    next_generation: list[list[Gate3V1NeuralCandidate]] = [[] for _ in worlds]
    generated_terminals: list[list[tuple[int, ...]]] = [[] for _ in worlds]
    terminal_slots: list[list[int]] = [[] for _ in worlds]
    live_populations: list[list[int]] = [[] for _ in worlds]
    productive_by_depth: list[list[int]] = [[0 for _ in range(GATE4_DEPTH)] for _ in worlds]
    activated_depths: list[list[int]] = [[] for _ in worlds]
    productive = [0 for _ in worlds]

    def prepare_generation(world_offset: int) -> None:
        if current[world_offset] or not next_generation[world_offset]:
            return
        parent_depth = next_generation[world_offset][0].depth
        ranked = _score_rank(
            next_generation[world_offset],
            world_seed=worlds[world_offset].public.seed,
            expansion_index=_static_generation_rank_index(parent_depth),
        )
        current[world_offset] = list(ranked)
        next_generation[world_offset] = []

    with torch.inference_mode():
        for slot_index in range(GATE4_SCHEDULED_SLOTS):
            selected: list[Gate3V1NeuralCandidate | None] = []
            for world_offset in range(len(worlds)):
                prepare_generation(world_offset)
                if not current[world_offset]:
                    selected.append(None)
                    continue
                selected.append(current[world_offset].pop(0))

            children_by_world = _advance_slot_batch(
                model,
                worlds,
                tuple(selected),
                device=device,
            )

            for world_offset, (parent, children) in enumerate(zip(selected, children_by_world, strict=True)):
                if parent is None:
                    activated_depths[world_offset].append(-1)
                    live_populations[world_offset].append(
                        len(current[world_offset]) + len(next_generation[world_offset])
                    )
                    continue

                productive[world_offset] += 1
                productive_by_depth[world_offset][parent.depth] += 1
                activated_depths[world_offset].append(parent.depth)
                next_depth = parent.depth + 1
                if next_depth == GATE4_DEPTH:
                    for child in children:
                        generated_terminals[world_offset].append(child.path)
                        terminal_slots[world_offset].append(slot_index)
                else:
                    next_generation[world_offset].extend(children)
                    if len(next_generation[world_offset]) > GATE4_RESERVE_CAPACITY:
                        raise RuntimeError("Gate-4 static next generation exceeded frozen L256 capacity")

                live_population = len(current[world_offset]) + len(next_generation[world_offset])
                if live_population > GATE4_RESERVE_CAPACITY:
                    raise RuntimeError("Gate-4 static live population exceeded frozen L256 capacity")
                live_populations[world_offset].append(live_population)

    results = tuple(
        _build_world_result(
            generated_terminals=generated_terminals[index],
            productive_slots=productive[index],
            live_population_by_slot=live_populations[index],
            productive_by_depth=productive_by_depth[index],
            activated_depth_by_slot=activated_depths[index],
            terminal_slot_indices=terminal_slots[index],
        )
        for index in range(len(worlds))
    )

    expected_depth_counts = (1, 2, 4, 8, 16, 32, 64, 32)
    for result in results:
        telemetry = result.telemetry
        if telemetry.productive_slots != GATE4_SCHEDULED_SLOTS or telemetry.sink_slots != 0:
            raise RuntimeError("Gate-4 static schedule did not use exactly 159 productive slots")
        if telemetry.productive_activations_by_parent_depth != expected_depth_counts:
            raise RuntimeError("Gate-4 static generation parent-depth schedule changed")
        if telemetry.generated_terminal_count != GATE4_STATIC_TERMINAL_COUNT:
            raise RuntimeError("Gate-4 static schedule did not generate exactly 64 terminals")
    return results


def run_gate4_world_batch(
    model: Gate3V1Scorer,
    worlds: Iterable[Gate4EvaluationWorld],
    *,
    mode: Gate4SchedulerMode,
    device: torch.device | str,
) -> tuple[Gate4WorldResult, ...]:
    world_tuple = tuple(worlds)
    if not world_tuple:
        return ()
    if mode not in GATE4_CONDITIONS:
        raise ValueError("Gate-4 scheduler mode is outside the frozen matrix")
    for world in world_tuple:
        world.validate()

    target_device = torch.device(device)
    model = model.to(target_device)
    if mode is Gate4SchedulerMode.STATIC_GENERATION:
        return _run_static_generation_batch(model, world_tuple, device=target_device)
    return _run_adaptive_batch(model, world_tuple, mode=mode, device=target_device)


def evaluate_gate4_condition(
    model: Gate3V1Scorer,
    *,
    checkpoint_index: int,
    mode: Gate4SchedulerMode,
    device: torch.device | str,
    world_count: int = GATE4_WORLD_COUNT,
    evaluation_batch_size: int = GATE4_EVAL_BATCH_SIZE,
) -> Gate4ConditionEvaluation:
    if checkpoint_index not in GATE4_CHECKPOINT_INDICES:
        raise ValueError("Gate-4 checkpoint index must be 0, 1 or 2")
    if mode not in GATE4_CONDITIONS:
        raise ValueError("Gate-4 condition is outside the frozen matrix")
    if world_count != GATE4_WORLD_COUNT:
        raise ValueError("Gate-4 development must use exactly 256 worlds")
    if evaluation_batch_size != GATE4_EVAL_BATCH_SIZE:
        raise ValueError("Gate-4 development batch size is frozen at 64")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    productive: list[int] = []
    sink: list[int] = []
    live_population: list[tuple[int, ...]] = []
    max_population: list[int] = []
    mean_population: list[float] = []
    distinct_depths: list[int] = []
    productive_by_depth: list[tuple[int, ...]] = []
    activated_depths: list[tuple[int, ...]] = []
    terminal_slots: list[tuple[int, ...]] = []
    terminal_count: list[int] = []
    unique_terminal_count: list[int] = []

    for start in range(0, world_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, world_count)
        worlds = tuple(generate_gate4_development_world(world_index=index) for index in range(start, stop))
        batch = run_gate4_world_batch(model, worlds, mode=mode, device=device)
        for world, result in zip(worlds, batch, strict=True):
            telemetry = result.telemetry
            covered.append(world.hidden_path in set(result.generated_terminal_paths))
            runtime_seeds.append(world.public.seed)
            productive.append(telemetry.productive_slots)
            sink.append(telemetry.sink_slots)
            live_population.append(telemetry.live_nonterminal_population_by_slot)
            max_population.append(telemetry.max_live_nonterminal_population)
            mean_population.append(telemetry.mean_live_nonterminal_population)
            distinct_depths.append(telemetry.distinct_parent_depths_activated)
            productive_by_depth.append(telemetry.productive_activations_by_parent_depth)
            activated_depths.append(telemetry.activated_parent_depth_by_slot)
            terminal_slots.append(telemetry.terminal_generation_slot_indices)
            terminal_count.append(telemetry.generated_terminal_count)
            unique_terminal_count.append(telemetry.unique_generated_terminal_count)

    vector = tuple(covered)
    return Gate4ConditionEvaluation(
        checkpoint_index=checkpoint_index,
        mode=mode,
        world_count=world_count,
        world_indices=tuple(range(world_count)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / world_count,
        productive_slots_by_world=tuple(productive),
        sink_slots_by_world=tuple(sink),
        live_nonterminal_population_by_slot_by_world=tuple(live_population),
        max_live_nonterminal_population_by_world=tuple(max_population),
        mean_live_nonterminal_population_by_world=tuple(mean_population),
        distinct_parent_depths_activated_by_world=tuple(distinct_depths),
        productive_activations_by_parent_depth_by_world=tuple(productive_by_depth),
        activated_parent_depth_by_slot_by_world=tuple(activated_depths),
        terminal_generation_slot_indices_by_world=tuple(terminal_slots),
        generated_terminal_count_by_world=tuple(terminal_count),
        unique_generated_terminal_count_by_world=tuple(unique_terminal_count),
        total_learned_updates_per_world=GATE4_TOTAL_LEARNED_UPDATES,
        reserve_capacity=GATE4_RESERVE_CAPACITY,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint_index: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts("gate4-adaptive-activation-bootstrap", checkpoint_index, comparison)
    )
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE4_BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (GATE4_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE4_BOOTSTRAP_SAMPLES - 1)))],
    )


def _paired_summary(
    *,
    comparison: str,
    checkpoint_index: int,
    treatment: Gate4ConditionEvaluation,
    reference: Gate4ConditionEvaluation,
) -> Gate4PairedSummary:
    if treatment.world_indices != reference.world_indices:
        raise ValueError("Gate-4 paired conditions use different world indices")
    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = GATE4_WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=checkpoint_index,
        comparison=comparison,
    )
    return Gate4PairedSummary(
        comparison=comparison,
        checkpoint_index=checkpoint_index,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        world_count=GATE4_WORLD_COUNT,
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=sum(differences) / GATE4_WORLD_COUNT,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate4_paired_summaries(
    conditions: Iterable[Gate4ConditionEvaluation],
) -> tuple[Gate4PairedSummary, ...]:
    rows = tuple(conditions)
    index = {(row.checkpoint_index, row.mode): row for row in rows}
    specs = (
        (
            "adaptive_score_vs_static_generation",
            Gate4SchedulerMode.ADAPTIVE_SCORE,
            Gate4SchedulerMode.STATIC_GENERATION,
        ),
        (
            "adaptive_score_vs_adaptive_hash",
            Gate4SchedulerMode.ADAPTIVE_SCORE,
            Gate4SchedulerMode.ADAPTIVE_HASH,
        ),
        (
            "static_generation_vs_adaptive_hash",
            Gate4SchedulerMode.STATIC_GENERATION,
            Gate4SchedulerMode.ADAPTIVE_HASH,
        ),
    )
    summaries: list[Gate4PairedSummary] = []
    for checkpoint_index in GATE4_CHECKPOINT_INDICES:
        for comparison, treatment_mode, reference_mode in specs:
            summaries.append(
                _paired_summary(
                    comparison=comparison,
                    checkpoint_index=checkpoint_index,
                    treatment=index[(checkpoint_index, treatment_mode)],
                    reference=index[(checkpoint_index, reference_mode)],
                )
            )
    return tuple(summaries)
