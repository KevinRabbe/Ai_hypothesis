"""Gate-6 v0 fixed-K routing under controlled live-population scaling.

No training occurs here. Every condition first builds the same complete depth-8 neural frontier,
then applies answer-blind population thinning/capacity pruning before a fixed 128-slot Stage-B
routing budget. The causal variables are Stage-B live-population capacity and scheduler visibility.
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

GATE6_EXPERIMENT_VERSION = "gate6-fixed-k-population-scaling-v0"
GATE6_DEPTH = 10
GATE6_FRONTIER_DEPTH = 8
GATE6_FULL_FRONTIER = 256
GATE6_POPULATION_LADDER = (64, 128, 256)
GATE6_WORLD_COUNT = 256
GATE6_EVAL_BATCH_SIZE = 64
GATE6_BOOTSTRAP_SAMPLES = 2_000
GATE6_HINT_RELIABILITY = 0.70
GATE6_STAGE_A_PARENT_SLOTS = 255
GATE6_STAGE_B_PARENT_SLOTS = 128
GATE6_SCHEDULED_PARENT_SLOTS = GATE6_STAGE_A_PARENT_SLOTS + GATE6_STAGE_B_PARENT_SLOTS
GATE6_TOTAL_LEARNED_UPDATES = GATE6_SCHEDULED_PARENT_SLOTS * GATE3_V1_UPDATES_PER_ROUND
GATE6_NONINFERIORITY_MARGIN = 0.05
GATE6_PRIMARY_K = 16
GATE6_DESCRIPTIVE_K = 8
GATE6_CHECKPOINT_INDICES = (0, 1, 2)
GATE6_PARAMETER_COUNT = 19_649


class Gate6SchedulerMode(str, Enum):
    GLOBAL_SCORE = "global_score"
    BOUNDED_SCORE_K16 = "bounded_score_k16"
    BOUNDED_HASH_K16 = "bounded_hash_k16"
    BOUNDED_SCORE_K8 = "bounded_score_k8"


GATE6_CONDITIONS = (
    Gate6SchedulerMode.GLOBAL_SCORE,
    Gate6SchedulerMode.BOUNDED_SCORE_K16,
    Gate6SchedulerMode.BOUNDED_HASH_K16,
    Gate6SchedulerMode.BOUNDED_SCORE_K8,
)

_BOUNDED_K = {
    Gate6SchedulerMode.BOUNDED_SCORE_K16: 16,
    Gate6SchedulerMode.BOUNDED_HASH_K16: 16,
    Gate6SchedulerMode.BOUNDED_SCORE_K8: 8,
}

_SAMPLING_GROUP = {
    Gate6SchedulerMode.BOUNDED_SCORE_K16: "k16",
    Gate6SchedulerMode.BOUNDED_HASH_K16: "k16",
    Gate6SchedulerMode.BOUNDED_SCORE_K8: "k8",
}


@dataclass(frozen=True, slots=True)
class Gate6EvaluationWorld:
    world_index: int
    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.world_index < GATE6_WORLD_COUNT:
            raise ValueError("Gate-6 world index is outside the frozen development domain")
        self.public.validate()
        if self.public.depth != GATE6_DEPTH:
            raise ValueError("Gate-6 must remain at depth 10")
        if len(self.hidden_path) != GATE6_DEPTH or any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("Gate-6 hidden path must contain ten binary decisions")


@dataclass(frozen=True, slots=True)
class Gate6CheckpointIdentity:
    checkpoint_index: int
    checkpoint_sha256: str
    parameter_fingerprint: str
    learned_parameter_count: int


@dataclass(frozen=True, slots=True)
class Gate6WorldTelemetry:
    stage_a_parent_slots: int
    stage_b_productive_slots: int
    total_learned_updates: int
    stage_a_frontier_width: int
    initial_stage_b_population_size: int
    population_capacity: int
    stage_b_live_population_by_slot: tuple[int, ...]
    stage_b_activated_parent_depth_by_slot: tuple[int, ...]
    stage_b_visible_candidate_count_by_slot: tuple[int, ...]
    stage_b_score_observation_count_by_slot: tuple[int, ...]
    total_stage_b_score_observations: int
    max_stage_b_score_observations: int
    selected_visible_score_rank_by_slot: tuple[int, ...]
    selected_global_score_rank_by_slot: tuple[int, ...]
    selected_parent_paths_by_slot: tuple[tuple[int, ...], ...]
    overflow_pruned_count_by_slot: tuple[int, ...]
    generated_terminal_count: int
    unique_generated_terminal_count: int


@dataclass(frozen=True, slots=True)
class Gate6WorldResult:
    generated_terminal_paths: tuple[tuple[int, ...], ...]
    telemetry: Gate6WorldTelemetry


@dataclass(frozen=True, slots=True)
class Gate6ConditionEvaluation:
    checkpoint_index: int
    population_size: int
    mode: Gate6SchedulerMode
    world_count: int
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    coverage_rate: float
    stage_a_parent_slots_by_world: tuple[int, ...]
    stage_b_productive_slots_by_world: tuple[int, ...]
    stage_a_frontier_width_by_world: tuple[int, ...]
    initial_stage_b_population_size_by_world: tuple[int, ...]
    stage_b_live_population_by_slot_by_world: tuple[tuple[int, ...], ...]
    stage_b_activated_parent_depth_by_slot_by_world: tuple[tuple[int, ...], ...]
    stage_b_visible_candidate_count_by_slot_by_world: tuple[tuple[int, ...], ...]
    stage_b_score_observation_count_by_slot_by_world: tuple[tuple[int, ...], ...]
    total_stage_b_score_observations_by_world: tuple[int, ...]
    max_stage_b_score_observations_by_world: tuple[int, ...]
    selected_visible_score_rank_by_slot_by_world: tuple[tuple[int, ...], ...]
    selected_global_score_rank_by_slot_by_world: tuple[tuple[int, ...], ...]
    selected_parent_paths_by_slot_by_world: tuple[tuple[tuple[int, ...], ...], ...]
    overflow_pruned_count_by_slot_by_world: tuple[tuple[int, ...], ...]
    generated_terminal_count_by_world: tuple[int, ...]
    unique_generated_terminal_count_by_world: tuple[int, ...]
    total_learned_updates_per_world: int
    learned_parameter_count: int
    parameter_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate6PairedSummary:
    comparison: str
    checkpoint_index: int
    population_size: int
    treatment_mode: Gate6SchedulerMode
    reference_mode: Gate6SchedulerMode
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
class Gate6DevelopmentResult:
    experiment_version: str
    scientific_status: str
    confirmation_opened: bool
    training_performed: bool
    checkpoints: tuple[Gate6CheckpointIdentity, ...]
    world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    depth: int
    frontier_depth: int
    hint_reliability: float
    population_ladder: tuple[int, ...]
    stage_a_parent_slots: int
    stage_b_parent_slots: int
    scheduled_parent_slots: int
    total_learned_updates_per_world: int
    primary_k: int
    descriptive_k: int
    noninferiority_margin: float
    conditions: tuple[Gate6ConditionEvaluation, ...]
    paired_summaries: tuple[Gate6PairedSummary, ...]

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
            "frontier_depth": self.frontier_depth,
            "hint_reliability": self.hint_reliability,
            "population_ladder": list(self.population_ladder),
            "stage_a_parent_slots": self.stage_a_parent_slots,
            "stage_b_parent_slots": self.stage_b_parent_slots,
            "scheduled_parent_slots": self.scheduled_parent_slots,
            "active_child_lanes": 2,
            "recurrent_updates_per_child": GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            "total_learned_updates_per_world": self.total_learned_updates_per_world,
            "primary_k": self.primary_k,
            "descriptive_k": self.descriptive_k,
            "noninferiority_margin": self.noninferiority_margin,
            "conditions": [row.to_dict() for row in self.conditions],
            "paired_summaries": [row.to_dict() for row in self.paired_summaries],
            "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
        }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def gate6_runtime_seed(*, world_index: int) -> int:
    return _seed_from_parts("gate6-fixed-k-population-scaling-development-runtime", world_index, GATE6_DEPTH)


def generate_gate6_development_world(*, world_index: int) -> Gate6EvaluationWorld:
    if not 0 <= world_index < GATE6_WORLD_COUNT:
        raise ValueError("Gate-6 world index is outside 0..255")
    hidden_rng = random.Random(
        _seed_from_parts("gate6-fixed-k-population-scaling-development-hidden", world_index, GATE6_DEPTH)
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE6_DEPTH))
    hint_rng = random.Random(
        _seed_from_parts("gate6-fixed-k-population-scaling-development-hints", world_index, GATE6_DEPTH)
    )
    noisy_hints = tuple(
        hidden_bit if hint_rng.random() < GATE6_HINT_RELIABILITY else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    world = Gate6EvaluationWorld(
        world_index=world_index,
        public=Gate3V1PublicWorld(
            seed=gate6_runtime_seed(world_index=world_index),
            depth=GATE6_DEPTH,
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


def _answer_blind_order_key(
    namespace: str, *, world_seed: int, slot_index: int, path: tuple[int, ...]
) -> tuple[bytes, tuple[int, ...]]:
    encoded = "".join(str(bit) for bit in path)
    digest = hashlib.sha256(f"{namespace}:{world_seed}:{slot_index}:{encoded}".encode("ascii")).digest()
    return digest, path


def _initial_thinning(
    population: tuple[Gate3V1NeuralCandidate, ...], *, world_seed: int, population_size: int
) -> tuple[Gate3V1NeuralCandidate, ...]:
    if population_size not in GATE6_POPULATION_LADDER:
        raise ValueError("Gate-6 population size is outside the frozen ladder")
    if len(population) != GATE6_FULL_FRONTIER:
        raise ValueError("Gate-6 initial thinning requires the complete 256-candidate frontier")
    ordered = tuple(
        sorted(
            population,
            key=lambda candidate: _answer_blind_order_key(
                "gate6-fixed-k-population-scaling-initial-thinning",
                world_seed=world_seed,
                slot_index=-1,
                path=candidate.path,
            ),
        )
    )
    return ordered[:population_size]


def _prune_to_capacity(
    population: tuple[Gate3V1NeuralCandidate, ...],
    *,
    world_seed: int,
    slot_index: int,
    population_size: int,
) -> tuple[tuple[Gate3V1NeuralCandidate, ...], int]:
    if len(population) <= population_size:
        return population, 0
    ordered = tuple(
        sorted(
            population,
            key=lambda candidate: _answer_blind_order_key(
                "gate6-fixed-k-population-scaling-stage-b-retention",
                world_seed=world_seed,
                slot_index=slot_index,
                path=candidate.path,
            ),
        )
    )
    return ordered[:population_size], len(population) - population_size


def _bounded_indices(*, count: int, k: int, world_seed: int, slot_index: int, group: str) -> tuple[int, ...]:
    if count <= 0 or k <= 0:
        return ()
    visible = min(k, count)
    digest = hashlib.sha256(
        f"{GATE6_EXPERIMENT_VERSION}:{world_seed}:{slot_index}:{group}".encode("ascii")
    ).digest()
    start = int.from_bytes(digest[:8], "big") % count
    stride = int.from_bytes(digest[8:16], "big") % count
    if stride == 0:
        stride = 1
    while math.gcd(stride, count) != 1:
        stride += 1
        if stride >= count:
            stride = 1
    return tuple((start + offset * stride) % count for offset in range(visible))


def _bounded_visible_candidates(
    reserve: Iterable[Gate3V1NeuralCandidate],
    *,
    mode: Gate6SchedulerMode,
    world_seed: int,
    slot_index: int,
) -> tuple[Gate3V1NeuralCandidate, ...]:
    if mode not in _BOUNDED_K:
        raise ValueError("Gate-6 bounded visibility requested for global mode")
    ordered = tuple(sorted(reserve, key=lambda candidate: candidate.path))
    indices = _bounded_indices(
        count=len(ordered),
        k=_BOUNDED_K[mode],
        world_seed=world_seed,
        slot_index=slot_index,
        group=_SAMPLING_GROUP[mode],
    )
    return tuple(ordered[index] for index in indices)


def _hash_select(
    visible: tuple[Gate3V1NeuralCandidate, ...], *, world_seed: int, slot_index: int
) -> Gate3V1NeuralCandidate:
    if not visible:
        raise ValueError("Gate-6 hash selector received no visible candidates")
    return min(
        visible,
        key=lambda candidate: (
            _seed_from_parts(
                "gate6-fixed-k-population-scaling-bounded-hash-selection",
                world_seed,
                slot_index,
                "".join(str(bit) for bit in candidate.path),
            ),
            candidate.path,
        ),
    )


def _advance_parent_batch(
    model: Gate3V1Scorer,
    worlds: tuple[Gate6EvaluationWorld, ...],
    selected_by_world: tuple[tuple[Gate3V1NeuralCandidate, ...], ...],
    *,
    device: torch.device,
) -> tuple[tuple[Gate3V1NeuralCandidate, ...], ...]:
    owner: list[tuple[int, tuple[int, ...], int, int]] = []
    states: list[torch.Tensor] = []
    inputs: list[torch.Tensor] = []
    for world_offset, (world, parents) in enumerate(zip(worlds, selected_by_world, strict=True)):
        for parent in parents:
            next_depth = parent.depth + 1
            if not 1 <= next_depth <= GATE6_DEPTH:
                raise RuntimeError("Gate-6 attempted to expand a terminal/nonexistent parent")
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

    children: list[list[Gate3V1NeuralCandidate]] = [[] for _ in worlds]
    if not states:
        return tuple(tuple(row) for row in children)

    state_batch = torch.stack(states, dim=0)
    input_batch = torch.stack(inputs, dim=0)
    advanced = model.advance(
        state_batch,
        input_batch,
        repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    )
    scores = model.score(advanced)
    for row_index, (world_offset, parent_path, action, _next_depth) in enumerate(owner):
        children[world_offset].append(
            Gate3V1NeuralCandidate(
                path=parent_path + (action,),
                state=advanced[row_index].clone(),
                score=float(scores[row_index].item()),
            )
        )
    return tuple(tuple(row) for row in children)


def run_gate6_world_batch(
    model: Gate3V1Scorer,
    worlds: Iterable[Gate6EvaluationWorld],
    *,
    population_size: int,
    mode: Gate6SchedulerMode,
    device: torch.device | str,
) -> tuple[Gate6WorldResult, ...]:
    world_tuple = tuple(worlds)
    if not world_tuple:
        return ()
    if population_size not in GATE6_POPULATION_LADDER:
        raise ValueError("Gate-6 population size is outside the frozen ladder")
    if mode not in GATE6_CONDITIONS:
        raise ValueError("Gate-6 scheduler mode is outside the frozen matrix")
    for world in world_tuple:
        world.validate()

    target_device = torch.device(device)
    model = model.to(target_device)
    populations: list[tuple[Gate3V1NeuralCandidate, ...]] = [
        (
            Gate3V1NeuralCandidate(
                path=(),
                state=model.initial_state(1, device=target_device)[0],
                score=0.0,
            ),
        )
        for _ in world_tuple
    ]
    generated_terminals: list[list[tuple[int, ...]]] = [[] for _ in world_tuple]
    stage_a_productive = [0 for _ in world_tuple]
    stage_b_productive = [0 for _ in world_tuple]

    stage_b_live: list[list[int]] = [[] for _ in world_tuple]
    stage_b_depths: list[list[int]] = [[] for _ in world_tuple]
    stage_b_visible: list[list[int]] = [[] for _ in world_tuple]
    stage_b_score_obs: list[list[int]] = [[] for _ in world_tuple]
    selected_visible_rank: list[list[int]] = [[] for _ in world_tuple]
    selected_global_rank: list[list[int]] = [[] for _ in world_tuple]
    selected_paths: list[list[tuple[int, ...]]] = [[] for _ in world_tuple]
    overflow_pruned: list[list[int]] = [[] for _ in world_tuple]

    with torch.inference_mode():
        # Common Stage A: every condition builds exactly the same complete depth-8 frontier.
        for _parent_depth in range(GATE6_FRONTIER_DEPTH):
            selected = tuple(populations)
            children_by_world = _advance_parent_batch(
                model,
                world_tuple,
                selected,
                device=target_device,
            )
            next_populations: list[tuple[Gate3V1NeuralCandidate, ...]] = []
            for world_offset, (parents, children) in enumerate(zip(selected, children_by_world, strict=True)):
                stage_a_productive[world_offset] += len(parents)
                next_populations.append(tuple(children))
            populations = next_populations

        if any(len(population) != GATE6_FULL_FRONTIER for population in populations):
            raise RuntimeError("Gate-6 Stage A did not create exactly 256 depth-8 hypotheses")
        if any(count != GATE6_STAGE_A_PARENT_SLOTS for count in stage_a_productive):
            raise RuntimeError("Gate-6 Stage A did not consume exactly 255 parent slots")

        populations = [
            _initial_thinning(
                population,
                world_seed=world.public.seed,
                population_size=population_size,
            )
            for world, population in zip(world_tuple, populations, strict=True)
        ]
        if any(len(population) != population_size for population in populations):
            raise RuntimeError("Gate-6 initial answer-blind thinning produced the wrong population size")

        # Stage B: fixed 128 productive parent activations in every condition.
        for local_slot in range(GATE6_STAGE_B_PARENT_SLOTS):
            absolute_slot = GATE6_STAGE_A_PARENT_SLOTS + local_slot
            selected_rows: list[tuple[Gate3V1NeuralCandidate, ...]] = []
            for world_offset, (world, population) in enumerate(zip(world_tuple, populations, strict=True)):
                if not population:
                    raise RuntimeError("Gate-6 Stage-B reserve exhausted before the frozen 128 slots")
                if len(population) > population_size:
                    raise RuntimeError("Gate-6 live population exceeded its hard N capacity")

                if mode is Gate6SchedulerMode.GLOBAL_SCORE:
                    global_ranked = _score_rank(
                        population,
                        world_seed=world.public.seed,
                        expansion_index=absolute_slot,
                    )
                    visible = global_ranked
                    parent = global_ranked[0]
                    score_observations = len(visible)
                    visible_rank = 1
                    global_rank = 1
                else:
                    visible = _bounded_visible_candidates(
                        population,
                        mode=mode,
                        world_seed=world.public.seed,
                        slot_index=absolute_slot,
                    )
                    if mode is Gate6SchedulerMode.BOUNDED_HASH_K16:
                        parent = _hash_select(
                            visible,
                            world_seed=world.public.seed,
                            slot_index=absolute_slot,
                        )
                        score_observations = 0
                        visible_score_order = _score_rank(
                            visible,
                            world_seed=world.public.seed,
                            expansion_index=absolute_slot,
                        )
                        visible_rank = 1 + next(
                            index
                            for index, candidate in enumerate(visible_score_order)
                            if candidate.path == parent.path
                        )
                    else:
                        visible_score_order = _score_rank(
                            visible,
                            world_seed=world.public.seed,
                            expansion_index=absolute_slot,
                        )
                        parent = visible_score_order[0]
                        score_observations = len(visible)
                        visible_rank = 1

                    # Full-reserve score access happens only after bounded parent selection is fixed.
                    global_ranked_after_decision = _score_rank(
                        population,
                        world_seed=world.public.seed,
                        expansion_index=absolute_slot,
                    )
                    global_rank = 1 + next(
                        index
                        for index, candidate in enumerate(global_ranked_after_decision)
                        if candidate.path == parent.path
                    )

                stage_b_live[world_offset].append(len(population))
                stage_b_depths[world_offset].append(parent.depth)
                stage_b_visible[world_offset].append(len(visible))
                stage_b_score_obs[world_offset].append(score_observations)
                selected_visible_rank[world_offset].append(visible_rank)
                selected_global_rank[world_offset].append(global_rank)
                selected_paths[world_offset].append(parent.path)

                selected_rows.append((parent,))
                populations[world_offset] = tuple(
                    candidate for candidate in population if candidate.path != parent.path
                )

            children_by_world = _advance_parent_batch(
                model,
                world_tuple,
                tuple(selected_rows),
                device=target_device,
            )
            next_populations = []
            for world_offset, (world, population, parent_tuple, children) in enumerate(
                zip(world_tuple, populations, selected_rows, children_by_world, strict=True)
            ):
                parent = parent_tuple[0]
                stage_b_productive[world_offset] += 1
                if parent.depth + 1 == GATE6_DEPTH:
                    generated_terminals[world_offset].extend(child.path for child in children)
                    updated = population
                else:
                    updated = population + tuple(children)
                retained, pruned = _prune_to_capacity(
                    updated,
                    world_seed=world.public.seed,
                    slot_index=local_slot,
                    population_size=population_size,
                )
                overflow_pruned[world_offset].append(pruned)
                next_populations.append(retained)
            populations = next_populations

    results: list[Gate6WorldResult] = []
    for world_offset in range(len(world_tuple)):
        if stage_a_productive[world_offset] != GATE6_STAGE_A_PARENT_SLOTS:
            raise RuntimeError("Gate-6 Stage-A accounting changed")
        if stage_b_productive[world_offset] != GATE6_STAGE_B_PARENT_SLOTS:
            raise RuntimeError("Gate-6 Stage-B accounting changed")
        total_updates = (
            stage_a_productive[world_offset] + stage_b_productive[world_offset]
        ) * GATE3_V1_UPDATES_PER_ROUND
        if total_updates != GATE6_TOTAL_LEARNED_UPDATES:
            raise RuntimeError("Gate-6 learned-work identity was violated")
        if len(stage_b_live[world_offset]) != GATE6_STAGE_B_PARENT_SLOTS:
            raise RuntimeError("Gate-6 Stage-B telemetry is incomplete")
        if any(value > population_size or value < 1 for value in stage_b_live[world_offset]):
            raise RuntimeError("Gate-6 Stage-B live population violated hard N capacity")
        if mode in _BOUNDED_K:
            k = _BOUNDED_K[mode]
            for live_count, visible_count in zip(
                stage_b_live[world_offset], stage_b_visible[world_offset], strict=True
            ):
                if visible_count != min(k, live_count):
                    raise RuntimeError("Gate-6 bounded visibility violated frozen K")
        if mode is Gate6SchedulerMode.GLOBAL_SCORE:
            if stage_b_visible[world_offset] != stage_b_live[world_offset]:
                raise RuntimeError("Gate-6 global scheduler did not observe complete live reserve")
        if mode is Gate6SchedulerMode.BOUNDED_HASH_K16:
            if any(value != 0 for value in stage_b_score_obs[world_offset]):
                raise RuntimeError("Gate-6 hash control consumed neural-score observations")

        score_observations = tuple(stage_b_score_obs[world_offset])
        results.append(
            Gate6WorldResult(
                generated_terminal_paths=tuple(generated_terminals[world_offset]),
                telemetry=Gate6WorldTelemetry(
                    stage_a_parent_slots=stage_a_productive[world_offset],
                    stage_b_productive_slots=stage_b_productive[world_offset],
                    total_learned_updates=total_updates,
                    stage_a_frontier_width=GATE6_FULL_FRONTIER,
                    initial_stage_b_population_size=population_size,
                    population_capacity=population_size,
                    stage_b_live_population_by_slot=tuple(stage_b_live[world_offset]),
                    stage_b_activated_parent_depth_by_slot=tuple(stage_b_depths[world_offset]),
                    stage_b_visible_candidate_count_by_slot=tuple(stage_b_visible[world_offset]),
                    stage_b_score_observation_count_by_slot=score_observations,
                    total_stage_b_score_observations=sum(score_observations),
                    max_stage_b_score_observations=max(score_observations, default=0),
                    selected_visible_score_rank_by_slot=tuple(selected_visible_rank[world_offset]),
                    selected_global_score_rank_by_slot=tuple(selected_global_rank[world_offset]),
                    selected_parent_paths_by_slot=tuple(selected_paths[world_offset]),
                    overflow_pruned_count_by_slot=tuple(overflow_pruned[world_offset]),
                    generated_terminal_count=len(generated_terminals[world_offset]),
                    unique_generated_terminal_count=len(set(generated_terminals[world_offset])),
                ),
            )
        )
    return tuple(results)


def evaluate_gate6_condition(
    model: Gate3V1Scorer,
    *,
    checkpoint_index: int,
    population_size: int,
    mode: Gate6SchedulerMode,
    device: torch.device | str,
    world_count: int = GATE6_WORLD_COUNT,
    evaluation_batch_size: int = GATE6_EVAL_BATCH_SIZE,
) -> Gate6ConditionEvaluation:
    if checkpoint_index not in GATE6_CHECKPOINT_INDICES:
        raise ValueError("Gate-6 checkpoint index must be 0, 1 or 2")
    if population_size not in GATE6_POPULATION_LADDER:
        raise ValueError("Gate-6 population size is outside the frozen ladder")
    if mode not in GATE6_CONDITIONS:
        raise ValueError("Gate-6 scheduler mode is outside the frozen matrix")
    if world_count != GATE6_WORLD_COUNT:
        raise ValueError("Gate-6 development must use exactly 256 worlds")
    if evaluation_batch_size != GATE6_EVAL_BATCH_SIZE:
        raise ValueError("Gate-6 development batch size is frozen at 64")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    stage_a_slots: list[int] = []
    stage_b_slots: list[int] = []
    frontier: list[int] = []
    initial_population: list[int] = []
    live_rows: list[tuple[int, ...]] = []
    depth_rows: list[tuple[int, ...]] = []
    visible_rows: list[tuple[int, ...]] = []
    score_obs_rows: list[tuple[int, ...]] = []
    total_score_obs: list[int] = []
    max_score_obs: list[int] = []
    visible_rank_rows: list[tuple[int, ...]] = []
    global_rank_rows: list[tuple[int, ...]] = []
    selected_path_rows: list[tuple[tuple[int, ...], ...]] = []
    pruned_rows: list[tuple[int, ...]] = []
    terminal_count: list[int] = []
    unique_terminal_count: list[int] = []

    for start in range(0, world_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, world_count)
        worlds = tuple(generate_gate6_development_world(world_index=index) for index in range(start, stop))
        batch = run_gate6_world_batch(
            model,
            worlds,
            population_size=population_size,
            mode=mode,
            device=device,
        )
        for world, result in zip(worlds, batch, strict=True):
            telemetry = result.telemetry
            covered.append(world.hidden_path in set(result.generated_terminal_paths))
            runtime_seeds.append(world.public.seed)
            stage_a_slots.append(telemetry.stage_a_parent_slots)
            stage_b_slots.append(telemetry.stage_b_productive_slots)
            frontier.append(telemetry.stage_a_frontier_width)
            initial_population.append(telemetry.initial_stage_b_population_size)
            live_rows.append(telemetry.stage_b_live_population_by_slot)
            depth_rows.append(telemetry.stage_b_activated_parent_depth_by_slot)
            visible_rows.append(telemetry.stage_b_visible_candidate_count_by_slot)
            score_obs_rows.append(telemetry.stage_b_score_observation_count_by_slot)
            total_score_obs.append(telemetry.total_stage_b_score_observations)
            max_score_obs.append(telemetry.max_stage_b_score_observations)
            visible_rank_rows.append(telemetry.selected_visible_score_rank_by_slot)
            global_rank_rows.append(telemetry.selected_global_score_rank_by_slot)
            selected_path_rows.append(telemetry.selected_parent_paths_by_slot)
            pruned_rows.append(telemetry.overflow_pruned_count_by_slot)
            terminal_count.append(telemetry.generated_terminal_count)
            unique_terminal_count.append(telemetry.unique_generated_terminal_count)

    vector = tuple(covered)
    return Gate6ConditionEvaluation(
        checkpoint_index=checkpoint_index,
        population_size=population_size,
        mode=mode,
        world_count=world_count,
        world_indices=tuple(range(world_count)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / world_count,
        stage_a_parent_slots_by_world=tuple(stage_a_slots),
        stage_b_productive_slots_by_world=tuple(stage_b_slots),
        stage_a_frontier_width_by_world=tuple(frontier),
        initial_stage_b_population_size_by_world=tuple(initial_population),
        stage_b_live_population_by_slot_by_world=tuple(live_rows),
        stage_b_activated_parent_depth_by_slot_by_world=tuple(depth_rows),
        stage_b_visible_candidate_count_by_slot_by_world=tuple(visible_rows),
        stage_b_score_observation_count_by_slot_by_world=tuple(score_obs_rows),
        total_stage_b_score_observations_by_world=tuple(total_score_obs),
        max_stage_b_score_observations_by_world=tuple(max_score_obs),
        selected_visible_score_rank_by_slot_by_world=tuple(visible_rank_rows),
        selected_global_score_rank_by_slot_by_world=tuple(global_rank_rows),
        selected_parent_paths_by_slot_by_world=tuple(selected_path_rows),
        overflow_pruned_count_by_slot_by_world=tuple(pruned_rows),
        generated_terminal_count_by_world=tuple(terminal_count),
        unique_generated_terminal_count_by_world=tuple(unique_terminal_count),
        total_learned_updates_per_world=GATE6_TOTAL_LEARNED_UPDATES,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint_index: int, population_size: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts(
            "gate6-fixed-k-population-scaling-bootstrap",
            checkpoint_index,
            population_size,
            comparison,
        )
    )
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE6_BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (GATE6_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE6_BOOTSTRAP_SAMPLES - 1)))],
    )


def _paired_summary(
    *,
    comparison: str,
    checkpoint_index: int,
    population_size: int,
    treatment: Gate6ConditionEvaluation,
    reference: Gate6ConditionEvaluation,
) -> Gate6PairedSummary:
    if treatment.world_indices != reference.world_indices:
        raise ValueError("Gate-6 paired conditions use different world indices")
    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = GATE6_WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=checkpoint_index,
        population_size=population_size,
        comparison=comparison,
    )
    return Gate6PairedSummary(
        comparison=comparison,
        checkpoint_index=checkpoint_index,
        population_size=population_size,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        world_count=GATE6_WORLD_COUNT,
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=sum(differences) / GATE6_WORLD_COUNT,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate6_paired_summaries(
    conditions: Iterable[Gate6ConditionEvaluation],
) -> tuple[Gate6PairedSummary, ...]:
    rows = tuple(conditions)
    index = {(row.checkpoint_index, row.population_size, row.mode): row for row in rows}
    specs = (
        (
            "bounded_score_k16_vs_bounded_hash_k16",
            Gate6SchedulerMode.BOUNDED_SCORE_K16,
            Gate6SchedulerMode.BOUNDED_HASH_K16,
        ),
        (
            "bounded_score_k16_vs_global_score",
            Gate6SchedulerMode.BOUNDED_SCORE_K16,
            Gate6SchedulerMode.GLOBAL_SCORE,
        ),
        (
            "bounded_score_k8_vs_global_score",
            Gate6SchedulerMode.BOUNDED_SCORE_K8,
            Gate6SchedulerMode.GLOBAL_SCORE,
        ),
    )
    summaries: list[Gate6PairedSummary] = []
    for checkpoint_index in GATE6_CHECKPOINT_INDICES:
        for population_size in GATE6_POPULATION_LADDER:
            for comparison, treatment_mode, reference_mode in specs:
                summaries.append(
                    _paired_summary(
                        comparison=comparison,
                        checkpoint_index=checkpoint_index,
                        population_size=population_size,
                        treatment=index[(checkpoint_index, population_size, treatment_mode)],
                        reference=index[(checkpoint_index, population_size, reference_mode)],
                    )
                )
    return tuple(summaries)


def classify_gate6(
    lows: dict[str, float], highs: dict[str, float]
) -> str:
    def key(checkpoint: int, population_size: int, comparison: str) -> str:
        return f"c{checkpoint}_n{population_size}_{comparison}"

    learned_comparison = "bounded_score_k16_vs_bounded_hash_k16"
    global_comparison = "bounded_score_k16_vs_global_score"

    learned_keys = [
        key(checkpoint, population_size, learned_comparison)
        for checkpoint in GATE6_CHECKPOINT_INDICES
        for population_size in GATE6_POPULATION_LADDER
    ]
    if any(highs[item] < 0.0 for item in learned_keys):
        return "G6_S4_BOUNDED_ROUTING_HARMFUL_AT_SCALE"

    pass_by_tier: dict[int, tuple[bool, ...]] = {}
    for population_size in GATE6_POPULATION_LADDER:
        pass_by_tier[population_size] = tuple(
            lows[key(checkpoint, population_size, learned_comparison)] > 0.0
            and lows[key(checkpoint, population_size, global_comparison)] > -GATE6_NONINFERIORITY_MARGIN
            for checkpoint in GATE6_CHECKPOINT_INDICES
        )

    if any(len(set(statuses)) > 1 for statuses in pass_by_tier.values()):
        return "G6_S3_CHECKPOINT_SENSITIVE_SCALING"
    if not all(pass_by_tier[64]):
        return "G6_S0_FIXED_K_NOT_ESTABLISHED"
    if all(all(statuses) for statuses in pass_by_tier.values()):
        return "G6_S2_ROBUST_FIXED_K_POPULATION_SCALING"
    return "G6_S1_FIXED_K_DEGRADES_WITH_POPULATION"
