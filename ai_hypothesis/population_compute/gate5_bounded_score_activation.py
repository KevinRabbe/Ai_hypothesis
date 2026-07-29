"""Gate-5 v0 bounded score-visibility adaptive activation.

No training occurs here.  Gate-5 reuses the exact frozen Gate-3 v1 scorer/checkpoints and
holds L256 plus the 159-slot / 2,544-update learned-work budget fixed.  The treatment is
only how many live candidate scores may be inspected before each Stage-B activation.
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

GATE5_EXPERIMENT_VERSION = "gate5-bounded-score-activation-v0"
GATE5_DEPTH = 8
GATE5_HINT_RELIABILITY = 0.70
GATE5_WORLD_COUNT = 256
GATE5_EVAL_BATCH_SIZE = 64
GATE5_BOOTSTRAP_SAMPLES = 2_000
GATE5_RESERVE_CAPACITY = 256
GATE5_STAGE_A_SLOTS = 63
GATE5_STAGE_B_SLOTS = 96
GATE5_SCHEDULED_SLOTS = GATE5_STAGE_A_SLOTS + GATE5_STAGE_B_SLOTS
GATE5_TOTAL_LEARNED_UPDATES = GATE5_SCHEDULED_SLOTS * GATE3_V1_UPDATES_PER_ROUND
GATE5_STAGE_A_FRONTIER = 64
GATE5_NONINFERIORITY_MARGIN = 0.05
GATE5_CHECKPOINT_INDICES = (0, 1, 2)
GATE5_PARAMETER_COUNT = 19_649


class Gate5SchedulerMode(str, Enum):
    GLOBAL_SCORE = "global_score"
    BOUNDED_SCORE_K4 = "bounded_score_k4"
    BOUNDED_SCORE_K8 = "bounded_score_k8"
    BOUNDED_SCORE_K16 = "bounded_score_k16"
    BOUNDED_SCORE_K32 = "bounded_score_k32"
    BOUNDED_HASH_K16 = "bounded_hash_k16"


GATE5_CONDITIONS = (
    Gate5SchedulerMode.GLOBAL_SCORE,
    Gate5SchedulerMode.BOUNDED_SCORE_K4,
    Gate5SchedulerMode.BOUNDED_SCORE_K8,
    Gate5SchedulerMode.BOUNDED_SCORE_K16,
    Gate5SchedulerMode.BOUNDED_SCORE_K32,
    Gate5SchedulerMode.BOUNDED_HASH_K16,
)

_BOUNDED_K = {
    Gate5SchedulerMode.BOUNDED_SCORE_K4: 4,
    Gate5SchedulerMode.BOUNDED_SCORE_K8: 8,
    Gate5SchedulerMode.BOUNDED_SCORE_K16: 16,
    Gate5SchedulerMode.BOUNDED_SCORE_K32: 32,
    Gate5SchedulerMode.BOUNDED_HASH_K16: 16,
}

_SAMPLING_GROUP = {
    Gate5SchedulerMode.BOUNDED_SCORE_K4: "k4",
    Gate5SchedulerMode.BOUNDED_SCORE_K8: "k8",
    Gate5SchedulerMode.BOUNDED_SCORE_K16: "k16",
    Gate5SchedulerMode.BOUNDED_SCORE_K32: "k32",
    Gate5SchedulerMode.BOUNDED_HASH_K16: "k16",
}


@dataclass(frozen=True, slots=True)
class Gate5EvaluationWorld:
    world_index: int
    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.world_index < GATE5_WORLD_COUNT:
            raise ValueError("Gate-5 world index is outside the frozen development domain")
        self.public.validate()
        if self.public.depth != GATE5_DEPTH:
            raise ValueError("Gate-5 must remain at depth 8")
        if len(self.hidden_path) != GATE5_DEPTH or any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("Gate-5 hidden path must contain eight binary decisions")


@dataclass(frozen=True, slots=True)
class Gate5CheckpointIdentity:
    checkpoint_index: int
    checkpoint_sha256: str
    parameter_fingerprint: str
    learned_parameter_count: int


@dataclass(frozen=True, slots=True)
class Gate5WorldTelemetry:
    productive_slots: int
    sink_slots: int
    total_learned_updates: int
    stage_a_frontier_width: int
    stage_b_live_population_by_slot: tuple[int, ...]
    stage_b_activated_parent_depth_by_slot: tuple[int, ...]
    stage_b_visible_candidate_count_by_slot: tuple[int, ...]
    stage_b_score_observation_count_by_slot: tuple[int, ...]
    total_stage_b_score_observations: int
    max_stage_b_score_observations: int
    selected_visible_score_rank_by_slot: tuple[int, ...]
    selected_global_score_rank_by_slot: tuple[int, ...]
    selected_parent_paths_by_slot: tuple[tuple[int, ...], ...]
    generated_terminal_count: int
    unique_generated_terminal_count: int


@dataclass(frozen=True, slots=True)
class Gate5WorldResult:
    generated_terminal_paths: tuple[tuple[int, ...], ...]
    telemetry: Gate5WorldTelemetry


@dataclass(frozen=True, slots=True)
class Gate5ConditionEvaluation:
    checkpoint_index: int
    mode: Gate5SchedulerMode
    world_count: int
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    coverage_rate: float
    productive_slots_by_world: tuple[int, ...]
    sink_slots_by_world: tuple[int, ...]
    stage_a_frontier_width_by_world: tuple[int, ...]
    stage_b_live_population_by_slot_by_world: tuple[tuple[int, ...], ...]
    stage_b_activated_parent_depth_by_slot_by_world: tuple[tuple[int, ...], ...]
    stage_b_visible_candidate_count_by_slot_by_world: tuple[tuple[int, ...], ...]
    stage_b_score_observation_count_by_slot_by_world: tuple[tuple[int, ...], ...]
    total_stage_b_score_observations_by_world: tuple[int, ...]
    max_stage_b_score_observations_by_world: tuple[int, ...]
    selected_visible_score_rank_by_slot_by_world: tuple[tuple[int, ...], ...]
    selected_global_score_rank_by_slot_by_world: tuple[tuple[int, ...], ...]
    selected_parent_paths_by_slot_by_world: tuple[tuple[tuple[int, ...], ...], ...]
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
class Gate5PairedSummary:
    comparison: str
    checkpoint_index: int
    treatment_mode: Gate5SchedulerMode
    reference_mode: Gate5SchedulerMode
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
class Gate5DevelopmentResult:
    experiment_version: str
    scientific_status: str
    confirmation_opened: bool
    training_performed: bool
    checkpoints: tuple[Gate5CheckpointIdentity, ...]
    world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    depth: int
    hint_reliability: float
    reserve_capacity: int
    stage_a_slots: int
    stage_b_slots: int
    scheduled_slots: int
    total_learned_updates_per_world: int
    noninferiority_margin: float
    conditions: tuple[Gate5ConditionEvaluation, ...]
    paired_summaries: tuple[Gate5PairedSummary, ...]

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
            "stage_a_slots": self.stage_a_slots,
            "stage_b_slots": self.stage_b_slots,
            "scheduled_slots": self.scheduled_slots,
            "active_child_lanes": 2,
            "recurrent_updates_per_child": GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            "total_learned_updates_per_world": self.total_learned_updates_per_world,
            "noninferiority_margin": self.noninferiority_margin,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "paired_summaries": [summary.to_dict() for summary in self.paired_summaries],
            "scientific_decision": "DEVELOPMENT_ONLY_NOT_ASSIGNED",
        }


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def gate5_runtime_seed(*, world_index: int) -> int:
    return _seed_from_parts("gate5-bounded-score-activation-development-runtime", world_index, GATE5_DEPTH)


def generate_gate5_development_world(*, world_index: int) -> Gate5EvaluationWorld:
    if not 0 <= world_index < GATE5_WORLD_COUNT:
        raise ValueError("Gate-5 world index is outside 0..255")
    hidden_rng = random.Random(
        _seed_from_parts("gate5-bounded-score-activation-development-hidden", world_index, GATE5_DEPTH)
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE5_DEPTH))
    hint_rng = random.Random(
        _seed_from_parts("gate5-bounded-score-activation-development-hints", world_index, GATE5_DEPTH)
    )
    noisy_hints = tuple(
        hidden_bit if hint_rng.random() < GATE5_HINT_RELIABILITY else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    world = Gate5EvaluationWorld(
        world_index=world_index,
        public=Gate3V1PublicWorld(
            seed=gate5_runtime_seed(world_index=world_index),
            depth=GATE5_DEPTH,
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


def _bounded_indices(*, count: int, k: int, world_seed: int, slot_index: int, group: str) -> tuple[int, ...]:
    if count <= 0 or k <= 0:
        return ()
    visible = min(k, count)
    digest = hashlib.sha256(
        f"{GATE5_EXPERIMENT_VERSION}:{world_seed}:{slot_index}:{group}".encode("ascii")
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
    mode: Gate5SchedulerMode,
    world_seed: int,
    slot_index: int,
) -> tuple[Gate3V1NeuralCandidate, ...]:
    if mode not in _BOUNDED_K:
        raise ValueError("Gate-5 bounded visibility requested for global mode")
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
        raise ValueError("Gate-5 hash selector received no visible candidates")
    return min(
        visible,
        key=lambda candidate: (
            _seed_from_parts(
                "gate5-bounded-hash-selection",
                world_seed,
                slot_index,
                "".join(str(bit) for bit in candidate.path),
            ),
            candidate.path,
        ),
    )


def _advance_parent_batch(
    model: Gate3V1Scorer,
    worlds: tuple[Gate5EvaluationWorld, ...],
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
            if not 1 <= next_depth <= GATE5_DEPTH:
                raise RuntimeError("Gate-5 attempted to expand a terminal/nonexistent parent")
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
    for row_index, (world_offset, parent_path, action, next_depth) in enumerate(owner):
        children[world_offset].append(
            Gate3V1NeuralCandidate(
                path=parent_path + (action,),
                state=advanced[row_index].clone(),
                score=float(scores[row_index].item()),
            )
        )
    return tuple(tuple(row) for row in children)


def run_gate5_world_batch(
    model: Gate3V1Scorer,
    worlds: Iterable[Gate5EvaluationWorld],
    *,
    mode: Gate5SchedulerMode,
    device: torch.device | str,
) -> tuple[Gate5WorldResult, ...]:
    world_tuple = tuple(worlds)
    if not world_tuple:
        return ()
    if mode not in GATE5_CONDITIONS:
        raise ValueError("Gate-5 scheduler mode is outside the frozen matrix")
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
    productive = [0 for _ in world_tuple]

    stage_b_live: list[list[int]] = [[] for _ in world_tuple]
    stage_b_depths: list[list[int]] = [[] for _ in world_tuple]
    stage_b_visible: list[list[int]] = [[] for _ in world_tuple]
    stage_b_score_obs: list[list[int]] = [[] for _ in world_tuple]
    selected_visible_rank: list[list[int]] = [[] for _ in world_tuple]
    selected_global_rank: list[list[int]] = [[] for _ in world_tuple]
    selected_paths: list[list[tuple[int, ...]]] = [[] for _ in world_tuple]

    with torch.inference_mode():
        # Stage A: complete generation-synchronous build to 64 depth-6 hypotheses.
        for _parent_depth in range(6):
            selected = tuple(populations)
            children_by_world = _advance_parent_batch(
                model,
                world_tuple,
                selected,
                device=target_device,
            )
            next_populations: list[tuple[Gate3V1NeuralCandidate, ...]] = []
            for world_offset, (parents, children) in enumerate(
                zip(selected, children_by_world, strict=True)
            ):
                productive[world_offset] += len(parents)
                next_populations.append(tuple(children))
            populations = next_populations

        if any(len(population) != GATE5_STAGE_A_FRONTIER for population in populations):
            raise RuntimeError("Gate-5 Stage A did not create exactly 64 depth-6 hypotheses")
        if any(count != GATE5_STAGE_A_SLOTS for count in productive):
            raise RuntimeError("Gate-5 Stage A did not consume exactly 63 parent slots")

        # Stage B: one selected parent/world/slot under the treatment visibility rule.
        for local_slot in range(GATE5_STAGE_B_SLOTS):
            absolute_slot = GATE5_STAGE_A_SLOTS + local_slot
            selected_rows: list[tuple[Gate3V1NeuralCandidate, ...]] = []
            for world_offset, (world, population) in enumerate(zip(world_tuple, populations, strict=True)):
                if not population:
                    raise RuntimeError("Gate-5 Stage B unexpectedly exhausted the live reserve")

                global_ranked = _score_rank(
                    population,
                    world_seed=world.public.seed,
                    expansion_index=absolute_slot,
                )

                if mode is Gate5SchedulerMode.GLOBAL_SCORE:
                    visible = global_ranked
                    parent = global_ranked[0]
                    score_observations = len(visible)
                    visible_rank = 1
                else:
                    visible = _bounded_visible_candidates(
                        population,
                        mode=mode,
                        world_seed=world.public.seed,
                        slot_index=absolute_slot,
                    )
                    if mode is Gate5SchedulerMode.BOUNDED_HASH_K16:
                        parent = _hash_select(
                            visible,
                            world_seed=world.public.seed,
                            slot_index=absolute_slot,
                        )
                        score_observations = 0
                        # Evaluation-only diagnostic, computed after the answer-blind selection.
                        visible_score_order = _score_rank(
                            visible,
                            world_seed=world.public.seed,
                            expansion_index=absolute_slot,
                        )
                        visible_rank = 1 + next(
                            index for index, candidate in enumerate(visible_score_order) if candidate.path == parent.path
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

                # Evaluation-only global-rank diagnostic after parent selection; never feeds back.
                global_rank = 1 + next(
                    index for index, candidate in enumerate(global_ranked) if candidate.path == parent.path
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
            for world_offset, (population, parent_tuple, children) in enumerate(
                zip(populations, selected_rows, children_by_world, strict=True)
            ):
                parent = parent_tuple[0]
                productive[world_offset] += 1
                if parent.depth + 1 == GATE5_DEPTH:
                    generated_terminals[world_offset].extend(child.path for child in children)
                    updated = population
                else:
                    updated = population + tuple(children)
                if len(updated) > GATE5_RESERVE_CAPACITY:
                    raise RuntimeError("Gate-5 live reserve exceeded frozen L256 capacity")
                next_populations.append(updated)
            populations = next_populations

    results: list[Gate5WorldResult] = []
    for world_offset in range(len(world_tuple)):
        sink_slots = GATE5_SCHEDULED_SLOTS - productive[world_offset]
        if sink_slots != 0:
            raise RuntimeError("Gate-5 admitted topology should use all 159 slots productively")
        total_updates = productive[world_offset] * GATE3_V1_UPDATES_PER_ROUND
        if total_updates != GATE5_TOTAL_LEARNED_UPDATES:
            raise RuntimeError("Gate-5 learned-work identity was violated")
        if len(stage_b_live[world_offset]) != GATE5_STAGE_B_SLOTS:
            raise RuntimeError("Gate-5 Stage-B telemetry is incomplete")
        if mode in _BOUNDED_K:
            k = _BOUNDED_K[mode]
            for live_count, visible_count in zip(
                stage_b_live[world_offset], stage_b_visible[world_offset], strict=True
            ):
                if visible_count != min(k, live_count):
                    raise RuntimeError("Gate-5 bounded visibility exceeded or undershot frozen K")
        if mode is Gate5SchedulerMode.GLOBAL_SCORE:
            if stage_b_visible[world_offset] != stage_b_live[world_offset]:
                raise RuntimeError("Gate-5 global scheduler did not observe the complete live reserve")
        if mode is Gate5SchedulerMode.BOUNDED_HASH_K16:
            if any(value != 0 for value in stage_b_score_obs[world_offset]):
                raise RuntimeError("Gate-5 hash control observed neural scores before selection")

        score_observations = tuple(stage_b_score_obs[world_offset])
        results.append(
            Gate5WorldResult(
                generated_terminal_paths=tuple(generated_terminals[world_offset]),
                telemetry=Gate5WorldTelemetry(
                    productive_slots=productive[world_offset],
                    sink_slots=0,
                    total_learned_updates=total_updates,
                    stage_a_frontier_width=GATE5_STAGE_A_FRONTIER,
                    stage_b_live_population_by_slot=tuple(stage_b_live[world_offset]),
                    stage_b_activated_parent_depth_by_slot=tuple(stage_b_depths[world_offset]),
                    stage_b_visible_candidate_count_by_slot=tuple(stage_b_visible[world_offset]),
                    stage_b_score_observation_count_by_slot=score_observations,
                    total_stage_b_score_observations=sum(score_observations),
                    max_stage_b_score_observations=max(score_observations, default=0),
                    selected_visible_score_rank_by_slot=tuple(selected_visible_rank[world_offset]),
                    selected_global_score_rank_by_slot=tuple(selected_global_rank[world_offset]),
                    selected_parent_paths_by_slot=tuple(selected_paths[world_offset]),
                    generated_terminal_count=len(generated_terminals[world_offset]),
                    unique_generated_terminal_count=len(set(generated_terminals[world_offset])),
                ),
            )
        )
    return tuple(results)


def evaluate_gate5_condition(
    model: Gate3V1Scorer,
    *,
    checkpoint_index: int,
    mode: Gate5SchedulerMode,
    device: torch.device | str,
    world_count: int = GATE5_WORLD_COUNT,
    evaluation_batch_size: int = GATE5_EVAL_BATCH_SIZE,
) -> Gate5ConditionEvaluation:
    if checkpoint_index not in GATE5_CHECKPOINT_INDICES:
        raise ValueError("Gate-5 checkpoint index must be 0, 1 or 2")
    if mode not in GATE5_CONDITIONS:
        raise ValueError("Gate-5 condition is outside the frozen matrix")
    if world_count != GATE5_WORLD_COUNT:
        raise ValueError("Gate-5 development must use exactly 256 worlds")
    if evaluation_batch_size != GATE5_EVAL_BATCH_SIZE:
        raise ValueError("Gate-5 development batch size is frozen at 64")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    productive: list[int] = []
    sink: list[int] = []
    frontier: list[int] = []
    live_rows: list[tuple[int, ...]] = []
    depth_rows: list[tuple[int, ...]] = []
    visible_rows: list[tuple[int, ...]] = []
    score_obs_rows: list[tuple[int, ...]] = []
    total_score_obs: list[int] = []
    max_score_obs: list[int] = []
    visible_rank_rows: list[tuple[int, ...]] = []
    global_rank_rows: list[tuple[int, ...]] = []
    selected_path_rows: list[tuple[tuple[int, ...], ...]] = []
    terminal_count: list[int] = []
    unique_terminal_count: list[int] = []

    for start in range(0, world_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, world_count)
        worlds = tuple(generate_gate5_development_world(world_index=index) for index in range(start, stop))
        batch = run_gate5_world_batch(model, worlds, mode=mode, device=device)
        for world, result in zip(worlds, batch, strict=True):
            telemetry = result.telemetry
            covered.append(world.hidden_path in set(result.generated_terminal_paths))
            runtime_seeds.append(world.public.seed)
            productive.append(telemetry.productive_slots)
            sink.append(telemetry.sink_slots)
            frontier.append(telemetry.stage_a_frontier_width)
            live_rows.append(telemetry.stage_b_live_population_by_slot)
            depth_rows.append(telemetry.stage_b_activated_parent_depth_by_slot)
            visible_rows.append(telemetry.stage_b_visible_candidate_count_by_slot)
            score_obs_rows.append(telemetry.stage_b_score_observation_count_by_slot)
            total_score_obs.append(telemetry.total_stage_b_score_observations)
            max_score_obs.append(telemetry.max_stage_b_score_observations)
            visible_rank_rows.append(telemetry.selected_visible_score_rank_by_slot)
            global_rank_rows.append(telemetry.selected_global_score_rank_by_slot)
            selected_path_rows.append(telemetry.selected_parent_paths_by_slot)
            terminal_count.append(telemetry.generated_terminal_count)
            unique_terminal_count.append(telemetry.unique_generated_terminal_count)

    vector = tuple(covered)
    return Gate5ConditionEvaluation(
        checkpoint_index=checkpoint_index,
        mode=mode,
        world_count=world_count,
        world_indices=tuple(range(world_count)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / world_count,
        productive_slots_by_world=tuple(productive),
        sink_slots_by_world=tuple(sink),
        stage_a_frontier_width_by_world=tuple(frontier),
        stage_b_live_population_by_slot_by_world=tuple(live_rows),
        stage_b_activated_parent_depth_by_slot_by_world=tuple(depth_rows),
        stage_b_visible_candidate_count_by_slot_by_world=tuple(visible_rows),
        stage_b_score_observation_count_by_slot_by_world=tuple(score_obs_rows),
        total_stage_b_score_observations_by_world=tuple(total_score_obs),
        max_stage_b_score_observations_by_world=tuple(max_score_obs),
        selected_visible_score_rank_by_slot_by_world=tuple(visible_rank_rows),
        selected_global_score_rank_by_slot_by_world=tuple(global_rank_rows),
        selected_parent_paths_by_slot_by_world=tuple(selected_path_rows),
        generated_terminal_count_by_world=tuple(terminal_count),
        unique_generated_terminal_count_by_world=tuple(unique_terminal_count),
        total_learned_updates_per_world=GATE5_TOTAL_LEARNED_UPDATES,
        reserve_capacity=GATE5_RESERVE_CAPACITY,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint_index: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts("gate5-bounded-score-activation-bootstrap", checkpoint_index, comparison)
    )
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE5_BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (GATE5_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE5_BOOTSTRAP_SAMPLES - 1)))],
    )


def _paired_summary(
    *,
    comparison: str,
    checkpoint_index: int,
    treatment: Gate5ConditionEvaluation,
    reference: Gate5ConditionEvaluation,
) -> Gate5PairedSummary:
    if treatment.world_indices != reference.world_indices:
        raise ValueError("Gate-5 paired conditions use different world indices")
    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = GATE5_WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=checkpoint_index,
        comparison=comparison,
    )
    return Gate5PairedSummary(
        comparison=comparison,
        checkpoint_index=checkpoint_index,
        treatment_mode=treatment.mode,
        reference_mode=reference.mode,
        world_count=GATE5_WORLD_COUNT,
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=sum(differences) / GATE5_WORLD_COUNT,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate5_paired_summaries(
    conditions: Iterable[Gate5ConditionEvaluation],
) -> tuple[Gate5PairedSummary, ...]:
    rows = tuple(conditions)
    index = {(row.checkpoint_index, row.mode): row for row in rows}
    specs = (
        (
            "bounded_score_k4_vs_global_score",
            Gate5SchedulerMode.BOUNDED_SCORE_K4,
            Gate5SchedulerMode.GLOBAL_SCORE,
        ),
        (
            "bounded_score_k8_vs_global_score",
            Gate5SchedulerMode.BOUNDED_SCORE_K8,
            Gate5SchedulerMode.GLOBAL_SCORE,
        ),
        (
            "bounded_score_k16_vs_global_score",
            Gate5SchedulerMode.BOUNDED_SCORE_K16,
            Gate5SchedulerMode.GLOBAL_SCORE,
        ),
        (
            "bounded_score_k32_vs_global_score",
            Gate5SchedulerMode.BOUNDED_SCORE_K32,
            Gate5SchedulerMode.GLOBAL_SCORE,
        ),
        (
            "bounded_score_k16_vs_bounded_hash_k16",
            Gate5SchedulerMode.BOUNDED_SCORE_K16,
            Gate5SchedulerMode.BOUNDED_HASH_K16,
        ),
    )
    summaries: list[Gate5PairedSummary] = []
    for checkpoint_index in GATE5_CHECKPOINT_INDICES:
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
