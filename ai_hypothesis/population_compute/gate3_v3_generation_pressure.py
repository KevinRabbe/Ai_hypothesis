"""Gate-3 v3 generation-synchronous frontier-pressure evaluation.

No training occurs in this experiment.  It reuses the frozen Gate-3 v1 scorer/checkpoints and
changes only the answer-blind search scheduler so a 128-hypothesis depth-7 generation is created
before the final search phase.  Learned parameters, active neural width and total recurrent work
remain fixed across conditions.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .gate3_v1_model import (
    Gate3V1NeuralCandidate,
    Gate3V1Scorer,
    encode_gate3_v1_child_input,
)
from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    GATE3_V1_UPDATES_PER_ROUND,
    Gate3V1ControlMode,
    Gate3V1PublicWorld,
    deterministic_gate3_v1_tie_break,
    quantize_gate3_v1_score,
)

GATE3_V3_EXPERIMENT_VERSION = "gate3-v3-generation-pressure-v0"
GATE3_V3_DEPTH = 8
GATE3_V3_HINT_RELIABILITY = 0.70
GATE3_V3_WORLD_COUNT = 256
GATE3_V3_EVAL_BATCH_SIZE = 64
GATE3_V3_BOOTSTRAP_SAMPLES = 2_000
GATE3_V3_SCHEDULED_SLOTS = 223
GATE3_V3_TOTAL_LEARNED_UPDATES = GATE3_V3_SCHEDULED_SLOTS * GATE3_V1_UPDATES_PER_ROUND
GATE3_V3_STABLE_CAPACITIES = (16, 64, 256)
GATE3_V3_CONTROL_CAPACITY = 256
GATE3_V3_CHECKPOINT_INDICES = (0, 1, 2)
GATE3_V3_EXPECTED_DEPTH7_PREPRUNE = 128
GATE3_V3_L64_FINAL_PRODUCTIVE = 64
GATE3_V3_L64_FINAL_SINK = 32
GATE3_V3_L256_FINAL_PRODUCTIVE = 96
GATE3_V3_L256_FINAL_SINK = 0

GATE3_V3_CONDITIONS: tuple[tuple[int, Gate3V1ControlMode], ...] = (
    (16, Gate3V1ControlMode.STABLE_RESERVE),
    (64, Gate3V1ControlMode.STABLE_RESERVE),
    (256, Gate3V1ControlMode.STABLE_RESERVE),
    (256, Gate3V1ControlMode.COLLAPSED_DIVERSITY),
    (256, Gate3V1ControlMode.RESHUFFLED_CONTINUITY),
)


@dataclass(frozen=True, slots=True)
class Gate3V3EvaluationWorld:
    world_index: int
    public: Gate3V1PublicWorld
    hidden_path: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.world_index < GATE3_V3_WORLD_COUNT:
            raise ValueError("Gate-3 v3 world index is outside the frozen development domain")
        self.public.validate()
        if self.public.depth != GATE3_V3_DEPTH:
            raise ValueError("Gate-3 v3 must remain at depth 8")
        if len(self.hidden_path) != GATE3_V3_DEPTH or any(bit not in (0, 1) for bit in self.hidden_path):
            raise ValueError("Gate-3 v3 hidden path must be eight binary decisions")


@dataclass(frozen=True, slots=True)
class Gate3V3CheckpointIdentity:
    checkpoint_index: int
    checkpoint_sha256: str
    parameter_fingerprint: str
    learned_parameter_count: int


@dataclass(frozen=True, slots=True)
class Gate3V3WorldTelemetry:
    productive_slots: int
    sink_slots: int
    total_learned_updates: int
    preprune_widths: tuple[int, ...]
    retained_widths: tuple[int, ...]
    unique_retained_widths: tuple[int, ...]
    binding_by_generation: tuple[bool, ...]
    depth7_preprune_width: int
    depth7_retained_width: int
    depth7_expanded_parents: int
    generated_terminal_count: int
    unique_generated_terminal_count: int


@dataclass(frozen=True, slots=True)
class Gate3V3WorldResult:
    generated_terminal_paths: tuple[tuple[int, ...], ...]
    telemetry: Gate3V3WorldTelemetry


@dataclass(frozen=True, slots=True)
class Gate3V3ConditionEvaluation:
    checkpoint_index: int
    reserve_capacity: int
    mode: Gate3V1ControlMode
    world_count: int
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    covered_by_world: tuple[bool, ...]
    coverage_rate: float
    productive_slots_by_world: tuple[int, ...]
    sink_slots_by_world: tuple[int, ...]
    preprune_widths_by_world: tuple[tuple[int, ...], ...]
    retained_widths_by_world: tuple[tuple[int, ...], ...]
    unique_retained_widths_by_world: tuple[tuple[int, ...], ...]
    binding_by_generation_by_world: tuple[tuple[bool, ...], ...]
    depth7_preprune_width_by_world: tuple[int, ...]
    depth7_retained_width_by_world: tuple[int, ...]
    depth7_expanded_parents_by_world: tuple[int, ...]
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
class Gate3V3PairedSummary:
    comparison: str
    checkpoint_index: int
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
        payload["treatment_mode"] = self.treatment_mode.value
        payload["reference_mode"] = self.reference_mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Gate3V3DevelopmentResult:
    experiment_version: str
    scientific_status: str
    confirmation_opened: bool
    training_performed: bool
    checkpoints: tuple[Gate3V3CheckpointIdentity, ...]
    world_count: int
    evaluation_batch_size: int
    bootstrap_samples: int
    depth: int
    hint_reliability: float
    scheduled_slots: int
    total_learned_updates_per_world: int
    conditions: tuple[Gate3V3ConditionEvaluation, ...]
    paired_summaries: tuple[Gate3V3PairedSummary, ...]

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


def gate3_v3_runtime_seed(*, world_index: int) -> int:
    return _seed_from_parts("gate3-v3-generation-pressure-development-runtime", world_index, GATE3_V3_DEPTH)


def generate_gate3_v3_development_world(*, world_index: int) -> Gate3V3EvaluationWorld:
    if not 0 <= world_index < GATE3_V3_WORLD_COUNT:
        raise ValueError("Gate-3 v3 world index is outside 0..255")
    hidden_rng = random.Random(
        _seed_from_parts("gate3-v3-generation-pressure-development-hidden", world_index, GATE3_V3_DEPTH)
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(GATE3_V3_DEPTH))
    hint_rng = random.Random(
        _seed_from_parts("gate3-v3-generation-pressure-development-hints", world_index, GATE3_V3_DEPTH)
    )
    noisy_hints = tuple(
        hidden_bit if hint_rng.random() < GATE3_V3_HINT_RELIABILITY else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    result = Gate3V3EvaluationWorld(
        world_index=world_index,
        public=Gate3V1PublicWorld(
            seed=gate3_v3_runtime_seed(world_index=world_index),
            depth=GATE3_V3_DEPTH,
            noisy_hints=noisy_hints,
        ),
        hidden_path=hidden_path,
    )
    result.validate()
    return result


def _rank_candidates(
    candidates: Iterable[Gate3V1NeuralCandidate], *, world_seed: int, generation: int
) -> tuple[Gate3V1NeuralCandidate, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -quantize_gate3_v1_score(candidate.score),
                deterministic_gate3_v1_tie_break(
                    world_seed=world_seed,
                    expansion_index=generation,
                    candidate_path=candidate.path,
                ),
            ),
        )
    )


def _reshuffle_histories(
    candidates: tuple[Gate3V1NeuralCandidate, ...], *, world_seed: int, generation: int
) -> tuple[Gate3V1NeuralCandidate, ...]:
    if len(candidates) <= 1:
        return candidates
    rng = random.Random(
        _seed_from_parts("gate3-v3-generation-pressure-reshuffle", world_seed, generation, len(candidates))
    )
    permutation = list(range(len(candidates)))
    rng.shuffle(permutation)
    histories = tuple((candidate.state, candidate.score) for candidate in candidates)
    return tuple(
        Gate3V1NeuralCandidate(
            path=candidate.path,
            state=histories[permutation[index]][0].clone(),
            score=histories[permutation[index]][1],
        )
        for index, candidate in enumerate(candidates)
    )


def _apply_generation_control(
    candidates: tuple[Gate3V1NeuralCandidate, ...],
    *,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    world_seed: int,
    generation: int,
) -> tuple[Gate3V1NeuralCandidate, ...]:
    ranked = _rank_candidates(candidates, world_seed=world_seed, generation=generation)
    retained = ranked[:reserve_capacity]
    if mode is Gate3V1ControlMode.STABLE_RESERVE:
        return retained
    if mode is Gate3V1ControlMode.COLLAPSED_DIVERSITY:
        return retained[:1]
    if mode is Gate3V1ControlMode.RESHUFFLED_CONTINUITY:
        return _reshuffle_histories(retained, world_seed=world_seed, generation=generation)
    raise ValueError(f"unknown Gate-3 v3 control mode: {mode}")


def _expand_parent_batch(
    model: Gate3V1Scorer,
    worlds: tuple[Gate3V3EvaluationWorld, ...],
    selected_by_world: tuple[tuple[Gate3V1NeuralCandidate, ...], ...],
    *,
    child_depth: int,
    device: torch.device,
) -> tuple[tuple[Gate3V1NeuralCandidate, ...], ...]:
    owner_rows: list[tuple[int, tuple[int, ...], int]] = []
    states: list[torch.Tensor] = []
    inputs: list[torch.Tensor] = []
    for world_offset, (world, parents) in enumerate(zip(worlds, selected_by_world, strict=True)):
        hint = world.public.noisy_hints[child_depth - 1]
        for parent in parents:
            for action in (0, 1):
                states.append(parent.state.clone())
                inputs.append(
                    encode_gate3_v1_child_input(
                        world=world.public,
                        child_depth=child_depth,
                        observed_hint=hint,
                        branch_action=action,
                        sink=False,
                        device=device,
                    )
                )
                owner_rows.append((world_offset, parent.path, action))

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
    for row_index, (world_offset, parent_path, action) in enumerate(owner_rows):
        children[world_offset].append(
            Gate3V1NeuralCandidate(
                path=parent_path + (action,),
                state=advanced[row_index].clone(),
                score=float(scores[row_index].item()),
            )
        )
    return tuple(tuple(row) for row in children)


def _execute_sink_work(
    model: Gate3V1Scorer,
    worlds: tuple[Gate3V3EvaluationWorld, ...],
    sink_slots: tuple[int, ...],
    *,
    device: torch.device,
) -> None:
    rows: list[tuple[int, Gate3V3EvaluationWorld]] = []
    for world_offset, (world, slots) in enumerate(zip(worlds, sink_slots, strict=True)):
        for _ in range(slots * 2):
            rows.append((world_offset, world))
    if not rows:
        return

    # Bound transient tensors while preserving one eager-FP32 recurrent call per sink lane.
    chunk_size = 8192
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        state_batch = model.initial_state(len(chunk), device=device)
        input_batch = torch.stack(
            [
                encode_gate3_v1_child_input(
                    world=world.public,
                    child_depth=GATE3_V3_DEPTH,
                    observed_hint=None,
                    branch_action=None,
                    sink=True,
                    device=device,
                )
                for _, world in chunk
            ],
            dim=0,
        )
        model.advance(
            state_batch,
            input_batch,
            repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
        )


def run_gate3_v3_world_batch(
    model: Gate3V1Scorer,
    worlds: Iterable[Gate3V3EvaluationWorld],
    *,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    device: torch.device | str,
) -> tuple[Gate3V3WorldResult, ...]:
    world_tuple = tuple(worlds)
    if not world_tuple:
        return ()
    if (reserve_capacity, mode) not in GATE3_V3_CONDITIONS:
        raise ValueError("condition is outside the frozen Gate-3 v3 matrix")
    for world in world_tuple:
        world.validate()

    target_device = torch.device(device)
    model = model.to(target_device)
    batch_size = len(world_tuple)
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
    productive = [0 for _ in world_tuple]
    preprune_widths: list[list[int]] = [[] for _ in world_tuple]
    retained_widths: list[list[int]] = [[] for _ in world_tuple]
    unique_widths: list[list[int]] = [[] for _ in world_tuple]
    binding: list[list[bool]] = [[] for _ in world_tuple]
    terminals: list[list[tuple[int, ...]]] = [[] for _ in world_tuple]
    depth7_expanded = [0 for _ in world_tuple]

    with torch.inference_mode():
        # Build complete nonterminal generations through the depth-7 frontier.
        for parent_depth in range(0, GATE3_V3_DEPTH - 1):
            selected = tuple(populations)
            children_by_world = _expand_parent_batch(
                model,
                world_tuple,
                selected,
                child_depth=parent_depth + 1,
                device=target_device,
            )
            next_populations: list[tuple[Gate3V1NeuralCandidate, ...]] = []
            for world_offset, (world, parents, children) in enumerate(
                zip(world_tuple, selected, children_by_world, strict=True)
            ):
                productive[world_offset] += len(parents)
                pre_width = len(children)
                retained = _apply_generation_control(
                    children,
                    reserve_capacity=reserve_capacity,
                    mode=mode,
                    world_seed=world.public.seed,
                    generation=parent_depth + 1,
                )
                preprune_widths[world_offset].append(pre_width)
                retained_widths[world_offset].append(len(retained))
                unique_widths[world_offset].append(len({candidate.path for candidate in retained}))
                binding[world_offset].append(pre_width > len(retained))
                next_populations.append(retained)
            populations = next_populations

        # Final depth-7 parent generation: use only the remaining scheduled slots.
        selected_final: list[tuple[Gate3V1NeuralCandidate, ...]] = []
        for world_offset, (world, population) in enumerate(zip(world_tuple, populations, strict=True)):
            remaining_slots = GATE3_V3_SCHEDULED_SLOTS - productive[world_offset]
            if remaining_slots < 0:
                raise RuntimeError("Gate-3 v3 generation work exceeded the frozen slot budget")
            ranked = _rank_candidates(
                population,
                world_seed=world.public.seed,
                generation=GATE3_V3_DEPTH,
            )
            selected_final.append(ranked[:remaining_slots])

        terminal_children = _expand_parent_batch(
            model,
            world_tuple,
            tuple(selected_final),
            child_depth=GATE3_V3_DEPTH,
            device=target_device,
        )
        for world_offset, (selected, children) in enumerate(
            zip(selected_final, terminal_children, strict=True)
        ):
            productive[world_offset] += len(selected)
            depth7_expanded[world_offset] = len(selected)
            terminals[world_offset].extend(candidate.path for candidate in children)

        sink_slots = tuple(GATE3_V3_SCHEDULED_SLOTS - count for count in productive)
        if any(value < 0 for value in sink_slots):
            raise RuntimeError("Gate-3 v3 productive work exceeded frozen scheduled slots")
        _execute_sink_work(model, world_tuple, sink_slots, device=target_device)

    results: list[Gate3V3WorldResult] = []
    for world_offset, population in enumerate(populations):
        updates = (productive[world_offset] + sink_slots[world_offset]) * GATE3_V1_UPDATES_PER_ROUND
        if updates != GATE3_V3_TOTAL_LEARNED_UPDATES:
            raise RuntimeError("Gate-3 v3 learned-work identity was violated")
        if len(preprune_widths[world_offset]) != GATE3_V3_DEPTH - 1:
            raise RuntimeError("Gate-3 v3 generation telemetry is incomplete")
        results.append(
            Gate3V3WorldResult(
                generated_terminal_paths=tuple(terminals[world_offset]),
                telemetry=Gate3V3WorldTelemetry(
                    productive_slots=productive[world_offset],
                    sink_slots=sink_slots[world_offset],
                    total_learned_updates=updates,
                    preprune_widths=tuple(preprune_widths[world_offset]),
                    retained_widths=tuple(retained_widths[world_offset]),
                    unique_retained_widths=tuple(unique_widths[world_offset]),
                    binding_by_generation=tuple(binding[world_offset]),
                    depth7_preprune_width=preprune_widths[world_offset][-1],
                    depth7_retained_width=len(population),
                    depth7_expanded_parents=depth7_expanded[world_offset],
                    generated_terminal_count=len(terminals[world_offset]),
                    unique_generated_terminal_count=len(set(terminals[world_offset])),
                ),
            )
        )

    # Hard preregistered structural invariants for the primary stable frontier conditions.
    if mode is Gate3V1ControlMode.STABLE_RESERVE and reserve_capacity in (64, 256):
        for result in results:
            telemetry = result.telemetry
            if telemetry.depth7_preprune_width != GATE3_V3_EXPECTED_DEPTH7_PREPRUNE:
                raise RuntimeError("Gate-3 v3 did not create the frozen 128-hypothesis depth-7 frontier")
            if reserve_capacity == 64:
                if telemetry.depth7_retained_width != 64:
                    raise RuntimeError("Gate-3 v3 L64 did not retain exactly 64 depth-7 hypotheses")
                if telemetry.depth7_expanded_parents != GATE3_V3_L64_FINAL_PRODUCTIVE:
                    raise RuntimeError("Gate-3 v3 L64 final productive-slot identity failed")
                if telemetry.sink_slots != GATE3_V3_L64_FINAL_SINK:
                    raise RuntimeError("Gate-3 v3 L64 final sink-slot identity failed")
            else:
                if telemetry.depth7_retained_width != 128:
                    raise RuntimeError("Gate-3 v3 L256 did not retain all 128 depth-7 hypotheses")
                if telemetry.depth7_expanded_parents != GATE3_V3_L256_FINAL_PRODUCTIVE:
                    raise RuntimeError("Gate-3 v3 L256 final productive-slot identity failed")
                if telemetry.sink_slots != GATE3_V3_L256_FINAL_SINK:
                    raise RuntimeError("Gate-3 v3 L256 final sink-slot identity failed")
    return tuple(results)


def evaluate_gate3_v3_condition(
    model: Gate3V1Scorer,
    *,
    checkpoint_index: int,
    reserve_capacity: int,
    mode: Gate3V1ControlMode,
    device: torch.device | str,
    world_count: int = GATE3_V3_WORLD_COUNT,
    evaluation_batch_size: int = GATE3_V3_EVAL_BATCH_SIZE,
) -> Gate3V3ConditionEvaluation:
    if checkpoint_index not in GATE3_V3_CHECKPOINT_INDICES:
        raise ValueError("checkpoint index must be 0, 1 or 2")
    if world_count != GATE3_V3_WORLD_COUNT:
        raise ValueError("Gate-3 v3 development must use exactly 256 worlds")
    if evaluation_batch_size != GATE3_V3_EVAL_BATCH_SIZE:
        raise ValueError("Gate-3 v3 evaluation batch size is frozen at 64")
    if (reserve_capacity, mode) not in GATE3_V3_CONDITIONS:
        raise ValueError("condition is outside the frozen Gate-3 v3 matrix")

    covered: list[bool] = []
    runtime_seeds: list[int] = []
    productive: list[int] = []
    sink: list[int] = []
    preprune: list[tuple[int, ...]] = []
    retained: list[tuple[int, ...]] = []
    unique: list[tuple[int, ...]] = []
    binding: list[tuple[bool, ...]] = []
    depth7_pre: list[int] = []
    depth7_retained: list[int] = []
    depth7_expanded: list[int] = []
    terminal_count: list[int] = []
    unique_terminal_count: list[int] = []

    for start in range(0, world_count, evaluation_batch_size):
        stop = min(start + evaluation_batch_size, world_count)
        worlds = tuple(generate_gate3_v3_development_world(world_index=index) for index in range(start, stop))
        batch = run_gate3_v3_world_batch(
            model,
            worlds,
            reserve_capacity=reserve_capacity,
            mode=mode,
            device=device,
        )
        for world, runtime_result in zip(worlds, batch, strict=True):
            telemetry = runtime_result.telemetry
            covered.append(world.hidden_path in set(runtime_result.generated_terminal_paths))
            runtime_seeds.append(world.public.seed)
            productive.append(telemetry.productive_slots)
            sink.append(telemetry.sink_slots)
            preprune.append(telemetry.preprune_widths)
            retained.append(telemetry.retained_widths)
            unique.append(telemetry.unique_retained_widths)
            binding.append(telemetry.binding_by_generation)
            depth7_pre.append(telemetry.depth7_preprune_width)
            depth7_retained.append(telemetry.depth7_retained_width)
            depth7_expanded.append(telemetry.depth7_expanded_parents)
            terminal_count.append(telemetry.generated_terminal_count)
            unique_terminal_count.append(telemetry.unique_generated_terminal_count)

    vector = tuple(covered)
    return Gate3V3ConditionEvaluation(
        checkpoint_index=checkpoint_index,
        reserve_capacity=reserve_capacity,
        mode=mode,
        world_count=world_count,
        world_indices=tuple(range(world_count)),
        runtime_seeds=tuple(runtime_seeds),
        covered_by_world=vector,
        coverage_rate=sum(int(value) for value in vector) / world_count,
        productive_slots_by_world=tuple(productive),
        sink_slots_by_world=tuple(sink),
        preprune_widths_by_world=tuple(preprune),
        retained_widths_by_world=tuple(retained),
        unique_retained_widths_by_world=tuple(unique),
        binding_by_generation_by_world=tuple(binding),
        depth7_preprune_width_by_world=tuple(depth7_pre),
        depth7_retained_width_by_world=tuple(depth7_retained),
        depth7_expanded_parents_by_world=tuple(depth7_expanded),
        generated_terminal_count_by_world=tuple(terminal_count),
        unique_generated_terminal_count_by_world=tuple(unique_terminal_count),
        total_learned_updates_per_world=GATE3_V3_TOTAL_LEARNED_UPDATES,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
    )


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint_index: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts("gate3-v3-generation-pressure-bootstrap", checkpoint_index, comparison)
    )
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(GATE3_V3_BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (GATE3_V3_BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (GATE3_V3_BOOTSTRAP_SAMPLES - 1)))],
    )


def _paired_summary(
    *,
    comparison: str,
    checkpoint_index: int,
    treatment: Gate3V3ConditionEvaluation,
    reference: Gate3V3ConditionEvaluation,
) -> Gate3V3PairedSummary:
    if treatment.world_indices != reference.world_indices:
        raise ValueError("Gate-3 v3 paired conditions must share world indices")
    pairs = tuple(zip(treatment.covered_by_world, reference.covered_by_world, strict=True))
    treatment_only = sum(int(a and not b) for a, b in pairs)
    reference_only = sum(int(b and not a) for a, b in pairs)
    both = sum(int(a and b) for a, b in pairs)
    neither = GATE3_V3_WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(a) - int(b) for a, b in pairs)
    low, high = _bootstrap_ci(
        differences,
        checkpoint_index=checkpoint_index,
        comparison=comparison,
    )
    return Gate3V3PairedSummary(
        comparison=comparison,
        checkpoint_index=checkpoint_index,
        treatment_capacity=treatment.reserve_capacity,
        treatment_mode=treatment.mode,
        reference_capacity=reference.reserve_capacity,
        reference_mode=reference.mode,
        world_count=GATE3_V3_WORLD_COUNT,
        treatment_only=treatment_only,
        reference_only=reference_only,
        both_covered=both,
        neither_covered=neither,
        coverage_delta=sum(differences) / GATE3_V3_WORLD_COUNT,
        bootstrap_ci_low=low,
        bootstrap_ci_high=high,
    )


def build_gate3_v3_paired_summaries(
    conditions: Iterable[Gate3V3ConditionEvaluation],
) -> tuple[Gate3V3PairedSummary, ...]:
    rows = tuple(conditions)
    index = {
        (row.checkpoint_index, row.reserve_capacity, row.mode): row
        for row in rows
    }
    specs = (
        ("stable_l256_vs_l64", 256, Gate3V1ControlMode.STABLE_RESERVE, 64, Gate3V1ControlMode.STABLE_RESERVE),
        ("stable_l64_vs_l16", 64, Gate3V1ControlMode.STABLE_RESERVE, 16, Gate3V1ControlMode.STABLE_RESERVE),
        ("stable_l256_vs_collapsed", 256, Gate3V1ControlMode.STABLE_RESERVE, 256, Gate3V1ControlMode.COLLAPSED_DIVERSITY),
        ("stable_l256_vs_reshuffled", 256, Gate3V1ControlMode.STABLE_RESERVE, 256, Gate3V1ControlMode.RESHUFFLED_CONTINUITY),
    )
    summaries: list[Gate3V3PairedSummary] = []
    for checkpoint_index in GATE3_V3_CHECKPOINT_INDICES:
        for comparison, t_cap, t_mode, r_cap, r_mode in specs:
            summaries.append(
                _paired_summary(
                    comparison=comparison,
                    checkpoint_index=checkpoint_index,
                    treatment=index[(checkpoint_index, t_cap, t_mode)],
                    reference=index[(checkpoint_index, r_cap, r_mode)],
                )
            )
    return tuple(summaries)
