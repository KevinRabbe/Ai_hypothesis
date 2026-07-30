"""Fresh world namespace and checkpoint bindings for Gate-7 continuation execution."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .gate7_high_scale_routing_bandwidth import load_verified_gate7_high_scale_checkpoint
from .gate7_high_scale_routing_bandwidth_continuation_protocol import (
    GATE7_CONTINUATION_CHECKPOINT_INDICES,
    GATE7_CONTINUATION_CHECKPOINTS,
    GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
    GATE7_CONTINUATION_HINT_RELIABILITY,
    GATE7_CONTINUATION_LEARNED_PARAMETER_COUNT,
    GATE7_CONTINUATION_POPULATIONS,
    GATE7_CONTINUATION_WORLD_COUNT,
)
from .gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer
from .gate7_scale_neutral_transition_bridge import Gate7TransitionCheckpointIdentity


@dataclass(frozen=True, slots=True)
class Gate7ContinuationWorld:
    population: int
    world_index: int
    task_depth: int
    runtime_seed: int
    noisy_hints: tuple[int, ...]
    hidden_path: tuple[int, ...]
    hidden_terminal_path_id: int

    def validate(self) -> None:
        if self.population not in GATE7_CONTINUATION_POPULATIONS:
            raise ValueError("continuation population is outside the frozen ladder")
        if not 0 <= self.world_index < GATE7_CONTINUATION_WORLD_COUNT:
            raise ValueError("continuation world index is outside 0..511")
        expected_depth = self.population.bit_length()
        if self.task_depth != expected_depth:
            raise ValueError("continuation task depth changed")
        if len(self.noisy_hints) != expected_depth or len(self.hidden_path) != expected_depth:
            raise ValueError("continuation path/hint length changed")
        if any(bit not in (0, 1) for bit in self.noisy_hints + self.hidden_path):
            raise ValueError("continuation paths and hints must remain binary")
        expected_id = 0
        for bit in self.hidden_path:
            expected_id = expected_id * 2 + bit
        if self.hidden_terminal_path_id != expected_id:
            raise ValueError("continuation hidden terminal path ID changed")
        if not 0 <= self.hidden_terminal_path_id < 2 * self.population:
            raise ValueError("continuation hidden terminal path is outside the task tree")


def continuation_seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def gate7_continuation_runtime_seed(*, population: int, world_index: int) -> int:
    if population not in GATE7_CONTINUATION_POPULATIONS:
        raise ValueError("continuation population is outside the frozen ladder")
    if not 0 <= world_index < GATE7_CONTINUATION_WORLD_COUNT:
        raise ValueError("continuation world index is outside 0..511")
    return continuation_seed_from_parts(
        "gate7-high-scale-routing-bandwidth-continuation-runtime-v0",
        population,
        world_index,
        population.bit_length(),
    )


def generate_gate7_continuation_world(
    *, population: int, world_index: int
) -> Gate7ContinuationWorld:
    """Generate one untouched continuation world from the frozen namespace."""

    if population not in GATE7_CONTINUATION_POPULATIONS:
        raise ValueError("continuation population is outside the frozen ladder")
    if not 0 <= world_index < GATE7_CONTINUATION_WORLD_COUNT:
        raise ValueError("continuation world index is outside 0..511")
    task_depth = population.bit_length()
    hidden_rng = random.Random(
        continuation_seed_from_parts(
            "gate7-high-scale-routing-bandwidth-continuation-hidden-v0",
            population,
            world_index,
            task_depth,
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(task_depth))
    hint_rng = random.Random(
        continuation_seed_from_parts(
            "gate7-high-scale-routing-bandwidth-continuation-hints-v0",
            population,
            world_index,
            task_depth,
        )
    )
    noisy_hints = tuple(
        hidden_bit
        if hint_rng.random() < GATE7_CONTINUATION_HINT_RELIABILITY
        else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    terminal_id = 0
    for bit in hidden_path:
        terminal_id = terminal_id * 2 + bit
    world = Gate7ContinuationWorld(
        population=population,
        world_index=world_index,
        task_depth=task_depth,
        runtime_seed=gate7_continuation_runtime_seed(
            population=population,
            world_index=world_index,
        ),
        noisy_hints=noisy_hints,
        hidden_path=hidden_path,
        hidden_terminal_path_id=terminal_id,
    )
    world.validate()
    return world


def load_verified_gate7_continuation_checkpoint(
    *, checkpoint_index: int, checkpoint_path: Path, device: torch.device | str
) -> tuple[Gate7ScaleNeutralScorer, Gate7TransitionCheckpointIdentity]:
    if checkpoint_index not in GATE7_CONTINUATION_CHECKPOINT_INDICES:
        raise ValueError("continuation checkpoint index must be 0, 1 or 2")
    model, identity = load_verified_gate7_high_scale_checkpoint(
        checkpoint_index=checkpoint_index,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    expected = GATE7_CONTINUATION_CHECKPOINTS[checkpoint_index]
    if identity.checkpoint_sha256 != expected["sha256"]:
        raise RuntimeError("continuation checkpoint SHA differs from the frozen protocol")
    if identity.parameter_fingerprint != expected["fingerprint"]:
        raise RuntimeError("continuation checkpoint fingerprint differs from the frozen protocol")
    if model.trainable_parameter_count() != GATE7_CONTINUATION_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("continuation scorer parameter count changed")
    return model, identity


def continuation_world_batch(
    *, population: int, batch_start: int
) -> tuple[Gate7ContinuationWorld, ...]:
    if batch_start % GATE7_CONTINUATION_EVALUATION_BATCH_SIZE:
        raise ValueError("continuation batch start must align to 64 worlds")
    batch_stop = batch_start + GATE7_CONTINUATION_EVALUATION_BATCH_SIZE
    if not 0 <= batch_start < batch_stop <= GATE7_CONTINUATION_WORLD_COUNT:
        raise ValueError("continuation batch range is outside 0..511")
    worlds = tuple(
        generate_gate7_continuation_world(population=population, world_index=index)
        for index in range(batch_start, batch_stop)
    )
    validate_continuation_world_batch(worlds)
    return worlds


def validate_continuation_world_batch(worlds: tuple[Gate7ContinuationWorld, ...]) -> int:
    if len(worlds) != GATE7_CONTINUATION_EVALUATION_BATCH_SIZE:
        raise ValueError("continuation execution requires exactly 64 physical worlds")
    population = worlds[0].population
    if population not in GATE7_CONTINUATION_POPULATIONS:
        raise ValueError("continuation batch population is outside the frozen ladder")
    indices = tuple(world.world_index for world in worlds)
    if indices != tuple(range(indices[0], indices[0] + GATE7_CONTINUATION_EVALUATION_BATCH_SIZE)):
        raise ValueError("continuation batch world indices must be contiguous")
    if indices[0] % GATE7_CONTINUATION_EVALUATION_BATCH_SIZE:
        raise ValueError("continuation batch world indices must align to 64")
    for world in worlds:
        world.validate()
        if world.population != population:
            raise ValueError("continuation batch mixes populations")
    return population
