"""Fresh worlds, public rank permutations, and checkpoint bindings for Gate-7 ceiling diagnosis."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .gate7_high_scale_routing_bandwidth import load_verified_gate7_high_scale_checkpoint
from .gate7_information_ceiling_decomposition_protocol import (
    GATE7_INFORMATION_CEILING_CHECKPOINTS,
    GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES,
    GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE,
    GATE7_INFORMATION_CEILING_HASH,
    GATE7_INFORMATION_CEILING_HIDDEN_NAMESPACE,
    GATE7_INFORMATION_CEILING_HINT_NAMESPACE,
    GATE7_INFORMATION_CEILING_HINT_RELIABILITY,
    GATE7_INFORMATION_CEILING_LEARNED_PARAMETER_COUNT,
    GATE7_INFORMATION_CEILING_POPULATIONS,
    GATE7_INFORMATION_CEILING_RUNTIME_NAMESPACE,
    GATE7_INFORMATION_CEILING_TIE_NAMESPACE,
    GATE7_INFORMATION_CEILING_WORLD_COUNT,
    gate7_information_ceiling_frontier_depth,
)
from .gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer
from .gate7_scale_neutral_transition_bridge import Gate7TransitionCheckpointIdentity


@dataclass(frozen=True, slots=True)
class Gate7InformationCeilingWorld:
    population: int
    world_index: int
    frontier_depth: int
    task_depth: int
    runtime_seed: int
    noisy_hints: tuple[int, ...]
    hidden_path: tuple[int, ...]
    hidden_parent_index: int
    hidden_terminal_path_id: int
    tie_multiplier: int
    tie_offset: int
    hash_multiplier: int
    hash_offset: int

    def validate(self) -> None:
        if self.population not in GATE7_INFORMATION_CEILING_POPULATIONS:
            raise ValueError("information-ceiling population is outside the frozen ladder")
        if not 0 <= self.world_index < GATE7_INFORMATION_CEILING_WORLD_COUNT:
            raise ValueError("information-ceiling world index is outside 0..511")
        expected_frontier_depth = gate7_information_ceiling_frontier_depth(self.population)
        if self.frontier_depth != expected_frontier_depth:
            raise ValueError("information-ceiling frontier depth changed")
        if self.task_depth != expected_frontier_depth + 1:
            raise ValueError("information-ceiling task depth changed")
        if len(self.noisy_hints) != self.task_depth or len(self.hidden_path) != self.task_depth:
            raise ValueError("information-ceiling path or hint length changed")
        if any(bit not in (0, 1) for bit in self.noisy_hints + self.hidden_path):
            raise ValueError("information-ceiling paths and hints must remain binary")
        parent = 0
        terminal = 0
        for index, bit in enumerate(self.hidden_path):
            terminal = terminal * 2 + bit
            if index < self.frontier_depth:
                parent = parent * 2 + bit
        if self.hidden_parent_index != parent:
            raise ValueError("hidden parent identity changed")
        if self.hidden_terminal_path_id != terminal:
            raise ValueError("hidden terminal identity changed")
        if terminal // 2 != parent:
            raise ValueError("terminal-to-parent identity changed")
        if not 0 <= parent < self.population:
            raise ValueError("hidden parent is outside the complete frontier")
        mask = self.population - 1
        for multiplier, offset in (
            (self.tie_multiplier, self.tie_offset),
            (self.hash_multiplier, self.hash_offset),
        ):
            if multiplier <= 0 or multiplier > mask or multiplier % 2 != 1:
                raise ValueError("public permutation multiplier must be odd and below population")
            if not 0 <= offset <= mask:
                raise ValueError("public permutation offset is outside population")


def information_ceiling_seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def gate7_information_ceiling_runtime_seed(*, population: int, world_index: int) -> int:
    return information_ceiling_seed_from_parts(
        GATE7_INFORMATION_CEILING_RUNTIME_NAMESPACE,
        population,
        world_index,
        gate7_information_ceiling_frontier_depth(population),
    )


def _public_affine_permutation(
    *, namespace: str, population: int, world_index: int
) -> tuple[int, int]:
    rng = random.Random(
        information_ceiling_seed_from_parts(
            namespace,
            population,
            world_index,
            gate7_information_ceiling_frontier_depth(population),
        )
    )
    multiplier = rng.randrange(1, population, 2)
    offset = rng.randrange(population)
    return multiplier, offset


def generate_gate7_information_ceiling_world(
    *, population: int, world_index: int
) -> Gate7InformationCeilingWorld:
    if population not in GATE7_INFORMATION_CEILING_POPULATIONS:
        raise ValueError("information-ceiling population is outside the frozen ladder")
    if not 0 <= world_index < GATE7_INFORMATION_CEILING_WORLD_COUNT:
        raise ValueError("information-ceiling world index is outside 0..511")
    frontier_depth = gate7_information_ceiling_frontier_depth(population)
    task_depth = frontier_depth + 1
    hidden_rng = random.Random(
        information_ceiling_seed_from_parts(
            GATE7_INFORMATION_CEILING_HIDDEN_NAMESPACE,
            population,
            world_index,
            task_depth,
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(task_depth))
    hint_rng = random.Random(
        information_ceiling_seed_from_parts(
            GATE7_INFORMATION_CEILING_HINT_NAMESPACE,
            population,
            world_index,
            task_depth,
        )
    )
    noisy_hints = tuple(
        hidden_bit
        if hint_rng.random() < GATE7_INFORMATION_CEILING_HINT_RELIABILITY
        else 1 - hidden_bit
        for hidden_bit in hidden_path
    )
    hidden_parent = 0
    hidden_terminal = 0
    for index, bit in enumerate(hidden_path):
        hidden_terminal = hidden_terminal * 2 + bit
        if index < frontier_depth:
            hidden_parent = hidden_parent * 2 + bit
    tie_multiplier, tie_offset = _public_affine_permutation(
        namespace=GATE7_INFORMATION_CEILING_TIE_NAMESPACE,
        population=population,
        world_index=world_index,
    )
    hash_multiplier, hash_offset = _public_affine_permutation(
        namespace=f"{GATE7_INFORMATION_CEILING_HASH}-v0",
        population=population,
        world_index=world_index,
    )
    world = Gate7InformationCeilingWorld(
        population=population,
        world_index=world_index,
        frontier_depth=frontier_depth,
        task_depth=task_depth,
        runtime_seed=gate7_information_ceiling_runtime_seed(
            population=population, world_index=world_index
        ),
        noisy_hints=noisy_hints,
        hidden_path=hidden_path,
        hidden_parent_index=hidden_parent,
        hidden_terminal_path_id=hidden_terminal,
        tie_multiplier=tie_multiplier,
        tie_offset=tie_offset,
        hash_multiplier=hash_multiplier,
        hash_offset=hash_offset,
    )
    world.validate()
    return world


def information_ceiling_world_batch(
    *, population: int, batch_start: int
) -> tuple[Gate7InformationCeilingWorld, ...]:
    if batch_start % GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE:
        raise ValueError("information-ceiling batch start must align to 64")
    stop = batch_start + GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE
    if not 0 <= batch_start < stop <= GATE7_INFORMATION_CEILING_WORLD_COUNT:
        raise ValueError("information-ceiling batch range is outside 0..511")
    worlds = tuple(
        generate_gate7_information_ceiling_world(
            population=population,
            world_index=world_index,
        )
        for world_index in range(batch_start, stop)
    )
    validate_information_ceiling_world_batch(worlds)
    return worlds


def validate_information_ceiling_world_batch(
    worlds: tuple[Gate7InformationCeilingWorld, ...],
) -> int:
    if len(worlds) != GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE:
        raise ValueError("information-ceiling execution requires exactly 64 worlds")
    population = worlds[0].population
    indices = tuple(world.world_index for world in worlds)
    if indices != tuple(range(indices[0], indices[0] + len(worlds))):
        raise ValueError("information-ceiling world indices must be contiguous")
    if indices[0] % GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE:
        raise ValueError("information-ceiling world indices must align to 64")
    for world in worlds:
        world.validate()
        if world.population != population:
            raise ValueError("information-ceiling batch mixes populations")
    return population


def load_verified_gate7_information_ceiling_checkpoint(
    *, checkpoint_index: int, checkpoint_path: Path, device: torch.device | str
) -> tuple[Gate7ScaleNeutralScorer, Gate7TransitionCheckpointIdentity]:
    if checkpoint_index not in GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES:
        raise ValueError("information-ceiling checkpoint must be T0, T1 or T2")
    model, identity = load_verified_gate7_high_scale_checkpoint(
        checkpoint_index=checkpoint_index,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    expected = GATE7_INFORMATION_CEILING_CHECKPOINTS[checkpoint_index]
    if identity.checkpoint_sha256 != expected["sha256"]:
        raise RuntimeError("information-ceiling checkpoint SHA changed")
    if identity.parameter_fingerprint != expected["fingerprint"]:
        raise RuntimeError("information-ceiling checkpoint fingerprint changed")
    if model.trainable_parameter_count() != GATE7_INFORMATION_CEILING_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("information-ceiling learned parameter count changed")
    return model, identity
