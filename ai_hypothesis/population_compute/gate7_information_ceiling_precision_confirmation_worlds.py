"""Fresh worlds and checkpoint bindings for Gate-7 precision confirmation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .gate7_high_scale_routing_bandwidth import load_verified_gate7_high_scale_checkpoint
from .gate7_information_ceiling_decomposition_protocol import (
    GATE7_INFORMATION_CEILING_CHECKPOINTS,
)
from .gate7_information_ceiling_precision_confirmation_protocol import (
    GATE7_PRECISION_CHECKPOINT_INDICES,
    GATE7_PRECISION_EVALUATION_BATCH_SIZE,
    GATE7_PRECISION_HASH_NAMESPACE,
    GATE7_PRECISION_HIDDEN_NAMESPACE,
    GATE7_PRECISION_HINT_NAMESPACE,
    GATE7_PRECISION_HINT_RELIABILITY,
    GATE7_PRECISION_LEARNED_PARAMETER_COUNT,
    GATE7_PRECISION_POPULATIONS,
    GATE7_PRECISION_RUNTIME_NAMESPACE,
    GATE7_PRECISION_TIE_NAMESPACE,
    GATE7_PRECISION_WORLD_COUNT,
)
from .gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer
from .gate7_scale_neutral_transition_bridge import Gate7TransitionCheckpointIdentity


@dataclass(frozen=True, slots=True)
class Gate7PrecisionWorld:
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
        if self.population not in GATE7_PRECISION_POPULATIONS:
            raise ValueError("precision population is outside the frozen ladder")
        if not 0 <= self.world_index < GATE7_PRECISION_WORLD_COUNT:
            raise ValueError("precision world index is outside the frozen range")
        expected_depth = self.population.bit_length() - 1
        if self.frontier_depth != expected_depth:
            raise ValueError("precision frontier depth changed")
        if self.task_depth != expected_depth + 1:
            raise ValueError("precision task depth changed")
        if len(self.noisy_hints) != self.task_depth or len(self.hidden_path) != self.task_depth:
            raise ValueError("precision path or hint length changed")
        if any(bit not in (0, 1) for bit in self.noisy_hints + self.hidden_path):
            raise ValueError("precision paths and hints must remain binary")
        parent = 0
        terminal = 0
        for index, bit in enumerate(self.hidden_path):
            terminal = terminal * 2 + bit
            if index < self.frontier_depth:
                parent = parent * 2 + bit
        if parent != self.hidden_parent_index or terminal != self.hidden_terminal_path_id:
            raise ValueError("precision hidden identity changed")
        if terminal // 2 != parent:
            raise ValueError("precision terminal-to-parent identity changed")
        mask = self.population - 1
        for multiplier, offset in (
            (self.tie_multiplier, self.tie_offset),
            (self.hash_multiplier, self.hash_offset),
        ):
            if multiplier <= 0 or multiplier > mask or multiplier % 2 != 1:
                raise ValueError("precision public multiplier must be odd and below population")
            if not 0 <= offset <= mask:
                raise ValueError("precision public offset is outside population")


def precision_seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _public_affine_permutation(
    *, namespace: str, population: int, world_index: int
) -> tuple[int, int]:
    depth = population.bit_length() - 1
    rng = random.Random(
        precision_seed_from_parts(namespace, population, world_index, depth)
    )
    return rng.randrange(1, population, 2), rng.randrange(population)


def generate_gate7_precision_world(
    *, population: int, world_index: int
) -> Gate7PrecisionWorld:
    if population not in GATE7_PRECISION_POPULATIONS:
        raise ValueError("precision population is outside the frozen ladder")
    if not 0 <= world_index < GATE7_PRECISION_WORLD_COUNT:
        raise ValueError("precision world index is outside the frozen range")
    frontier_depth = population.bit_length() - 1
    task_depth = frontier_depth + 1
    hidden_rng = random.Random(
        precision_seed_from_parts(
            GATE7_PRECISION_HIDDEN_NAMESPACE, population, world_index, task_depth
        )
    )
    hidden_path = tuple(hidden_rng.randrange(2) for _ in range(task_depth))
    hint_rng = random.Random(
        precision_seed_from_parts(
            GATE7_PRECISION_HINT_NAMESPACE, population, world_index, task_depth
        )
    )
    noisy_hints = tuple(
        bit if hint_rng.random() < GATE7_PRECISION_HINT_RELIABILITY else 1 - bit
        for bit in hidden_path
    )
    parent = 0
    terminal = 0
    for index, bit in enumerate(hidden_path):
        terminal = terminal * 2 + bit
        if index < frontier_depth:
            parent = parent * 2 + bit
    tie_multiplier, tie_offset = _public_affine_permutation(
        namespace=GATE7_PRECISION_TIE_NAMESPACE,
        population=population,
        world_index=world_index,
    )
    hash_multiplier, hash_offset = _public_affine_permutation(
        namespace=GATE7_PRECISION_HASH_NAMESPACE,
        population=population,
        world_index=world_index,
    )
    world = Gate7PrecisionWorld(
        population=population,
        world_index=world_index,
        frontier_depth=frontier_depth,
        task_depth=task_depth,
        runtime_seed=precision_seed_from_parts(
            GATE7_PRECISION_RUNTIME_NAMESPACE,
            population,
            world_index,
            frontier_depth,
        ),
        noisy_hints=noisy_hints,
        hidden_path=hidden_path,
        hidden_parent_index=parent,
        hidden_terminal_path_id=terminal,
        tie_multiplier=tie_multiplier,
        tie_offset=tie_offset,
        hash_multiplier=hash_multiplier,
        hash_offset=hash_offset,
    )
    world.validate()
    return world


def precision_world_batch(
    *, population: int, batch_start: int
) -> tuple[Gate7PrecisionWorld, ...]:
    if batch_start % GATE7_PRECISION_EVALUATION_BATCH_SIZE:
        raise ValueError("precision batch start must align to 64")
    stop = batch_start + GATE7_PRECISION_EVALUATION_BATCH_SIZE
    if not 0 <= batch_start < stop <= GATE7_PRECISION_WORLD_COUNT:
        raise ValueError("precision batch range is outside the frozen world range")
    worlds = tuple(
        generate_gate7_precision_world(population=population, world_index=index)
        for index in range(batch_start, stop)
    )
    validate_precision_world_batch(worlds)
    return worlds


def validate_precision_world_batch(
    worlds: tuple[Gate7PrecisionWorld, ...],
) -> int:
    if len(worlds) != GATE7_PRECISION_EVALUATION_BATCH_SIZE:
        raise ValueError("precision execution requires exactly 64 worlds")
    population = worlds[0].population
    indices = tuple(world.world_index for world in worlds)
    if indices != tuple(range(indices[0], indices[0] + len(worlds))):
        raise ValueError("precision world indices must be contiguous")
    if indices[0] % GATE7_PRECISION_EVALUATION_BATCH_SIZE:
        raise ValueError("precision world indices must align to 64")
    for world in worlds:
        world.validate()
        if world.population != population:
            raise ValueError("precision batch mixes populations")
    return population


def load_verified_gate7_precision_checkpoint(
    *, checkpoint_index: int, checkpoint_path: Path, device: torch.device | str
) -> tuple[Gate7ScaleNeutralScorer, Gate7TransitionCheckpointIdentity]:
    if checkpoint_index not in GATE7_PRECISION_CHECKPOINT_INDICES:
        raise ValueError("precision checkpoint must be T0, T1 or T2")
    model, identity = load_verified_gate7_high_scale_checkpoint(
        checkpoint_index=checkpoint_index,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    expected = GATE7_INFORMATION_CEILING_CHECKPOINTS[checkpoint_index]
    if identity.checkpoint_sha256 != expected["sha256"]:
        raise RuntimeError("precision checkpoint SHA changed")
    if identity.parameter_fingerprint != expected["fingerprint"]:
        raise RuntimeError("precision checkpoint fingerprint changed")
    if model.trainable_parameter_count() != GATE7_PRECISION_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("precision learned parameter count changed")
    return model, identity
