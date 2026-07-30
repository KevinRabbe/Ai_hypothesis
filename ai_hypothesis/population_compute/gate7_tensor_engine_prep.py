"""Data-blind tensor execution substrate for Gate-7 preparation.

This module contains no Gate-7 scientific world generator, checkpoint training, result classifier,
or scientific namespace.  It exists to remove object-per-candidate and per-child CUDA-synchronization
costs while preserving the recurrent scorer semantics used by the qualified eager reference.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .gate3_v1_model import GATE3_V1_INPUT_WIDTH, Gate3V1Scorer
from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_DEPTHS,
    GATE3_V1_MAX_DEPTH,
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    GATE3_V1_SCORE_QUANTIZATION,
    Gate3V1PublicWorld,
    deterministic_gate3_v1_tie_break,
)

GATE7_TENSOR_ENGINE_PREPARATION_ONLY = True
GATE7_TENSOR_STATE_WIDTH = 64
GATE7_TENSOR_INPUT_WIDTH = GATE3_V1_INPUT_WIDTH


@dataclass(frozen=True, slots=True)
class Gate7TensorFrontier:
    """Generation-synchronous candidate bank with no Python object per candidate."""

    states: torch.Tensor
    scores: torch.Tensor
    path_bits: torch.Tensor
    depth: int

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    @property
    def population(self) -> int:
        return int(self.states.shape[1])

    def validate(self) -> None:
        if self.states.ndim != 3 or self.states.shape[-1] != GATE7_TENSOR_STATE_WIDTH:
            raise ValueError("states must have shape [batch,population,64]")
        if self.scores.shape != self.states.shape[:2]:
            raise ValueError("scores must have shape [batch,population]")
        if self.path_bits.shape != self.states.shape[:2]:
            raise ValueError("path_bits must have shape [batch,population]")
        if self.path_bits.dtype != torch.int64:
            raise ValueError("path_bits must use int64")
        if not 0 <= self.depth <= GATE3_V1_MAX_DEPTH:
            raise ValueError("frontier depth is outside the frozen scorer representation")
        if self.population != (1 << self.depth):
            raise ValueError("complete frontier width must equal 2**depth")


def path_bits_to_tuple(path_bits: int, depth: int) -> tuple[int, ...]:
    if depth < 0 or depth > GATE3_V1_MAX_DEPTH:
        raise ValueError("path depth is outside the frozen scorer representation")
    if path_bits < 0 or path_bits >= (1 << depth):
        raise ValueError("path_bits is outside the requested depth")
    return tuple((path_bits >> shift) & 1 for shift in reversed(range(depth)))


def path_tuple_to_bits(path: Sequence[int]) -> int:
    if len(path) > GATE3_V1_MAX_DEPTH or any(bit not in (0, 1) for bit in path):
        raise ValueError("path must be a binary prefix within the frozen scorer depth")
    value = 0
    for bit in path:
        value = (value << 1) | int(bit)
    return value


def _validate_public_worlds(worlds: tuple[Gate3V1PublicWorld, ...], *, child_depth: int) -> None:
    if not worlds:
        raise ValueError("at least one public world is required")
    for world in worlds:
        world.validate()
        if not 1 <= child_depth <= world.depth:
            raise ValueError("child depth is outside a public world")


def build_productive_child_inputs(
    worlds: tuple[Gate3V1PublicWorld, ...],
    *,
    child_depth: int,
    parent_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Vectorize the exact productive child-input encoding.

    Output order is world-major, then parent-major, then action 0/1, matching the eager reference.
    No candidate-dependent value crosses from CUDA to Python.
    """

    _validate_public_worlds(worlds, child_depth=child_depth)
    if parent_count <= 0:
        raise ValueError("parent_count must be positive")

    target = torch.device(device)
    batch = len(worlds)
    per_world = torch.zeros((batch, 2, GATE7_TENSOR_INPUT_WIDTH), dtype=torch.float32, device=target)

    per_world[:, :, child_depth - 1] = 1.0

    world_depth_feature_offset = GATE3_V1_MAX_DEPTH
    for world_offset, world in enumerate(worlds):
        per_world[world_offset, :, world_depth_feature_offset + GATE3_V1_DEPTHS.index(world.depth)] = 1.0

    hint_feature_offset = GATE3_V1_MAX_DEPTH + len(GATE3_V1_DEPTHS)
    hint_indices = torch.tensor(
        [world.noisy_hints[child_depth - 1] for world in worlds],
        dtype=torch.int64,
        device=target,
    )
    per_world[
        torch.arange(batch, device=target),
        torch.zeros(batch, dtype=torch.int64, device=target),
        hint_feature_offset + hint_indices,
    ] = 1.0
    per_world[
        torch.arange(batch, device=target),
        torch.ones(batch, dtype=torch.int64, device=target),
        hint_feature_offset + hint_indices,
    ] = 1.0

    action_feature_offset = hint_feature_offset + 3
    per_world[:, 0, action_feature_offset] = 1.0
    per_world[:, 1, action_feature_offset + 1] = 1.0

    return (
        per_world[:, None, :, :]
        .expand(batch, parent_count, 2, GATE7_TENSOR_INPUT_WIDTH)
        .reshape(batch * parent_count * 2, GATE7_TENSOR_INPUT_WIDTH)
    )


def build_complete_tensor_frontier(
    model: Gate3V1Scorer,
    worlds: tuple[Gate3V1PublicWorld, ...],
    *,
    frontier_depth: int,
    device: torch.device | str,
) -> Gate7TensorFrontier:
    """Build a complete binary frontier with dense batched recurrent execution."""

    if not worlds:
        raise ValueError("at least one public world is required")
    if not 0 <= frontier_depth <= GATE3_V1_MAX_DEPTH:
        raise ValueError("frontier depth is outside the frozen scorer representation")
    if any(frontier_depth > world.depth for world in worlds):
        raise ValueError("frontier depth exceeds a public world")

    target = torch.device(device)
    model = model.to(target)
    batch = len(worlds)
    states = model.initial_state(batch, device=target).reshape(batch, 1, GATE7_TENSOR_STATE_WIDTH)
    scores = torch.zeros((batch, 1), dtype=torch.float32, device=target)
    path_bits = torch.zeros((batch, 1), dtype=torch.int64, device=target)

    with torch.inference_mode():
        for child_depth in range(1, frontier_depth + 1):
            parent_count = states.shape[1]
            parent_states = (
                states[:, :, None, :]
                .expand(batch, parent_count, 2, GATE7_TENSOR_STATE_WIDTH)
                .reshape(batch * parent_count * 2, GATE7_TENSOR_STATE_WIDTH)
            )
            child_inputs = build_productive_child_inputs(
                worlds,
                child_depth=child_depth,
                parent_count=parent_count,
                device=target,
            )
            advanced = model.advance(
                parent_states,
                child_inputs,
                repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            )
            child_scores = model.score(advanced)
            states = advanced.reshape(batch, parent_count * 2, GATE7_TENSOR_STATE_WIDTH)
            scores = child_scores.reshape(batch, parent_count * 2)

            actions = torch.tensor((0, 1), dtype=torch.int64, device=target)
            path_bits = (
                path_bits[:, :, None] * 2 + actions[None, None, :]
            ).reshape(batch, parent_count * 2)

    frontier = Gate7TensorFrontier(states=states, scores=scores, path_bits=path_bits, depth=frontier_depth)
    frontier.validate()
    return frontier


def _bounded_indices(*, count: int, k: int, world_seed: int, slot_index: int, group: str) -> tuple[int, ...]:
    """Reference-compatible bounded rank sampler; O(K) once candidates are path ordered."""

    if count <= 0 or k <= 0:
        return ()
    visible = min(k, count)
    digest = hashlib.sha256(
        f"gate6-fixed-k-population-scaling-v0:{world_seed}:{slot_index}:{group}".encode("ascii")
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


def _signed_unsigned_order_key(value: int) -> int:
    if not 0 <= value < (1 << 64):
        raise ValueError("tie key must be uint64")
    return value - (1 << 63)


def select_bounded_score_indices(
    scores: torch.Tensor,
    *,
    path_bits_by_world: tuple[tuple[int, ...], ...],
    path_depth: int,
    world_seeds: tuple[int, ...],
    slot_index: int,
    k: int,
    sampling_group: str,
) -> torch.Tensor:
    """Select one parent/world using only K gathered neural scores.

    `path_bits_by_world` must already be in lexicographic path order.  Candidate metadata remains on
    the host, while the only causal neural values read are the K gathered scores.  The return value
    stays on the same device as `scores` so downstream gather can remain device-side.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch,population]")
    batch, population = scores.shape
    if len(path_bits_by_world) != batch or len(world_seeds) != batch:
        raise ValueError("metadata batch does not match score batch")
    if k <= 0:
        raise ValueError("k must be positive")
    if any(len(row) != population for row in path_bits_by_world):
        raise ValueError("path metadata does not match population width")

    sampled_rows: list[tuple[int, ...]] = []
    tie_rows: list[tuple[int, ...]] = []
    for row_paths, world_seed in zip(path_bits_by_world, world_seeds, strict=True):
        sampled = _bounded_indices(
            count=population,
            k=k,
            world_seed=world_seed,
            slot_index=slot_index,
            group=sampling_group,
        )
        sampled_rows.append(sampled)
        tie_rows.append(
            tuple(
                _signed_unsigned_order_key(
                    deterministic_gate3_v1_tie_break(
                        world_seed=world_seed,
                        expansion_index=slot_index,
                        candidate_path=path_bits_to_tuple(row_paths[position], path_depth),
                    )
                )
                for position in sampled
            )
        )

    sample_index = torch.tensor(sampled_rows, dtype=torch.int64, device=scores.device)
    visible_scores = scores.gather(1, sample_index)
    quantized = torch.round(visible_scores / GATE3_V1_SCORE_QUANTIZATION).to(torch.int64)
    best_score = quantized.max(dim=1, keepdim=True).values

    tie_keys = torch.tensor(tie_rows, dtype=torch.int64, device=scores.device)
    sentinel = torch.full_like(tie_keys, torch.iinfo(torch.int64).max)
    eligible_ties = torch.where(quantized == best_score, tie_keys, sentinel)
    selected_visible = eligible_ties.argmin(dim=1, keepdim=True)
    return sample_index.gather(1, selected_visible).squeeze(1)


def gather_selected_states(states: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
    if states.ndim != 3 or selected_indices.ndim != 1 or states.shape[0] != selected_indices.shape[0]:
        raise ValueError("states/indices have incompatible batch shapes")
    batch = states.shape[0]
    return states[torch.arange(batch, device=states.device), selected_indices]
