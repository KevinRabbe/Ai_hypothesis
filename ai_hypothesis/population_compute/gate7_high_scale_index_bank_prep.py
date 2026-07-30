"""Data-blind immutable-frontier / live-index-bank mechanics for Gate-7 high-scale preparation.

The high-scale protocol terminates one decision beyond a complete Stage-A frontier.  Stage-B therefore
never retains child states: each activation selects one immutable frontier candidate, computes both
terminal children, and removes the selected frontier index.  Routing conditions can consequently clone
only an int64 index bank instead of cloning the multi-gigabyte recurrent-state frontier.

No scientific world generator, checkpoint loading, result classifier, or admitted runner lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gate3_v1_sparse_active_reserve import GATE3_V1_SCORE_QUANTIZATION
from .gate7_high_scale_routing_bandwidth_protocol import (
    GATE7_HIGH_SCALE_K_LADDER,
    GATE7_HIGH_SCALE_POPULATIONS,
)
from .gate7_native_tensor_bank_prep import (
    Gate7NativeSample,
    gate7_native_priority,
    sample_gate7_native_positions,
)

GATE7_HIGH_SCALE_INDEX_BANK_PREPARATION_ONLY = True
GATE7_HIGH_SCALE_FRONTIER_STATE_WIDTH = 64
_PRIORITY_NAMESPACE = 70_701


@dataclass(frozen=True, slots=True)
class Gate7HighScaleImmutableFrontier:
    states: torch.Tensor
    scores: torch.Tensor
    population: int

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    def validate(self) -> None:
        if self.population not in GATE7_HIGH_SCALE_POPULATIONS:
            raise ValueError("frontier population is outside the frozen high-scale ladder")
        if self.states.ndim != 3 or self.states.shape[1:] != (
            self.population,
            GATE7_HIGH_SCALE_FRONTIER_STATE_WIDTH,
        ):
            raise ValueError("frontier states must have shape [batch,population,64]")
        if self.scores.shape != self.states.shape[:2]:
            raise ValueError("frontier scores must have shape [batch,population]")
        if self.states.dtype != torch.float32 or self.scores.dtype != torch.float32:
            raise ValueError("frontier states/scores must remain FP32")
        if self.states.device != self.scores.device:
            raise ValueError("frontier states/scores must share one device")


@dataclass(frozen=True, slots=True)
class Gate7HighScaleLiveIndexBank:
    live_indices: torch.Tensor
    live_counts: torch.Tensor
    population: int

    @property
    def batch_size(self) -> int:
        return int(self.live_indices.shape[0])

    def validate(self) -> None:
        if self.population not in GATE7_HIGH_SCALE_POPULATIONS:
            raise ValueError("index-bank population is outside the frozen high-scale ladder")
        if self.live_indices.shape != (self.batch_size, self.population):
            raise ValueError("live_indices must have shape [batch,population]")
        if self.live_indices.dtype != torch.int64:
            raise ValueError("live_indices must use int64")
        if self.live_counts.shape != (self.batch_size,) or self.live_counts.dtype != torch.int64:
            raise ValueError("live_counts must use int64 [batch]")
        if self.live_indices.device != self.live_counts.device:
            raise ValueError("index-bank tensors must share one device")


@dataclass(frozen=True, slots=True)
class Gate7HighScaleIndexSelection:
    selected_live_positions: torch.Tensor
    selected_original_indices: torch.Tensor
    sampled_live_positions: torch.Tensor | None
    neural_scores_observed_per_world: torch.Tensor

    def validate(self, *, batch_size: int) -> None:
        expected = (batch_size,)
        if self.selected_live_positions.shape != expected:
            raise ValueError("selected live positions must have shape [batch]")
        if self.selected_original_indices.shape != expected:
            raise ValueError("selected original indices must have shape [batch]")
        if self.neural_scores_observed_per_world.shape != expected:
            raise ValueError("score-observation telemetry must have shape [batch]")
        if self.sampled_live_positions is not None and self.sampled_live_positions.shape[0] != batch_size:
            raise ValueError("sampled positions must preserve batch dimension")


def initialize_gate7_high_scale_live_index_bank(
    *, batch_size: int, population: int, device: torch.device | str
) -> Gate7HighScaleLiveIndexBank:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if population not in GATE7_HIGH_SCALE_POPULATIONS:
        raise ValueError("population is outside the frozen Gate-7 high-scale ladder")
    target = torch.device(device)
    indices = torch.arange(population, dtype=torch.int64, device=target)[None, :].expand(
        batch_size, population
    ).clone()
    bank = Gate7HighScaleLiveIndexBank(
        live_indices=indices,
        live_counts=torch.full((batch_size,), population, dtype=torch.int64, device=target),
        population=population,
    )
    bank.validate()
    return bank


def clone_gate7_high_scale_live_index_bank(
    bank: Gate7HighScaleLiveIndexBank,
) -> Gate7HighScaleLiveIndexBank:
    bank.validate()
    clone = Gate7HighScaleLiveIndexBank(
        live_indices=bank.live_indices.clone(),
        live_counts=bank.live_counts.clone(),
        population=bank.population,
    )
    clone.validate()
    return clone


def gate7_high_scale_index_bank_storage_bytes(bank: Gate7HighScaleLiveIndexBank) -> int:
    bank.validate()
    return (
        bank.live_indices.numel() * bank.live_indices.element_size()
        + bank.live_counts.numel() * bank.live_counts.element_size()
    )


def _validate_shared(
    frontier: Gate7HighScaleImmutableFrontier,
    bank: Gate7HighScaleLiveIndexBank,
    public_seeds: torch.Tensor,
) -> None:
    frontier.validate()
    bank.validate()
    if frontier.population != bank.population or frontier.batch_size != bank.batch_size:
        raise ValueError("frontier/index-bank geometry mismatch")
    if frontier.states.device != bank.live_indices.device:
        raise ValueError("frontier/index-bank device mismatch")
    if public_seeds.shape != (bank.batch_size,) or public_seeds.dtype != torch.int64:
        raise ValueError("public seeds must use int64 [batch]")
    if public_seeds.device != bank.live_indices.device:
        raise ValueError("public seeds must share the frontier device")


def _row_gather(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[rows, positions]


def _live_mask(bank: Gate7HighScaleLiveIndexBank) -> torch.Tensor:
    positions = torch.arange(bank.population, dtype=torch.int64, device=bank.live_indices.device)[None, :]
    return positions < bank.live_counts[:, None]


def _heap_ids(population: int, original_indices: torch.Tensor) -> torch.Tensor:
    return original_indices + population


def _selection(
    *,
    bank: Gate7HighScaleLiveIndexBank,
    selected_live_positions: torch.Tensor,
    sampled_live_positions: torch.Tensor | None,
    score_observations: torch.Tensor,
) -> Gate7HighScaleIndexSelection:
    rows = torch.arange(bank.batch_size, device=bank.live_indices.device)
    selected_original = bank.live_indices[rows, selected_live_positions]
    result = Gate7HighScaleIndexSelection(
        selected_live_positions=selected_live_positions,
        selected_original_indices=selected_original,
        sampled_live_positions=sampled_live_positions,
        neural_scores_observed_per_world=score_observations,
    )
    result.validate(batch_size=bank.batch_size)
    return result


def select_gate7_high_scale_global_score(
    frontier: Gate7HighScaleImmutableFrontier,
    bank: Gate7HighScaleLiveIndexBank,
    *,
    public_seeds: torch.Tensor,
    slot_index: int,
) -> Gate7HighScaleIndexSelection:
    _validate_shared(frontier, bank, public_seeds)
    live = _live_mask(bank)
    ordered_scores = frontier.scores.gather(1, bank.live_indices)
    quantized = torch.round(ordered_scores / GATE3_V1_SCORE_QUANTIZATION).to(torch.int64)
    quantized = torch.where(
        live,
        quantized,
        torch.full_like(quantized, torch.iinfo(torch.int64).min),
    )
    best = quantized.max(dim=1, keepdim=True).values
    heap_ids = _heap_ids(bank.population, bank.live_indices)
    priority = gate7_native_priority(
        heap_ids,
        public_seeds=public_seeds,
        slot_index=slot_index,
        namespace_code=_PRIORITY_NAMESPACE,
    )
    priority = torch.where(
        live & (quantized == best),
        priority,
        torch.full_like(priority, torch.iinfo(torch.int64).max),
    )
    selected_live = priority.argmin(dim=1)
    return _selection(
        bank=bank,
        selected_live_positions=selected_live,
        sampled_live_positions=None,
        score_observations=bank.live_counts.clone(),
    )


def select_gate7_high_scale_global_hash(
    frontier: Gate7HighScaleImmutableFrontier,
    bank: Gate7HighScaleLiveIndexBank,
    *,
    public_seeds: torch.Tensor,
    slot_index: int,
) -> Gate7HighScaleIndexSelection:
    _validate_shared(frontier, bank, public_seeds)
    live = _live_mask(bank)
    heap_ids = _heap_ids(bank.population, bank.live_indices)
    priority = gate7_native_priority(
        heap_ids,
        public_seeds=public_seeds,
        slot_index=slot_index,
        namespace_code=_PRIORITY_NAMESPACE,
    )
    priority = torch.where(live, priority, torch.full_like(priority, torch.iinfo(torch.int64).max))
    selected_live = priority.argmin(dim=1)
    return _selection(
        bank=bank,
        selected_live_positions=selected_live,
        sampled_live_positions=None,
        score_observations=torch.zeros_like(bank.live_counts),
    )


def _bounded_sample(
    bank: Gate7HighScaleLiveIndexBank,
    *,
    k: int,
    public_seeds: torch.Tensor,
    slot_index: int,
) -> Gate7NativeSample:
    if k not in GATE7_HIGH_SCALE_K_LADDER or k >= bank.population:
        raise ValueError("K is outside the frozen bounded ladder for this population")
    return sample_gate7_native_positions(
        population_capacity=bank.population,
        live_counts=bank.live_counts,
        k=k,
        public_seeds=public_seeds,
        slot_index=slot_index,
        sampling_group_code=k,
    )


def select_gate7_high_scale_bounded_score(
    frontier: Gate7HighScaleImmutableFrontier,
    bank: Gate7HighScaleLiveIndexBank,
    *,
    k: int,
    public_seeds: torch.Tensor,
    slot_index: int,
) -> Gate7HighScaleIndexSelection:
    _validate_shared(frontier, bank, public_seeds)
    sample = _bounded_sample(
        bank,
        k=k,
        public_seeds=public_seeds,
        slot_index=slot_index,
    )
    sampled_original = _row_gather(bank.live_indices, sample.positions)
    visible_scores = _row_gather(frontier.scores, sampled_original)
    quantized = torch.round(visible_scores / GATE3_V1_SCORE_QUANTIZATION).to(torch.int64)
    best = quantized.max(dim=1, keepdim=True).values
    visible_heap = _heap_ids(bank.population, sampled_original)
    priority = gate7_native_priority(
        visible_heap,
        public_seeds=public_seeds,
        slot_index=slot_index,
        namespace_code=_PRIORITY_NAMESPACE,
    )
    priority = torch.where(
        quantized == best,
        priority,
        torch.full_like(priority, torch.iinfo(torch.int64).max),
    )
    selected_visible = priority.argmin(dim=1, keepdim=True)
    selected_live = sample.positions.gather(1, selected_visible).squeeze(1)
    return _selection(
        bank=bank,
        selected_live_positions=selected_live,
        sampled_live_positions=sample.positions,
        score_observations=torch.full_like(bank.live_counts, k),
    )


def select_gate7_high_scale_bounded_hash(
    frontier: Gate7HighScaleImmutableFrontier,
    bank: Gate7HighScaleLiveIndexBank,
    *,
    k: int,
    public_seeds: torch.Tensor,
    slot_index: int,
) -> Gate7HighScaleIndexSelection:
    _validate_shared(frontier, bank, public_seeds)
    sample = _bounded_sample(
        bank,
        k=k,
        public_seeds=public_seeds,
        slot_index=slot_index,
    )
    sampled_original = _row_gather(bank.live_indices, sample.positions)
    visible_heap = _heap_ids(bank.population, sampled_original)
    priority = gate7_native_priority(
        visible_heap,
        public_seeds=public_seeds,
        slot_index=slot_index,
        namespace_code=_PRIORITY_NAMESPACE,
    )
    selected_visible = priority.argmin(dim=1, keepdim=True)
    selected_live = sample.positions.gather(1, selected_visible).squeeze(1)
    return _selection(
        bank=bank,
        selected_live_positions=selected_live,
        sampled_live_positions=sample.positions,
        score_observations=torch.zeros_like(bank.live_counts),
    )


def gather_gate7_high_scale_selected_states(
    frontier: Gate7HighScaleImmutableFrontier,
    selection: Gate7HighScaleIndexSelection,
) -> torch.Tensor:
    frontier.validate()
    selection.validate(batch_size=frontier.batch_size)
    rows = torch.arange(frontier.batch_size, device=frontier.states.device)
    return frontier.states[rows, selection.selected_original_indices]


def delete_gate7_high_scale_selected(
    bank: Gate7HighScaleLiveIndexBank,
    selection: Gate7HighScaleIndexSelection,
) -> Gate7HighScaleLiveIndexBank:
    bank.validate()
    selection.validate(batch_size=bank.batch_size)
    rows = torch.arange(bank.batch_size, device=bank.live_indices.device)
    last = bank.live_counts - 1
    last_values = bank.live_indices[rows, last]
    bank.live_indices[rows, selection.selected_live_positions] = last_values
    bank.live_indices[rows, last] = 0
    result = Gate7HighScaleLiveIndexBank(
        live_indices=bank.live_indices,
        live_counts=bank.live_counts - 1,
        population=bank.population,
    )
    result.validate()
    return result
