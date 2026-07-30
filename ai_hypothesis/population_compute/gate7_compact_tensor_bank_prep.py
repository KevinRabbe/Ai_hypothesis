"""Compact ordered Stage-B tensor bank for Gate-7 engineering preparation.

This is still preparation-only.  It preserves the already-qualified dynamic tensor transcript while
removing full-bank path sorting and top-k retention from every activation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gate3_v1_model import Gate3V1Scorer
from .gate3_v1_sparse_active_reserve import GATE3_V1_RECURRENT_UPDATES_PER_CHILD, GATE3_V1_SCORE_QUANTIZATION
from .gate6_fixed_k_population_scaling import Gate6EvaluationWorld
from .gate7_dynamic_tensor_bank_prep import (
    _INT64_MAX,
    Gate7DynamicTensorTranscript,
    Gate7ReferenceLookupTables,
    _build_selected_child_inputs,
    _gather_bank_rows,
    build_gate6_reference_lookup_tables,
    initialize_gate6_compatible_dynamic_bank,
)
from .gate7_tensor_engine_prep import GATE7_TENSOR_STATE_WIDTH, Gate7TensorFrontier

GATE7_COMPACT_TENSOR_BANK_PREPARATION_ONLY = True


@dataclass(frozen=True, slots=True)
class Gate7CompactTensorBank:
    states: torch.Tensor
    scores: torch.Tensor
    heap_ids: torch.Tensor
    live_counts: torch.Tensor
    population_capacity: int

    def validate(self) -> None:
        if self.states.ndim != 3 or self.states.shape[-1] != GATE7_TENSOR_STATE_WIDTH:
            raise ValueError("states must have shape [batch,capacity,64]")
        if self.scores.shape != self.states.shape[:2]:
            raise ValueError("scores must have shape [batch,capacity]")
        if self.heap_ids.shape != self.states.shape[:2] or self.heap_ids.dtype != torch.int64:
            raise ValueError("heap_ids must be int64 [batch,capacity]")
        if self.live_counts.shape != (self.states.shape[0],) or self.live_counts.dtype != torch.int64:
            raise ValueError("live_counts must be int64 [batch]")
        if self.states.shape[1] != self.population_capacity:
            raise ValueError("bank width must equal population_capacity")


def initialize_compact_bank(
    frontier: Gate7TensorFrontier,
    *,
    tables: Gate7ReferenceLookupTables,
    population_capacity: int,
) -> Gate7CompactTensorBank:
    dynamic = initialize_gate6_compatible_dynamic_bank(
        frontier,
        tables=tables,
        population_capacity=population_capacity,
    )
    bank = Gate7CompactTensorBank(
        states=dynamic.states,
        scores=dynamic.scores,
        heap_ids=dynamic.heap_ids,
        live_counts=dynamic.live_mask.sum(dim=1).to(torch.int64),
        population_capacity=population_capacity,
    )
    bank.validate()
    return bank


def _live_position_mask(bank: Gate7CompactTensorBank) -> torch.Tensor:
    positions = torch.arange(bank.population_capacity, device=bank.states.device)[None, :]
    return positions < bank.live_counts[:, None]


def _select_global_compact(
    bank: Gate7CompactTensorBank,
    *,
    tables: Gate7ReferenceLookupTables,
    local_slot: int,
) -> torch.Tensor:
    live = _live_position_mask(bank)
    quantized = torch.round(bank.scores / GATE3_V1_SCORE_QUANTIZATION).to(torch.int64)
    quantized = torch.where(live, quantized, torch.full_like(quantized, torch.iinfo(torch.int64).min))
    best = quantized.max(dim=1, keepdim=True).values
    tie = tables.score_tie_key[:, local_slot].gather(1, bank.heap_ids)
    tie = torch.where(live & (quantized == best), tie, torch.full_like(tie, _INT64_MAX))
    return tie.argmin(dim=1)


def _compact_sample_positions(
    bank: Gate7CompactTensorBank,
    *,
    tables: Gate7ReferenceLookupTables,
    local_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = bank.live_counts
    starts = tables.sampler_start_by_count[:, local_slot].gather(1, counts[:, None]).squeeze(1)
    strides = tables.sampler_stride_by_count[:, local_slot].gather(1, counts[:, None]).squeeze(1)
    offsets = torch.arange(tables.k, dtype=torch.int64, device=bank.states.device)[None, :]
    positions = (starts[:, None] + offsets * strides[:, None]) % counts[:, None]
    valid = offsets < counts[:, None]
    return positions, valid


def _select_bounded_score_compact(
    bank: Gate7CompactTensorBank,
    *,
    tables: Gate7ReferenceLookupTables,
    local_slot: int,
) -> torch.Tensor:
    positions, valid = _compact_sample_positions(bank, tables=tables, local_slot=local_slot)
    visible_scores = _gather_bank_rows(bank.scores, positions)
    quantized = torch.round(visible_scores / GATE3_V1_SCORE_QUANTIZATION).to(torch.int64)
    quantized = torch.where(valid, quantized, torch.full_like(quantized, torch.iinfo(torch.int64).min))
    best = quantized.max(dim=1, keepdim=True).values
    visible_heap = _gather_bank_rows(bank.heap_ids, positions)
    tie = tables.score_tie_key[:, local_slot].gather(1, visible_heap)
    tie = torch.where(valid & (quantized == best), tie, torch.full_like(tie, _INT64_MAX))
    selected_visible = tie.argmin(dim=1, keepdim=True)
    return positions.gather(1, selected_visible).squeeze(1)


def _select_bounded_hash_compact(
    bank: Gate7CompactTensorBank,
    *,
    tables: Gate7ReferenceLookupTables,
    local_slot: int,
) -> torch.Tensor:
    positions, valid = _compact_sample_positions(bank, tables=tables, local_slot=local_slot)
    visible_heap = _gather_bank_rows(bank.heap_ids, positions)
    rank = tables.hash_selection_rank[:, local_slot].gather(1, visible_heap)
    rank = torch.where(valid, rank, torch.full_like(rank, _INT64_MAX))
    selected_visible = rank.argmin(dim=1, keepdim=True)
    return positions.gather(1, selected_visible).squeeze(1)


def _ordered_expand(
    bank: Gate7CompactTensorBank,
    *,
    selected_pos: torch.Tensor,
    child_states: torch.Tensor,
    child_scores: torch.Tensor,
    child_heap_ids: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace a parent in an already path-ordered frontier without sorting.

    A live parent has no live descendants.  Therefore its two children occupy exactly the parent's
    lexicographic location, and removing a terminal parent only closes that one position.
    """

    batch, capacity, width = bank.states.shape
    out_width = capacity + 1
    target = torch.arange(out_width, dtype=torch.int64, device=bank.states.device)[None, :]
    parent_pos = selected_pos[:, None]
    old_count = bank.live_counts[:, None]

    nonterminal_src = torch.where(target < parent_pos, target, target - 1)
    terminal_src = torch.where(target < parent_pos, target, target + 1)
    source = torch.where(terminal[:, None], terminal_src, nonterminal_src).clamp(0, capacity - 1)

    gathered_states = _gather_bank_rows(bank.states, source)
    gathered_scores = _gather_bank_rows(bank.scores, source)
    gathered_heap = _gather_bank_rows(bank.heap_ids, source)

    child0 = (~terminal)[:, None] & (target == parent_pos)
    child1 = (~terminal)[:, None] & (target == parent_pos + 1)
    gathered_states = torch.where(child0[:, :, None], child_states[:, 0:1, :], gathered_states)
    gathered_states = torch.where(child1[:, :, None], child_states[:, 1:2, :], gathered_states)
    gathered_scores = torch.where(child0, child_scores[:, 0:1], gathered_scores)
    gathered_scores = torch.where(child1, child_scores[:, 1:2], gathered_scores)
    gathered_heap = torch.where(child0, child_heap_ids[:, 0:1], gathered_heap)
    gathered_heap = torch.where(child1, child_heap_ids[:, 1:2], gathered_heap)

    new_count = bank.live_counts + torch.where(
        terminal,
        torch.full_like(bank.live_counts, -1),
        torch.ones_like(bank.live_counts),
    )
    live = target < new_count[:, None]
    gathered_states = torch.where(live[:, :, None], gathered_states, torch.zeros_like(gathered_states))
    gathered_scores = torch.where(live, gathered_scores, torch.zeros_like(gathered_scores))
    gathered_heap = torch.where(live, gathered_heap, torch.zeros_like(gathered_heap))
    return gathered_states, gathered_scores, gathered_heap, new_count


def _drop_single_overflow(
    states: torch.Tensor,
    scores: torch.Tensor,
    heap_ids: torch.Tensor,
    counts: torch.Tensor,
    *,
    population_capacity: int,
    retention_rank_lookup: torch.Tensor,
) -> tuple[Gate7CompactTensorBank, torch.Tensor]:
    """Gate-6 can exceed capacity by at most one candidate per activation."""

    batch, pool_width, _ = states.shape
    positions = torch.arange(pool_width, dtype=torch.int64, device=states.device)[None, :]
    live = positions < counts[:, None]
    rank = retention_rank_lookup.gather(1, heap_ids)
    rank = torch.where(live, rank, torch.full_like(rank, -1))
    worst = rank.argmax(dim=1)
    overflow = counts > population_capacity

    target = torch.arange(population_capacity, dtype=torch.int64, device=states.device)[None, :]
    source_if_drop = torch.where(target < worst[:, None], target, target + 1)
    source = torch.where(overflow[:, None], source_if_drop, target)

    kept_states = _gather_bank_rows(states, source)
    kept_scores = _gather_bank_rows(scores, source)
    kept_heap = _gather_bank_rows(heap_ids, source)
    kept_counts = torch.minimum(counts, torch.full_like(counts, population_capacity))
    kept_live = target < kept_counts[:, None]
    kept_states = torch.where(kept_live[:, :, None], kept_states, torch.zeros_like(kept_states))
    kept_scores = torch.where(kept_live, kept_scores, torch.zeros_like(kept_scores))
    kept_heap = torch.where(kept_live, kept_heap, torch.zeros_like(kept_heap))

    bank = Gate7CompactTensorBank(
        states=kept_states,
        scores=kept_scores,
        heap_ids=kept_heap,
        live_counts=kept_counts,
        population_capacity=population_capacity,
    )
    bank.validate()
    return bank, overflow.to(torch.int64)


def run_gate6_compatible_compact_stage_b_tensor(
    model: Gate3V1Scorer,
    worlds: tuple[Gate6EvaluationWorld, ...],
    frontier: Gate7TensorFrontier,
    *,
    population_capacity: int,
    stage_b_slots: int,
    mode: str,
    k: int,
    sampling_group: str,
    device: torch.device | str,
) -> Gate7DynamicTensorTranscript:
    if mode not in {"global_score", "bounded_score", "bounded_hash"}:
        raise ValueError("unsupported preparation scheduler mode")

    target = torch.device(device)
    model = model.to(target)
    tables = build_gate6_reference_lookup_tables(
        worlds,
        population_capacity=population_capacity,
        stage_b_slots=stage_b_slots,
        k=k,
        sampling_group=sampling_group,
        device=target,
    )
    bank = initialize_compact_bank(frontier, tables=tables, population_capacity=population_capacity)

    selected_transcript: list[torch.Tensor] = []
    terminal_transcript: list[torch.Tensor] = []
    pruned_transcript: list[torch.Tensor] = []
    batch = len(worlds)
    rows = torch.arange(batch, device=target)
    actions = torch.tensor((0, 1), dtype=torch.int64, device=target)
    world_depths = torch.tensor([world.public.depth for world in worlds], dtype=torch.int64, device=target)

    with torch.inference_mode():
        for local_slot in range(stage_b_slots):
            if mode == "global_score":
                selected_pos = _select_global_compact(bank, tables=tables, local_slot=local_slot)
            elif mode == "bounded_score":
                selected_pos = _select_bounded_score_compact(bank, tables=tables, local_slot=local_slot)
            else:
                selected_pos = _select_bounded_hash_compact(bank, tables=tables, local_slot=local_slot)

            selected_heap = bank.heap_ids[rows, selected_pos]
            selected_state = bank.states[rows, selected_pos]
            selected_depth = tables.depth_by_heap_id[selected_heap]
            child_depth = selected_depth + 1
            selected_transcript.append(selected_heap)

            child_inputs = _build_selected_child_inputs(worlds, child_depths=child_depth, device=target)
            parent_states = selected_state[:, None, :].expand(batch, 2, GATE7_TENSOR_STATE_WIDTH)
            child_states = model.advance(
                parent_states.reshape(batch * 2, GATE7_TENSOR_STATE_WIDTH),
                child_inputs,
                repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            ).reshape(batch, 2, GATE7_TENSOR_STATE_WIDTH)
            child_scores = model.score(child_states.reshape(batch * 2, GATE7_TENSOR_STATE_WIDTH)).reshape(batch, 2)
            child_heap = selected_heap[:, None] * 2 + actions[None, :]
            terminal = child_depth == world_depths
            terminal_transcript.append(torch.where(terminal[:, None], child_heap, torch.zeros_like(child_heap)))

            expanded_states, expanded_scores, expanded_heap, expanded_counts = _ordered_expand(
                bank,
                selected_pos=selected_pos,
                child_states=child_states,
                child_scores=child_scores,
                child_heap_ids=child_heap,
                terminal=terminal,
            )
            bank, pruned = _drop_single_overflow(
                expanded_states,
                expanded_scores,
                expanded_heap,
                expanded_counts,
                population_capacity=population_capacity,
                retention_rank_lookup=tables.stage_b_retention_rank[:, local_slot],
            )
            pruned_transcript.append(pruned)

    live_mask = (
        torch.arange(population_capacity, device=target)[None, :] < bank.live_counts[:, None]
    )
    dynamic_bank = __import__(
        "ai_hypothesis.population_compute.gate7_dynamic_tensor_bank_prep",
        fromlist=["Gate7DynamicTensorBank"],
    ).Gate7DynamicTensorBank(
        states=bank.states,
        scores=bank.scores,
        heap_ids=bank.heap_ids,
        live_mask=live_mask,
        population_capacity=population_capacity,
    )
    return Gate7DynamicTensorTranscript(
        selected_heap_ids=torch.stack(selected_transcript, dim=1),
        terminal_child_heap_ids=torch.stack(terminal_transcript, dim=1),
        overflow_pruned_count=torch.stack(pruned_transcript, dim=1),
        final_bank=dynamic_bank,
    )
