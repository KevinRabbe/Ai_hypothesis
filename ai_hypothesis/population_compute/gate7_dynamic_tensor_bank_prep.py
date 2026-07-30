"""Reference-compatible dynamic Stage-B tensor bank for Gate-7 engineering preparation.

No Gate-7 scientific worlds, training, result classification, or scientific namespaces live here.
This module exists only to prove that dynamic candidate-bank mechanics can reproduce the qualified
Gate-6 eager scheduler without candidate objects or CUDA-to-Python scalar extraction in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gate3_v1_model import Gate3V1Scorer
from .gate3_v1_sparse_active_reserve import (
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    GATE3_V1_SCORE_QUANTIZATION,
    deterministic_gate3_v1_tie_break,
)
from .gate6_fixed_k_population_scaling import (
    GATE6_DEPTH,
    GATE6_EXPERIMENT_VERSION,
    GATE6_STAGE_A_PARENT_SLOTS,
    Gate6EvaluationWorld,
    _answer_blind_order_key,
    _bounded_indices,
    _seed_from_parts,
)
from .gate7_tensor_engine_prep import (
    GATE7_TENSOR_INPUT_WIDTH,
    GATE7_TENSOR_STATE_WIDTH,
    Gate7TensorFrontier,
    path_bits_to_tuple,
)

GATE7_DYNAMIC_TENSOR_BANK_PREPARATION_ONLY = True
_REFERENCE_MAX_WORLDS = 8
_REFERENCE_MAX_STAGE_B_SLOTS = 64
_INT64_MAX = torch.iinfo(torch.int64).max


@dataclass(frozen=True, slots=True)
class Gate7DynamicTensorBank:
    states: torch.Tensor
    scores: torch.Tensor
    heap_ids: torch.Tensor
    live_mask: torch.Tensor
    population_capacity: int

    def validate(self) -> None:
        if self.states.ndim != 3 or self.states.shape[-1] != GATE7_TENSOR_STATE_WIDTH:
            raise ValueError("states must have shape [batch,capacity,64]")
        if self.scores.shape != self.states.shape[:2]:
            raise ValueError("scores must have shape [batch,capacity]")
        if self.heap_ids.shape != self.states.shape[:2] or self.heap_ids.dtype != torch.int64:
            raise ValueError("heap_ids must be int64 [batch,capacity]")
        if self.live_mask.shape != self.states.shape[:2] or self.live_mask.dtype != torch.bool:
            raise ValueError("live_mask must be bool [batch,capacity]")
        if self.states.shape[1] != self.population_capacity:
            raise ValueError("bank width must equal population_capacity")
        if self.population_capacity <= 0:
            raise ValueError("population_capacity must be positive")


@dataclass(frozen=True, slots=True)
class Gate7ReferenceLookupTables:
    """Small-scale exact-reference tables.

    The old Gate-6 SHA ordering is intentionally precomputed on the host and uploaded once.  This is
    an equivalence device, not the intended high-scale Gate-7 retention primitive.
    """

    lex_rank: torch.Tensor
    depth_by_heap_id: torch.Tensor
    initial_retention_rank: torch.Tensor
    stage_b_retention_rank: torch.Tensor
    score_tie_key: torch.Tensor
    hash_selection_rank: torch.Tensor
    sampler_start_by_count: torch.Tensor
    sampler_stride_by_count: torch.Tensor
    stage_b_slots: int
    population_capacity: int
    k: int
    sampling_group: str


@dataclass(frozen=True, slots=True)
class Gate7DynamicTensorTranscript:
    selected_heap_ids: torch.Tensor
    terminal_child_heap_ids: torch.Tensor
    overflow_pruned_count: torch.Tensor
    final_bank: Gate7DynamicTensorBank


def _heap_id_to_path(heap_id: int) -> tuple[int, ...]:
    if heap_id <= 0:
        raise ValueError("heap_id must be positive")
    depth = heap_id.bit_length() - 1
    path_bits = heap_id - (1 << depth)
    return path_bits_to_tuple(path_bits, depth)


def _signed_unsigned_order_key(value: int) -> int:
    if not 0 <= value < (1 << 64):
        raise ValueError("value must fit uint64")
    return value - (1 << 63)


def _rank_lookup_from_order(order: list[int], *, max_heap_id: int) -> list[int]:
    result = [_INT64_MAX] * (max_heap_id + 1)
    for rank, heap_id in enumerate(order):
        result[heap_id] = rank
    return result


def build_gate6_reference_lookup_tables(
    worlds: tuple[Gate6EvaluationWorld, ...],
    *,
    population_capacity: int,
    stage_b_slots: int,
    k: int,
    sampling_group: str,
    device: torch.device | str,
) -> Gate7ReferenceLookupTables:
    """Precompute exact Gate-6 ordering/sampler metadata for small-scale equivalence tests."""

    if not worlds or len(worlds) > _REFERENCE_MAX_WORLDS:
        raise ValueError(f"reference-table preparation supports 1..{_REFERENCE_MAX_WORLDS} worlds")
    if not 1 <= stage_b_slots <= _REFERENCE_MAX_STAGE_B_SLOTS:
        raise ValueError(f"reference-table preparation supports 1..{_REFERENCE_MAX_STAGE_B_SLOTS} Stage-B slots")
    if population_capacity <= 0 or k <= 0:
        raise ValueError("population_capacity and k must be positive")
    if not sampling_group:
        raise ValueError("sampling_group must be non-empty")
    for world in worlds:
        world.validate()
        if world.public.depth != GATE6_DEPTH:
            raise ValueError("reference tables currently protect Gate-6 depth-10 semantics")

    target = torch.device(device)
    max_heap_id = (1 << (GATE6_DEPTH + 1)) - 1
    all_heap_ids = list(range(1, max_heap_id + 1))
    path_by_id = {heap_id: _heap_id_to_path(heap_id) for heap_id in all_heap_ids}

    lex_order = sorted(all_heap_ids, key=path_by_id.__getitem__)
    lex_rank = torch.tensor(
        _rank_lookup_from_order(lex_order, max_heap_id=max_heap_id), dtype=torch.int64, device=target
    )
    depth_lookup = torch.zeros(max_heap_id + 1, dtype=torch.int64, device=target)
    for heap_id in all_heap_ids:
        depth_lookup[heap_id] = heap_id.bit_length() - 1

    initial_rows: list[list[int]] = []
    retention_rows: list[list[list[int]]] = []
    tie_rows: list[list[list[int]]] = []
    hash_rows: list[list[list[int]]] = []
    sampler_start_rows: list[list[list[int]]] = []
    sampler_stride_rows: list[list[list[int]]] = []

    for world in worlds:
        world_seed = world.public.seed
        initial_order = sorted(
            all_heap_ids,
            key=lambda heap_id: _answer_blind_order_key(
                "gate6-fixed-k-population-scaling-initial-thinning",
                world_seed=world_seed,
                slot_index=-1,
                path=path_by_id[heap_id],
            ),
        )
        initial_rows.append(_rank_lookup_from_order(initial_order, max_heap_id=max_heap_id))

        world_retention: list[list[int]] = []
        world_ties: list[list[int]] = []
        world_hash: list[list[int]] = []
        world_starts: list[list[int]] = []
        world_strides: list[list[int]] = []

        for local_slot in range(stage_b_slots):
            absolute_slot = GATE6_STAGE_A_PARENT_SLOTS + local_slot
            retention_order = sorted(
                all_heap_ids,
                key=lambda heap_id: _answer_blind_order_key(
                    "gate6-fixed-k-population-scaling-stage-b-retention",
                    world_seed=world_seed,
                    slot_index=local_slot,
                    path=path_by_id[heap_id],
                ),
            )
            world_retention.append(_rank_lookup_from_order(retention_order, max_heap_id=max_heap_id))

            tie_values = [_INT64_MAX] * (max_heap_id + 1)
            for heap_id in all_heap_ids:
                tie_values[heap_id] = _signed_unsigned_order_key(
                    deterministic_gate3_v1_tie_break(
                        world_seed=world_seed,
                        expansion_index=absolute_slot,
                        candidate_path=path_by_id[heap_id],
                    )
                )
            world_ties.append(tie_values)

            hash_order = sorted(
                all_heap_ids,
                key=lambda heap_id: (
                    _seed_from_parts(
                        "gate6-fixed-k-population-scaling-bounded-hash-selection",
                        world_seed,
                        absolute_slot,
                        "".join(str(bit) for bit in path_by_id[heap_id]),
                    ),
                    path_by_id[heap_id],
                ),
            )
            world_hash.append(_rank_lookup_from_order(hash_order, max_heap_id=max_heap_id))

            starts = [0] * (population_capacity + 1)
            strides = [1] * (population_capacity + 1)
            for count in range(1, population_capacity + 1):
                indices = _bounded_indices(
                    count=count,
                    k=max(2, min(k, count)),
                    world_seed=world_seed,
                    slot_index=absolute_slot,
                    group=sampling_group,
                )
                starts[count] = indices[0]
                if len(indices) >= 2:
                    strides[count] = (indices[1] - indices[0]) % count
                else:
                    strides[count] = 1
            world_starts.append(starts)
            world_strides.append(strides)

        retention_rows.append(world_retention)
        tie_rows.append(world_ties)
        hash_rows.append(world_hash)
        sampler_start_rows.append(world_starts)
        sampler_stride_rows.append(world_strides)

    return Gate7ReferenceLookupTables(
        lex_rank=lex_rank,
        depth_by_heap_id=depth_lookup,
        initial_retention_rank=torch.tensor(initial_rows, dtype=torch.int64, device=target),
        stage_b_retention_rank=torch.tensor(retention_rows, dtype=torch.int64, device=target),
        score_tie_key=torch.tensor(tie_rows, dtype=torch.int64, device=target),
        hash_selection_rank=torch.tensor(hash_rows, dtype=torch.int64, device=target),
        sampler_start_by_count=torch.tensor(sampler_start_rows, dtype=torch.int64, device=target),
        sampler_stride_by_count=torch.tensor(sampler_stride_rows, dtype=torch.int64, device=target),
        stage_b_slots=stage_b_slots,
        population_capacity=population_capacity,
        k=k,
        sampling_group=sampling_group,
    )


def _gather_bank_rows(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch = values.shape[0]
    rows = torch.arange(batch, device=values.device)[:, None]
    return values[rows, positions]


def _canonicalize_bank(
    states: torch.Tensor,
    scores: torch.Tensor,
    heap_ids: torch.Tensor,
    live_mask: torch.Tensor,
    *,
    lex_rank: torch.Tensor,
) -> Gate7DynamicTensorBank:
    rank = lex_rank[heap_ids]
    rank = torch.where(live_mask, rank, torch.full_like(rank, _INT64_MAX))
    order = rank.argsort(dim=1)
    states = _gather_bank_rows(states, order)
    scores = _gather_bank_rows(scores, order)
    heap_ids = _gather_bank_rows(heap_ids, order)
    live_mask = _gather_bank_rows(live_mask, order)
    return Gate7DynamicTensorBank(
        states=states,
        scores=scores,
        heap_ids=heap_ids,
        live_mask=live_mask,
        population_capacity=states.shape[1],
    )


def initialize_gate6_compatible_dynamic_bank(
    frontier: Gate7TensorFrontier,
    *,
    tables: Gate7ReferenceLookupTables,
    population_capacity: int,
) -> Gate7DynamicTensorBank:
    frontier.validate()
    if frontier.depth != 8 or frontier.population != 256:
        raise ValueError("Gate-6 compatibility requires the complete depth-8 / 256 frontier")
    if population_capacity != tables.population_capacity:
        raise ValueError("population_capacity differs from reference tables")
    if frontier.batch_size != tables.initial_retention_rank.shape[0]:
        raise ValueError("frontier/reference-table batch mismatch")

    heap_ids = frontier.path_bits + (1 << frontier.depth)
    priority = tables.initial_retention_rank.gather(1, heap_ids)
    retained = priority.topk(k=population_capacity, dim=1, largest=False, sorted=False).indices
    states = _gather_bank_rows(frontier.states, retained)
    scores = _gather_bank_rows(frontier.scores, retained)
    heap_ids = _gather_bank_rows(heap_ids, retained)
    live_mask = torch.ones_like(heap_ids, dtype=torch.bool)
    bank = _canonicalize_bank(states, scores, heap_ids, live_mask, lex_rank=tables.lex_rank)
    bank.validate()
    return bank


def _select_global(
    bank: Gate7DynamicTensorBank,
    *,
    tables: Gate7ReferenceLookupTables,
    local_slot: int,
) -> torch.Tensor:
    quantized = torch.round(bank.scores / GATE3_V1_SCORE_QUANTIZATION).to(torch.int64)
    minimum = torch.full_like(quantized, torch.iinfo(torch.int64).min)
    quantized = torch.where(bank.live_mask, quantized, minimum)
    best_score = quantized.max(dim=1, keepdim=True).values
    tie = tables.score_tie_key[:, local_slot].gather(1, bank.heap_ids)
    tie = torch.where(
        bank.live_mask & (quantized == best_score),
        tie,
        torch.full_like(tie, _INT64_MAX),
    )
    return tie.argmin(dim=1)


def _bounded_sample_positions(
    bank: Gate7DynamicTensorBank,
    *,
    tables: Gate7ReferenceLookupTables,
    local_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = bank.live_mask.sum(dim=1).to(torch.int64)
    starts = tables.sampler_start_by_count[:, local_slot].gather(1, counts[:, None]).squeeze(1)
    strides = tables.sampler_stride_by_count[:, local_slot].gather(1, counts[:, None]).squeeze(1)
    offsets = torch.arange(tables.k, dtype=torch.int64, device=bank.states.device)[None, :]
    positions = (starts[:, None] + offsets * strides[:, None]) % counts[:, None]
    valid = offsets < counts[:, None]
    return positions, valid


def _select_bounded_score(
    bank: Gate7DynamicTensorBank,
    *,
    tables: Gate7ReferenceLookupTables,
    local_slot: int,
) -> torch.Tensor:
    positions, valid = _bounded_sample_positions(bank, tables=tables, local_slot=local_slot)
    visible_scores = _gather_bank_rows(bank.scores, positions)
    quantized = torch.round(visible_scores / GATE3_V1_SCORE_QUANTIZATION).to(torch.int64)
    minimum = torch.full_like(quantized, torch.iinfo(torch.int64).min)
    quantized = torch.where(valid, quantized, minimum)
    best_score = quantized.max(dim=1, keepdim=True).values
    visible_heap_ids = _gather_bank_rows(bank.heap_ids, positions)
    tie = tables.score_tie_key[:, local_slot].gather(1, visible_heap_ids)
    tie = torch.where(valid & (quantized == best_score), tie, torch.full_like(tie, _INT64_MAX))
    selected_visible = tie.argmin(dim=1, keepdim=True)
    return positions.gather(1, selected_visible).squeeze(1)


def _select_bounded_hash(
    bank: Gate7DynamicTensorBank,
    *,
    tables: Gate7ReferenceLookupTables,
    local_slot: int,
) -> torch.Tensor:
    positions, valid = _bounded_sample_positions(bank, tables=tables, local_slot=local_slot)
    visible_heap_ids = _gather_bank_rows(bank.heap_ids, positions)
    rank = tables.hash_selection_rank[:, local_slot].gather(1, visible_heap_ids)
    rank = torch.where(valid, rank, torch.full_like(rank, _INT64_MAX))
    selected_visible = rank.argmin(dim=1, keepdim=True)
    return positions.gather(1, selected_visible).squeeze(1)


def _build_selected_child_inputs(
    worlds: tuple[Gate6EvaluationWorld, ...],
    *,
    child_depths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch = len(worlds)
    if child_depths.shape != (batch,):
        raise ValueError("child_depths must have shape [batch]")

    inputs = torch.zeros((batch, 2, GATE7_TENSOR_INPUT_WIDTH), dtype=torch.float32, device=device)
    rows = torch.arange(batch, device=device)
    action_rows = rows[:, None].expand(batch, 2)
    actions = torch.tensor((0, 1), dtype=torch.int64, device=device)[None, :].expand(batch, 2)

    depth_positions = (child_depths - 1)[:, None].expand(batch, 2)
    inputs[action_rows, actions, depth_positions] = 1.0

    world_depth_offset = 10
    # Gate-6 uses depth 10, whose frozen Gate-3 v1 world-depth feature index is 2.
    inputs[:, :, world_depth_offset + 2] = 1.0

    hint_offset = world_depth_offset + 3
    hints = torch.tensor([world.public.noisy_hints for world in worlds], dtype=torch.int64, device=device)
    observed = hints.gather(1, (child_depths - 1)[:, None]).squeeze(1)
    inputs[rows, 0, hint_offset + observed] = 1.0
    inputs[rows, 1, hint_offset + observed] = 1.0

    action_offset = hint_offset + 3
    inputs[:, 0, action_offset] = 1.0
    inputs[:, 1, action_offset + 1] = 1.0
    return inputs.reshape(batch * 2, GATE7_TENSOR_INPUT_WIDTH)


def run_gate6_compatible_dynamic_stage_b_tensor(
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
    """Execute a small-scale Gate-6-compatible Stage B entirely through tensor bank mechanics."""

    if mode not in {"global_score", "bounded_score", "bounded_hash"}:
        raise ValueError("unsupported preparation scheduler mode")
    if stage_b_slots <= 0:
        raise ValueError("stage_b_slots must be positive")

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
    bank = initialize_gate6_compatible_dynamic_bank(
        frontier,
        tables=tables,
        population_capacity=population_capacity,
    )

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
                selected_pos = _select_global(bank, tables=tables, local_slot=local_slot)
            elif mode == "bounded_score":
                selected_pos = _select_bounded_score(bank, tables=tables, local_slot=local_slot)
            else:
                selected_pos = _select_bounded_hash(bank, tables=tables, local_slot=local_slot)

            selected_heap = bank.heap_ids[rows, selected_pos]
            selected_state = bank.states[rows, selected_pos]
            selected_depth = tables.depth_by_heap_id[selected_heap]
            child_depth = selected_depth + 1
            selected_transcript.append(selected_heap)

            child_inputs = _build_selected_child_inputs(worlds, child_depths=child_depth, device=target)
            parent_states = selected_state[:, None, :].expand(batch, 2, GATE7_TENSOR_STATE_WIDTH)
            advanced = model.advance(
                parent_states.reshape(batch * 2, GATE7_TENSOR_STATE_WIDTH),
                child_inputs,
                repeats=GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
            ).reshape(batch, 2, GATE7_TENSOR_STATE_WIDTH)
            child_scores = model.score(advanced.reshape(batch * 2, GATE7_TENSOR_STATE_WIDTH)).reshape(batch, 2)
            child_heap = selected_heap[:, None] * 2 + actions[None, :]
            terminal = child_depth == world_depths
            terminal_transcript.append(torch.where(terminal[:, None], child_heap, torch.zeros_like(child_heap)))

            remaining_mask = bank.live_mask.clone()
            remaining_mask[rows, selected_pos] = False
            child_live = (~terminal)[:, None].expand(batch, 2)

            pool_states = torch.cat((bank.states, advanced), dim=1)
            pool_scores = torch.cat((bank.scores, child_scores), dim=1)
            pool_heap = torch.cat((bank.heap_ids, child_heap), dim=1)
            pool_live = torch.cat((remaining_mask, child_live), dim=1)

            priority = tables.stage_b_retention_rank[:, local_slot].gather(1, pool_heap)
            priority = torch.where(pool_live, priority, torch.full_like(priority, _INT64_MAX))
            retained_pos = priority.topk(k=population_capacity, dim=1, largest=False, sorted=False).indices
            states = _gather_bank_rows(pool_states, retained_pos)
            scores = _gather_bank_rows(pool_scores, retained_pos)
            heap_ids = _gather_bank_rows(pool_heap, retained_pos)
            live_mask = _gather_bank_rows(pool_live, retained_pos)

            pool_count = pool_live.sum(dim=1)
            retained_count = live_mask.sum(dim=1)
            pruned_transcript.append(pool_count - retained_count)

            bank = _canonicalize_bank(states, scores, heap_ids, live_mask, lex_rank=tables.lex_rank)
            bank.validate()

    return Gate7DynamicTensorTranscript(
        selected_heap_ids=torch.stack(selected_transcript, dim=1),
        terminal_child_heap_ids=torch.stack(terminal_transcript, dim=1),
        overflow_pruned_count=torch.stack(pruned_transcript, dim=1),
        final_bank=bank,
    )
