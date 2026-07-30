"""Scale-oriented native live-bank mechanics for Gate-7 preparation.

This module is deliberately data-blind and NOT scientifically admitted. Unlike the exact Gate-6
compatibility engine, it prepares new Gate-7-native mechanics whose bounded routing cost is a function
of K rather than population N: dense live positions, swap-delete, append, deterministic answer-blind
victim selection, and an affine metadata sampler over power-of-two capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gate3_v1_sparse_active_reserve import GATE3_V1_SCORE_QUANTIZATION

GATE7_NATIVE_TENSOR_BANK_PREPARATION_ONLY = True
GATE7_NATIVE_MAX_STAGE_B_SLOTS = 128
GATE7_NATIVE_K_LADDER = (16, 32, 64, 128, 256, 512)
GATE7_NATIVE_POPULATION_LADDER = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
GATE7_NATIVE_STATE_WIDTH = 64

# Small arithmetic constants chosen so all prepared-domain products stay within signed int64.
_START_SLOT_MIX = 1_103_515_245
_START_GROUP_MIX = 12_345
_STRIDE_SEED_MIX = 1_664_525
_STRIDE_SLOT_MIX = 1_013_904_223
_STRIDE_GROUP_MIX = 69_069
_PRIORITY_PATH_MIX = 1_000_003
_PRIORITY_SEED_MIX = 97_003
_PRIORITY_SLOT_MIX = 9_973
_PRIORITY_GROUP_MIX = 389
_PRIORITY_MODULUS = 2_305_843_009_213_693_951  # 2**61 - 1
_VICTIM_SEED_MIX = 2_654_435_761
_VICTIM_SLOT_MIX = 40_503


@dataclass(frozen=True, slots=True)
class Gate7NativeTensorBank:
    """Dense live prefix plus one overflow scratch position."""

    states: torch.Tensor
    scores: torch.Tensor
    heap_ids: torch.Tensor
    live_counts: torch.Tensor
    population_capacity: int

    def validate(self) -> None:
        expected_width = self.population_capacity + 1
        if self.states.ndim != 3 or self.states.shape[1:] != (expected_width, GATE7_NATIVE_STATE_WIDTH):
            raise ValueError("states must have shape [batch,capacity+1,64]")
        if self.scores.shape != self.states.shape[:2]:
            raise ValueError("scores must have shape [batch,capacity+1]")
        if self.heap_ids.shape != self.states.shape[:2] or self.heap_ids.dtype != torch.int64:
            raise ValueError("heap_ids must be int64 [batch,capacity+1]")
        if self.live_counts.shape != (self.states.shape[0],) or self.live_counts.dtype != torch.int64:
            raise ValueError("live_counts must be int64 [batch]")
        if self.population_capacity <= 0 or self.population_capacity & (self.population_capacity - 1):
            raise ValueError("population_capacity must remain a positive power of two")


@dataclass(frozen=True, slots=True)
class Gate7NativeSample:
    positions: torch.Tensor
    metadata_candidates_examined: int


def gate7_native_public_seed_tensor(public_runtime_seeds: tuple[int, ...], *, device: torch.device | str) -> torch.Tensor:
    """Reduce public runtime seeds to a safe deterministic 31-bit arithmetic domain."""

    if not public_runtime_seeds:
        raise ValueError("at least one public runtime seed is required")
    reduced = [int(seed) % 2_147_483_647 for seed in public_runtime_seeds]
    return torch.tensor(reduced, dtype=torch.int64, device=device)


def _validate_native_scale(*, population_capacity: int, k: int) -> None:
    if population_capacity not in GATE7_NATIVE_POPULATION_LADDER:
        raise ValueError("population_capacity is outside the prepared Gate-7 ladder")
    if k not in GATE7_NATIVE_K_LADDER or k >= population_capacity:
        raise ValueError("K must be a prepared bounded value strictly below N")


def sample_gate7_native_positions(
    *,
    population_capacity: int,
    live_counts: torch.Tensor,
    k: int,
    public_seeds: torch.Tensor,
    slot_index: int,
    sampling_group_code: int,
) -> Gate7NativeSample:
    """Return exactly K unique live positions with work bounded by K+128 metadata candidates.

    Capacity N is a power of two. An odd stride therefore permutes all N positions. The admitted
    caller will maintain the frozen N-128..N live-count envelope; examining K+128 positions from the
    permutation then guarantees at least K live-prefix positions without scanning N.
    """

    _validate_native_scale(population_capacity=population_capacity, k=k)
    if live_counts.ndim != 1 or public_seeds.shape != live_counts.shape:
        raise ValueError("live_counts/public_seeds must be matching [batch] tensors")
    if not 0 <= slot_index < GATE7_NATIVE_MAX_STAGE_B_SLOTS:
        raise ValueError("slot_index is outside the prepared Stage-B horizon")
    if sampling_group_code < 0:
        raise ValueError("sampling_group_code must be non-negative")

    capacity_mask = population_capacity - 1
    start = (
        public_seeds
        + slot_index * _START_SLOT_MIX
        + sampling_group_code * _START_GROUP_MIX
    ) & capacity_mask
    stride_seed = (
        public_seeds * _STRIDE_SEED_MIX
        + slot_index * _STRIDE_SLOT_MIX
        + sampling_group_code * _STRIDE_GROUP_MIX
    )
    stride = ((stride_seed & (population_capacity // 2 - 1)) * 2 + 1) & capacity_mask

    examined = min(population_capacity, k + GATE7_NATIVE_MAX_STAGE_B_SLOTS)
    offsets = torch.arange(examined, dtype=torch.int64, device=live_counts.device)[None, :]
    permutation_prefix = (start[:, None] + offsets * stride[:, None]) & capacity_mask
    valid = permutation_prefix < live_counts[:, None]

    # Pack the first K valid metadata positions. This ordering operation is over at most K+128
    # metadata candidates, never over the full population and never over neural scores.
    candidate_order = torch.arange(examined, dtype=torch.int64, device=live_counts.device)[None, :]
    pack_key = torch.where(valid, candidate_order, candidate_order + examined)
    chosen_prefix = pack_key.topk(k=k, dim=1, largest=False, sorted=True).indices
    positions = permutation_prefix.gather(1, chosen_prefix)
    return Gate7NativeSample(positions=positions, metadata_candidates_examined=examined)


def gate7_native_priority(
    heap_ids: torch.Tensor,
    *,
    public_seeds: torch.Tensor,
    slot_index: int,
    namespace_code: int,
) -> torch.Tensor:
    """Cheap deterministic public priority; independent of neural state/score and hidden answer."""

    if heap_ids.ndim != 2 or public_seeds.shape != (heap_ids.shape[0],):
        raise ValueError("heap_ids/public_seeds have incompatible shapes")
    return (
        heap_ids * _PRIORITY_PATH_MIX
        + public_seeds[:, None] * _PRIORITY_SEED_MIX
        + slot_index * _PRIORITY_SLOT_MIX
        + namespace_code * _PRIORITY_GROUP_MIX
    ) % _PRIORITY_MODULUS


def select_gate7_native_bounded_score(
    bank: Gate7NativeTensorBank,
    *,
    k: int,
    public_seeds: torch.Tensor,
    slot_index: int,
    sampling_group_code: int,
    tie_namespace_code: int,
) -> tuple[torch.Tensor, Gate7NativeSample]:
    bank.validate()
    sample = sample_gate7_native_positions(
        population_capacity=bank.population_capacity,
        live_counts=bank.live_counts,
        k=k,
        public_seeds=public_seeds,
        slot_index=slot_index,
        sampling_group_code=sampling_group_code,
    )
    batch = bank.states.shape[0]
    rows = torch.arange(batch, device=bank.states.device)[:, None]
    visible_scores = bank.scores[rows, sample.positions]
    quantized = torch.round(visible_scores / GATE3_V1_SCORE_QUANTIZATION).to(torch.int64)
    best = quantized.max(dim=1, keepdim=True).values
    visible_heap = bank.heap_ids[rows, sample.positions]
    priority = gate7_native_priority(
        visible_heap,
        public_seeds=public_seeds,
        slot_index=slot_index,
        namespace_code=tie_namespace_code,
    )
    priority = torch.where(
        quantized == best,
        priority,
        torch.full_like(priority, torch.iinfo(torch.int64).max),
    )
    selected_visible = priority.argmin(dim=1, keepdim=True)
    return sample.positions.gather(1, selected_visible).squeeze(1), sample


def select_gate7_native_bounded_hash(
    bank: Gate7NativeTensorBank,
    *,
    k: int,
    public_seeds: torch.Tensor,
    slot_index: int,
    sampling_group_code: int,
    hash_namespace_code: int,
) -> tuple[torch.Tensor, Gate7NativeSample]:
    bank.validate()
    sample = sample_gate7_native_positions(
        population_capacity=bank.population_capacity,
        live_counts=bank.live_counts,
        k=k,
        public_seeds=public_seeds,
        slot_index=slot_index,
        sampling_group_code=sampling_group_code,
    )
    batch = bank.states.shape[0]
    rows = torch.arange(batch, device=bank.states.device)[:, None]
    visible_heap = bank.heap_ids[rows, sample.positions]
    priority = gate7_native_priority(
        visible_heap,
        public_seeds=public_seeds,
        slot_index=slot_index,
        namespace_code=hash_namespace_code,
    )
    selected_visible = priority.argmin(dim=1, keepdim=True)
    return sample.positions.gather(1, selected_visible).squeeze(1), sample


def gate7_native_victim_positions(
    *,
    live_counts_with_overflow: torch.Tensor,
    public_seeds: torch.Tensor,
    slot_index: int,
) -> torch.Tensor:
    """Answer/score-blind victim position for rows that temporarily hold N+1 candidates."""

    if live_counts_with_overflow.ndim != 1 or public_seeds.shape != live_counts_with_overflow.shape:
        raise ValueError("live_counts/public_seeds must be matching [batch] tensors")
    mixed = public_seeds * _VICTIM_SEED_MIX + slot_index * _VICTIM_SLOT_MIX
    return mixed % live_counts_with_overflow


def swap_delete_gate7_native_parent(
    bank: Gate7NativeTensorBank,
    *,
    selected_positions: torch.Tensor,
) -> Gate7NativeTensorBank:
    """Remove one selected live candidate/world with dense-prefix swap-delete."""

    bank.validate()
    batch = bank.states.shape[0]
    if selected_positions.shape != (batch,):
        raise ValueError("selected_positions must have shape [batch]")
    rows = torch.arange(batch, device=bank.states.device)
    last = bank.live_counts - 1

    states = bank.states.clone()
    scores = bank.scores.clone()
    heap_ids = bank.heap_ids.clone()
    states[rows, selected_positions] = states[rows, last]
    scores[rows, selected_positions] = scores[rows, last]
    heap_ids[rows, selected_positions] = heap_ids[rows, last]
    states[rows, last] = 0
    scores[rows, last] = 0
    heap_ids[rows, last] = 0

    result = Gate7NativeTensorBank(
        states=states,
        scores=scores,
        heap_ids=heap_ids,
        live_counts=bank.live_counts - 1,
        population_capacity=bank.population_capacity,
    )
    result.validate()
    return result


def append_gate7_native_children(
    bank: Gate7NativeTensorBank,
    *,
    child_states: torch.Tensor,
    child_scores: torch.Tensor,
    child_heap_ids: torch.Tensor,
    terminal: torch.Tensor,
) -> Gate7NativeTensorBank:
    """Append two nonterminal children/world into the dense prefix and overflow scratch slot."""

    bank.validate()
    batch = bank.states.shape[0]
    if child_states.shape != (batch, 2, GATE7_NATIVE_STATE_WIDTH):
        raise ValueError("child_states must have shape [batch,2,64]")
    if child_scores.shape != (batch, 2) or child_heap_ids.shape != (batch, 2):
        raise ValueError("child scores/ids must have shape [batch,2]")
    if terminal.shape != (batch,):
        raise ValueError("terminal must have shape [batch]")

    rows = torch.arange(batch, device=bank.states.device)
    first = bank.live_counts
    second = bank.live_counts + 1
    states = bank.states.clone()
    scores = bank.scores.clone()
    heap_ids = bank.heap_ids.clone()

    # Fixed-shape writes for every row. Terminal rows keep these positions outside their live prefix,
    # so their contents are irrelevant and no dynamically sized boolean-index result is required.
    states[rows, first] = child_states[:, 0]
    states[rows, second] = child_states[:, 1]
    scores[rows, first] = child_scores[:, 0]
    scores[rows, second] = child_scores[:, 1]
    heap_ids[rows, first] = child_heap_ids[:, 0]
    heap_ids[rows, second] = child_heap_ids[:, 1]

    nonterminal = (~terminal).to(torch.int64)
    result = Gate7NativeTensorBank(
        states=states,
        scores=scores,
        heap_ids=heap_ids,
        live_counts=bank.live_counts + nonterminal * 2,
        population_capacity=bank.population_capacity,
    )
    result.validate()
    return result


def prune_gate7_native_overflow(
    bank: Gate7NativeTensorBank,
    *,
    public_seeds: torch.Tensor,
    slot_index: int,
) -> tuple[Gate7NativeTensorBank, torch.Tensor]:
    """Drop one deterministic score-blind victim only on rows with N+1 live candidates."""

    bank.validate()
    batch = bank.states.shape[0]
    rows = torch.arange(batch, device=bank.states.device)
    overflow = bank.live_counts > bank.population_capacity
    victim = gate7_native_victim_positions(
        live_counts_with_overflow=bank.live_counts,
        public_seeds=public_seeds,
        slot_index=slot_index,
    )
    last = bank.live_counts - 1

    states = bank.states.clone()
    scores = bank.scores.clone()
    heap_ids = bank.heap_ids.clone()

    victim_states = states[rows, victim]
    victim_scores = scores[rows, victim]
    victim_heap = heap_ids[rows, victim]
    last_states = states[rows, last]
    last_scores = scores[rows, last]
    last_heap = heap_ids[rows, last]

    states[rows, victim] = torch.where(overflow[:, None], last_states, victim_states)
    scores[rows, victim] = torch.where(overflow, last_scores, victim_scores)
    heap_ids[rows, victim] = torch.where(overflow, last_heap, victim_heap)

    kept_last_states = torch.where(overflow[:, None], torch.zeros_like(last_states), states[rows, last])
    kept_last_scores = torch.where(overflow, torch.zeros_like(last_scores), scores[rows, last])
    kept_last_heap = torch.where(overflow, torch.zeros_like(last_heap), heap_ids[rows, last])
    states[rows, last] = kept_last_states
    scores[rows, last] = kept_last_scores
    heap_ids[rows, last] = kept_last_heap

    result = Gate7NativeTensorBank(
        states=states,
        scores=scores,
        heap_ids=heap_ids,
        live_counts=bank.live_counts - overflow.to(torch.int64),
        population_capacity=bank.population_capacity,
    )
    result.validate()
    return result, overflow.to(torch.int64)
