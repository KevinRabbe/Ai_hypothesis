"""GPU frontier construction and exact hidden-parent rank decomposition for Gate-7."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .gate7_high_scale_frontier_prep import build_gate7_high_scale_immutable_frontier
from .gate7_information_ceiling_decomposition_protocol import (
    GATE7_INFORMATION_CEILING_BAYES,
    GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES,
    GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE,
    GATE7_INFORMATION_CEILING_HASH,
    GATE7_INFORMATION_CEILING_LEARNED,
    GATE7_INFORMATION_CEILING_RANKERS,
    GATE7_INFORMATION_CEILING_WORLD_COUNT,
)
from .gate7_information_ceiling_decomposition_worlds import (
    Gate7InformationCeilingWorld,
    validate_information_ceiling_world_batch,
)
from .gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer


@dataclass(frozen=True, slots=True)
class Gate7InformationCeilingBatchRanks:
    checkpoint_index: int
    population: int
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    learned_ranks: tuple[int, ...]
    bayes_ranks: tuple[int, ...]
    hash_ranks: tuple[int, ...]
    learned_parameter_count: int
    parameter_fingerprint: str
    frontier_wall_seconds: float
    frontier_peak_allocated_bytes: int
    frontier_storage_bytes: int
    frontier_score_checksum: float
    hidden_score_checksum: float
    tie_priority_checksum: int
    hash_priority_checksum: int

    def validate(self) -> None:
        if self.checkpoint_index not in GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES:
            raise ValueError("rank batch checkpoint is outside T0/T1/T2")
        count = GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE
        if any(
            len(values) != count
            for values in (
                self.world_indices,
                self.runtime_seeds,
                self.learned_ranks,
                self.bayes_ranks,
                self.hash_ranks,
            )
        ):
            raise ValueError("rank batch must preserve exactly 64 worlds")
        if self.world_indices != tuple(range(self.world_indices[0], self.world_indices[0] + count)):
            raise ValueError("rank batch world indices are not contiguous")
        if self.world_indices[0] % count:
            raise ValueError("rank batch world indices are not B64-aligned")
        for ranks in (self.learned_ranks, self.bayes_ranks, self.hash_ranks):
            if any(not 1 <= rank <= self.population for rank in ranks):
                raise ValueError("hidden-parent rank is outside 1..N")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7InformationCeilingCheckpointRanks:
    checkpoint_index: int
    population: int
    world_indices: tuple[int, ...]
    runtime_seeds: tuple[int, ...]
    ranks_by_ranker: dict[str, tuple[int, ...]]
    learned_parameter_count: int
    parameter_fingerprint: str
    batch_count: int
    frontier_wall_seconds: float
    frontier_peak_allocated_bytes: int
    frontier_storage_bytes: int
    frontier_score_checksum: float
    hidden_score_checksum: float
    tie_priority_checksum: int
    hash_priority_checksum: int

    def validate(self) -> None:
        if self.checkpoint_index not in GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES:
            raise ValueError("checkpoint rank result is outside T0/T1/T2")
        if self.world_indices != tuple(range(GATE7_INFORMATION_CEILING_WORLD_COUNT)):
            raise ValueError("checkpoint ranks must cover exact worlds 0..511")
        if len(self.runtime_seeds) != GATE7_INFORMATION_CEILING_WORLD_COUNT:
            raise ValueError("checkpoint runtime-seed vector changed")
        if tuple(self.ranks_by_ranker) != GATE7_INFORMATION_CEILING_RANKERS:
            raise ValueError("checkpoint ranker order changed")
        for ranks in self.ranks_by_ranker.values():
            if len(ranks) != GATE7_INFORMATION_CEILING_WORLD_COUNT:
                raise ValueError("checkpoint rank vector changed length")
            if any(not 1 <= rank <= self.population for rank in ranks):
                raise ValueError("checkpoint hidden-parent rank is outside 1..N")
        expected_batches = (
            GATE7_INFORMATION_CEILING_WORLD_COUNT
            // GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE
        )
        if self.batch_count != expected_batches:
            raise ValueError("checkpoint rank batch count changed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prefix_ids(worlds: tuple[Gate7InformationCeilingWorld, ...]) -> torch.Tensor:
    values = []
    for world in worlds:
        value = 0
        for bit in world.noisy_hints[: world.frontier_depth]:
            value = value * 2 + bit
        values.append(value)
    return torch.tensor(values, dtype=torch.int64)


def _affine_priorities(
    *,
    population: int,
    multipliers: torch.Tensor,
    offsets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if multipliers.dtype != torch.int64 or offsets.dtype != torch.int64:
        raise ValueError("public permutation parameters must use int64")
    candidate_ids = torch.arange(population, dtype=torch.int64, device=device)
    return (
        candidate_ids[None, :] * multipliers[:, None].to(device)
        + offsets[:, None].to(device)
    ) & (population - 1)


def _hamming_distance_matrix(
    *, population: int, hint_prefix_ids: torch.Tensor, device: torch.device
) -> torch.Tensor:
    depth = population.bit_length() - 1
    candidate_ids = torch.arange(population, dtype=torch.int64, device=device)
    hint_prefix_ids = hint_prefix_ids.to(device)
    distances = torch.zeros(
        (hint_prefix_ids.shape[0], population),
        dtype=torch.int16,
        device=device,
    )
    for bit_position in range(depth):
        candidate_bits = ((candidate_ids >> bit_position) & 1).to(torch.int16)
        hint_bits = ((hint_prefix_ids >> bit_position) & 1).to(torch.int16)
        distances.add_((candidate_bits[None, :] != hint_bits[:, None]).to(torch.int16))
    return distances


def evaluate_information_ceiling_rank_batch(
    model: Gate7ScaleNeutralScorer,
    *,
    checkpoint_index: int,
    worlds: tuple[Gate7InformationCeilingWorld, ...],
    device: torch.device | str,
) -> Gate7InformationCeilingBatchRanks:
    population = validate_information_ceiling_world_batch(worlds)
    if checkpoint_index not in GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES:
        raise ValueError("rank evaluation checkpoint must be T0, T1 or T2")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("admitted information-ceiling rank evaluation requires CUDA")
    _release_cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frontier = build_gate7_high_scale_immutable_frontier(
        model,
        population=population,
        noisy_hints_by_world=tuple(world.noisy_hints for world in worlds),
        device=target,
    )
    torch.cuda.synchronize()
    frontier_wall = time.perf_counter() - started
    frontier_peak = int(torch.cuda.max_memory_allocated())
    frontier_storage = int(
        frontier.states.numel() * frontier.states.element_size()
        + frontier.scores.numel() * frontier.scores.element_size()
    )

    hidden = torch.tensor(
        [world.hidden_parent_index for world in worlds],
        dtype=torch.int64,
        device=target,
    )
    batch_rows = torch.arange(len(worlds), dtype=torch.int64, device=target)
    tie_priorities = _affine_priorities(
        population=population,
        multipliers=torch.tensor([world.tie_multiplier for world in worlds], dtype=torch.int64),
        offsets=torch.tensor([world.tie_offset for world in worlds], dtype=torch.int64),
        device=target,
    )
    hash_priorities = _affine_priorities(
        population=population,
        multipliers=torch.tensor([world.hash_multiplier for world in worlds], dtype=torch.int64),
        offsets=torch.tensor([world.hash_offset for world in worlds], dtype=torch.int64),
        device=target,
    )

    scores = frontier.scores
    hidden_scores = scores[batch_rows, hidden]
    hidden_tie = tie_priorities[batch_rows, hidden]
    learned_ranks_tensor = 1 + (
        (scores > hidden_scores[:, None])
        | ((scores == hidden_scores[:, None]) & (tie_priorities < hidden_tie[:, None]))
    ).sum(dim=1, dtype=torch.int64)

    hint_prefix_ids = _prefix_ids(worlds)
    distances = _hamming_distance_matrix(
        population=population,
        hint_prefix_ids=hint_prefix_ids,
        device=target,
    )
    hidden_distances = distances[batch_rows, hidden]
    bayes_ranks_tensor = 1 + (
        (distances < hidden_distances[:, None])
        | (
            (distances == hidden_distances[:, None])
            & (tie_priorities < hidden_tie[:, None])
        )
    ).sum(dim=1, dtype=torch.int64)

    hash_ranks_tensor = hash_priorities[batch_rows, hidden] + 1
    learned_ranks = tuple(int(value) for value in learned_ranks_tensor.cpu().tolist())
    bayes_ranks = tuple(int(value) for value in bayes_ranks_tensor.cpu().tolist())
    hash_ranks = tuple(int(value) for value in hash_ranks_tensor.cpu().tolist())

    row = Gate7InformationCeilingBatchRanks(
        checkpoint_index=checkpoint_index,
        population=population,
        world_indices=tuple(world.world_index for world in worlds),
        runtime_seeds=tuple(world.runtime_seed for world in worlds),
        learned_ranks=learned_ranks,
        bayes_ranks=bayes_ranks,
        hash_ranks=hash_ranks,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=model.parameter_fingerprint(),
        frontier_wall_seconds=frontier_wall,
        frontier_peak_allocated_bytes=frontier_peak,
        frontier_storage_bytes=frontier_storage,
        frontier_score_checksum=float(scores.sum().detach().cpu()),
        hidden_score_checksum=float(hidden_scores.sum().detach().cpu()),
        tie_priority_checksum=int(tie_priorities.sum().detach().cpu()),
        hash_priority_checksum=int(hash_priorities.sum().detach().cpu()),
    )
    row.validate()
    del (
        frontier,
        hidden,
        batch_rows,
        tie_priorities,
        hash_priorities,
        scores,
        hidden_scores,
        hidden_tie,
        learned_ranks_tensor,
        hint_prefix_ids,
        distances,
        hidden_distances,
        bayes_ranks_tensor,
        hash_ranks_tensor,
    )
    _release_cuda()
    return row


def aggregate_information_ceiling_rank_batches(
    rows: tuple[Gate7InformationCeilingBatchRanks, ...],
) -> Gate7InformationCeilingCheckpointRanks:
    expected_batches = (
        GATE7_INFORMATION_CEILING_WORLD_COUNT
        // GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE
    )
    if len(rows) != expected_batches:
        raise ValueError("rank aggregation requires exactly eight B64 rows")
    first = rows[0]
    for row in rows:
        row.validate()
        if (
            row.checkpoint_index,
            row.population,
            row.learned_parameter_count,
            row.parameter_fingerprint,
        ) != (
            first.checkpoint_index,
            first.population,
            first.learned_parameter_count,
            first.parameter_fingerprint,
        ):
            raise ValueError("rank batch identity changed during aggregation")
    world_indices = tuple(index for row in rows for index in row.world_indices)
    runtime_seeds = tuple(seed for row in rows for seed in row.runtime_seeds)
    result = Gate7InformationCeilingCheckpointRanks(
        checkpoint_index=first.checkpoint_index,
        population=first.population,
        world_indices=world_indices,
        runtime_seeds=runtime_seeds,
        ranks_by_ranker={
            GATE7_INFORMATION_CEILING_LEARNED: tuple(
                rank for row in rows for rank in row.learned_ranks
            ),
            GATE7_INFORMATION_CEILING_BAYES: tuple(
                rank for row in rows for rank in row.bayes_ranks
            ),
            GATE7_INFORMATION_CEILING_HASH: tuple(
                rank for row in rows for rank in row.hash_ranks
            ),
        },
        learned_parameter_count=first.learned_parameter_count,
        parameter_fingerprint=first.parameter_fingerprint,
        batch_count=expected_batches,
        frontier_wall_seconds=sum(row.frontier_wall_seconds for row in rows),
        frontier_peak_allocated_bytes=max(row.frontier_peak_allocated_bytes for row in rows),
        frontier_storage_bytes=max(row.frontier_storage_bytes for row in rows),
        frontier_score_checksum=sum(row.frontier_score_checksum for row in rows),
        hidden_score_checksum=sum(row.hidden_score_checksum for row in rows),
        tie_priority_checksum=sum(row.tie_priority_checksum for row in rows),
        hash_priority_checksum=sum(row.hash_priority_checksum for row in rows),
    )
    result.validate()
    return result
