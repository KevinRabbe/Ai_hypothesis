"""Engineering-only CUDA scaling profile for Gate-7 native routing-bank mechanics.

This profile deliberately contains no scientific worlds, hidden answers, checkpoint loading, training,
or capability outcome.  It measures whether the prepared bounded-routing data structure itself remains
bounded as N grows, before the Gate-7 scientific protocol is admitted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .gate7_native_tensor_bank_prep import (
    GATE7_NATIVE_K_LADDER,
    GATE7_NATIVE_MAX_STAGE_B_SLOTS,
    GATE7_NATIVE_POPULATION_LADDER,
    GATE7_NATIVE_STATE_WIDTH,
    Gate7NativeTensorBank,
    append_gate7_native_children,
    gate7_native_priority,
    gate7_native_public_seed_tensor,
    prune_gate7_native_overflow,
    select_gate7_native_bounded_hash,
    select_gate7_native_bounded_score,
    swap_delete_gate7_native_parent,
)

PROFILE_VERSION = "gate7-native-bank-scaling-profile-v0"
DEFAULT_BATCH_SIZE = 32
DEFAULT_SLOTS = 128
DEFAULT_REPEATS = 3
_PROFILE_K = (16, 64, 256, 512)


def _synchronize() -> None:
    torch.cuda.synchronize()


def _allocate_bank(*, batch_size: int, population: int, device: torch.device) -> Gate7NativeTensorBank:
    width = population + 1
    # State contents are irrelevant to this mechanics-only profile; storage size and indexed mutation
    # are what matter. Scores/IDs are deterministic public synthetic values.
    states = torch.empty((batch_size, width, GATE7_NATIVE_STATE_WIDTH), dtype=torch.float32, device=device)
    base_scores = torch.linspace(-1.0, 1.0, steps=width, dtype=torch.float32, device=device)
    scores = base_scores[None, :].expand(batch_size, width).clone()
    heap_ids = torch.arange(1, width + 1, dtype=torch.int64, device=device)[None, :].expand(batch_size, width).clone()
    live_counts = torch.full((batch_size,), population, dtype=torch.int64, device=device)
    bank = Gate7NativeTensorBank(
        states=states,
        scores=scores,
        heap_ids=heap_ids,
        live_counts=live_counts,
        population_capacity=population,
    )
    bank.validate()
    return bank


def _synthetic_children(*, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    child_states = torch.zeros((batch_size, 2, GATE7_NATIVE_STATE_WIDTH), dtype=torch.float32, device=device)
    child_scores = torch.tensor((-0.125, 0.125), dtype=torch.float32, device=device)[None, :].expand(batch_size, 2)
    return child_states, child_scores


def _child_ids(*, batch_size: int, slot_index: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(batch_size, dtype=torch.int64, device=device)
    base = 10_000_000 + slot_index * 10_000 + rows * 2
    return torch.stack((base, base + 1), dim=1)


def _select_global(bank: Gate7NativeTensorBank, *, public_seeds: torch.Tensor, slot_index: int) -> torch.Tensor:
    """Engineering reference: deliberately inspect every live score."""

    population = bank.population_capacity
    live_scores = bank.scores[:, :population]
    best = live_scores.max(dim=1, keepdim=True).values
    tie = gate7_native_priority(
        bank.heap_ids[:, :population],
        public_seeds=public_seeds,
        slot_index=slot_index,
        namespace_code=7_001,
    )
    tie = torch.where(live_scores == best, tie, torch.full_like(tie, torch.iinfo(torch.int64).max))
    return tie.argmin(dim=1)


def _run_cycle(
    *,
    population: int,
    batch_size: int,
    slots: int,
    mode: str,
    k: int | None,
    device: torch.device,
) -> dict[str, float | int | str]:
    torch.cuda.reset_peak_memory_stats(device)
    bank = _allocate_bank(batch_size=batch_size, population=population, device=device)
    public_seeds = gate7_native_public_seed_tensor(
        tuple(100_003 + index * 7_919 for index in range(batch_size)), device=device
    )
    child_states, child_scores = _synthetic_children(batch_size=batch_size, device=device)
    terminal = torch.zeros(batch_size, dtype=torch.bool, device=device)

    _synchronize()
    start = time.perf_counter()
    metadata_examined_per_world = 0
    for slot_index in range(slots):
        if mode == "bounded_score":
            if k is None:
                raise RuntimeError("bounded_score requires K")
            selected, sample = select_gate7_native_bounded_score(
                bank,
                k=k,
                public_seeds=public_seeds,
                slot_index=slot_index,
                sampling_group_code=k,
                tie_namespace_code=8_001,
            )
            metadata_examined_per_world += sample.metadata_candidates_examined
        elif mode == "bounded_hash":
            if k is None:
                raise RuntimeError("bounded_hash requires K")
            selected, sample = select_gate7_native_bounded_hash(
                bank,
                k=k,
                public_seeds=public_seeds,
                slot_index=slot_index,
                sampling_group_code=k,
                hash_namespace_code=8_002,
            )
            metadata_examined_per_world += sample.metadata_candidates_examined
        elif mode == "global_score":
            selected = _select_global(bank, public_seeds=public_seeds, slot_index=slot_index)
            metadata_examined_per_world += population
        else:
            raise ValueError(mode)

        bank = swap_delete_gate7_native_parent(bank, selected_positions=selected)
        bank = append_gate7_native_children(
            bank,
            child_states=child_states,
            child_scores=child_scores,
            child_heap_ids=_child_ids(batch_size=batch_size, slot_index=slot_index, device=device),
            terminal=terminal,
        )
        bank, _overflow = prune_gate7_native_overflow(
            bank,
            public_seeds=public_seeds,
            slot_index=slot_index,
        )

    _synchronize()
    elapsed_s = time.perf_counter() - start
    world_decisions = batch_size * slots
    return {
        "mode": mode,
        "k": -1 if k is None else k,
        "elapsed_ms": elapsed_s * 1000.0,
        "world_decisions_per_second": world_decisions / elapsed_s,
        "microseconds_per_world_decision": elapsed_s * 1_000_000.0 / world_decisions,
        "metadata_candidates_examined_per_world": metadata_examined_per_world,
        "neural_score_reads_per_world": population * slots if mode == "global_score" else (0 if mode == "bounded_hash" else int(k) * slots),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _measure_condition(
    *,
    population: int,
    batch_size: int,
    slots: int,
    mode: str,
    k: int | None,
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    # One warmup isolates allocator/kernel first-use noise from timed repetitions.
    _run_cycle(
        population=population,
        batch_size=batch_size,
        slots=min(8, slots),
        mode=mode,
        k=k,
        device=device,
    )
    samples = [
        _run_cycle(
            population=population,
            batch_size=batch_size,
            slots=slots,
            mode=mode,
            k=k,
            device=device,
        )
        for _ in range(repeats)
    ]
    elapsed = [float(row["elapsed_ms"]) for row in samples]
    throughput = [float(row["world_decisions_per_second"]) for row in samples]
    latency = [float(row["microseconds_per_world_decision"]) for row in samples]
    return {
        "population": population,
        "batch_size": batch_size,
        "slots": slots,
        "mode": mode,
        "k": None if k is None else k,
        "mean_elapsed_ms": statistics.mean(elapsed),
        "min_elapsed_ms": min(elapsed),
        "max_elapsed_ms": max(elapsed),
        "mean_world_decisions_per_second": statistics.mean(throughput),
        "mean_microseconds_per_world_decision": statistics.mean(latency),
        "metadata_candidates_examined_per_world": samples[0]["metadata_candidates_examined_per_world"],
        "neural_score_reads_per_world": samples[0]["neural_score_reads_per_world"],
        "peak_allocated_bytes_max": max(int(row["peak_allocated_bytes"]) for row in samples),
        "samples_elapsed_ms": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineering-only Gate-7 native bank scaling profile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the native-bank scaling profile")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not 1 <= args.slots <= GATE7_NATIVE_MAX_STAGE_B_SLOTS:
        raise SystemExit(f"--slots must be in 1..{GATE7_NATIVE_MAX_STAGE_B_SLOTS}")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    device = torch.device("cuda:0")
    rows: list[dict[str, object]] = []
    for population in GATE7_NATIVE_POPULATION_LADDER:
        rows.append(
            _measure_condition(
                population=population,
                batch_size=args.batch_size,
                slots=args.slots,
                mode="global_score",
                k=None,
                repeats=args.repeats,
                device=device,
            )
        )
        for k in _PROFILE_K:
            if k >= population:
                continue
            for mode in ("bounded_score", "bounded_hash"):
                rows.append(
                    _measure_condition(
                        population=population,
                        batch_size=args.batch_size,
                        slots=args.slots,
                        mode=mode,
                        k=k,
                        repeats=args.repeats,
                        device=device,
                    )
                )

    payload = {
        "profile_version": PROFILE_VERSION,
        "status": "ENGINEERING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
        "scientific_worlds_used": False,
        "checkpoint_used": False,
        "batch_size": args.batch_size,
        "slots": args.slots,
        "repeats": args.repeats,
        "population_ladder": list(GATE7_NATIVE_POPULATION_LADDER),
        "profile_k_values": list(_PROFILE_K),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
