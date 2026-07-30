"""Engineering-only CUDA profile for the Gate-7 high-scale execution substrate.

This profile uses one deterministic randomly initialized scale-neutral scorer and synthetic public hints.
It loads no trained checkpoint, generates no hidden path, assigns no coverage or scientific outcome, and
cannot open the Gate-7 screening namespace.  Its only purpose is to establish end-to-end wall time and
peak CUDA allocation for the frozen physical batch and prepared population ladder.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable, TypeVar

import torch

from .gate7_high_scale_frontier_prep import build_gate7_high_scale_immutable_frontier
from .gate7_high_scale_index_bank_prep import (
    gate7_high_scale_index_bank_storage_bytes,
    initialize_gate7_high_scale_live_index_bank,
)
from .gate7_high_scale_routing_bandwidth_protocol import (
    GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE,
    GATE7_HIGH_SCALE_POPULATIONS,
    GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
    build_gate7_high_scale_tier_plan,
)
from .gate7_high_scale_terminal_stage_b_prep import (
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH,
    GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH,
    GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE,
    run_gate7_high_scale_terminal_stage_b_preparation,
)
from .gate7_scale_neutral_memory_bounded_prep import (
    advance_gate7_scale_neutral_memory_bounded,
)
from .gate7_scale_neutral_model_prep import Gate7ScaleNeutralScorer

GATE7_HIGH_SCALE_ENGINEERING_PROFILE_ONLY = True
GATE7_HIGH_SCALE_ENGINEERING_PROFILE_VERSION = "gate7-high-scale-execution-engineering-profile-v0"
GATE7_HIGH_SCALE_ENGINEERING_MODEL_SEED = 71_007
GATE7_HIGH_SCALE_ENGINEERING_CONDITIONS = (
    (GATE7_HIGH_SCALE_STAGE_B_GLOBAL_SCORE, None),
    (GATE7_HIGH_SCALE_STAGE_B_GLOBAL_HASH, None),
    (GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE, 16),
    (GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH, 16),
    (GATE7_HIGH_SCALE_STAGE_B_BOUNDED_SCORE, 512),
    (GATE7_HIGH_SCALE_STAGE_B_BOUNDED_HASH, 512),
)
GATE7_HIGH_SCALE_ENGINEERING_EQUIVALENCE_TOLERANCE = 5e-6

T = TypeVar("T")


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def gate7_high_scale_engineering_public_hints(
    *, population: int
) -> tuple[tuple[int, ...], ...]:
    """Deterministic public-only synthetic inputs; no latent or hidden answer is constructed."""

    plan = build_gate7_high_scale_tier_plan(population)
    return tuple(
        tuple(
            _seed_from_parts(
                "gate7-high-scale-engineering-public-hint",
                population,
                world_index,
                depth_index,
            )
            & 1
            for depth_index in range(plan.world_depth)
        )
        for world_index in range(GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE)
    )


def gate7_high_scale_engineering_public_seeds(
    *, population: int, device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [
            _seed_from_parts(
                "gate7-high-scale-engineering-public-seed",
                population,
                world_index,
            )
            for world_index in range(GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE)
        ],
        dtype=torch.int64,
        device=device,
    )


def _condition_name(mode: str, k: int | None) -> str:
    return mode if k is None else f"{mode}_k{k}"


def _cuda_measure(function: Callable[[], T]) -> tuple[T, dict[str, float | int]]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    current_before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return result, {
        "wall_seconds": elapsed,
        "current_allocated_before_bytes": current_before,
        "current_allocated_after_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _equivalence_preflight(model: Gate7ScaleNeutralScorer, device: torch.device) -> dict[str, float | int]:
    generator = torch.Generator(device=device)
    generator.manual_seed(GATE7_HIGH_SCALE_ENGINEERING_MODEL_SEED + 1)
    state = torch.randn((257, 64), dtype=torch.float32, device=device, generator=generator)
    phase_input = torch.randn((257, 19), dtype=torch.float32, device=device, generator=generator)
    with torch.inference_mode():
        reference = model.advance(state.clone(), phase_input.clone(), repeats=8)
        bounded = advance_gate7_scale_neutral_memory_bounded(
            model,
            state.clone(),
            phase_input.clone(),
            repeats=8,
        )
        max_abs = float((reference - bounded).abs().max().detach().cpu())
        score_abs = float(
            (model.score(reference) - model.score(bounded)).abs().max().detach().cpu()
        )
    if max_abs > GATE7_HIGH_SCALE_ENGINEERING_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            f"memory-bounded recurrent state differs from reference by {max_abs}, "
            f"above {GATE7_HIGH_SCALE_ENGINEERING_EQUIVALENCE_TOLERANCE}"
        )
    if score_abs > GATE7_HIGH_SCALE_ENGINEERING_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            f"memory-bounded recurrent score differs from reference by {score_abs}, "
            f"above {GATE7_HIGH_SCALE_ENGINEERING_EQUIVALENCE_TOLERANCE}"
        )
    del state, phase_input, reference, bounded
    torch.cuda.empty_cache()
    return {
        "batch_size": 257,
        "recurrent_updates": 8,
        "state_max_abs_difference": max_abs,
        "score_max_abs_difference": score_abs,
        "tolerance": GATE7_HIGH_SCALE_ENGINEERING_EQUIVALENCE_TOLERANCE,
    }


def run_gate7_high_scale_engineering_profile(*, output_root: Path) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-7 engineering profile output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate-7 high-scale engineering profile requires CUDA")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    summary_path = output_root / "summary.json"
    device = torch.device("cuda")
    torch.manual_seed(GATE7_HIGH_SCALE_ENGINEERING_MODEL_SEED)
    torch.cuda.manual_seed_all(GATE7_HIGH_SCALE_ENGINEERING_MODEL_SEED)
    model = Gate7ScaleNeutralScorer().to(device).eval()
    if model.trainable_parameter_count() != 19_649:
        raise RuntimeError("engineering scorer parameter count differs from 19,649")

    print("Gate-7 high-scale execution profile — ENGINEERING ONLY", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Checkpoint loading: NONE", flush=True)
    print("Hidden/scientific worlds: NONE", flush=True)
    print(f"Physical batch: {GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE}", flush=True)
    print(f"Population ladder: {GATE7_HIGH_SCALE_POPULATIONS}", flush=True)
    print("Compiler/CUDA graphs/mixed precision: OFF", flush=True)

    payload: dict[str, object] = {
        "profile_version": GATE7_HIGH_SCALE_ENGINEERING_PROFILE_VERSION,
        "engineering_only": True,
        "scientific_evidence": False,
        "checkpoint_loading_performed": False,
        "hidden_path_constructed": False,
        "gate7_high_scale_science_opened": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "model_seed": GATE7_HIGH_SCALE_ENGINEERING_MODEL_SEED,
        "learned_parameter_count": model.trainable_parameter_count(),
        "parameter_fingerprint": model.parameter_fingerprint(),
        "physical_batch_size": GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE,
        "stage_b_parent_slots": GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
        "population_ladder": list(GATE7_HIGH_SCALE_POPULATIONS),
        "conditions": [
            {"mode": mode, "k": k, "name": _condition_name(mode, k)}
            for mode, k in GATE7_HIGH_SCALE_ENGINEERING_CONDITIONS
        ],
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
        "equivalence_preflight": _equivalence_preflight(model, device),
        "tiers": [],
        "status": "RUNNING",
    }
    _write_summary(summary_path, payload)

    tiers: list[dict[str, object]] = []
    stopped = False
    for tier_index, population in enumerate(GATE7_HIGH_SCALE_POPULATIONS, start=1):
        plan = build_gate7_high_scale_tier_plan(population)
        hints = gate7_high_scale_engineering_public_hints(population=population)
        public_seeds = gate7_high_scale_engineering_public_seeds(
            population=population,
            device=device,
        )
        terminal_hints = tuple(row[-1] for row in hints)
        tier: dict[str, object] = {
            "population": population,
            "frontier_depth": plan.frontier_depth,
            "world_depth": plan.world_depth,
            "logical_stage_a_parent_slots": plan.stage_a_parent_slots,
            "logical_stage_a_learned_updates": plan.stage_a_learned_updates,
            "logical_stage_b_parent_slots": plan.stage_b_parent_slots,
            "logical_stage_b_learned_updates": plan.stage_b_learned_updates,
            "logical_total_learned_updates": plan.total_logical_learned_updates,
            "reference_repeated_sequence_bytes_at_last_layer": (
                GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE
                * population
                * 8
                * 32
                * 4
            ),
            "memory_bounded_projected_action_bytes_at_last_layer": (
                GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE
                * (population // 2)
                * 32
                * 4
            ),
            "conditions": [],
            "status": "RUNNING",
        }
        print(
            f"N={population:6d} ({tier_index}/{len(GATE7_HIGH_SCALE_POPULATIONS)}) "
            f"building B{GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE} immutable frontier...",
            flush=True,
        )

        try:
            frontier, build_metrics = _cuda_measure(
                lambda: build_gate7_high_scale_immutable_frontier(
                    model,
                    population=population,
                    noisy_hints_by_world=hints,
                    device=device,
                )
            )
        except torch.OutOfMemoryError as exc:
            tier["status"] = "CUDA_OUT_OF_MEMORY_DURING_FRONTIER_BUILD"
            tier["error"] = str(exc)
            tiers.append(tier)
            stopped = True
            payload["tiers"] = tiers
            payload["status"] = "RESOURCE_FRONTIER_REACHED"
            _write_summary(summary_path, payload)
            break

        tier["frontier_build"] = build_metrics
        tier["frontier_storage_bytes"] = (
            frontier.states.numel() * frontier.states.element_size()
            + frontier.scores.numel() * frontier.scores.element_size()
        )
        probe_bank = initialize_gate7_high_scale_live_index_bank(
            batch_size=GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE,
            population=population,
            device=device,
        )
        tier["condition_index_bank_storage_bytes"] = gate7_high_scale_index_bank_storage_bytes(
            probe_bank
        )
        del probe_bank
        print(
            f"  frontier {build_metrics['wall_seconds']:.3f}s, "
            f"peak={build_metrics['peak_allocated_bytes'] / 2**30:.2f} GiB",
            flush=True,
        )

        condition_rows: list[dict[str, object]] = []
        for mode, k in GATE7_HIGH_SCALE_ENGINEERING_CONDITIONS:
            name = _condition_name(mode, k)
            try:
                transcript, condition_metrics = _cuda_measure(
                    lambda mode=mode, k=k: run_gate7_high_scale_terminal_stage_b_preparation(
                        model,
                        frontier,
                        terminal_hints_by_world=terminal_hints,
                        public_seeds=public_seeds,
                        mode=mode,
                        k=k,
                        stage_b_slots=GATE7_HIGH_SCALE_STAGE_B_PARENT_SLOTS,
                    )
                )
            except torch.OutOfMemoryError as exc:
                condition_rows.append(
                    {
                        "name": name,
                        "mode": mode,
                        "k": k,
                        "status": "CUDA_OUT_OF_MEMORY",
                        "error": str(exc),
                    }
                )
                tier["status"] = "CUDA_OUT_OF_MEMORY_DURING_CONDITION"
                stopped = True
                break

            totals = transcript.total_neural_score_observations_per_world()
            condition_row = {
                "name": name,
                "mode": mode,
                "k": k,
                "status": "SUCCESS",
                **condition_metrics,
                "score_observations_per_world_min": int(totals.min().detach().cpu()),
                "score_observations_per_world_max": int(totals.max().detach().cpu()),
                "selected_frontier_index_checksum": int(
                    transcript.selected_frontier_indices.sum().detach().cpu()
                ),
                "terminal_score_checksum": float(
                    transcript.terminal_child_scores.sum().detach().cpu()
                ),
                "final_live_count_min": int(
                    transcript.final_bank.live_counts.min().detach().cpu()
                ),
                "final_live_count_max": int(
                    transcript.final_bank.live_counts.max().detach().cpu()
                ),
            }
            condition_rows.append(condition_row)
            print(
                f"  {name:<24s} {condition_metrics['wall_seconds']:.3f}s, "
                f"peak={condition_metrics['peak_allocated_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
            del transcript, totals
            gc.collect()
            torch.cuda.empty_cache()

        tier["conditions"] = condition_rows
        if tier.get("status") == "RUNNING":
            tier["status"] = "SUCCESS"
        tiers.append(tier)
        payload["tiers"] = tiers
        payload["status"] = "RESOURCE_FRONTIER_REACHED" if stopped else "RUNNING"
        _write_summary(summary_path, payload)

        del frontier, public_seeds
        gc.collect()
        torch.cuda.empty_cache()
        if stopped:
            break

    if not stopped:
        payload["status"] = "COMPLETE"
    payload["tiers"] = tiers
    _write_summary(summary_path, payload)
    print(f"Profile status: {payload['status']}", flush=True)
    print(f"Full JSON: {summary_path}", flush=True)
    print("This is engineering evidence only, not Gate-7 scientific evidence.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return run_gate7_high_scale_engineering_profile(output_root=args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
