"""Engineering-only CUDA profiler for eager-vs-tensor Gate-7 execution preparation.

No Gate-7 scientific worlds or outcomes are generated here.  The harness uses deterministic synthetic
public hints and one already-frozen checkpoint only to make the execution shape representative.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from .gate3_v1_model import Gate3V1NeuralCandidate
from .gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from .gate6_fixed_k_population_scaling import Gate6EvaluationWorld, _advance_parent_batch
from .gate7_tensor_engine_prep import build_complete_tensor_frontier
from .run_gate3_v2_frontier import load_verified_checkpoint

ENGINEERING_NAMESPACE = "gate7-execution-engineering-profile-v0"
DEFAULT_WORLD_COUNT = 64
DEFAULT_FRONTIER_DEPTH = 8
DEFAULT_REPEATS = 3


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def build_engineering_worlds(world_count: int) -> tuple[Gate6EvaluationWorld, ...]:
    if not 1 <= world_count <= 256:
        raise ValueError("engineering world_count must be in 1..256")
    rows = []
    for index in range(world_count):
        rng = random.Random(_seed_from_parts(ENGINEERING_NAMESPACE, index))
        hints = tuple(rng.randrange(2) for _ in range(10))
        public = Gate3V1PublicWorld(
            seed=_seed_from_parts(ENGINEERING_NAMESPACE, "runtime", index),
            depth=10,
            noisy_hints=hints,
        )
        world = Gate6EvaluationWorld(
            world_index=index,
            public=public,
            # Evaluation-only dummy; the profile never reads hidden_path.
            hidden_path=(0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        )
        world.validate()
        rows.append(world)
    return tuple(rows)


def build_eager_frontier(
    model: torch.nn.Module,
    worlds: tuple[Gate6EvaluationWorld, ...],
    *,
    frontier_depth: int,
    device: torch.device,
) -> tuple[tuple[Gate3V1NeuralCandidate, ...], ...]:
    populations: tuple[tuple[Gate3V1NeuralCandidate, ...], ...] = tuple(
        (
            Gate3V1NeuralCandidate(
                path=(),
                state=model.initial_state(1, device=device)[0],
                score=0.0,
            ),
        )
        for _ in worlds
    )
    with torch.inference_mode():
        for _ in range(frontier_depth):
            populations = _advance_parent_batch(
                model,
                worlds,
                populations,
                device=device,
            )
    return populations


def _cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _time_cuda(fn: Callable[[], object], *, repeats: int) -> dict[str, object]:
    samples_ms: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        _cleanup_cuda()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        samples_ms.append(elapsed_ms)
        peaks.append(torch.cuda.max_memory_allocated())
        del result
    return {
        "samples_ms": samples_ms,
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "peak_allocated_bytes_max": max(peaks),
    }


def _profile_once(fn: Callable[[], object], *, table_path: Path, trace_path: Path) -> None:
    _cleanup_cuda()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        result = fn()
        torch.cuda.synchronize()
    table_path.write_text(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40),
        encoding="utf-8",
    )
    prof.export_chrome_trace(str(trace_path))
    del result


def run_profile(
    *,
    output_root: Path,
    checkpoint_path: Path,
    world_count: int,
    frontier_depth: int,
    repeats: int,
) -> int:
    if output_root.exists():
        raise FileExistsError(f"engineering profile output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate-7 execution profile requires CUDA")
    if not 1 <= frontier_depth <= 9:
        raise ValueError("existing frozen checkpoint profile supports frontier depth 1..9")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    device = torch.device("cuda")
    model, identity = load_verified_checkpoint(
        checkpoint_index=0,
        checkpoint_path=checkpoint_path,
        device="cuda",
    )
    model.eval()
    worlds = build_engineering_worlds(world_count)
    public_worlds = tuple(world.public for world in worlds)

    eager_fn = lambda: build_eager_frontier(
        model,
        worlds,
        frontier_depth=frontier_depth,
        device=device,
    )
    tensor_fn = lambda: build_complete_tensor_frontier(
        model,
        public_worlds,
        frontier_depth=frontier_depth,
        device=device,
    )

    # One warmup per path before timed measurements.
    warm = eager_fn()
    torch.cuda.synchronize()
    del warm
    _cleanup_cuda()
    warm = tensor_fn()
    torch.cuda.synchronize()
    del warm
    _cleanup_cuda()

    eager = _time_cuda(eager_fn, repeats=repeats)
    tensor = _time_cuda(tensor_fn, repeats=repeats)

    generated_children_per_run = world_count * ((1 << (frontier_depth + 1)) - 2)
    eager_seconds = float(eager["mean_ms"]) / 1000.0
    tensor_seconds = float(tensor["mean_ms"]) / 1000.0
    summary = {
        "status": "ENGINEERING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "namespace": ENGINEERING_NAMESPACE,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "checkpoint_index": identity.checkpoint_index,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "parameter_fingerprint": identity.parameter_fingerprint,
        "learned_parameter_count": identity.learned_parameter_count,
        "world_count": world_count,
        "frontier_depth": frontier_depth,
        "final_population_per_world": 1 << frontier_depth,
        "generated_children_per_run": generated_children_per_run,
        "repeats": repeats,
        "eager_object": {
            **eager,
            "generated_children_per_second": generated_children_per_run / eager_seconds,
        },
        "tensorized_eager": {
            **tensor,
            "generated_children_per_second": generated_children_per_run / tensor_seconds,
        },
        "wall_speedup_tensor_over_object": float(eager["mean_ms"]) / float(tensor["mean_ms"]),
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _profile_once(
        eager_fn,
        table_path=output_root / "eager-profiler-table.txt",
        trace_path=output_root / "eager-trace.json",
    )
    _profile_once(
        tensor_fn,
        table_path=output_root / "tensor-profiler-table.txt",
        trace_path=output_root / "tensor-trace.json",
    )

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--world-count", type=int, default=DEFAULT_WORLD_COUNT)
    parser.add_argument("--frontier-depth", type=int, default=DEFAULT_FRONTIER_DEPTH)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    return run_profile(
        output_root=args.output_root,
        checkpoint_path=args.checkpoint,
        world_count=args.world_count,
        frontier_depth=args.frontier_depth,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    raise SystemExit(main())
