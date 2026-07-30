"""Run the one admitted Gate-7 post-confirmation continuation campaign."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from .gate7_high_scale_routing_bandwidth_continuation import (
    GATE7_CONTINUATION_EXECUTION_ADMITTED,
    GATE7_CONTINUATION_SCIENTIFIC_STATUS,
    continuation_provenance,
    load_verified_gate7_continuation_checkpoint,
)
from .gate7_high_scale_routing_bandwidth_continuation_campaign import (
    execute_gate7_continuation_tier,
)
from .gate7_high_scale_routing_bandwidth_continuation_protocol import (
    GATE7_CONTINUATION_BOOTSTRAP_SAMPLES,
    GATE7_CONTINUATION_CHECKPOINT_INDICES,
    GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
    GATE7_CONTINUATION_HINT_RELIABILITY,
    GATE7_CONTINUATION_K_LADDER,
    GATE7_CONTINUATION_NONINFERIORITY_MARGIN,
    GATE7_CONTINUATION_POPULATIONS,
    GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS,
    GATE7_CONTINUATION_VERSION,
    GATE7_CONTINUATION_WORLD_COUNT,
    classify_continuation_campaign,
)
from .gate7_scale_neutral_transition_bridge import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    return (
        args.transition_checkpoint0,
        args.transition_checkpoint1,
        args.transition_checkpoint2,
    )


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_gate7_high_scale_routing_bandwidth_continuation(
    *,
    output_root: Path,
    transition_checkpoint_paths: tuple[Path, Path, Path],
) -> int:
    if output_root.exists():
        raise FileExistsError(f"Gate-7 continuation output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted Gate-7 continuation requires CUDA")
    if not GATE7_CONTINUATION_EXECUTION_ADMITTED:
        raise RuntimeError("Gate-7 continuation execution is not admitted")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    result_path = output_root / "gate7-high-scale-routing-bandwidth-continuation.json"
    runtime_path = output_root / "runtime.json"
    started = time.monotonic()

    checkpoint_identities = []
    for checkpoint, path in zip(
        GATE7_CONTINUATION_CHECKPOINT_INDICES,
        transition_checkpoint_paths,
        strict=True,
    ):
        model, identity = load_verified_gate7_continuation_checkpoint(
            checkpoint_index=checkpoint,
            checkpoint_path=path,
            device="cpu",
        )
        checkpoint_identities.append(identity)
        del model

    result: dict[str, Any] = {
        "experiment_version": GATE7_CONTINUATION_VERSION,
        "scientific_status": GATE7_CONTINUATION_SCIENTIFIC_STATUS,
        "execution_admitted": True,
        "continuation_opened": True,
        "second_confirmation_opened": False,
        "second_continuation_opened": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        **continuation_provenance(),
        "hint_reliability": GATE7_CONTINUATION_HINT_RELIABILITY,
        "noninferiority_margin": GATE7_CONTINUATION_NONINFERIORITY_MARGIN,
        "populations": list(GATE7_CONTINUATION_POPULATIONS),
        "k_ladder": list(GATE7_CONTINUATION_K_LADDER),
        "world_count_per_checkpoint_population": GATE7_CONTINUATION_WORLD_COUNT,
        "evaluation_batch_size": GATE7_CONTINUATION_EVALUATION_BATCH_SIZE,
        "bootstrap_samples": GATE7_CONTINUATION_BOOTSTRAP_SAMPLES,
        "stage_b_parent_slots": GATE7_CONTINUATION_STAGE_B_PARENT_SLOTS,
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
        "transition_checkpoints": [identity.to_dict() for identity in checkpoint_identities],
        "tiers": [],
        "completed_populations": [],
        "resource_frontier_population": None,
        "resource_error": None,
        "campaign_outcome": "RUNNING",
    }
    _write_json(result_path, result)

    print("Gate-7 post-confirmation routing-bandwidth continuation — FRESH SCIENTIFIC EVIDENCE", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print("Training/checkpoint selection: NONE", flush=True)
    print("Worlds: fresh continuation namespace 0..511 per N", flush=True)
    print(f"Populations: {GATE7_CONTINUATION_POPULATIONS}", flush=True)
    print(f"Complete K ladder at every N: {GATE7_CONTINUATION_K_LADDER}", flush=True)
    print("Physical world batch: 64 x 8", flush=True)
    print("Scientific early stopping: NONE", flush=True)
    print("Second confirmation/continuation: CLOSED", flush=True)

    tiers: list[dict[str, Any]] = []
    completed_populations: list[int] = []
    resource_frontier_population: int | None = None
    resource_error: str | None = None

    for population in GATE7_CONTINUATION_POPULATIONS:
        try:
            tier = execute_gate7_continuation_tier(
                population=population,
                transition_checkpoint_paths=transition_checkpoint_paths,
            )
        except torch.cuda.OutOfMemoryError as exc:
            resource_frontier_population = population
            resource_error = f"{type(exc).__name__}: {exc}"
            _release_cuda()
            print(
                f"\nN={population} resource frontier reached: {resource_error}",
                flush=True,
            )
            break
        tiers.append(tier)
        completed_populations.append(population)
        result["tiers"] = tiers
        result["completed_populations"] = completed_populations
        _write_json(result_path, result)

    campaign_outcome = classify_continuation_campaign(
        completed_populations=tuple(completed_populations),
        resource_frontier_population=resource_frontier_population,
    )
    result["tiers"] = tiers
    result["completed_populations"] = completed_populations
    result["resource_frontier_population"] = resource_frontier_population
    result["resource_error"] = resource_error
    result["campaign_outcome"] = campaign_outcome
    _write_json(result_path, result)

    k_required_by_population = {
        str(tier["population"]): tier["smallest_passing_k"] for tier in tiers
    }
    passing_k_by_population = {
        str(tier["population"]): tier["passing_k"] for tier in tiers
    }
    runtime = {
        "scientific_status": GATE7_CONTINUATION_SCIENTIFIC_STATUS,
        "campaign_outcome": campaign_outcome,
        "continuation_opened": True,
        "second_confirmation_opened": False,
        "second_continuation_opened": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - started,
        "result_sha256": sha256_file(result_path),
        "transition_checkpoint_paths": [str(path.resolve()) for path in transition_checkpoint_paths],
    }
    _write_json(runtime_path, runtime)
    print(
        json.dumps(
            {
                "status": "GATE7_ROUTING_BANDWIDTH_CONTINUATION_COMPLETE",
                "campaign_outcome": campaign_outcome,
                "completed_populations": completed_populations,
                "resource_frontier_population": resource_frontier_population,
                "k_required_by_population": k_required_by_population,
                "passing_k_by_population": passing_k_by_population,
                "second_continuation_opened": False,
                "result": str(result_path),
                "result_sha256": runtime["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transition-checkpoint0", type=Path, required=True)
    parser.add_argument("--transition-checkpoint1", type=Path, required=True)
    parser.add_argument("--transition-checkpoint2", type=Path, required=True)
    args = parser.parse_args()
    return run_gate7_high_scale_routing_bandwidth_continuation(
        output_root=args.output_root,
        transition_checkpoint_paths=_checkpoint_paths(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
