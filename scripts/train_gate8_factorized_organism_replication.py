#!/usr/bin/env python3
"""Execute frozen Gate-8 v1 factorized replication seeds 1 or 2 on CUDA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_RUNNER_PATH = REPO_ROOT / "scripts/train_gate8_factorized_organism.py"

REPLICATION_EXECUTION_VERSION = (
    "gate8-factorized-message-training-replication-execution-v1"
)
QUALIFIED_SEED0_RESULT_HEAD = (
    "f259620f7d3beab2f886c76271c753e9ebf96dc9"
)
ALLOWED_REPLICATION_SEEDS = (1, 2)


def _load_base_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "gate8_v1_seed0_qualified_runner",
        BASE_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load qualified Gate8 v1 seed-0 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def train_gate8_v1_replication(
    *,
    seed: int,
    output_root: pathlib.Path,
    device_name: str,
) -> int:
    if seed not in ALLOWED_REPLICATION_SEEDS:
        raise ValueError(
            "Gate8 v1 replication execution admits seeds 1 and 2 only"
        )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    base = _load_base_runner()
    if device_name != "cuda":
        raise ValueError(
            "Gate8 v1 replication execution requires --device cuda"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Gate8 v1 replication training requires an available CUDA device"
        )
    device = torch.device("cuda", 0)
    base._configure_determinism(seed=seed, torch=torch)

    worlds = base._load(
        base.WORLD_PATH,
        f"gate8_v1_replication_worlds_seed_{seed}",
    )
    architecture = base._load(
        base.ARCHITECTURE_PATH,
        f"gate8_v1_replication_architecture_seed_{seed}",
    )
    protocol = base._load(
        base.PROTOCOL_PATH,
        f"gate8_v1_replication_protocol_seed_{seed}",
    )
    training = base._load(
        base.TRAINING_PATH,
        f"gate8_v1_replication_mechanics_seed_{seed}",
    )
    development_runtime = base._load(
        base.DEVELOPMENT_RUNTIME_PATH,
        f"gate8_v1_replication_development_runtime_seed_{seed}",
    )

    if (
        protocol.GATE8_V1_TRAINING_PROTOCOL_RUNTIME_HEAD
        != base.RUNTIME_HEAD
    ):
        raise RuntimeError("Gate8 v1 replication protocol runtime drifted")
    if (
        protocol.GATE8_V1_TRAINING_PROTOCOL_ARCHITECTURE_HEAD
        != base.ARCHITECTURE_HEAD
    ):
        raise RuntimeError(
            "Gate8 v1 replication protocol architecture drifted"
        )
    if (
        training.GATE8_V1_TRAINING_EXECUTION_PROTOCOL_HEAD
        != base.PROTOCOL_HEAD
    ):
        raise RuntimeError(
            "Gate8 v1 replication training-mechanics binding drifted"
        )
    if (
        development_runtime.GATE8_V1_DEVELOPMENT_RUNTIME_PROTOCOL_HEAD
        != base.PROTOCOL_HEAD
    ):
        raise RuntimeError(
            "Gate8 v1 replication development-protocol binding drifted"
        )
    if (
        development_runtime
        .GATE8_V1_DEVELOPMENT_RUNTIME_QUALIFIED_RUNTIME_HEAD
        != base.RUNTIME_HEAD
    ):
        raise RuntimeError(
            "Gate8 v1 replication development-runtime binding drifted"
        )

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Gate8 v1 replication output already exists: {output_root}"
        )
    output_root.mkdir(parents=True)
    checkpoints_root = output_root / "checkpoints"
    checkpoints_root.mkdir()

    model = architecture.Gate8V1SharedWorkerCore().to(
        device=device,
        dtype=torch.float32,
    )
    if sum(parameter.numel() for parameter in model.parameters()) != 19_649:
        raise RuntimeError(
            "Gate8 v1 replication model parameter count drifted"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.GATE8_V1_LEARNING_RATE,
        betas=protocol.GATE8_V1_ADAM_BETAS,
        eps=protocol.GATE8_V1_ADAM_EPSILON,
        weight_decay=protocol.GATE8_V1_WEIGHT_DECAY,
    )

    head = base._git_head()
    status = base._git_status()
    if status:
        raise RuntimeError(
            "Gate8 v1 replication runner requires a clean Git working tree"
        )

    environment = base._environment(torch=torch, device=device)
    validation_cache: dict[tuple[int, int], tuple[Any, ...]] | None = None
    telemetry: list[dict[str, Any]] = []
    candidates = []
    candidate_details = []
    checkpoint_records = []
    cumulative_inbox_codes: set[int] = set()
    cumulative_target_codes: set[int] = set()
    cumulative_target_carriers: set[int] = set()
    cumulative_target_symbols: set[int] = set()
    model.train()

    for step in range(1, protocol.GATE8_V1_OPTIMIZER_STEPS + 1):
        start_world = (
            step - 1
        ) * protocol.GATE8_V1_TRAINING_WORLD_BATCH_SIZE
        end_world = (
            start_world + protocol.GATE8_V1_TRAINING_WORLD_BATCH_SIZE
        )
        generated = []
        for global_world_index in range(start_world, end_world):
            address = protocol.gate8_v1_training_world_address(
                global_world_index
            )
            generated.append(
                worlds.generate_gate8_world(
                    split="train",
                    seed=seed,
                    world_index=address.condition_world_index,
                    population=address.population,
                    depth=address.depth,
                )
            )

        batch = training.collate_gate8_v1_local_batch(
            worlds=generated,
            transform_permutations=worlds.GATE8_TRANSFORM_PERMUTATIONS,
            protocol=protocol,
            device=device,
        )
        cumulative_inbox_codes.update(
            int(value) for value in batch.inbox_code.cpu().tolist()
        )
        cumulative_target_codes.update(
            int(value) for value in batch.message_target.cpu().tolist()
        )
        cumulative_target_carriers.update(
            int(value) for value in batch.carrier_target.cpu().tolist()
        )
        cumulative_target_symbols.update(
            int(value) for value in batch.symbol_target.cpu().tolist()
        )

        learning_rate = protocol.gate8_v1_learning_rate(step)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()

        output = training.gate8_v1_local_forward(model, batch)
        losses = training.gate8_v1_local_loss(
            output=output,
            batch=batch,
            protocol=protocol,
        )
        losses.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            protocol.GATE8_V1_GRADIENT_CLIP_NORM,
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        duration = time.perf_counter() - started

        row = {
            "step": step,
            "global_world_start": start_world,
            "global_world_end_exclusive": end_world,
            "world_count": batch.world_count,
            "edge_count": batch.edge_count,
            "learning_rate": learning_rate,
            "gradient_norm_before_clip": float(
                gradient_norm.detach().item()
            ),
            "duration_seconds": duration,
            "cumulative_inbox_code_coverage": len(
                cumulative_inbox_codes
            ),
            "cumulative_target_code_coverage": len(
                cumulative_target_codes
            ),
            "cumulative_target_carrier_coverage": len(
                cumulative_target_carriers
            ),
            "cumulative_target_symbol_coverage": len(
                cumulative_target_symbols
            ),
            **losses.detached_metrics(),
        }
        telemetry.append(row)
        if (
            step == 1
            or step % 16 == 0
            or step in protocol.GATE8_V1_CHECKPOINT_STEPS
        ):
            print(json.dumps(row, sort_keys=True), flush=True)

        if step in protocol.GATE8_V1_CHECKPOINT_STEPS:
            checkpoint_path = checkpoints_root / f"step-{step:04d}.pt"
            torch.save(
                base._checkpoint_payload(
                    model=model,
                    seed=seed,
                    step=step,
                ),
                checkpoint_path,
            )
            checkpoint_hash = base._sha256(checkpoint_path)
            if validation_cache is None:
                validation_cache = base._build_validation_cache(
                    worlds=worlds,
                    protocol=protocol,
                    seed=seed,
                )
            candidate, details = base._validate_candidate(
                model=model,
                cache=validation_cache,
                worlds=worlds,
                protocol=protocol,
                training=training,
                development_runtime=development_runtime,
                device=device,
                step=step,
            )
            candidates.append(candidate)
            details["checkpoint"] = str(
                checkpoint_path.relative_to(output_root)
            ).replace("\\", "/")
            details["checkpoint_sha256"] = checkpoint_hash
            candidate_details.append(details)
            checkpoint_records.append(
                {
                    "step": step,
                    "path": details["checkpoint"],
                    "sha256": checkpoint_hash,
                }
            )
            print(
                json.dumps(
                    {
                        "checkpoint_step": step,
                        "checkpoint_sha256": checkpoint_hash,
                        **candidate.to_dict(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    candidate_tuple = tuple(candidates)
    selected = protocol.select_gate8_v1_checkpoint(candidate_tuple)
    outcome = protocol.classify_gate8_v1_training(candidate_tuple)
    selected_record = next(
        record
        for record in checkpoint_records
        if record["step"] == selected.step
    )
    selected_source = output_root / selected_record["path"]
    selected_path = output_root / "selected-checkpoint.pt"
    shutil.copy2(selected_source, selected_path)
    selected_hash = base._sha256(selected_path)
    if selected_hash != selected_record["sha256"]:
        raise RuntimeError(
            "Gate8 v1 replication selected checkpoint copy changed bytes"
        )

    result = {
        "experiment_version": base.EXPERIMENT_VERSION,
        "replication_execution_version": REPLICATION_EXECUTION_VERSION,
        "qualified_seed0_result_head": QUALIFIED_SEED0_RESULT_HEAD,
        "scientific_status": outcome,
        "source_head": head,
        "protocol_head": base.PROTOCOL_HEAD,
        "runtime_head": base.RUNTIME_HEAD,
        "architecture_head": base.ARCHITECTURE_HEAD,
        "seed": seed,
        "environment": environment,
        "learned_parameter_count": 19_649,
        "training_worlds": protocol.GATE8_V1_TRAINING_WORLDS_PER_SEED,
        "optimizer_steps": protocol.GATE8_V1_OPTIMIZER_STEPS,
        "training_conditions": [
            list(condition)
            for condition in protocol.GATE8_V1_TRAINING_CONDITIONS
        ],
        "validation": {
            "namespace": "validation",
            "world_index_start": 512,
            "world_index_end_inclusive": 1_023,
            "worlds_per_condition": 512,
            "disjoint_from_v0_indices_0_through_511": True,
        },
        "optimizer": {
            "name": protocol.GATE8_V1_OPTIMIZER,
            "initial_learning_rate": protocol.GATE8_V1_LEARNING_RATE,
            "minimum_learning_rate": (
                protocol.GATE8_V1_MINIMUM_LEARNING_RATE
            ),
            "warmup_steps": protocol.GATE8_V1_WARMUP_STEPS,
            "betas": list(protocol.GATE8_V1_ADAM_BETAS),
            "epsilon": protocol.GATE8_V1_ADAM_EPSILON,
            "weight_decay": protocol.GATE8_V1_WEIGHT_DECAY,
            "gradient_clip_norm": (
                protocol.GATE8_V1_GRADIENT_CLIP_NORM
            ),
        },
        "loss_weights": {
            "carrier_cross_entropy": (
                protocol.GATE8_V1_CARRIER_LOSS_WEIGHT
            ),
            "symbol_cross_entropy": (
                protocol.GATE8_V1_SYMBOL_LOSS_WEIGHT
            ),
        },
        "training_code_coverage": {
            "inbox": sorted(cumulative_inbox_codes),
            "target": sorted(cumulative_target_codes),
            "target_carrier": sorted(cumulative_target_carriers),
            "target_symbol": sorted(cumulative_target_symbols),
        },
        "telemetry": telemetry,
        "checkpoint_candidates": candidate_details,
        "selected_checkpoint": {
            "step": selected.step,
            "path": "selected-checkpoint.pt",
            "sha256": selected_hash,
            "admitted": selected.admitted(),
            "selection_metrics": selected.to_dict(),
        },
        "training_performed": True,
        "validation_performed": True,
        "scientific_test_worlds_generated": False,
        "reference_tokenizer_loaded": False,
        "reference_model_weights_loaded": False,
        "reference_inference_performed": False,
        "seeds_1_and_2_executed": True,
    }

    result_path = output_root / (
        "gate8-factorized-organism-training-result.json"
    )
    base._write_json(result_path, result)
    print(
        json.dumps(
            {
                "status": outcome,
                "seed": seed,
                "selected_step": selected.step,
                "selected_checkpoint_sha256": selected_hash,
                "result": str(result_path),
                "result_sha256": base._sha256(result_path),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=ALLOWED_REPLICATION_SEEDS,
    )
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args()
    return train_gate8_v1_replication(
        seed=args.seed,
        output_root=args.output_root,
        device_name=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
