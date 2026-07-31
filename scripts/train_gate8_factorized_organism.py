#!/usr/bin/env python3
"""Execute one frozen Gate-8 v1 factorized organism training seed on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_distributed_transformation_worlds.py"
)
ARCHITECTURE_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_factorized_organism_architecture.py"
)
PROTOCOL_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_factorized_organism_training_protocol.py"
)
TRAINING_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_factorized_organism_training.py"
)
DEVELOPMENT_RUNTIME_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_factorized_organism_development_runtime.py"
)

PROTOCOL_HEAD = "a33dc123d090268a531d112251ea3ab53cb50062"
RUNTIME_HEAD = "333d88ac4fc52f1651741fba224e0b4605feedd3"
ARCHITECTURE_HEAD = "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8"
EXPERIMENT_VERSION = "gate8-factorized-message-training-execution-v1"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 v1 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
    )


def _checkpoint_payload(
    *,
    model: Any,
    seed: int,
    step: int,
) -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "protocol_head": PROTOCOL_HEAD,
        "runtime_head": RUNTIME_HEAD,
        "architecture_head": ARCHITECTURE_HEAD,
        "seed": seed,
        "step": step,
        "learned_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
    }


def _configure_determinism(*, seed: int, torch: Any) -> None:
    expected_hash_seed = str(seed)
    if os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
        raise RuntimeError(
            f"PYTHONHASHSEED must equal the training seed ({expected_hash_seed})"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _environment(*, torch: Any, device: Any) -> dict[str, Any]:
    index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )
    properties = torch.cuda.get_device_properties(index)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(index)),
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def _build_validation_cache(
    *,
    worlds: Any,
    protocol: Any,
    seed: int,
) -> dict[tuple[int, int], tuple[Any, ...]]:
    indices = protocol.gate8_v1_validation_world_indices()
    if indices != tuple(range(512, 1_024)):
        raise RuntimeError("Gate8 v1 validation index range drifted")
    cache: dict[tuple[int, int], tuple[Any, ...]] = {}
    for population, depth in protocol.GATE8_V1_TRAINING_CONDITIONS:
        cache[(population, depth)] = tuple(
            worlds.generate_gate8_world(
                split="validation",
                seed=seed,
                world_index=world_index,
                population=population,
                depth=depth,
            )
            for world_index in indices
        )
    return cache


def _validate_candidate(
    *,
    model: Any,
    cache: dict[tuple[int, int], tuple[Any, ...]],
    worlds: Any,
    protocol: Any,
    training: Any,
    development_runtime: Any,
    device: Any,
    step: int,
) -> tuple[Any, dict[str, Any]]:
    model.eval()
    condition_rows = []
    local_rows = []
    condition_details = []
    validation_batch_worlds = 64
    for population, depth in protocol.GATE8_V1_TRAINING_CONDITIONS:
        generated_worlds = cache[(population, depth)]
        correct = 0
        reached = 0
        updates = 0
        emitted = 0
        messages = 0
        bits = 0
        for generated in generated_worlds:
            result = development_runtime.run_gate8_v1_development_runtime(
                model=model,
                world=generated.public,
            )
            reached += int(result.target_reached)
            correct += int(
                result.target_reached
                and result.predicted_symbol == generated.truth.answer_symbol
            )
            updates += result.recurrent_updates
            emitted += result.emitted_messages
            messages += result.delivered_messages
            bits += result.communicated_bits
        target_accuracy = correct / len(generated_worlds)
        condition_rows.append(
            protocol.Gate8V1ValidationConditionRow(
                population=population,
                depth=depth,
                target_accuracy=target_accuracy,
            )
        )
        condition_details.append(
            {
                "population": population,
                "depth": depth,
                "world_index_start": 512,
                "world_index_end_inclusive": 1_023,
                "worlds": len(generated_worlds),
                "target_reached": reached,
                "target_correct": correct,
                "target_accuracy": target_accuracy,
                "recurrent_updates": updates,
                "emitted_messages": emitted,
                "delivered_messages": messages,
                "communicated_bits": bits,
            }
        )
        for start in range(
            0,
            len(generated_worlds),
            validation_batch_worlds,
        ):
            batch = training.collate_gate8_v1_local_batch(
                worlds=generated_worlds[
                    start : start + validation_batch_worlds
                ],
                transform_permutations=(
                    worlds.GATE8_TRANSFORM_PERMUTATIONS
                ),
                protocol=protocol,
                device=device,
            )
            local_rows.append(
                training.evaluate_gate8_v1_local_batch(
                    model=model,
                    batch=batch,
                    protocol=protocol,
                )
            )
    merged = training.merge_gate8_v1_local_evaluations(local_rows)
    candidate = protocol.Gate8V1CheckpointCandidate(
        step=step,
        conditions=tuple(condition_rows),
        exact_message_accuracy=(
            merged.exact_message_correct / merged.edge_count
        ),
        carrier_accuracy=merged.carrier_correct / merged.edge_count,
        symbol_accuracy=merged.symbol_correct / merged.edge_count,
        validation_loss=merged.total_loss_sum / merged.edge_count,
        inbox_code_coverage=len(merged.inbox_codes),
        target_code_coverage=len(merged.target_codes),
        target_carrier_coverage=len(merged.target_carriers),
        target_symbol_coverage=len(merged.target_symbols),
    )
    candidate.validate()
    details = {
        "candidate": candidate.to_dict(),
        "edge_count": merged.edge_count,
        "carrier_loss": merged.carrier_loss_sum / merged.edge_count,
        "symbol_loss": merged.symbol_loss_sum / merged.edge_count,
        "condition_details": condition_details,
        "inbox_codes": sorted(merged.inbox_codes),
        "target_codes": sorted(merged.target_codes),
        "target_carriers": sorted(merged.target_carriers),
        "target_symbols": sorted(merged.target_symbols),
    }
    model.train()
    return candidate, details


def train_gate8_v1_organism(
    *,
    seed: int,
    output_root: pathlib.Path,
    device_name: str,
) -> int:
    if seed not in (0, 1, 2):
        raise ValueError("Gate8 v1 training seed must be 0, 1 or 2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if device_name != "cuda":
        raise ValueError(
            "Gate8 v1 frozen training execution requires --device cuda"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Gate8 v1 training requires an available CUDA device")
    device = torch.device("cuda", 0)
    _configure_determinism(seed=seed, torch=torch)

    worlds = _load(WORLD_PATH, "gate8_v1_training_worlds")
    architecture = _load(
        ARCHITECTURE_PATH,
        "gate8_v1_training_architecture",
    )
    protocol = _load(PROTOCOL_PATH, "gate8_v1_training_protocol")
    training = _load(TRAINING_PATH, "gate8_v1_training_mechanics")
    development_runtime = _load(
        DEVELOPMENT_RUNTIME_PATH,
        "gate8_v1_training_development_runtime",
    )

    if protocol.GATE8_V1_TRAINING_PROTOCOL_RUNTIME_HEAD != RUNTIME_HEAD:
        raise RuntimeError("Gate8 v1 frozen protocol runtime binding drifted")
    if (
        protocol.GATE8_V1_TRAINING_PROTOCOL_ARCHITECTURE_HEAD
        != ARCHITECTURE_HEAD
    ):
        raise RuntimeError(
            "Gate8 v1 frozen protocol architecture binding drifted"
        )
    if training.GATE8_V1_TRAINING_EXECUTION_PROTOCOL_HEAD != PROTOCOL_HEAD:
        raise RuntimeError("Gate8 v1 training mechanics binding drifted")
    if (
        development_runtime.GATE8_V1_DEVELOPMENT_RUNTIME_PROTOCOL_HEAD
        != PROTOCOL_HEAD
    ):
        raise RuntimeError("Gate8 v1 development protocol binding drifted")
    if (
        development_runtime.GATE8_V1_DEVELOPMENT_RUNTIME_QUALIFIED_RUNTIME_HEAD
        != RUNTIME_HEAD
    ):
        raise RuntimeError("Gate8 v1 development runtime binding drifted")

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Gate8 v1 training output already exists: {output_root}"
        )
    output_root.mkdir(parents=True)
    checkpoints_root = output_root / "checkpoints"
    checkpoints_root.mkdir()

    model = architecture.Gate8V1SharedWorkerCore().to(
        device=device,
        dtype=torch.float32,
    )
    if sum(parameter.numel() for parameter in model.parameters()) != 19_649:
        raise RuntimeError("Gate8 v1 training model parameter count drifted")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.GATE8_V1_LEARNING_RATE,
        betas=protocol.GATE8_V1_ADAM_BETAS,
        eps=protocol.GATE8_V1_ADAM_EPSILON,
        weight_decay=protocol.GATE8_V1_WEIGHT_DECAY,
    )

    head = _git_head()
    status = _git_status()
    if status:
        raise RuntimeError(
            "Gate8 v1 training runner requires a clean Git working tree"
        )
    environment = _environment(torch=torch, device=device)
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
                _checkpoint_payload(
                    model=model,
                    seed=seed,
                    step=step,
                ),
                checkpoint_path,
            )
            checkpoint_hash = _sha256(checkpoint_path)
            if validation_cache is None:
                validation_cache = _build_validation_cache(
                    worlds=worlds,
                    protocol=protocol,
                    seed=seed,
                )
            candidate, details = _validate_candidate(
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
    selected_hash = _sha256(selected_path)
    if selected_hash != selected_record["sha256"]:
        raise RuntimeError("Gate8 v1 selected checkpoint copy changed bytes")

    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "scientific_status": outcome,
        "source_head": head,
        "protocol_head": PROTOCOL_HEAD,
        "runtime_head": RUNTIME_HEAD,
        "architecture_head": ARCHITECTURE_HEAD,
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
        "seeds_1_and_2_executed": False if seed == 0 else None,
    }
    result_path = output_root / "gate8-factorized-organism-training-result.json"
    _write_json(result_path, result)
    print(
        json.dumps(
            {
                "status": outcome,
                "seed": seed,
                "selected_step": selected.step,
                "selected_checkpoint_sha256": selected_hash,
                "result": str(result_path),
                "result_sha256": _sha256(result_path),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args()
    return train_gate8_v1_organism(
        seed=args.seed,
        output_root=args.output_root,
        device_name=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
