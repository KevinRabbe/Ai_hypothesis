#!/usr/bin/env python3
"""Execute the preregistered Gate-8 seed-0 causal diagnostic on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py"
ARCHITECTURE_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_organism_architecture.py"
BASE_PROTOCOL_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_organism_training_protocol.py"
TRAINING_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_organism_training.py"
DIAGNOSTIC_PROTOCOL_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_seed0_causal_diagnostic_protocol.py"
DIAGNOSTIC_RUNTIME_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_seed0_causal_diagnostic_runtime.py"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 diagnostic module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _configure_determinism(*, torch: Any) -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED must equal 0 for the seed-0 diagnostic")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _environment(*, torch: Any, device: Any) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
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
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def _validate_source_artifacts(
    *,
    checkpoint_path: pathlib.Path,
    result_path: pathlib.Path,
    manifest_path: pathlib.Path,
    diagnostic_protocol: Any,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (checkpoint_path, result_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Gate8 diagnostic source artifact is missing: {path}")
    observed = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "result_sha256": _sha256(result_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    expected = {
        "checkpoint_sha256": diagnostic_protocol.GATE8_SEED0_CHECKPOINT_SHA256,
        "result_sha256": diagnostic_protocol.GATE8_SEED0_RESULT_SHA256,
        "manifest_sha256": diagnostic_protocol.GATE8_SEED0_MANIFEST_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"Gate8 diagnostic source artifact hash mismatch: {observed} != {expected}"
        )
    source_result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if source_result.get("scientific_status") != "G8_TRAINING_CHECKPOINT_NOT_ADMITTED":
        raise ValueError("Gate8 diagnostic source result is not the frozen non-admission")
    if source_result.get("seed") != 0 or source_result.get("learned_parameter_count") != 19_649:
        raise ValueError("Gate8 diagnostic source result identity drifted")
    selected = source_result.get("selected_checkpoint", {})
    if selected.get("sha256") != expected["checkpoint_sha256"] or selected.get("step") != 1024:
        raise ValueError("Gate8 diagnostic selected checkpoint identity drifted")
    if source_result.get("source_head") != diagnostic_protocol.GATE8_TRAINING_EXECUTION_HEAD:
        raise ValueError("Gate8 diagnostic source execution head drifted")
    if source_result.get("scientific_test_worlds_generated") is not False:
        raise ValueError("Gate8 diagnostic source crossed the scientific-test boundary")
    if any(
        source_result.get(name) is not False
        for name in (
            "reference_tokenizer_loaded",
            "reference_model_weights_loaded",
            "reference_inference_performed",
        )
    ):
        raise ValueError("Gate8 diagnostic source crossed the reference-model boundary")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Gate8 diagnostic checkpoint payload is not a dictionary")
    for name, value in (
        ("experiment_version", "gate8-organism-training-execution-v0"),
        ("protocol_head", diagnostic_protocol.GATE8_TRAINING_PROTOCOL_HEAD),
        ("seed", 0),
        ("step", 1024),
        ("learned_parameter_count", 19_649),
    ):
        if checkpoint.get(name) != value:
            raise ValueError(f"Gate8 diagnostic checkpoint {name} drifted")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or len(state_dict) != 15:
        raise ValueError("Gate8 diagnostic checkpoint state dictionary drifted")
    if sum(int(tensor.numel()) for tensor in state_dict.values()) != 19_649:
        raise ValueError("Gate8 diagnostic checkpoint tensor ledger drifted")
    return source_result, checkpoint


def _new_model(*, architecture: Any, checkpoint: dict[str, Any], torch: Any, device: Any):
    model = architecture.Gate8SharedWorkerCore()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32)
    if sum(parameter.numel() for parameter in model.parameters()) != 19_649:
        raise RuntimeError("Gate8 diagnostic model parameter count drifted")
    return model


def _build_validation_cache(*, worlds: Any, diagnostic_protocol: Any) -> dict[tuple[int, int], tuple[Any, ...]]:
    return {
        (population, depth): tuple(
            worlds.generate_gate8_world(
                split="validation",
                seed=0,
                world_index=world_index,
                population=population,
                depth=depth,
            )
            for world_index in range(
                diagnostic_protocol.GATE8_VALIDATION_WORLDS_PER_CONDITION
            )
        )
        for population, depth in diagnostic_protocol.GATE8_VALIDATION_CONDITIONS
    }


def _runtime_probe(
    *,
    model: Any,
    cache: dict[tuple[int, int], tuple[Any, ...]],
    diagnostic_protocol: Any,
    diagnostic_runtime: Any,
    probe: str,
) -> tuple[Any, dict[str, Any]]:
    model.eval()
    condition_accuracies: list[float] = []
    details: list[dict[str, Any]] = []
    for population, depth in diagnostic_protocol.GATE8_VALIDATION_CONDITIONS:
        generated_worlds = cache[(population, depth)]
        correct = 0
        reached = 0
        updates = 0
        messages = 0
        bits = 0
        for generated in generated_worlds:
            result = diagnostic_runtime.run_gate8_seed0_diagnostic_runtime(
                model=model,
                world=generated.public,
                probe=probe,
            )
            reached += int(result.target_reached)
            correct += int(
                result.target_reached
                and result.predicted_symbol == generated.truth.answer_symbol
            )
            updates += result.recurrent_updates
            messages += result.delivered_messages
            bits += result.communicated_bits
        accuracy = correct / len(generated_worlds)
        condition_accuracies.append(accuracy)
        details.append(
            {
                "population": population,
                "depth": depth,
                "worlds": len(generated_worlds),
                "target_reached": reached,
                "target_correct": correct,
                "target_accuracy": accuracy,
                "recurrent_updates": updates,
                "delivered_messages": messages,
                "communicated_bits": bits,
            }
        )
    metrics = diagnostic_protocol.Gate8RuntimeProbeMetrics(
        probe=probe,
        mean_target_accuracy=sum(condition_accuracies) / len(condition_accuracies),
        minimum_target_accuracy=min(condition_accuracies),
        condition_target_accuracies=tuple(condition_accuracies),
    )
    metrics.validate()
    return metrics, {"metrics": {
        "probe": metrics.probe,
        "mean_target_accuracy": metrics.mean_target_accuracy,
        "minimum_target_accuracy": metrics.minimum_target_accuracy,
        "condition_target_accuracies": list(metrics.condition_target_accuracies),
    }, "condition_details": details}


def _local_metrics(
    *,
    model: Any,
    cache: dict[tuple[int, int], tuple[Any, ...]],
    worlds: Any,
    base_protocol: Any,
    training: Any,
    device: Any,
) -> dict[str, Any]:
    rows = []
    model.eval()
    for condition in base_protocol.GATE8_TRAINING_CONDITIONS:
        generated_worlds = cache[condition]
        for start in range(0, len(generated_worlds), 64):
            batch = training.collate_gate8_local_batch(
                worlds=generated_worlds[start : start + 64],
                transform_permutations=worlds.GATE8_TRANSFORM_PERMUTATIONS,
                protocol=base_protocol,
                device=device,
            )
            rows.append(
                training.evaluate_gate8_local_batch(
                    model=model,
                    batch=batch,
                    protocol=base_protocol,
                )
            )
    merged = training.merge_gate8_local_evaluations(rows)
    return {
        "edge_count": merged.edge_count,
        "message_accuracy": merged.message_correct / merged.edge_count,
        "answer_accuracy": merged.answer_correct / merged.edge_count,
        "activity_accuracy": merged.activity_correct / merged.edge_count,
        "validation_loss": merged.total_loss_sum / merged.edge_count,
        "message_loss": merged.message_loss_sum / merged.edge_count,
        "answer_loss": merged.answer_loss_sum / merged.edge_count,
        "activity_loss": merged.activity_loss_sum / merged.edge_count,
        "inbox_code_coverage": len(merged.inbox_codes),
        "target_code_coverage": len(merged.target_codes),
    }


def _training_probe_metrics(
    *,
    probe: str,
    step: int,
    model: Any,
    cache: dict[tuple[int, int], tuple[Any, ...]],
    worlds: Any,
    base_protocol: Any,
    diagnostic_protocol: Any,
    diagnostic_runtime: Any,
    training: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    local = _local_metrics(
        model=model,
        cache=cache,
        worlds=worlds,
        base_protocol=base_protocol,
        training=training,
        device=device,
    )
    runtime, runtime_details = _runtime_probe(
        model=model,
        cache=cache,
        diagnostic_protocol=diagnostic_protocol,
        diagnostic_runtime=diagnostic_runtime,
        probe="baseline",
    )
    invariance = diagnostic_runtime.evaluate_gate8_nonroot_target_root_invariance(
        model=model
    )
    metrics = diagnostic_protocol.Gate8TrainingProbeMetrics(
        probe=probe,
        step=step,
        message_accuracy=local["message_accuracy"],
        answer_accuracy=local["answer_accuracy"],
        activity_accuracy=local["activity_accuracy"],
        mean_target_accuracy=runtime.mean_target_accuracy,
        minimum_target_accuracy=runtime.minimum_target_accuracy,
        message_root_invariance=invariance.message_root_invariance,
        answer_root_invariance=invariance.answer_root_invariance,
    )
    metrics.validate()
    return metrics, {
        "metrics": {
            "probe": metrics.probe,
            "step": metrics.step,
            "message_accuracy": metrics.message_accuracy,
            "answer_accuracy": metrics.answer_accuracy,
            "activity_accuracy": metrics.activity_accuracy,
            "mean_target_accuracy": metrics.mean_target_accuracy,
            "minimum_target_accuracy": metrics.minimum_target_accuracy,
            "message_root_invariance": metrics.message_root_invariance,
            "answer_root_invariance": metrics.answer_root_invariance,
        },
        "local": local,
        "runtime": runtime_details,
        "root_invariance": {
            "cases": invariance.cases,
            "message_invariant_cases": invariance.message_invariant_cases,
            "answer_invariant_cases": invariance.answer_invariant_cases,
            "activity_invariant_cases": invariance.activity_invariant_cases,
            "message_root_invariance": invariance.message_root_invariance,
            "answer_root_invariance": invariance.answer_root_invariance,
            "activity_root_invariance": invariance.activity_root_invariance,
        },
    }


def _training_worlds(
    *,
    global_start: int,
    worlds: Any,
    diagnostic_protocol: Any,
) -> list[Any]:
    generated = []
    conditions = diagnostic_protocol.GATE8_VALIDATION_CONDITIONS
    for global_world_index in range(
        global_start,
        global_start + diagnostic_protocol.GATE8_WORLD_BATCH_SIZE,
    ):
        condition_index = global_world_index % len(conditions)
        population, depth = conditions[condition_index]
        generated.append(
            worlds.generate_gate8_world(
                split="train",
                seed=0,
                world_index=global_world_index // len(conditions),
                population=population,
                depth=depth,
            )
        )
    return generated


def _checkpoint_payload(
    *,
    model: Any,
    probe: str,
    step: int,
    trainable_parameter_count: int,
    diagnostic_protocol: Any,
) -> dict[str, Any]:
    return {
        "experiment_version": "gate8-seed0-causal-diagnostic-execution-v0",
        "diagnostic_protocol_head": diagnostic_protocol.GATE8_SEED0_RESULT_HEAD,
        "source_checkpoint_sha256": diagnostic_protocol.GATE8_SEED0_CHECKPOINT_SHA256,
        "seed": 0,
        "probe": probe,
        "step": step,
        "learned_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": trainable_parameter_count,
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
    }


def _full_resume_learning_rate(*, step: int, diagnostic_protocol: Any) -> float:
    if not 1 <= step <= diagnostic_protocol.GATE8_FULL_RESUME_STEPS:
        raise ValueError("Gate8 full-resume step is outside its frozen schedule")
    if diagnostic_protocol.GATE8_FULL_RESUME_STEPS == 1:
        return diagnostic_protocol.GATE8_FULL_RESUME_MINIMUM_LEARNING_RATE
    progress = (step - 1) / (diagnostic_protocol.GATE8_FULL_RESUME_STEPS - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return diagnostic_protocol.GATE8_FULL_RESUME_MINIMUM_LEARNING_RATE + (
        diagnostic_protocol.GATE8_FULL_RESUME_INITIAL_LEARNING_RATE
        - diagnostic_protocol.GATE8_FULL_RESUME_MINIMUM_LEARNING_RATE
    ) * cosine


def _run_training_probe(
    *,
    probe: str,
    model: Any,
    output_root: pathlib.Path,
    world_start: int,
    steps: int,
    checkpoint_steps: tuple[int, ...],
    cache: dict[tuple[int, int], tuple[Any, ...]],
    worlds: Any,
    base_protocol: Any,
    diagnostic_protocol: Any,
    diagnostic_runtime: Any,
    training: Any,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    if probe == "head_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(
                name.startswith(prefix)
                for prefix in diagnostic_protocol.GATE8_HEAD_ONLY_TRAINABLE_PREFIXES
            )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=diagnostic_protocol.GATE8_HEAD_ONLY_LEARNING_RATE,
            betas=diagnostic_protocol.GATE8_HEAD_ONLY_BETAS,
            eps=diagnostic_protocol.GATE8_HEAD_ONLY_EPSILON,
            weight_decay=diagnostic_protocol.GATE8_HEAD_ONLY_WEIGHT_DECAY,
        )
        clip_norm = diagnostic_protocol.GATE8_HEAD_ONLY_GRADIENT_CLIP_NORM
    elif probe == "full_resume":
        for parameter in model.parameters():
            parameter.requires_grad = True
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(
            trainable,
            lr=diagnostic_protocol.GATE8_FULL_RESUME_INITIAL_LEARNING_RATE,
            betas=diagnostic_protocol.GATE8_FULL_RESUME_BETAS,
            eps=diagnostic_protocol.GATE8_FULL_RESUME_EPSILON,
            weight_decay=diagnostic_protocol.GATE8_FULL_RESUME_WEIGHT_DECAY,
        )
        clip_norm = diagnostic_protocol.GATE8_FULL_RESUME_GRADIENT_CLIP_NORM
    else:
        raise ValueError("Gate8 diagnostic training probe is unknown")

    trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
    if probe == "head_only" and trainable_parameter_count != 9_009:
        raise RuntimeError("Gate8 diagnostic head-only trainable ledger drifted")
    if probe == "full_resume" and trainable_parameter_count != 19_649:
        raise RuntimeError("Gate8 diagnostic full-resume trainable ledger drifted")

    checkpoints_root = output_root / "checkpoints"
    checkpoints_root.mkdir(parents=True)
    telemetry: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    final_metrics = None
    model.train()

    for step in range(1, steps + 1):
        global_start = world_start + (step - 1) * diagnostic_protocol.GATE8_WORLD_BATCH_SIZE
        generated = _training_worlds(
            global_start=global_start,
            worlds=worlds,
            diagnostic_protocol=diagnostic_protocol,
        )
        batch = training.collate_gate8_local_batch(
            worlds=generated,
            transform_permutations=worlds.GATE8_TRANSFORM_PERMUTATIONS,
            protocol=base_protocol,
            device=device,
        )
        learning_rate = (
            diagnostic_protocol.GATE8_HEAD_ONLY_LEARNING_RATE
            if probe == "head_only"
            else _full_resume_learning_rate(
                step=step,
                diagnostic_protocol=diagnostic_protocol,
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = training.gate8_local_forward(model, batch)
        losses = training.gate8_local_loss(
            output=output,
            batch=batch,
            protocol=base_protocol,
        )
        losses.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, clip_norm)
        optimizer.step()
        torch.cuda.synchronize(device)
        row = {
            "probe": probe,
            "step": step,
            "global_world_start": global_start,
            "global_world_end_exclusive": (
                global_start + diagnostic_protocol.GATE8_WORLD_BATCH_SIZE
            ),
            "world_count": batch.world_count,
            "edge_count": batch.edge_count,
            "learning_rate": learning_rate,
            "gradient_norm_before_clip": float(gradient_norm.detach().item()),
            "duration_seconds": time.perf_counter() - started,
            **losses.detached_metrics(),
        }
        telemetry.append(row)
        if step == 1 or step % 16 == 0 or step in checkpoint_steps:
            print(json.dumps(row, sort_keys=True), flush=True)

        if step in checkpoint_steps:
            model.eval()
            checkpoint_path = checkpoints_root / f"step-{step:04d}.pt"
            torch.save(
                _checkpoint_payload(
                    model=model,
                    probe=probe,
                    step=step,
                    trainable_parameter_count=trainable_parameter_count,
                    diagnostic_protocol=diagnostic_protocol,
                ),
                checkpoint_path,
            )
            metrics, details = _training_probe_metrics(
                probe=probe,
                step=step,
                model=model,
                cache=cache,
                worlds=worlds,
                base_protocol=base_protocol,
                diagnostic_protocol=diagnostic_protocol,
                diagnostic_runtime=diagnostic_runtime,
                training=training,
                device=device,
            )
            record = {
                "step": step,
                "path": str(checkpoint_path.relative_to(output_root)).replace("\\", "/"),
                "sha256": _sha256(checkpoint_path),
                "details": details,
            }
            checkpoint_records.append(record)
            print(
                json.dumps(
                    {
                        "probe": probe,
                        "checkpoint_step": step,
                        "checkpoint_sha256": record["sha256"],
                        **details["metrics"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if step == steps:
                final_metrics = metrics
            model.train()

    if final_metrics is None or tuple(record["step"] for record in checkpoint_records) != checkpoint_steps:
        raise RuntimeError("Gate8 diagnostic did not preserve every checkpoint")
    _write_json(output_root / "telemetry.json", telemetry)
    _write_json(output_root / "checkpoint-metrics.json", checkpoint_records)
    return {
        "probe": probe,
        "steps": steps,
        "world_start": world_start,
        "world_end_exclusive": (
            world_start + steps * diagnostic_protocol.GATE8_WORLD_BATCH_SIZE
        ),
        "trainable_parameter_count": trainable_parameter_count,
        "telemetry": telemetry,
        "checkpoints": checkpoint_records,
        "final_metrics_object": final_metrics,
        "final_metrics": checkpoint_records[-1]["details"]["metrics"],
    }


def diagnose_gate8_seed0(
    *,
    checkpoint_path: pathlib.Path,
    source_result_path: pathlib.Path,
    source_manifest_path: pathlib.Path,
    output_root: pathlib.Path,
    device_name: str,
) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if device_name != "cuda":
        raise ValueError("Gate8 seed-0 diagnostic requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate8 seed-0 diagnostic requires an available CUDA device")
    device = torch.device("cuda", 0)
    _configure_determinism(torch=torch)

    worlds = _load(WORLD_PATH, "gate8_seed0_diagnostic_worlds")
    architecture = _load(ARCHITECTURE_PATH, "gate8_seed0_diagnostic_architecture")
    base_protocol = _load(BASE_PROTOCOL_PATH, "gate8_seed0_diagnostic_base_protocol")
    training = _load(TRAINING_PATH, "gate8_seed0_diagnostic_training")
    diagnostic_protocol = _load(
        DIAGNOSTIC_PROTOCOL_PATH,
        "gate8_seed0_diagnostic_protocol",
    )
    diagnostic_runtime = _load(
        DIAGNOSTIC_RUNTIME_PATH,
        "gate8_seed0_diagnostic_runtime",
    )

    if diagnostic_runtime.GATE8_SEED0_DIAGNOSTIC_PROTOCOL_HEAD != (
        "0fa9ec48c31b36c90d58da827139457fd812b98c"
    ):
        raise RuntimeError("Gate8 diagnostic runtime protocol binding drifted")
    if base_protocol.GATE8_ORGANISM_TRAINING_PROTOCOL_RUNTIME_HEAD != (
        diagnostic_protocol.GATE8_RUNTIME_HEAD
    ):
        raise RuntimeError("Gate8 diagnostic base runtime binding drifted")

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Gate8 diagnostic output already exists: {output_root}")
    output_root.mkdir(parents=True)

    source_result, checkpoint = _validate_source_artifacts(
        checkpoint_path=checkpoint_path.resolve(),
        result_path=source_result_path.resolve(),
        manifest_path=source_manifest_path.resolve(),
        diagnostic_protocol=diagnostic_protocol,
        torch=torch,
    )
    head = _git_head()
    if _git_status():
        raise RuntimeError("Gate8 diagnostic runner requires a clean Git working tree")
    environment = _environment(torch=torch, device=device)
    cache = _build_validation_cache(
        worlds=worlds,
        diagnostic_protocol=diagnostic_protocol,
    )

    original_model = _new_model(
        architecture=architecture,
        checkpoint=checkpoint,
        torch=torch,
        device=device,
    )
    original_model.eval()
    runtime_rows = []
    runtime_details = []
    for probe in diagnostic_protocol.GATE8_RUNTIME_PROBES:
        metrics, details = _runtime_probe(
            model=original_model,
            cache=cache,
            diagnostic_protocol=diagnostic_protocol,
            diagnostic_runtime=diagnostic_runtime,
            probe=probe,
        )
        runtime_rows.append(metrics)
        runtime_details.append(details)
        print(json.dumps(details["metrics"], sort_keys=True), flush=True)

    baseline_local = _local_metrics(
        model=original_model,
        cache=cache,
        worlds=worlds,
        base_protocol=base_protocol,
        training=training,
        device=device,
    )
    baseline_invariance = diagnostic_runtime.evaluate_gate8_nonroot_target_root_invariance(
        model=original_model
    )
    baseline_runtime = runtime_rows[0]
    exact_baselines = (
        (baseline_local["message_accuracy"], diagnostic_protocol.GATE8_BASELINE_MESSAGE_ACCURACY),
        (baseline_local["answer_accuracy"], diagnostic_protocol.GATE8_BASELINE_ANSWER_ACCURACY),
        (baseline_local["activity_accuracy"], diagnostic_protocol.GATE8_BASELINE_ACTIVITY_ACCURACY),
        (baseline_runtime.mean_target_accuracy, diagnostic_protocol.GATE8_BASELINE_MEAN_TARGET_ACCURACY),
        (baseline_runtime.minimum_target_accuracy, diagnostic_protocol.GATE8_BASELINE_MIN_TARGET_ACCURACY),
        (baseline_invariance.message_root_invariance, diagnostic_protocol.GATE8_BASELINE_MESSAGE_ROOT_INVARIANCE),
        (baseline_invariance.answer_root_invariance, diagnostic_protocol.GATE8_BASELINE_ANSWER_ROOT_INVARIANCE),
    )
    if any(abs(observed - expected) > 1.0e-12 for observed, expected in exact_baselines):
        raise RuntimeError(f"Gate8 diagnostic failed to reproduce frozen baselines: {exact_baselines}")

    head_only_model = _new_model(
        architecture=architecture,
        checkpoint=checkpoint,
        torch=torch,
        device=device,
    )
    full_resume_model = _new_model(
        architecture=architecture,
        checkpoint=checkpoint,
        torch=torch,
        device=device,
    )
    head_only = _run_training_probe(
        probe="head_only",
        model=head_only_model,
        output_root=output_root / "head-only",
        world_start=diagnostic_protocol.GATE8_HEAD_ONLY_WORLD_START,
        steps=diagnostic_protocol.GATE8_HEAD_ONLY_STEPS,
        checkpoint_steps=diagnostic_protocol.GATE8_HEAD_ONLY_CHECKPOINT_STEPS,
        cache=cache,
        worlds=worlds,
        base_protocol=base_protocol,
        diagnostic_protocol=diagnostic_protocol,
        diagnostic_runtime=diagnostic_runtime,
        training=training,
        torch=torch,
        device=device,
    )
    full_resume = _run_training_probe(
        probe="full_resume",
        model=full_resume_model,
        output_root=output_root / "full-resume",
        world_start=diagnostic_protocol.GATE8_FULL_RESUME_WORLD_START,
        steps=diagnostic_protocol.GATE8_FULL_RESUME_STEPS,
        checkpoint_steps=diagnostic_protocol.GATE8_FULL_RESUME_CHECKPOINT_STEPS,
        cache=cache,
        worlds=worlds,
        base_protocol=base_protocol,
        diagnostic_protocol=diagnostic_protocol,
        diagnostic_runtime=diagnostic_runtime,
        training=training,
        torch=torch,
        device=device,
    )

    findings = diagnostic_protocol.gate8_classify_diagnostic(
        runtime_rows=tuple(runtime_rows),
        head_only=head_only["final_metrics_object"],
        full_resume=full_resume["final_metrics_object"],
    )
    result = {
        "experiment_version": "gate8-seed0-causal-diagnostic-execution-v0",
        "diagnostic_status": "G8_SEED0_CAUSAL_DIAGNOSTIC_COMPLETE",
        "source_head": head,
        "diagnostic_protocol_head": "0fa9ec48c31b36c90d58da827139457fd812b98c",
        "seed0_result_head": diagnostic_protocol.GATE8_SEED0_RESULT_HEAD,
        "training_execution_head": diagnostic_protocol.GATE8_TRAINING_EXECUTION_HEAD,
        "architecture_head": diagnostic_protocol.GATE8_ARCHITECTURE_HEAD,
        "runtime_head": diagnostic_protocol.GATE8_RUNTIME_HEAD,
        "training_protocol_head": diagnostic_protocol.GATE8_TRAINING_PROTOCOL_HEAD,
        "source_artifacts": {
            "checkpoint_sha256": diagnostic_protocol.GATE8_SEED0_CHECKPOINT_SHA256,
            "result_sha256": diagnostic_protocol.GATE8_SEED0_RESULT_SHA256,
            "manifest_sha256": diagnostic_protocol.GATE8_SEED0_MANIFEST_SHA256,
        },
        "source_scientific_status": source_result["scientific_status"],
        "seed": 0,
        "learned_parameter_count": 19_649,
        "environment": environment,
        "baseline_reproduced": True,
        "baseline_local": baseline_local,
        "baseline_root_invariance": {
            "message": baseline_invariance.message_root_invariance,
            "answer": baseline_invariance.answer_root_invariance,
            "activity": baseline_invariance.activity_root_invariance,
        },
        "runtime_probes": runtime_details,
        "head_only": {
            key: value
            for key, value in head_only.items()
            if key != "final_metrics_object"
        },
        "full_resume": {
            key: value
            for key, value in full_resume.items()
            if key != "final_metrics_object"
        },
        "findings": findings.to_dict(),
        "diagnostic_performed": True,
        "training_seed0_additional_worlds_generated": True,
        "training_seeds_1_2_performed": False,
        "scientific_test_worlds_generated": False,
        "reference_tokenizer_loaded": False,
        "reference_model_weights_loaded": False,
        "reference_inference_performed": False,
        "original_checkpoint_modified": False,
        "original_non_admission_changed": False,
    }
    result_path = output_root / "gate8-seed0-causal-diagnostic-result.json"
    _write_json(result_path, result)
    print(
        json.dumps(
            {
                "status": result["diagnostic_status"],
                "result": str(result_path),
                "result_sha256": _sha256(result_path),
                "findings": result["findings"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True, type=pathlib.Path)
    parser.add_argument("--source-result-path", required=True, type=pathlib.Path)
    parser.add_argument("--source-manifest-path", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return diagnose_gate8_seed0(
        checkpoint_path=args.checkpoint_path,
        source_result_path=args.source_result_path,
        source_manifest_path=args.source_manifest_path,
        output_root=args.output_root,
        device_name=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
