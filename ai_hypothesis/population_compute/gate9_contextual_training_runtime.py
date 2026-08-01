"""Exact Gate-9 seed trainer and immutable validation evidence writer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parent
_ARCHITECTURE_PATH = _ROOT / "gate9_contextual_worker_architecture.py"
_DATA_PATH = _ROOT / "gate9_contextual_training_data.py"


def _load(path: pathlib.Path, name: str):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate9 training dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


architecture = _load(
    _ARCHITECTURE_PATH, "gate9_training_runtime_architecture_dependency"
)
data = _load(_DATA_PATH, "gate9_training_runtime_data_dependency")
protocol = data.protocol

GATE9_TRAINING_EXECUTION_VERSION = "gate9-contextual-training-execution-v0"
GATE9_TRAINING_SEED_STATUS = "G9_CONTEXTUAL_TRAINING_SEED_COMPLETE"
GATE9_TRAINING_EXECUTION_BRANCH = "agent/gate9-contextual-training-execution-v0"
GATE9_TRAINING_PROTOCOL_HEAD = "1228c19cbf85da4ab738c3355c58f946cd6a965c"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _valid_head(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("Gate9 training execution head is malformed")


def _checkpoint_payload(model: torch.nn.Module, seed: int) -> dict[str, Any]:
    data.validate_model_state(model)
    payload = {
        "experiment_version": GATE9_TRAINING_EXECUTION_VERSION,
        "architecture_head": protocol.GATE9_ARCHITECTURE_HEAD,
        "training_protocol_head": GATE9_TRAINING_PROTOCOL_HEAD,
        "seed": seed,
        "initialization_seed": protocol.GATE9_INITIALIZATION_SEEDS[seed],
        "step": protocol.GATE9_CHECKPOINT_STEP,
        "train_episodes": protocol.GATE9_TRAIN_EPISODES,
        "learned_parameter_count": protocol.GATE9_LEARNED_PARAMETER_COUNT,
        "tensor_count": protocol.GATE9_STATE_TENSOR_COUNT,
        "state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
    }
    if set(payload) != set(protocol.GATE9_CHECKPOINT_REQUIRED_FIELDS):
        raise RuntimeError("Gate9 selected checkpoint fields drifted")
    return payload


def run_training_seed(
    *, seed: int, output_root: pathlib.Path, execution_head: str
) -> dict[str, Any]:
    if seed not in protocol.GATE9_CHECKPOINT_SEEDS:
        raise ValueError("Gate9 training seed is outside 0..2")
    _valid_head(execution_head)
    if output_root.exists():
        raise FileExistsError(f"Gate9 training output exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate9 training requires CUDA")

    data.configure_determinism(seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    output_root.mkdir(parents=True)
    seed_root = output_root / f"seed-{seed}"
    seed_root.mkdir()
    steps_path = seed_root / "train-steps.jsonl"
    validation_path = seed_root / "validation-per-episode.jsonl"
    checkpoint_path = seed_root / "selected-checkpoint.pt"
    summary_path = seed_root / "summary.json"

    model = architecture.Gate9ContextualWorker().to(
        device=device, dtype=torch.float32
    )
    data.validate_model_state(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.GATE9_BASE_LEARNING_RATE,
        betas=protocol.GATE9_ADAM_BETAS,
        eps=protocol.GATE9_ADAM_EPSILON,
        weight_decay=protocol.GATE9_WEIGHT_DECAY,
    )
    train_digest = hashlib.sha256()
    train_seen = np.zeros(
        protocol.GATE9_TRAIN_OPERATOR_COUNT, dtype=np.bool_
    )
    final_loss = math.nan
    started_training = time.perf_counter()
    with steps_path.open("w", encoding="utf-8", newline="\n") as handle:
        model.train()
        for step in range(1, protocol.GATE9_TRAIN_STEPS + 1):
            started = time.perf_counter()
            ordinals, counters, arrays = data.training_batch_arrays(seed, step)
            if bool(train_seen[ordinals].any()):
                raise RuntimeError("Gate9 training operator ordinal repeated")
            train_seen[ordinals] = True
            data.digest_update(train_digest, ordinals, counters, *arrays[:4])
            support_inputs, support_outputs, queries, targets, _ = (
                data.tensor_batch(arrays, device)
            )
            learning_rate = protocol.learning_rate_at_step(step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            logits = model(support_inputs, support_outputs, queries)
            loss = F.binary_cross_entropy_with_logits(
                logits, data.target_bits(targets)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Gate9 training loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.GATE9_GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    "Gate9 training gradient norm became non-finite"
                )
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            row = {
                "seed": seed,
                "step": step,
                "episodes_seen": step * 512,
                "learning_rate": learning_rate,
                "loss": final_loss,
                "pre_clip_gradient_norm": float(
                    gradient_norm.detach().cpu()
                ),
                "batch_operator_ordinal_sha256": hashlib.sha256(
                    ordinals.tobytes(order="C")
                ).hexdigest(),
                "batch_query_sha256": hashlib.sha256(
                    arrays[2].tobytes(order="C")
                ).hexdigest(),
                "wall_seconds": time.perf_counter() - started,
            }
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            if step == 1 or step % 32 == 0 or step == 512:
                print(
                    f"seed={seed} step={step:3d}/512 "
                    f"loss={final_loss:.8f} lr={learning_rate:.8g}",
                    flush=True,
                )
    training_seconds = time.perf_counter() - started_training
    if int(train_seen.sum()) != protocol.GATE9_TRAIN_OPERATOR_COUNT:
        raise RuntimeError("Gate9 training operator coverage is incomplete")
    data.validate_model_state(model)
    torch.save(_checkpoint_payload(model, seed), checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    metrics = _validate(
        model=model,
        device=device,
        validation_path=validation_path,
    )
    evidence = protocol.Gate9CheckpointValidationEvidence(
        seed=seed,
        initialization_seed=protocol.GATE9_INITIALIZATION_SEEDS[seed],
        checkpoint_step=512,
        train_episodes=262_144,
        unique_train_operators=int(train_seen.sum()),
        validation_episodes=32_768,
        unique_validation_operators=metrics[
            "unique_validation_operators"
        ],
        learned_parameter_count=19_649,
        tensor_count=17,
        checkpoint_sha256=checkpoint_sha256,
        parameters_finite=True,
        final_train_loss=final_loss,
        validation_exact_accuracy=metrics["exact_accuracy"],
        validation_bit_accuracy=metrics["bit_accuracy"],
        shuffled_context_accuracy=metrics["shuffled_accuracy"],
        query_only_accuracy=metrics["query_only_accuracy"],
        oracle_accuracy=metrics["oracle_accuracy"],
    )
    evidence.validate()
    summary = {
        "experiment_version": GATE9_TRAINING_EXECUTION_VERSION,
        "scientific_status": GATE9_TRAINING_SEED_STATUS,
        "execution_head": execution_head,
        "training_protocol_head": GATE9_TRAINING_PROTOCOL_HEAD,
        "architecture_head": protocol.GATE9_ARCHITECTURE_HEAD,
        "seed": seed,
        "validation_evidence": evidence.to_dict(),
        "training_episode_sha256": train_digest.hexdigest(),
        "validation_episode_sha256": metrics["episode_sha256"],
        "artifacts": {
            "train_steps": steps_path.name,
            "validation_per_episode": validation_path.name,
            "selected_checkpoint": checkpoint_path.name,
            "selected_checkpoint_sha256": checkpoint_sha256,
        },
        "execution": {
            "device": torch.cuda.get_device_name(0),
            "training_seconds": training_seconds,
            "validation_seconds": metrics["seconds"],
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "amp_enabled": False,
            "tf32_enabled": False,
            "compile_enabled": False,
        },
        "boundaries": {
            "training_performed": True,
            "validation_performed": True,
            "checkpoint_serialized": True,
            "local_test_operator_accessed": False,
            "graph_test_operator_accessed": False,
            "scientific_assignment_key_accessed": False,
            "scientific_test_generated": False,
            "scientific_execution_performed": False,
            "result_classification_performed": False,
        },
    }
    write_json(summary_path, summary)
    return summary


def _validate(
    *,
    model: torch.nn.Module,
    device: torch.device,
    validation_path: pathlib.Path,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    seen = np.zeros(
        protocol.GATE9_VALIDATION_OPERATOR_COUNT, dtype=np.bool_
    )
    exact = bits = shuffled = query_only = oracle = 0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad(), validation_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for batch_index in range(protocol.GATE9_VALIDATION_BATCHES):
            (
                indices,
                ordinals,
                counters,
                arrays,
                shuffled_ordinals,
                shuffled_outputs,
            ) = data.validation_batch_arrays(batch_index)
            if bool(seen[ordinals].any()):
                raise RuntimeError(
                    "Gate9 validation operator ordinal repeated"
                )
            seen[ordinals] = True
            data.digest_update(
                digest,
                ordinals,
                counters,
                *arrays,
                shuffled_ordinals,
                shuffled_outputs,
            )
            support_inputs, support_outputs, queries, targets, oracle_targets = (
                data.tensor_batch(arrays, device)
            )
            shuffled_outputs_tensor = torch.as_tensor(
                shuffled_outputs, dtype=torch.long, device=device
            )
            full_predictions = model.decode_bytes(
                model(support_inputs, support_outputs, queries)
            )
            shuffled_predictions = model.decode_bytes(
                model(support_inputs, shuffled_outputs_tensor, queries)
            )
            query_predictions = model.decode_bytes(
                model.forward_query_only(queries)
            )
            full_correct = full_predictions == targets
            shuffled_correct = shuffled_predictions == targets
            query_correct = query_predictions == targets
            oracle_correct = oracle_targets == targets
            full_bits = (
                (
                    full_predictions.unsqueeze(-1)
                    >> torch.arange(8, device=device)
                )
                & 1
            )
            exact += int(full_correct.sum().cpu())
            bits += int(
                (
                    full_bits
                    == data.target_bits(targets).to(torch.long)
                ).sum().cpu()
            )
            shuffled += int(shuffled_correct.sum().cpu())
            query_only += int(query_correct.sum().cpu())
            oracle += int(oracle_correct.sum().cpu())
            columns = tuple(
                tensor.cpu().tolist()
                for tensor in (
                    queries,
                    targets,
                    full_predictions,
                    shuffled_predictions,
                    query_predictions,
                    oracle_targets,
                )
            )
            for local_index, episode_index in enumerate(indices):
                (
                    query,
                    answer,
                    full,
                    shuffled_value,
                    query_value,
                    oracle_value,
                ) = (column[local_index] for column in columns)
                row = {
                    "episode_index": episode_index,
                    "operator_ordinal": int(ordinals[local_index]),
                    "operator_counter": int(counters[local_index]),
                    "query": int(query),
                    "answer": int(answer),
                    "full_prediction": int(full),
                    "shuffled_context_operator_ordinal": int(
                        shuffled_ordinals[local_index]
                    ),
                    "shuffled_context_prediction": int(shuffled_value),
                    "query_only_prediction": int(query_value),
                    "oracle_prediction": int(oracle_value),
                    "full_correct": bool(full == answer),
                    "shuffled_context_correct": bool(
                        shuffled_value == answer
                    ),
                    "query_only_correct": bool(query_value == answer),
                    "oracle_correct": bool(oracle_value == answer),
                }
                handle.write(
                    json.dumps(
                        row, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
    unique = int(seen.sum())
    if unique != protocol.GATE9_VALIDATION_OPERATOR_COUNT:
        raise RuntimeError("Gate9 validation operator coverage is incomplete")
    denominator = protocol.GATE9_VALIDATION_EPISODES
    return {
        "unique_validation_operators": unique,
        "exact_accuracy": exact / denominator,
        "bit_accuracy": bits / (denominator * 8),
        "shuffled_accuracy": shuffled / denominator,
        "query_only_accuracy": query_only / denominator,
        "oracle_accuracy": oracle / denominator,
        "episode_sha256": digest.hexdigest(),
        "seconds": time.perf_counter() - started,
    }
