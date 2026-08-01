"""Gate-9 failure decomposition stage-1 execution runtime.

This runtime executes only ``single_operator_query_fit``.  It cannot select or
run later diagnostic stages and has no Gate-9 v0 scientific-world surface.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parent
_ARCHITECTURE_PATH = _ROOT / "gate9_contextual_worker_architecture.py"
_OPERATOR_PATH = _ROOT / "gate9_contextual_operator_contract.py"
_PROTOCOL_PATH = _ROOT / "gate9_contextual_failure_decomposition_protocol.py"


def _load(path: pathlib.Path, name: str):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate9D stage-1 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


architecture = _load(
    _ARCHITECTURE_PATH, "gate9d_stage1_architecture_dependency"
)
operators = _load(_OPERATOR_PATH, "gate9d_stage1_operator_dependency")
protocol = _load(_PROTOCOL_PATH, "gate9d_stage1_protocol_dependency")

GATE9D_STAGE1_EXECUTION_VERSION = (
    "gate9-contextual-failure-decomposition-stage1-execution-v0"
)
GATE9D_STAGE1_EXECUTION_STATUS = "G9D_STAGE1_SEED_COMPLETE"
GATE9D_STAGE1_EXECUTION_BRANCH = (
    "agent/gate9-contextual-failure-decomposition-stage1-execution-v0"
)
GATE9D_PROTOCOL_HEAD = "8deca15aef78d8636b07570aff044f9b7ae31928"
GATE9D_STAGE_NAME = "single_operator_query_fit"
GATE9D_STAGE_INDEX = 0
GATE9D_STAGE = protocol.GATE9D_STAGES[GATE9D_STAGE_INDEX]
GATE9D_TRAIN_EXAMPLES = 247
GATE9D_TRAIN_STEPS = 1_024
GATE9D_BATCH_SIZE = 247
GATE9D_STATE_TENSOR_SHAPES = {
    "support_slot_modulation": (9, 24),
    "output_scale": (),
    "pair_projection.weight": (48, 16),
    "pair_projection.bias": (48,),
    "query_projection.weight": (48, 8),
    "query_projection.bias": (48,),
    "support_attention.in_proj_weight": (144, 48),
    "support_attention.in_proj_bias": (144,),
    "support_attention.out_proj.weight": (48, 48),
    "support_attention.out_proj.bias": (48,),
    "support_ff_in.weight": (64, 48),
    "support_ff_in.bias": (64,),
    "support_ff_out.weight": (48, 64),
    "support_ff_out.bias": (48,),
    "query_support_fusion.weight": (24, 96),
    "query_support_fusion.bias": (24,),
    "output_bit_head.weight": (8, 24),
}

if GATE9D_STAGE.name != GATE9D_STAGE_NAME:
    raise RuntimeError("Gate9D stage-1 protocol identity drifted")
if GATE9D_STAGE.order != 1 or GATE9D_STAGE.requires_context_causality:
    raise RuntimeError("Gate9D stage-1 semantics drifted")
if GATE9D_STAGE.unseen_operator_evaluation:
    raise RuntimeError("Gate9D stage 1 cannot evaluate unseen operators")
if (
    GATE9D_STAGE.steps != GATE9D_TRAIN_STEPS
    or GATE9D_STAGE.batch_size != GATE9D_BATCH_SIZE
    or GATE9D_STAGE.train_examples != GATE9D_TRAIN_EXAMPLES
    or GATE9D_STAGE.evaluation_examples != GATE9D_TRAIN_EXAMPLES
):
    raise RuntimeError("Gate9D stage-1 schedule drifted")
if len(GATE9D_STAGE.train_operator_counters) != 1:
    raise RuntimeError("Gate9D stage 1 requires exactly one operator")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def learning_rate_at_step(step: int) -> float:
    if type(step) is not int or not 1 <= step <= GATE9D_TRAIN_STEPS:
        raise ValueError("Gate9D stage-1 step lies outside 1..1024")
    if step <= protocol.GATE9D_WARMUP_STEPS:
        return (
            protocol.GATE9D_BASE_LEARNING_RATE
            * step
            / protocol.GATE9D_WARMUP_STEPS
        )
    progress = (
        step - protocol.GATE9D_WARMUP_STEPS
    ) / (GATE9D_TRAIN_STEPS - protocol.GATE9D_WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return protocol.GATE9D_MIN_LEARNING_RATE + (
        protocol.GATE9D_BASE_LEARNING_RATE
        - protocol.GATE9D_MIN_LEARNING_RATE
    ) * cosine


def configure_determinism(seed_index: int) -> int:
    if seed_index not in (0, 1, 2):
        raise ValueError("Gate9D stage-1 seed index lies outside 0..2")
    initialization_seed = protocol.GATE9D_INITIALIZATION_SEEDS[seed_index]
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return initialization_seed


def stage1_material() -> dict[str, Any]:
    operator_counter = GATE9D_STAGE.train_operator_counters[0]
    operator = operators.operator_from_counter(operator_counter)
    support = operators.public_support_pairs(operator)
    queries = tuple(protocol.GATE9D_QUERY_VALUES)
    targets = tuple(operator.apply(query) for query in queries)
    oracle_targets = tuple(
        operators.apply_public_support_oracle(support, query)
        for query in queries
    )
    if targets != oracle_targets:
        raise RuntimeError("Gate9D stage-1 oracle disagrees with operator")
    if len(queries) != GATE9D_TRAIN_EXAMPLES:
        raise RuntimeError("Gate9D stage-1 query coverage drifted")
    if set(queries) & set(protocol.GATE9D_SUPPORT_INPUTS):
        raise RuntimeError("Gate9D stage-1 query escaped non-support domain")
    return {
        "operator_counter": operator_counter,
        "operator_key": operator.key,
        "support": support,
        "queries": queries,
        "targets": targets,
        "oracle_targets": oracle_targets,
    }


def stage1_dataset_sha256(material: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(int(material["operator_counter"]).to_bytes(8, "little"))
    digest.update(int(material["operator_key"]).to_bytes(8, "little"))
    for source, target in material["support"]:
        digest.update(bytes((source, target)))
    digest.update(bytes(material["queries"]))
    digest.update(bytes(material["targets"]))
    digest.update(bytes(material["oracle_targets"]))
    return digest.hexdigest()


def tensor_batch(
    material: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support = tuple(material["support"])
    queries = tuple(material["queries"])
    targets = tuple(material["targets"])
    support_sets = (support,) * len(queries)
    support_inputs, support_outputs, query_tensor = (
        architecture.serialize_gate9_worker_batch(
            support_sets,
            queries,
            device=device,
        )
    )
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return support_inputs, support_outputs, query_tensor, target_tensor


def target_bits(targets: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, dtype=torch.long, device=targets.device)
    return ((targets.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def validate_model_state(model: torch.nn.Module) -> dict[str, Any]:
    state = model.state_dict()
    if set(state) != set(GATE9D_STATE_TENSOR_SHAPES):
        raise RuntimeError("Gate9D stage-1 model tensor names drifted")
    parameters = 0
    for name, shape in GATE9D_STATE_TENSOR_SHAPES.items():
        tensor = state[name]
        if tuple(tensor.shape) != shape:
            raise RuntimeError(f"Gate9D tensor shape drifted: {name}")
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"Gate9D tensor dtype drifted: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Gate9D tensor became non-finite: {name}")
        parameters += tensor.numel()
    if parameters != protocol.GATE9D_LEARNED_PARAMETER_COUNT:
        raise RuntimeError("Gate9D stage-1 parameter count drifted")
    return {
        "tensor_count": len(state),
        "learned_parameter_count": parameters,
        "all_finite_float32": True,
    }


def _checkpoint_payload(
    model: torch.nn.Module,
    *,
    seed_index: int,
    initialization_seed: int,
    dataset_sha256: str,
) -> dict[str, Any]:
    state_evidence = validate_model_state(model)
    return {
        "experiment_version": GATE9D_STAGE1_EXECUTION_VERSION,
        "protocol_head": GATE9D_PROTOCOL_HEAD,
        "architecture_head": protocol.GATE9D_ARCHITECTURE_HEAD,
        "operator_contract_head": protocol.GATE9D_OPERATOR_CONTRACT_HEAD,
        "stage": GATE9D_STAGE_NAME,
        "seed_index": seed_index,
        "initialization_seed": initialization_seed,
        "step": GATE9D_TRAIN_STEPS,
        "unique_train_examples": GATE9D_TRAIN_EXAMPLES,
        "dataset_sha256": dataset_sha256,
        "learned_parameter_count": state_evidence["learned_parameter_count"],
        "tensor_count": state_evidence["tensor_count"],
        "state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
    }


def evaluate_stage1(
    *,
    model: torch.nn.Module,
    material: dict[str, Any],
    device: torch.device,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    support_inputs, support_outputs, queries, targets = tensor_batch(
        material, device
    )
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        full_predictions = model.decode_bytes(
            model(support_inputs, support_outputs, queries)
        )
        query_only_predictions = model.decode_bytes(
            model.forward_query_only(queries)
        )
    full_correct_flags = full_predictions == targets
    query_only_flags = query_only_predictions == targets
    full_bits = (
        (
            full_predictions.unsqueeze(-1)
            >> torch.arange(8, device=device, dtype=torch.long)
        )
        & 1
    )
    bit_correct = int(
        (full_bits == target_bits(targets).to(torch.long)).sum().cpu()
    )
    rows = GATE9D_TRAIN_EXAMPLES
    full_correct = int(full_correct_flags.sum().cpu())
    query_only_correct = int(query_only_flags.sum().cpu())
    oracle_correct = rows
    query_values = queries.cpu().tolist()
    target_values = targets.cpu().tolist()
    full_values = full_predictions.cpu().tolist()
    query_only_values = query_only_predictions.cpu().tolist()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(rows):
            row = {
                "episode_index": index,
                "operator_counter": material["operator_counter"],
                "query": int(query_values[index]),
                "answer": int(target_values[index]),
                "full_prediction": int(full_values[index]),
                "query_only_prediction": int(query_only_values[index]),
                "oracle_prediction": int(target_values[index]),
                "full_correct": bool(full_values[index] == target_values[index]),
                "query_only_correct": bool(
                    query_only_values[index] == target_values[index]
                ),
                "oracle_correct": True,
            }
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    exact_accuracy = full_correct / rows
    bit_accuracy = bit_correct / (rows * 8)
    query_only_accuracy = query_only_correct / rows
    oracle_accuracy = oracle_correct / rows
    stage_passes = (
        exact_accuracy >= protocol.GATE9D_EXACT_ACCURACY_MIN
        and bit_accuracy >= protocol.GATE9D_BIT_ACCURACY_MIN
        and oracle_accuracy == protocol.GATE9D_ORACLE_ACCURACY_REQUIRED
    )
    return {
        "rows": rows,
        "full_correct": full_correct,
        "exact_accuracy": exact_accuracy,
        "bit_accuracy": bit_accuracy,
        "query_only_correct": query_only_correct,
        "query_only_accuracy": query_only_accuracy,
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_accuracy,
        "stage_passes": stage_passes,
        "seconds": time.perf_counter() - started,
    }


def run_stage1_seed(
    *,
    seed_index: int,
    output_root: pathlib.Path,
    execution_head: str,
) -> dict[str, Any]:
    if seed_index not in (0, 1, 2):
        raise ValueError("Gate9D stage-1 seed index lies outside 0..2")
    if len(execution_head) != 40 or any(
        character not in "0123456789abcdef" for character in execution_head
    ):
        raise ValueError("Gate9D stage-1 execution head is malformed")
    if output_root.exists():
        raise FileExistsError(
            f"Gate9D stage-1 output already exists: {output_root}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Gate9D stage-1 execution requires CUDA")

    initialization_seed = configure_determinism(seed_index)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    output_root.mkdir(parents=True)
    seed_root = output_root / f"seed-{seed_index}"
    seed_root.mkdir()
    train_path = seed_root / "train-steps.jsonl"
    evaluation_path = seed_root / "evaluation-per-episode.jsonl"
    checkpoint_path = seed_root / "selected-checkpoint.pt"
    summary_path = seed_root / "summary.json"

    material = stage1_material()
    dataset_sha256 = stage1_dataset_sha256(material)
    support_inputs, support_outputs, queries, targets = tensor_batch(
        material, device
    )
    targets_as_bits = target_bits(targets)

    model = architecture.Gate9ContextualWorker().to(
        device=device, dtype=torch.float32
    )
    validate_model_state(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.GATE9D_BASE_LEARNING_RATE,
        betas=protocol.GATE9D_ADAM_BETAS,
        eps=protocol.GATE9D_ADAM_EPSILON,
        weight_decay=protocol.GATE9D_WEIGHT_DECAY,
    )

    final_loss = math.nan
    started_training = time.perf_counter()
    with train_path.open("w", encoding="utf-8", newline="\n") as handle:
        model.train()
        for step in range(1, GATE9D_TRAIN_STEPS + 1):
            started_step = time.perf_counter()
            learning_rate = learning_rate_at_step(step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            logits = model(support_inputs, support_outputs, queries)
            loss = F.binary_cross_entropy_with_logits(logits, targets_as_bits)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Gate9D stage-1 loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.GATE9D_GRADIENT_CLIP_NORM
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(
                    "Gate9D stage-1 gradient norm became non-finite"
                )
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            row = {
                "stage": GATE9D_STAGE_NAME,
                "seed_index": seed_index,
                "initialization_seed": initialization_seed,
                "step": step,
                "examples_seen": step * GATE9D_BATCH_SIZE,
                "learning_rate": learning_rate,
                "loss": final_loss,
                "pre_clip_gradient_norm": float(
                    gradient_norm.detach().cpu()
                ),
                "wall_seconds": time.perf_counter() - started_step,
            }
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            if step == 1 or step % 64 == 0 or step == GATE9D_TRAIN_STEPS:
                print(
                    f"stage=1 seed={seed_index} step={step:4d}/1024 "
                    f"loss={final_loss:.8f} lr={learning_rate:.8g}",
                    flush=True,
                )
    training_seconds = time.perf_counter() - started_training

    validate_model_state(model)
    torch.save(
        _checkpoint_payload(
            model,
            seed_index=seed_index,
            initialization_seed=initialization_seed,
            dataset_sha256=dataset_sha256,
        ),
        checkpoint_path,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    evaluation = evaluate_stage1(
        model=model,
        material=material,
        device=device,
        output_path=evaluation_path,
    )
    summary = {
        "experiment_version": GATE9D_STAGE1_EXECUTION_VERSION,
        "diagnostic_status": GATE9D_STAGE1_EXECUTION_STATUS,
        "execution_head": execution_head,
        "protocol_head": GATE9D_PROTOCOL_HEAD,
        "architecture_head": protocol.GATE9D_ARCHITECTURE_HEAD,
        "operator_contract_head": protocol.GATE9D_OPERATOR_CONTRACT_HEAD,
        "stage": GATE9D_STAGE_NAME,
        "seed_index": seed_index,
        "initialization_seed": initialization_seed,
        "dataset_sha256": dataset_sha256,
        "operator_counter": material["operator_counter"],
        "operator_key": material["operator_key"],
        "training": {
            "steps": GATE9D_TRAIN_STEPS,
            "batch_size": GATE9D_BATCH_SIZE,
            "unique_examples": GATE9D_TRAIN_EXAMPLES,
            "examples_seen": GATE9D_TRAIN_STEPS * GATE9D_BATCH_SIZE,
            "final_loss": final_loss,
            "seconds": training_seconds,
        },
        "evaluation": evaluation,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha256,
            **validate_model_state(model),
            "fixed_final_step": GATE9D_TRAIN_STEPS,
        },
        "execution": {
            "device": torch.cuda.get_device_name(0),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "amp_enabled": False,
            "tf32_enabled": False,
            "compile_enabled": False,
        },
        "boundaries": {
            "stage1_training_performed": True,
            "stage1_evaluation_performed": True,
            "checkpoint_serialized": True,
            "stage2_accessed": False,
            "stage3_accessed": False,
            "stage4_accessed": False,
            "gate9_v0_local_science_accessed": False,
            "gate9_v0_graph_science_accessed": False,
            "population_execution_performed": False,
            "diagnostic_classification_performed": False,
            "gate9_v0_result_modified": False,
        },
    }
    _write_json(summary_path, summary)
    return summary
