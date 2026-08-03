"""CUDA memory/runtime preflight for the matched Population Language L0 models."""
from __future__ import annotations

import gc
import json
import math
import pathlib
import time
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import l0_protocol as protocol
from .l0_data import LanguageBatch, materialize_batch
from .l0_models import (
    MatchedCausalTransformer,
    PopulationLanguageOrganism,
    count_parameters,
)

VERSION = "population-language-l0-reference-preflight-v0"
BRANCH = "agent/population-language-l0-reference-preflight-v0"
BASE_HEAD = "8f6ba0456cf57dad7209e44f62cac0e40331e8f4"
STATUS = "DEVELOPMENT_ONLY_REFERENCE_SCALE_ENGINEERING_PREFLIGHT"
PASS = "POPULATION_LANGUAGE_L0_REFERENCE_PREFLIGHT_PASSES"
FAIL = "POPULATION_LANGUAGE_L0_REFERENCE_PREFLIGHT_FAILED"
MICROBATCH_CANDIDATES = (1, 2, 4, 8)
MINIMUM_COMMON_MICROBATCH = 4
GLOBAL_BATCH_SIZE = 256
LEARNING_RATE = 3e-4


def transformer_kv_cache_bytes(
    *,
    sequence_length: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
    config: protocol.TransformerConfig = protocol.TransformerConfig(),
) -> int:
    for value, label in (
        (sequence_length, "sequence length"),
        (batch_size, "batch size"),
        (bytes_per_element, "bytes per element"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    return (
        2
        * config.layers
        * batch_size
        * sequence_length
        * config.d_model
        * bytes_per_element
    )


def organism_state_bytes(
    *,
    worker_count: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
    config: protocol.PopulationConfig = protocol.PopulationConfig(),
) -> int:
    for value, label in (
        (worker_count, "worker count"),
        (batch_size, "batch size"),
        (bytes_per_element, "bytes per element"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    return batch_size * worker_count * config.worker_width * bytes_per_element


def full_next_token_loss(logits: Tensor, batch: LanguageBatch) -> Tensor:
    if logits.ndim != 3 or logits.shape[:2] != batch.target_ids.shape:
        raise ValueError("reference preflight logit/target shapes disagree")
    if not bool(torch.any(batch.loss_mask)):
        raise ValueError("reference preflight loss mask is empty")
    return F.cross_entropy(logits[batch.loss_mask], batch.target_ids[batch.loss_mask])


def _finite_gradient_norm(model: nn.Module) -> float:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    value = float(norm.detach().to(torch.float64).cpu().item())
    if not math.isfinite(value):
        raise RuntimeError("reference preflight produced a non-finite gradient norm")
    return value


def _cuda_row(
    *,
    name: str,
    microbatch: int,
    model: nn.Module,
    batch: LanguageBatch,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    started = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(batch.input_ids)
        loss = full_next_token_loss(logits, batch)
    if not bool(torch.isfinite(loss.detach())):
        raise RuntimeError("reference preflight produced a non-finite loss")
    loss.backward()
    gradient_norm = _finite_gradient_norm(model)
    optimizer.step()
    torch.cuda.synchronize()

    return {
        "model": name,
        "microbatch": microbatch,
        "success": True,
        "loss": float(loss.detach().to(torch.float64).cpu().item()),
        "gradient_norm_before_clip": gradient_norm,
        "seconds": time.perf_counter() - started,
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _profile_model(
    *,
    name: str,
    factory: Callable[[], nn.Module],
    expected_parameters: int,
    device: torch.device,
    candidates: tuple[int, ...],
) -> tuple[list[dict[str, Any]], int]:
    gc.collect()
    torch.cuda.empty_cache()
    model = factory().to(device)
    actual_parameters = count_parameters(model)
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"{name} parameter count drifted: {actual_parameters} != {expected_parameters}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    rows: list[dict[str, Any]] = []

    for microbatch in candidates:
        batch = materialize_batch("train", range(microbatch), device=device)
        try:
            row = _cuda_row(
                name=name,
                microbatch=microbatch,
                model=model,
                batch=batch,
                optimizer=optimizer,
            )
        except torch.OutOfMemoryError as error:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            rows.append(
                {
                    "model": name,
                    "microbatch": microbatch,
                    "success": False,
                    "failure": "CUDA_OUT_OF_MEMORY",
                    "error": str(error),
                }
            )
            break
        rows.append(row)

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return rows, actual_parameters


def classify(rows: list[dict[str, Any]]) -> tuple[str, int | None]:
    expected_order = [
        (name, candidate)
        for name in ("transformer", "population")
        for candidate in MICROBATCH_CANDIDATES
    ]
    observed = [(row.get("model"), row.get("microbatch")) for row in rows]
    if observed != expected_order[: len(observed)]:
        raise ValueError("reference preflight row order drifted")

    successful_by_model: dict[str, set[int]] = {
        "transformer": set(),
        "population": set(),
    }
    for row in rows:
        name = row.get("model")
        if name not in successful_by_model:
            raise ValueError("reference preflight contains an unknown model row")
        if row.get("success") is True:
            for field in (
                "loss",
                "gradient_norm_before_clip",
                "seconds",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ):
                value = row.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    return FAIL, None
            successful_by_model[name].add(int(row["microbatch"]))

    common = sorted(
        successful_by_model["transformer"] & successful_by_model["population"]
    )
    recommended = common[-1] if common else None
    if recommended is None or recommended < MINIMUM_COMMON_MICROBATCH:
        return FAIL, recommended
    if GLOBAL_BATCH_SIZE % recommended:
        return FAIL, recommended
    return PASS, recommended


def run(
    output_root: pathlib.Path,
    execution_head: str,
    *,
    candidates: tuple[int, ...] = MICROBATCH_CANDIDATES,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"reference preflight output exists: {output_root}")
    if candidates != MICROBATCH_CANDIDATES:
        raise ValueError("reference preflight microbatch candidates are locked")
    if len(execution_head) != 40 or any(
        character not in "0123456789abcdef" for character in execution_head
    ):
        raise ValueError("reference preflight execution head is malformed")
    if not torch.cuda.is_available():
        raise RuntimeError("reference preflight requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("reference preflight requires CUDA BF16 support")

    torch.cuda.set_device(0)
    torch.manual_seed(protocol.INITIALIZATION_SEEDS[0])
    torch.cuda.manual_seed_all(protocol.INITIALIZATION_SEEDS[0])
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda", 0)
    output_root.mkdir(parents=True)

    transformer_rows, transformer_parameters = _profile_model(
        name="transformer",
        factory=MatchedCausalTransformer,
        expected_parameters=protocol.transformer_parameter_count(),
        device=device,
        candidates=candidates,
    )
    (output_root / "transformer-rows.json").write_text(
        json.dumps(transformer_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    population_rows, population_parameters = _profile_model(
        name="population",
        factory=PopulationLanguageOrganism,
        expected_parameters=protocol.population_parameter_count(),
        device=device,
        candidates=candidates,
    )
    (output_root / "population-rows.json").write_text(
        json.dumps(population_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = transformer_rows + population_rows
    diagnosis, recommended = classify(rows)
    summary = {
        "status": STATUS,
        "version": VERSION,
        "branch": BRANCH,
        "base_head": BASE_HEAD,
        "execution_head": execution_head,
        "diagnosis": diagnosis,
        "microbatch_candidates": list(candidates),
        "minimum_common_microbatch": MINIMUM_COMMON_MICROBATCH,
        "recommended_common_microbatch": recommended,
        "gradient_accumulation_steps": (
            GLOBAL_BATCH_SIZE // recommended if recommended else None
        ),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "transformer_parameters": transformer_parameters,
        "population_parameters": population_parameters,
        "cuda": {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "cache_state_estimates_bf16_bytes_per_sample": {
            "transformer_kv_at_32_tokens": transformer_kv_cache_bytes(
                sequence_length=protocol.MAX_SEQUENCE_LENGTH
            ),
            "organism_state_by_workers": {
                str(worker_count): organism_state_bytes(worker_count=worker_count)
                for worker_count in protocol.EVAL_WORKERS
            },
        },
        "rows": rows,
        "boundaries": {
            "engineering_preflight_only": True,
            "optimizer_steps_per_candidate": 1,
            "full_next_token_objective_used": True,
            "reference_training_performed": False,
            "validation_or_test_metrics_computed": False,
            "population_scaling_claimed": False,
            "cache_latency_benchmarked": False,
            "gate9_evidence_modified": False,
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
