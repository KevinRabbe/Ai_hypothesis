"""Development-only Gate-9D query-path capacity diagnostic.

Distinguishes a short training budget, a 24-unit raw-bit bottleneck, a wider
raw-bit capacity threshold, and a parity-representation mismatch. This module
cannot open later diagnostic stages, Gate-9 science, population execution, or
confirmation.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_ROOT = pathlib.Path(__file__).resolve().parent
_STAGE1_RUNTIME_PATH = _ROOT / (
    "gate9_contextual_failure_decomposition_stage1_runtime.py"
)
_FAST_HARNESS_PATH = _ROOT / "gate9d_fast_diagnostic_harness.py"

GATE9D_QUERY_CAPACITY_VERSION = "gate9d-query-capacity-diagnostic-v0"
GATE9D_QUERY_CAPACITY_STATUS = "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
GATE9D_QUERY_CAPACITY_BRANCH = "agent/gate9d-query-capacity-diagnostic-v0"
GATE9D_QUERY_CAPACITY_BASE_HEAD = (
    "84f6038dc58547718d1d1ab9df7ef11538f21fb8"
)
GATE9D_QUERY_CAPACITY_VARIANTS = (
    "current_query_only_1024",
    "current_query_only_4096",
    "raw_bits_tanh_32",
    "raw_bits_tanh_64",
    "walsh_tanh_24",
)
GATE9D_QUERY_CAPACITY_STEPS = {
    "current_query_only_1024": 1024,
    "current_query_only_4096": 4096,
    "raw_bits_tanh_32": 1024,
    "raw_bits_tanh_64": 1024,
    "walsh_tanh_24": 1024,
}
GATE9D_QUERY_CAPACITY_CHECKPOINTS = {
    "current_query_only_1024": (0, 1, 16, 64, 128, 256, 512, 1024),
    "current_query_only_4096": (
        0,
        1,
        16,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ),
    "raw_bits_tanh_32": (0, 1, 16, 64, 128, 256, 512, 1024),
    "raw_bits_tanh_64": (0, 1, 16, 64, 128, 256, 512, 1024),
    "walsh_tanh_24": (0, 1, 16, 64, 128, 256, 512, 1024),
}

for _variant in GATE9D_QUERY_CAPACITY_VARIANTS:
    if _variant not in GATE9D_QUERY_CAPACITY_STEPS:
        raise RuntimeError("Gate9D query-capacity schedule is incomplete")
    checkpoints = GATE9D_QUERY_CAPACITY_CHECKPOINTS[_variant]
    if checkpoints[0] != 0 or checkpoints[-1] != GATE9D_QUERY_CAPACITY_STEPS[_variant]:
        raise RuntimeError("Gate9D query-capacity checkpoints drifted")


def _load_module(name: str, path: pathlib.Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate9D dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_stage1_runtime():
    return _load_module("gate9d_query_capacity_stage1_runtime", _STAGE1_RUNTIME_PATH)


def _load_fast_harness():
    return _load_module("gate9d_query_capacity_fast_harness", _FAST_HARNESS_PATH)


def _validate_byte_vector(values: Tensor, label: str) -> None:
    if values.dtype != torch.long or values.ndim != 1:
        raise ValueError(f"Gate9D {label} must be one long byte vector")
    if bool(torch.any((values < 0) | (values > 255))):
        raise ValueError(f"Gate9D {label} lies outside 0..255")


def byte_bits(values: Tensor) -> Tensor:
    _validate_byte_vector(values, "byte input")
    shifts = torch.arange(8, dtype=torch.long, device=values.device)
    return ((values.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


class RawBitsTanh(nn.Module):
    """One hidden tanh layer over the eight raw query bits."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("Gate9D raw-bit width must be positive")
        self.width = width
        self.hidden = nn.Linear(8, width)
        self.output = nn.Linear(width, 8, bias=False)

    def forward(self, query: Tensor) -> Tensor:
        return self.output(torch.tanh(self.hidden(byte_bits(query))))


class WalshTanh24(nn.Module):
    """A 24-unit tanh bottleneck over all 256 fixed Walsh parity features."""

    def __init__(self) -> None:
        super().__init__()
        parity = torch.tensor(
            [
                1.0 if value.bit_count() % 2 == 0 else -1.0
                for value in range(256)
            ],
            dtype=torch.float32,
        )
        self.register_buffer("parity_lut", parity, persistent=False)
        self.hidden = nn.Linear(256, 24)
        self.output = nn.Linear(24, 8, bias=False)

    def features(self, query: Tensor) -> Tensor:
        _validate_byte_vector(query, "Walsh query")
        masks = torch.arange(256, dtype=torch.long, device=query.device)
        return self.parity_lut[query.unsqueeze(1) & masks.unsqueeze(0)]

    def forward(self, query: Tensor) -> Tensor:
        return self.output(torch.tanh(self.hidden(self.features(query))))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _parameter_l2(model: nn.Module) -> float:
    device = next(model.parameters()).device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in model.parameters():
        value = parameter.detach().to(torch.float64)
        total += torch.sum(value * value)
    return float(torch.sqrt(total).cpu())


def _parameter_update_l2(model: nn.Module, initial: tuple[Tensor, ...]) -> float:
    device = next(model.parameters()).device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for parameter, start in zip(model.parameters(), initial, strict=True):
        difference = parameter.detach().to(torch.float64) - start
        total += torch.sum(difference * difference)
    return float(torch.sqrt(total).cpu())


def _active_gradient_elements(model: nn.Module) -> int:
    return sum(
        int(torch.count_nonzero(parameter.grad).cpu())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _model_for(variant: str, runtime: Any, device: torch.device) -> nn.Module:
    if variant in ("current_query_only_1024", "current_query_only_4096"):
        model: nn.Module = runtime.architecture.Gate9ContextualWorker()
    elif variant == "raw_bits_tanh_32":
        model = RawBitsTanh(32)
    elif variant == "raw_bits_tanh_64":
        model = RawBitsTanh(64)
    elif variant == "walsh_tanh_24":
        model = WalshTanh24()
    else:
        raise ValueError(f"unknown Gate9D query-capacity variant: {variant}")
    return model.to(device=device, dtype=torch.float32)


def _logits_for(variant: str, model: nn.Module, queries: Tensor) -> Tensor:
    if variant in ("current_query_only_1024", "current_query_only_4096"):
        return model.forward_query_only(queries)
    return model(queries)


def _learning_rate(runtime: Any, step: int) -> float:
    if step <= runtime.GATE9D_TRAIN_STEPS:
        return runtime.learning_rate_at_step(step)
    return float(runtime.protocol.GATE9D_MIN_LEARNING_RATE)


def _per_bit_metrics(predictions: Tensor, targets: Tensor) -> dict[str, Any]:
    prediction_bits = byte_bits(predictions).to(torch.long)
    target_bit_values = byte_bits(targets).to(torch.long)
    correct = (prediction_bits == target_bit_values).sum(dim=0).cpu().tolist()
    rows = int(targets.numel())
    return {
        "per_bit_correct": [int(value) for value in correct],
        "per_bit_accuracy": [int(value) / rows for value in correct],
    }


def _affine_parity_contract(queries: Tensor, targets: Tensor) -> list[dict[str, Any]]:
    query_values = [int(value) for value in queries.detach().cpu().tolist()]
    target_values = [int(value) for value in targets.detach().cpu().tolist()]
    contract = []
    for output_bit in range(8):
        solutions: list[tuple[int, int]] = []
        for mask in range(256):
            for bias in (0, 1):
                if all(
                    ((((query & mask).bit_count() & 1) ^ bias)
                    == ((answer >> output_bit) & 1))
                    for query, answer in zip(query_values, target_values, strict=True)
                ):
                    solutions.append((mask, bias))
        if len(solutions) != 1:
            raise RuntimeError("Gate9D affine parity reconstruction is not unique")
        mask, bias = solutions[0]
        contract.append(
            {
                "output_bit": output_bit,
                "input_mask": mask,
                "input_indices": [index for index in range(8) if mask & (1 << index)],
                "parity_weight": mask.bit_count(),
                "bias": bias,
            }
        )
    return contract


def classify_query_capacity(final_rows: Iterable[dict[str, Any]]) -> str:
    rows = tuple(final_rows)
    by_variant: dict[str, list[bool]] = {
        variant: [] for variant in GATE9D_QUERY_CAPACITY_VARIANTS
    }
    seen: set[tuple[str, int]] = set()
    for row in rows:
        variant = row.get("variant")
        seed_index = row.get("seed_index")
        passes = row.get("passes")
        if variant not in by_variant:
            raise ValueError(f"unknown Gate9D query-capacity variant: {variant!r}")
        if type(seed_index) is not int or seed_index not in (0, 1, 2):
            raise ValueError("Gate9D query-capacity seed index is invalid")
        if type(passes) is not bool:
            raise ValueError("Gate9D query-capacity pass flag is not Boolean")
        identity = (variant, seed_index)
        if identity in seen:
            raise ValueError("Gate9D query-capacity result repeats one run")
        seen.add(identity)
        by_variant[variant].append(passes)
    if len(rows) != 15 or any(len(values) != 3 for values in by_variant.values()):
        raise ValueError("Gate9D query-capacity result requires five variants by three seeds")

    current_1024 = tuple(by_variant["current_query_only_1024"])
    current_4096 = tuple(by_variant["current_query_only_4096"])
    raw_32 = tuple(by_variant["raw_bits_tanh_32"])
    raw_64 = tuple(by_variant["raw_bits_tanh_64"])
    walsh_24 = tuple(by_variant["walsh_tanh_24"])

    if all(current_1024):
        return "G9D_QUERY_CAPACITY_FAILURE_NOT_REPRODUCED"
    if all(current_4096):
        return "G9D_QUERY_CAPACITY_TRAINING_BUDGET_LIMIT"
    if any(current_4096):
        return "G9D_QUERY_CAPACITY_TRAINING_BUDGET_MIXED"
    if all(raw_32):
        return "G9D_QUERY_CAPACITY_24_UNIT_BOTTLENECK"
    if any(raw_32):
        return "G9D_QUERY_CAPACITY_32_UNIT_MIXED"
    if all(raw_64):
        return "G9D_QUERY_CAPACITY_BETWEEN_32_AND_64"
    if any(raw_64):
        return "G9D_QUERY_CAPACITY_64_UNIT_MIXED"
    if all(walsh_24):
        return "G9D_QUERY_CAPACITY_RAW_BIT_PARITY_MISMATCH"
    if any(walsh_24):
        return "G9D_QUERY_CAPACITY_WALSH_MIXED"
    return "G9D_QUERY_CAPACITY_UNRESOLVED"


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def run_query_capacity_diagnostic(
    *, output_root: pathlib.Path, execution_head: str
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"Gate9D query-capacity output already exists: {output_root}"
        )
    if len(execution_head) != 40 or any(
        character not in "0123456789abcdef" for character in execution_head
    ):
        raise ValueError("Gate9D query-capacity execution head is malformed")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate9D query-capacity diagnostic requires CUDA")

    runtime = _load_stage1_runtime()
    fast = _load_fast_harness()
    if runtime.GATE9D_TRAIN_STEPS != 1024:
        raise RuntimeError("Gate9D frozen stage-1 schedule drifted")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    material = runtime.stage1_material()
    _, _, queries, targets = runtime.tensor_batch(material, device)
    expected_bits = fast.target_bits(targets)
    parity_contract = _affine_parity_contract(queries, targets)
    output_root.mkdir(parents=True)

    curves: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for seed_index in (0, 1, 2):
        for variant in GATE9D_QUERY_CAPACITY_VARIANTS:
            initialization_seed = runtime.configure_determinism(seed_index)
            model = _model_for(variant, runtime, device)
            initial = tuple(
                parameter.detach().to(torch.float64).clone()
                for parameter in model.parameters()
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=runtime.protocol.GATE9D_BASE_LEARNING_RATE,
                betas=runtime.protocol.GATE9D_ADAM_BETAS,
                eps=runtime.protocol.GATE9D_ADAM_EPSILON,
                weight_decay=runtime.protocol.GATE9D_WEIGHT_DECAY,
            )
            total_steps = GATE9D_QUERY_CAPACITY_STEPS[variant]
            checkpoints = set(GATE9D_QUERY_CAPACITY_CHECKPOINTS[variant])

            model.eval()
            with torch.no_grad():
                initial_logits = _logits_for(variant, model, queries)
                initial_loss = F.binary_cross_entropy_with_logits(
                    initial_logits, expected_bits
                )
                initial_metrics = fast.metrics_from_logits(initial_logits, targets)
                initial_predictions = fast.decode_bytes(initial_logits)
            curves.append(
                {
                    "seed_index": seed_index,
                    "initialization_seed": initialization_seed,
                    "variant": variant,
                    "step": 0,
                    "learning_rate": None,
                    "loss": float(initial_loss.cpu()),
                    "gradient_norm": None,
                    "active_gradient_elements": 0,
                    "parameter_count": _parameter_count(model),
                    "parameter_l2": _parameter_l2(model),
                    "parameter_update_l2": 0.0,
                    **initial_metrics,
                    **_per_bit_metrics(initial_predictions, targets),
                }
            )

            started = time.perf_counter()
            last_gradient_norm = math.nan
            last_active_gradients = 0
            for step in range(1, total_steps + 1):
                learning_rate = _learning_rate(runtime, step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = _logits_for(variant, model, queries)
                loss = F.binary_cross_entropy_with_logits(logits, expected_bits)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        f"Gate9D query-capacity loss became non-finite: "
                        f"{seed_index}:{variant}:{step}"
                    )
                loss.backward()
                last_active_gradients = _active_gradient_elements(model)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), runtime.protocol.GATE9D_GRADIENT_CLIP_NORM
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError(
                        f"Gate9D query-capacity gradient became non-finite: "
                        f"{seed_index}:{variant}:{step}"
                    )
                last_gradient_norm = float(gradient_norm.detach().cpu())
                optimizer.step()

                if step in checkpoints:
                    model.eval()
                    with torch.no_grad():
                        checkpoint_logits = _logits_for(variant, model, queries)
                        checkpoint_loss = F.binary_cross_entropy_with_logits(
                            checkpoint_logits, expected_bits
                        )
                        checkpoint_metrics = fast.metrics_from_logits(
                            checkpoint_logits, targets
                        )
                        checkpoint_predictions = fast.decode_bytes(checkpoint_logits)
                    curves.append(
                        {
                            "seed_index": seed_index,
                            "initialization_seed": initialization_seed,
                            "variant": variant,
                            "step": step,
                            "learning_rate": learning_rate,
                            "loss": float(checkpoint_loss.cpu()),
                            "gradient_norm": last_gradient_norm,
                            "active_gradient_elements": last_active_gradients,
                            "parameter_count": _parameter_count(model),
                            "parameter_l2": _parameter_l2(model),
                            "parameter_update_l2": _parameter_update_l2(model, initial),
                            **checkpoint_metrics,
                            **_per_bit_metrics(checkpoint_predictions, targets),
                        }
                    )

            elapsed = time.perf_counter() - started
            model.eval()
            with torch.no_grad():
                final_logits = _logits_for(variant, model, queries)
                final_loss = F.binary_cross_entropy_with_logits(
                    final_logits, expected_bits
                )
                final_predictions = fast.decode_bytes(final_logits)
                final_metrics = fast.metrics_from_logits(final_logits, targets)
            final_row = {
                "seed_index": seed_index,
                "initialization_seed": initialization_seed,
                "variant": variant,
                "steps": total_steps,
                "seconds": elapsed,
                "final_loss": float(final_loss.cpu()),
                "parameter_count": _parameter_count(model),
                "active_gradient_elements_final": last_active_gradients,
                "gradient_norm_final": last_gradient_norm,
                "parameter_update_l2": _parameter_update_l2(model, initial),
                **final_metrics,
                **_per_bit_metrics(final_predictions, targets),
            }
            finals.append(final_row)
            for index, (query, answer, prediction) in enumerate(
                zip(
                    queries.detach().cpu().tolist(),
                    targets.detach().cpu().tolist(),
                    final_predictions.detach().cpu().tolist(),
                    strict=True,
                )
            ):
                predictions.append(
                    {
                        "seed_index": seed_index,
                        "variant": variant,
                        "episode_index": index,
                        "query": int(query),
                        "answer": int(answer),
                        "prediction": int(prediction),
                        "correct": bool(answer == prediction),
                        "bit_correct": 8
                        - int((int(answer) ^ int(prediction)).bit_count()),
                    }
                )

    diagnosis = classify_query_capacity(finals)
    variant_summaries = {}
    for variant in GATE9D_QUERY_CAPACITY_VARIANTS:
        rows = [row for row in finals if row["variant"] == variant]
        variant_summaries[variant] = {
            "steps": GATE9D_QUERY_CAPACITY_STEPS[variant],
            "parameter_counts": [row["parameter_count"] for row in rows],
            "seed_passes": [row["passes"] for row in rows],
            "exact_accuracies": [row["exact_accuracy"] for row in rows],
            "bit_accuracies": [row["bit_accuracy"] for row in rows],
            "per_bit_accuracies": [row["per_bit_accuracy"] for row in rows],
            "mean_exact_accuracy": sum(row["exact_accuracy"] for row in rows) / 3,
            "mean_bit_accuracy": sum(row["bit_accuracy"] for row in rows) / 3,
        }

    summary = {
        "status": "G9D_QUERY_CAPACITY_DIAGNOSTIC_COMPLETE_DEVELOPMENT_ONLY",
        "version": GATE9D_QUERY_CAPACITY_VERSION,
        "development_only": True,
        "confirmation_result": False,
        "execution_head": execution_head,
        "fast_harness_base_head": GATE9D_QUERY_CAPACITY_BASE_HEAD,
        "frozen_stage1_execution_head": fast.GATE9D_FROZEN_STAGE1_EXECUTION_HEAD,
        "dataset_sha256": runtime.stage1_dataset_sha256(material),
        "operator_counter": material["operator_counter"],
        "operator_key": material["operator_key"],
        "affine_parity_contract": parity_contract,
        "seeds": [0, 1, 2],
        "initialization_seeds": list(runtime.protocol.GATE9D_INITIALIZATION_SEEDS),
        "variants": list(GATE9D_QUERY_CAPACITY_VARIANTS),
        "total_training_runs": len(finals),
        "diagnosis": diagnosis,
        "variant_summaries": variant_summaries,
        "boundaries": {
            "gate9_v0_science_executed": False,
            "population_execution_performed": False,
            "later_diagnostic_stage_executed": False,
            "frozen_gate9_result_modified": False,
            "confirmation_claimed": False,
            "checkpoint_selection_performed": False,
            "early_stopping_performed": False,
        },
    }
    _write_jsonl(output_root / "curves.jsonl", curves)
    _write_jsonl(output_root / "final-runs.jsonl", finals)
    _write_jsonl(output_root / "predictions.jsonl", predictions)
    _write_json(output_root / "aggregate-summary.json", summary)
    return summary
