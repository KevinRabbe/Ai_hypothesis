"""Development-only fast Gate-9 stage-1 failure diagnostic harness.

Compares an optimizer-wiring byte lookup, an affine-compatible Walsh parity
baseline, the frozen worker's query-only path, and its full support-conditioned
path. This module cannot open later stages, Gate-9 science, population
execution, or confirmation.
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

GATE9D_FAST_HARNESS_VERSION = "gate9d-fast-diagnostic-harness-v0"
GATE9D_FAST_HARNESS_STATUS = "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
GATE9D_FAST_HARNESS_BRANCH = "agent/gate9d-fast-diagnostic-harness-v0"
GATE9D_FROZEN_STAGE1_EXECUTION_HEAD = (
    "2e1b91d578e7bf9b4c54aa2ee1c120a9ec01b21c"
)
GATE9D_FAST_VARIANTS = (
    "byte_lookup_zero_init",
    "parity_feature_linear",
    "current_query_only",
    "current_full_context",
)
GATE9D_FAST_CHECKPOINT_STEPS = (0, 1, 16, 64, 128, 256, 512, 1024)
GATE9D_FAST_TRAIN_STEPS = 1024
GATE9D_FAST_EXACT_ACCURACY_MIN = 0.995
GATE9D_FAST_BIT_ACCURACY_MIN = 0.999

if len(GATE9D_FROZEN_STAGE1_EXECUTION_HEAD) != 40 or any(
    character not in "0123456789abcdef"
    for character in GATE9D_FROZEN_STAGE1_EXECUTION_HEAD
):
    raise RuntimeError("Gate9D fast harness source head is malformed")


def _load_stage1_runtime():
    name = "gate9d_fast_harness_stage1_runtime"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _STAGE1_RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen Gate9D stage-1 runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_byte_vector(values: Tensor, label: str) -> None:
    if values.dtype != torch.long or values.ndim != 1:
        raise ValueError(f"Gate9D {label} must be one long byte vector")
    if bool(torch.any((values < 0) | (values > 255))):
        raise ValueError(f"Gate9D {label} lies outside 0..255")


class ByteLookupBaseline(nn.Module):
    """One zero-initialized eight-logit row per input byte."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Embedding(256, 8)
        nn.init.zeros_(self.logits.weight)

    def forward(self, query: Tensor) -> Tensor:
        _validate_byte_vector(query, "lookup query")
        return self.logits(query)


class ParityFeatureLinear(nn.Module):
    """Linear output over all 256 Walsh parity features of one byte."""

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
        self.output = nn.Linear(256, 8, bias=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(self, query: Tensor) -> Tensor:
        _validate_byte_vector(query, "parity query")
        masks = torch.arange(256, dtype=torch.long, device=query.device)
        return self.parity_lut[query.unsqueeze(1) & masks.unsqueeze(0)]

    def forward(self, query: Tensor) -> Tensor:
        return self.output(self.features(query))


def target_bits(targets: Tensor) -> Tensor:
    _validate_byte_vector(targets, "target")
    shifts = torch.arange(8, dtype=torch.long, device=targets.device)
    return ((targets.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def decode_bytes(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] != 8:
        raise ValueError("Gate9D logits must have shape [batch,8]")
    weights = 1 << torch.arange(8, dtype=torch.long, device=logits.device)
    return ((logits >= 0).to(torch.long) * weights).sum(dim=-1)


def metrics_from_logits(logits: Tensor, targets: Tensor) -> dict[str, Any]:
    predictions = decode_bytes(logits)
    rows = int(targets.numel())
    exact_correct = int((predictions == targets).sum().cpu())
    bit_correct = int(
        (
            target_bits(predictions).to(torch.long)
            == target_bits(targets).to(torch.long)
        )
        .sum()
        .cpu()
    )
    exact_accuracy = exact_correct / rows
    bit_accuracy = bit_correct / (rows * 8)
    return {
        "rows": rows,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_accuracy,
        "bit_correct": bit_correct,
        "bit_total": rows * 8,
        "bit_accuracy": bit_accuracy,
        "passes": (
            exact_accuracy >= GATE9D_FAST_EXACT_ACCURACY_MIN
            and bit_accuracy >= GATE9D_FAST_BIT_ACCURACY_MIN
        ),
    }


def classify_fast_results(final_rows: Iterable[dict[str, Any]]) -> str:
    rows = tuple(final_rows)
    by_variant: dict[str, list[bool]] = {
        variant: [] for variant in GATE9D_FAST_VARIANTS
    }
    seen: set[tuple[str, int]] = set()
    for row in rows:
        variant = row.get("variant")
        seed_index = row.get("seed_index")
        passes = row.get("passes")
        if variant not in by_variant:
            raise ValueError(f"unknown Gate9D fast variant: {variant!r}")
        if type(seed_index) is not int or seed_index not in (0, 1, 2):
            raise ValueError("Gate9D fast result has invalid seed index")
        if type(passes) is not bool:
            raise ValueError("Gate9D fast pass flag is not Boolean")
        identity = (variant, seed_index)
        if identity in seen:
            raise ValueError("Gate9D fast result repeats one seed/variant")
        seen.add(identity)
        by_variant[variant].append(passes)
    if len(rows) != 12 or any(len(values) != 3 for values in by_variant.values()):
        raise ValueError("Gate9D fast result requires four variants by three seeds")

    lookup = tuple(by_variant["byte_lookup_zero_init"])
    parity = tuple(by_variant["parity_feature_linear"])
    query = tuple(by_variant["current_query_only"])
    full = tuple(by_variant["current_full_context"])
    if not all(lookup):
        return (
            "G9D_FAST_LOOKUP_PIPELINE_MIXED"
            if any(lookup)
            else "G9D_FAST_LOOKUP_PIPELINE_FAILED"
        )
    if not all(parity):
        return (
            "G9D_FAST_PARITY_REPRESENTATION_MIXED"
            if any(parity)
            else "G9D_FAST_PARITY_REPRESENTATION_FAILED"
        )
    if all(query) and all(full):
        return "G9D_FAST_STAGE1_FAILURE_NOT_REPRODUCED"
    if all(query) and not any(full):
        return "G9D_FAST_SUPPORT_PATH_INTERFERENCE"
    if all(query):
        return "G9D_FAST_SUPPORT_PATH_MIXED"
    if not any(query) and all(full):
        return "G9D_FAST_SUPPORT_CONTEXT_RESCUES_QUERY_PATH"
    if not any(query) and not any(full):
        return "G9D_FAST_CURRENT_QUERY_PATH_FAILED"
    return "G9D_FAST_CURRENT_WORKER_MIXED"


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


def _variant_model(variant: str, runtime: Any, device: torch.device) -> nn.Module:
    if variant == "byte_lookup_zero_init":
        model: nn.Module = ByteLookupBaseline()
    elif variant == "parity_feature_linear":
        model = ParityFeatureLinear()
    elif variant in ("current_query_only", "current_full_context"):
        model = runtime.architecture.Gate9ContextualWorker()
    else:
        raise ValueError(f"unknown Gate9D fast variant: {variant}")
    return model.to(device=device, dtype=torch.float32)


def _variant_logits(
    variant: str,
    model: nn.Module,
    *,
    support_inputs: Tensor,
    support_outputs: Tensor,
    queries: Tensor,
) -> Tensor:
    if variant in ("byte_lookup_zero_init", "parity_feature_linear"):
        return model(queries)
    if variant == "current_query_only":
        return model.forward_query_only(queries)
    if variant == "current_full_context":
        return model(support_inputs, support_outputs, queries)
    raise ValueError(f"unknown Gate9D fast variant: {variant}")


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def run_fast_diagnostic(
    *, output_root: pathlib.Path, execution_head: str
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Gate9D fast output already exists: {output_root}")
    if len(execution_head) != 40 or any(
        character not in "0123456789abcdef" for character in execution_head
    ):
        raise ValueError("Gate9D fast execution head is malformed")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate9D fast diagnostic requires CUDA")

    runtime = _load_stage1_runtime()
    if runtime.GATE9D_STAGE1_EXECUTION_VERSION != (
        "gate9-contextual-failure-decomposition-stage1-execution-v0"
    ):
        raise RuntimeError("Gate9D frozen stage-1 runtime drifted")
    if runtime.GATE9D_TRAIN_STEPS != GATE9D_FAST_TRAIN_STEPS:
        raise RuntimeError("Gate9D fast training schedule drifted")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    material = runtime.stage1_material()
    support_inputs, support_outputs, queries, targets = runtime.tensor_batch(
        material, device
    )
    expected_bits = target_bits(targets)
    output_root.mkdir(parents=True)
    curves: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for seed_index in (0, 1, 2):
        for variant in GATE9D_FAST_VARIANTS:
            initialization_seed = runtime.configure_determinism(seed_index)
            model = _variant_model(variant, runtime, device)
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

            model.eval()
            with torch.no_grad():
                initial_logits = _variant_logits(
                    variant,
                    model,
                    support_inputs=support_inputs,
                    support_outputs=support_outputs,
                    queries=queries,
                )
                initial_loss = F.binary_cross_entropy_with_logits(
                    initial_logits, expected_bits
                )
                initial_metrics = metrics_from_logits(initial_logits, targets)
            curves.append(
                {
                    "seed_index": seed_index,
                    "initialization_seed": initialization_seed,
                    "variant": variant,
                    "step": 0,
                    "loss": float(initial_loss.cpu()),
                    "gradient_norm": None,
                    "active_gradient_elements": 0,
                    "parameter_count": _parameter_count(model),
                    "parameter_l2": _parameter_l2(model),
                    "parameter_update_l2": 0.0,
                    **initial_metrics,
                }
            )

            started = time.perf_counter()
            last_gradient_norm = math.nan
            last_active_gradients = 0
            for step in range(1, GATE9D_FAST_TRAIN_STEPS + 1):
                learning_rate = runtime.learning_rate_at_step(step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = _variant_logits(
                    variant,
                    model,
                    support_inputs=support_inputs,
                    support_outputs=support_outputs,
                    queries=queries,
                )
                loss = F.binary_cross_entropy_with_logits(logits, expected_bits)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        f"Gate9D fast loss became non-finite: {seed_index}:{variant}"
                    )
                loss.backward()
                last_active_gradients = _active_gradient_elements(model)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), runtime.protocol.GATE9D_GRADIENT_CLIP_NORM
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError(
                        f"Gate9D fast gradient became non-finite: {seed_index}:{variant}"
                    )
                last_gradient_norm = float(gradient_norm.detach().cpu())
                optimizer.step()

                if step in GATE9D_FAST_CHECKPOINT_STEPS:
                    model.eval()
                    with torch.no_grad():
                        checkpoint_logits = _variant_logits(
                            variant,
                            model,
                            support_inputs=support_inputs,
                            support_outputs=support_outputs,
                            queries=queries,
                        )
                        checkpoint_loss = F.binary_cross_entropy_with_logits(
                            checkpoint_logits, expected_bits
                        )
                        checkpoint_metrics = metrics_from_logits(
                            checkpoint_logits, targets
                        )
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
                            "parameter_update_l2": _parameter_update_l2(
                                model, initial
                            ),
                            **checkpoint_metrics,
                        }
                    )

            elapsed = time.perf_counter() - started
            model.eval()
            with torch.no_grad():
                final_logits = _variant_logits(
                    variant,
                    model,
                    support_inputs=support_inputs,
                    support_outputs=support_outputs,
                    queries=queries,
                )
                final_loss = F.binary_cross_entropy_with_logits(
                    final_logits, expected_bits
                )
                final_predictions = decode_bytes(final_logits)
                final_metrics = metrics_from_logits(final_logits, targets)
            finals.append(
                {
                    "seed_index": seed_index,
                    "initialization_seed": initialization_seed,
                    "variant": variant,
                    "steps": GATE9D_FAST_TRAIN_STEPS,
                    "seconds": elapsed,
                    "final_loss": float(final_loss.cpu()),
                    "parameter_count": _parameter_count(model),
                    "active_gradient_elements_final": last_active_gradients,
                    "gradient_norm_final": last_gradient_norm,
                    "parameter_update_l2": _parameter_update_l2(model, initial),
                    **final_metrics,
                }
            )
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

    diagnosis = classify_fast_results(finals)
    variant_summaries = {}
    for variant in GATE9D_FAST_VARIANTS:
        rows = [row for row in finals if row["variant"] == variant]
        variant_summaries[variant] = {
            "seed_passes": [row["passes"] for row in rows],
            "exact_accuracies": [row["exact_accuracy"] for row in rows],
            "bit_accuracies": [row["bit_accuracy"] for row in rows],
            "mean_exact_accuracy": sum(
                row["exact_accuracy"] for row in rows
            )
            / 3,
            "mean_bit_accuracy": sum(row["bit_accuracy"] for row in rows) / 3,
        }
    summary = {
        "status": "G9D_FAST_DIAGNOSTIC_COMPLETE_DEVELOPMENT_ONLY",
        "version": GATE9D_FAST_HARNESS_VERSION,
        "development_only": True,
        "confirmation_result": False,
        "execution_head": execution_head,
        "frozen_stage1_execution_head": GATE9D_FROZEN_STAGE1_EXECUTION_HEAD,
        "dataset_sha256": runtime.stage1_dataset_sha256(material),
        "operator_counter": material["operator_counter"],
        "operator_key": material["operator_key"],
        "seeds": [0, 1, 2],
        "initialization_seeds": list(
            runtime.protocol.GATE9D_INITIALIZATION_SEEDS
        ),
        "variants": list(GATE9D_FAST_VARIANTS),
        "steps_per_variant": GATE9D_FAST_TRAIN_STEPS,
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
