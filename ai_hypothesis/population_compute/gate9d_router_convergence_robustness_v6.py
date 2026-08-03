"""Development-only convergence study for the representable local-summary router."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import torch

_ROOT = pathlib.Path(__file__).resolve().parent
_V5_PATH = _ROOT / "gate9d_router_factorization_sweep_v5.py"

VERSION = "gate9d-router-convergence-robustness-v6"
STATUS = "DEVELOPMENT_ONLY_SUPERVISED_ROUTING_CONVERGENCE"
BRANCH = "agent/gate9d-router-convergence-robustness-v6"
BASE_HEAD = "90315b5b078dd92c55d46698af1d1c0659d25f8c"
VARIANTS = ("adamw_2048", "lbfgs_full_batch", "analytic_separator")
PASS = "G9D_ROUTER_CONVERGENCE_ROBUSTNESS_PASSES"
FAIL = "G9D_ROUTER_CONVERGENCE_ROBUSTNESS_FAILED"


def _load_v5():
    name = "gate9d_router_convergence_v5_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _V5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load factorization sweep dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v5 = _load_v5()
v3 = v5.v3
v2 = v5.v2
v0 = v5.v0
LocalSummaryLinear = v5.LocalSummaryLinear


def analytic_model(device: torch.device) -> LocalSummaryLinear:
    """Exact unit-margin separator over [zero, popcount, overlap]."""
    model = LocalSummaryLinear().to(device)
    with torch.no_grad():
        model.output.weight[0].copy_(torch.tensor([4.0, 0.0, 0.0], device=device))
        model.output.bias[0] = -2.0
        model.output.weight[1].copy_(torch.tensor([-4.0, -4.0, 2.0], device=device))
        model.output.bias[1] = 3.0
    return model


def _adamw(seed: int, device: torch.device):
    v0._configure(seed)
    model = LocalSummaryLinear().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    worker, query, targets = v0.exhaustive_router_domain(device)
    curves: list[dict[str, Any]] = []
    checkpoints = {1, 64, 256, 512, 1024, 2048}
    for step in range(1, 2049):
        optimizer.zero_grad(set_to_none=True)
        loss = v3.exhaustive_margin_loss(model(worker, query), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            calibration = v2.calibrate_thresholds(model, device)
            curves.append({"step": step, "loss": float(loss.detach().cpu()), "bias_margin": calibration["gates"]["bias"]["margin"], "contribution_margin": calibration["gates"]["contribution"]["margin"]})
    return model, curves


def _lbfgs(seed: int, device: torch.device):
    v0._configure(seed)
    model = LocalSummaryLinear().to(device)
    worker, query, targets = v0.exhaustive_router_domain(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=256, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    calls = 0
    started = time.perf_counter()

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        loss = v3.exhaustive_margin_loss(model(worker, query), targets)
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)
    calibration = v2.calibrate_thresholds(model, device)
    return model, [{"step": calls, "loss": float(final_loss.detach().cpu()), "seconds": time.perf_counter() - started, "bias_margin": calibration["gates"]["bias"]["margin"], "contribution_margin": calibration["gates"]["contribution"]["margin"]}]


def train(variant: str, seed: int, device: torch.device):
    if variant == "adamw_2048":
        return _adamw(seed, device)
    if variant == "lbfgs_full_batch":
        return _lbfgs(seed, device)
    if variant == "analytic_separator":
        model = analytic_model(device)
        calibration = v2.calibrate_thresholds(model, device)
        return model, [{"step": 0, "loss": None, "bias_margin": calibration["gates"]["bias"]["margin"], "contribution_margin": calibration["gates"]["contribution"]["margin"]}]
    raise ValueError(f"unknown convergence variant: {variant}")


def run(output_root: pathlib.Path, execution_head: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output exists: {output_root}")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed_index, seed in enumerate(v0.SEEDS):
            model, history = train(variant, seed, device)
            calibration = v2.calibrate_thresholds(model, device)
            states = v3.worst_states(model, device)
            rows.append({"variant": variant, "seed_index": seed_index, "initialization_seed": seed, "parameter_count": v0._parameter_count(model), "separable": calibration["separable"], "bias_margin": calibration["gates"]["bias"]["margin"], "contribution_margin": calibration["gates"]["contribution"]["margin"], "worst_states": states})
            curves.extend({"variant": variant, "seed_index": seed_index, **item} for item in history)
    reliable = [variant for variant in VARIANTS if all(row["separable"] for row in rows if row["variant"] == variant)]
    summary = {"status": "G9D_ROUTER_CONVERGENCE_ROBUSTNESS_COMPLETE_DEVELOPMENT_ONLY", "version": VERSION, "diagnosis": PASS if reliable else FAIL, "execution_head": execution_head, "base_head": BASE_HEAD, "variants": list(VARIANTS), "reliable_variants": reliable, "rows": rows, "boundaries": {"supervised_routing_labels_used": True, "end_to_end_answer_loss_used": False, "support_output_used_by_router": False, "operator_identity_visible": False, "automatic_coordinate_discovery_claimed": False, "population_confirmation_claimed": False, "frozen_result_modified": False}}
    (output_root / "aggregate-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for name, records in (("final-rows.jsonl", rows), ("curves.jsonl", curves)):
        with (output_root / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return summary
