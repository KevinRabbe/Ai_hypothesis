"""Development-only factorized routing sweep after raw MLP architecture failure."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import torch
from torch import nn

_ROOT = pathlib.Path(__file__).resolve().parent
_V4_PATH = _ROOT / "gate9d_router_architecture_sweep_v4.py"

VERSION = "gate9d-router-factorization-sweep-v5"
STATUS = "DEVELOPMENT_ONLY_SUPERVISED_ROUTING_FACTORIZATION_SWEEP"
BRANCH = "agent/gate9d-router-factorization-sweep-v5"
BASE_HEAD = "ac7fc7ecaadcf46d4f55db6586984dd719eb5100"
VARIANTS = ("decoupled_raw64", "factorized_overlap16", "local_summary_linear")
TRAIN_STEPS = 512
LEARNING_RATE = 0.003
PASS = "G9D_ROUTER_FACTORIZATION_SWEEP_PASSES"
FAIL = "G9D_ROUTER_FACTORIZATION_SWEEP_NO_VARIANT_SEPARATES"


def _load_v4():
    name = "gate9d_router_factorization_v4_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _V4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load architecture-sweep dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v4 = _load_v4()
v3 = v4.v3
v2 = v3.v2
v0 = v3.v0


def _bits(values: torch.Tensor) -> torch.Tensor:
    return v0.sparse.byte_bits(values).to(torch.float32)


def _overlap(worker: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    return (_bits(worker) * _bits(query)).sum(dim=1, keepdim=True)


class DecoupledRaw64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        self.contribution = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, worker: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        worker_bits = _bits(worker)
        raw = torch.cat((worker_bits, _bits(query)), dim=1)
        return torch.cat((self.bias(worker_bits), self.contribution(raw)), dim=1)


class FactorizedOverlap16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        self.contribution = nn.Sequential(
            nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, worker: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        worker_bits = _bits(worker)
        contribution_features = torch.cat((worker_bits, _overlap(worker, query)), dim=1)
        return torch.cat(
            (self.bias(worker_bits), self.contribution(contribution_features)), dim=1
        )


class LocalSummaryLinear(nn.Module):
    """Linear gates over zero, worker-popcount, and local worker/query overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(3, 2)

    def features(self, worker: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        worker_bits = _bits(worker)
        zero = (worker == 0).to(torch.float32).unsqueeze(1)
        popcount = worker_bits.sum(dim=1, keepdim=True)
        overlap = (worker_bits * _bits(query)).sum(dim=1, keepdim=True)
        return torch.cat((zero, popcount, overlap), dim=1)

    def forward(self, worker: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(worker, query))


def make_model(variant: str) -> nn.Module:
    if variant == "decoupled_raw64":
        return DecoupledRaw64()
    if variant == "factorized_overlap16":
        return FactorizedOverlap16()
    if variant == "local_summary_linear":
        return LocalSummaryLinear()
    raise ValueError(f"unknown factorization variant: {variant}")


def train_variant(variant: str, seed: int, device: torch.device):
    v0._configure(seed)
    model = make_model(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    worker, query, targets = v0.exhaustive_router_domain(device)
    checkpoints = {1, 16, 64, 128, 256, 512}
    curves: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(worker, query)
        loss = v3.exhaustive_margin_loss(logits, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            calibration = v2.calibrate_thresholds(model, device)
            curves.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "seconds": time.perf_counter() - started,
                "separable": calibration["separable"],
                "bias_margin": calibration["gates"]["bias"]["margin"],
                "contribution_margin": calibration["gates"]["contribution"]["margin"],
            })
    return model, curves


def run(output_root: pathlib.Path, execution_head: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output exists: {output_root}")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed_index, seed in enumerate(v0.SEEDS):
            model, variant_curves = train_variant(variant, seed, device)
            curves.extend(
                {"variant": variant, "seed_index": seed_index, **row}
                for row in variant_curves
            )
            calibration = v2.calibrate_thresholds(model, device)
            states = v3.worst_states(model, device)
            rows.append({
                "variant": variant,
                "seed_index": seed_index,
                "initialization_seed": seed,
                "parameter_count": v0._parameter_count(model),
                "separable": calibration["separable"],
                "bias_margin": calibration["gates"]["bias"]["margin"],
                "contribution_margin": calibration["gates"]["contribution"]["margin"],
            })
            worst.append({"variant": variant, "seed_index": seed_index, **states})
    winning_variant = next(
        (
            variant
            for variant in VARIANTS
            if all(row["separable"] for row in rows if row["variant"] == variant)
        ),
        None,
    )
    summary = {
        "status": "G9D_ROUTER_FACTORIZATION_SWEEP_COMPLETE_DEVELOPMENT_ONLY",
        "version": VERSION,
        "diagnosis": PASS if winning_variant is not None else FAIL,
        "execution_head": execution_head,
        "base_head": BASE_HEAD,
        "variants": list(VARIANTS),
        "training_steps": TRAIN_STEPS,
        "winning_variant": winning_variant,
        "rows": rows,
        "worst_states": worst,
        "boundaries": {
            "supervised_routing_labels_used": True,
            "end_to_end_answer_loss_used": False,
            "support_output_used_by_router": False,
            "operator_identity_visible": False,
            "automatic_coordinate_discovery_claimed": False,
            "population_confirmation_claimed": False,
            "frozen_result_modified": False,
        },
    }
    (output_root / "aggregate-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, records in (("final-rows.jsonl", rows), ("curves.jsonl", curves)):
        with (output_root / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return summary
