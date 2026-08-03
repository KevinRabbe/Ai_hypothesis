"""Development-only architecture sweep for exact Gate9D routing separation."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import torch
from torch import Tensor, nn

_ROOT = pathlib.Path(__file__).resolve().parent
_V3_PATH = _ROOT / "gate9d_learned_shared_router_exhaustive_margin_v3.py"

VERSION = "gate9d-router-architecture-sweep-v4"
STATUS = "DEVELOPMENT_ONLY_SUPERVISED_ROUTER_ARCHITECTURE_SWEEP"
BRANCH = "agent/gate9d-router-architecture-sweep-v4"
BASE_HEAD = "b501783bababfa4fd82763441aea60620b2c2de9"
TRAIN_STEPS = 512
LEARNING_RATE = 0.003
VARIANTS = ("raw_width128", "raw_deep64", "interaction16")
PASS = "G9D_ROUTER_ARCHITECTURE_SWEEP_PASSES"
FAIL = "G9D_ROUTER_ARCHITECTURE_SWEEP_NO_VARIANT_SEPARATES"


def _load_v3():
    name = "gate9d_router_sweep_v3_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load exhaustive-margin v3 dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = _load_v3()
v2, v0 = v3.v2, v3.v0


def raw_features(worker: Tensor, query: Tensor) -> Tensor:
    return torch.cat((v0.sparse.byte_bits(worker).float(), v0.sparse.byte_bits(query).float()), dim=-1)


def interaction_features(worker: Tensor, query: Tensor) -> Tensor:
    worker_bits = v0.sparse.byte_bits(worker).float()
    query_bits = v0.sparse.byte_bits(query).float()
    return torch.cat((worker_bits, query_bits, worker_bits * query_bits), dim=-1)


class RawWidth128(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 2))
    def forward(self, worker: Tensor, query: Tensor) -> Tensor:
        return self.net(raw_features(worker, query))


class RawDeep64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
        )
    def forward(self, worker: Tensor, query: Tensor) -> Tensor:
        return self.net(raw_features(worker, query))


class Interaction16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, worker: Tensor, query: Tensor) -> Tensor:
        return self.net(interaction_features(worker, query))


def make_model(variant: str) -> nn.Module:
    if variant == "raw_width128": return RawWidth128()
    if variant == "raw_deep64": return RawDeep64()
    if variant == "interaction16": return Interaction16()
    raise ValueError(f"unknown router sweep variant: {variant}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_variant(variant: str, seed: int, device: torch.device) -> tuple[nn.Module, list[dict[str, Any]]]:
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
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            calibration = v2.calibrate_thresholds(model, device)
            curves.append({
                "variant": variant,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "seconds": time.perf_counter() - started,
                "separable": calibration["separable"],
                "bias_margin": calibration["gates"]["bias"]["margin"],
                "contribution_margin": calibration["gates"]["contribution"]["margin"],
            })
    return model, curves


def classify(rows: list[dict[str, Any]]) -> tuple[str, str | None]:
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if len(selected) == 3 and all(row["separable"] for row in selected):
            return PASS, variant
    return FAIL, None


def run(output_root: pathlib.Path, execution_head: str) -> dict[str, Any]:
    if output_root.exists(): raise FileExistsError(f"output exists: {output_root}")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed_index, seed in enumerate(v0.SEEDS):
            model, run_curves = train_variant(variant, seed, device)
            curves.extend({"seed_index": seed_index, **row} for row in run_curves)
            calibration = v2.calibrate_thresholds(model, device)
            state = v3.worst_states(model, device)
            rows.append({
                "variant": variant,
                "seed_index": seed_index,
                "initialization_seed": seed,
                "parameter_count": parameter_count(model),
                "separable": calibration["separable"],
                "bias_margin": calibration["gates"]["bias"]["margin"],
                "contribution_margin": calibration["gates"]["contribution"]["margin"],
            })
            worst.append({"variant": variant, "seed_index": seed_index, **state})
    diagnosis, winning_variant = classify(rows)
    summary = {
        "status": "G9D_ROUTER_ARCHITECTURE_SWEEP_COMPLETE_DEVELOPMENT_ONLY",
        "version": VERSION,
        "diagnosis": diagnosis,
        "winning_variant": winning_variant,
        "execution_head": execution_head,
        "base_head": BASE_HEAD,
        "variants": list(VARIANTS),
        "training_steps": TRAIN_STEPS,
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
    (output_root / "aggregate-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for name, data in (("final-rows.jsonl", rows), ("curves.jsonl", curves)):
        with (output_root / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in data: handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return summary
