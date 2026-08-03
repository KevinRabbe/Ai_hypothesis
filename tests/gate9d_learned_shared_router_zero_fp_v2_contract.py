from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch
from torch import Tensor, nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ai_hypothesis/population_compute/gate9d_learned_shared_router_zero_fp_v2.py"
RUNNER_PATH = ROOT / "scripts/run_gate9d_learned_shared_router_zero_fp_v2.py"
WRAPPER_PATH = ROOT / "scripts/run_gate9d_learned_shared_router_zero_fp_v2.ps1"
DOC_PATH = ROOT / "experiments/population_compute_scaling_v0/gate9d_learned_shared_router_zero_fp_v2.md"


def _load():
    name = "gate9d_router_zero_fp_contract_module"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load zero-FP router contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ExactRouter(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, worker: Tensor, query: Tensor) -> Tensor:
        targets = self.module.v0.routing_targets(worker, query)
        return torch.where(targets.bool(), torch.full_like(targets, 4.0), torch.full_like(targets, -4.0))


class FlatRouter(nn.Module):
    def forward(self, worker: Tensor, query: Tensor) -> Tensor:
        return torch.zeros((worker.numel(), 2), dtype=torch.float32, device=worker.device)


def verify() -> None:
    module = _load()
    assert module.VERSION == "gate9d-learned-shared-router-zero-fp-v2"
    assert module.STATUS == "DEVELOPMENT_ONLY_SUPERVISED_ROUTING_ZERO_FP"
    assert module.BRANCH == "agent/gate9d-learned-shared-router-zero-fp-v2"
    assert module.BASE_HEAD == "5e89fb42d6a84e32f163d3309abbb2294206f9a1"

    device = torch.device("cpu")
    exact = ExactRouter(module)
    calibration = module.calibrate_thresholds(exact, device)
    assert calibration["separable"] is True
    assert calibration["gates"]["bias"]["margin"] == 8.0
    assert calibration["gates"]["contribution"]["margin"] == 8.0
    assert calibration["gates"]["bias"]["threshold"] == 0.0
    assert calibration["gates"]["contribution"]["threshold"] == 0.0
    assert calibration["gates"]["bias"]["positive_count"] == 256
    assert calibration["gates"]["contribution"]["positive_count"] == 1024

    worker, query, targets = module.v0.exhaustive_router_domain(device)
    predictions = module.threshold_predictions(exact(worker, query), calibration)
    assert torch.equal(predictions, targets.bool())
    metrics = module.calibrated_class_metrics(exact, device, calibration)
    assert all(value == 1.0 for value in metrics.values())

    flat_calibration = module.calibrate_thresholds(FlatRouter(), device)
    assert flat_calibration["separable"] is False
    assert flat_calibration["gates"]["bias"]["threshold"] is None
    assert flat_calibration["gates"]["contribution"]["threshold"] is None
    try:
        module.threshold_predictions(torch.zeros((1, 2)), flat_calibration)
    except RuntimeError as error:
        assert "not strictly separable" in str(error)
    else:
        raise AssertionError("non-separable router did not fail closed")

    operator = module.v0.sparse.operators.operator_from_counter(module.v0.COUNTER_START)
    supports = module.v0.sparse.operators.public_support_pairs(operator)
    worker_inputs = torch.tensor([[source for source, _ in supports]], dtype=torch.long)
    worker_outputs = torch.tensor([[target for _, target in supports]], dtype=torch.long)
    queries = torch.tensor([173], dtype=torch.long)
    population_inputs, population_outputs = module.v0.sparse.augment_population(
        worker_inputs,
        worker_outputs,
        torch.tensor([module.v0.COUNTER_START], dtype=torch.long),
        256,
    )
    result, stats = module.calibrated_execute(
        exact, calibration, population_inputs, population_outputs, queries
    )
    assert int(result[0]) == operator.apply(173)
    assert stats["bias_messages"] == 1
    assert stats["contribution_messages"] == 173 .bit_count()

    source = MODULE_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    wrapper = WRAPPER_PATH.read_bytes().decode("ascii")
    document = DOC_PATH.read_text(encoding="utf-8")
    assert "max_negative < min_positive" not in source
    assert "margin > 0.0" in source
    assert "threshold_predictions" in source
    assert "zipfile.ZipFile" in runner
    assert "git-status.txt" in runner
    assert runner.index('status = _git("status", "--porcelain")') < runner.index("summary = module.run")
    assert "GATE9D_ROUTER_ZERO_FP_WRAPPER_SMOKE" in wrapper
    assert "Development-only" in document
    for forbidden in (
        "scientific_assignment_key",
        "generate_gate9_test_world(",
        "optimizer_from_answer_loss",
        "automatic_coordinate_discovery = True",
    ):
        assert forbidden not in source
        assert forbidden not in runner


verify()
