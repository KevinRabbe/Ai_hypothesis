from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ai_hypothesis/population_compute/gate9d_learned_shared_router.py"
CLI_PATH = ROOT / "scripts/run_gate9d_learned_shared_router.py"
WRAPPER_PATH = ROOT / "scripts/run_gate9d_learned_shared_router.ps1"
DOC_PATH = ROOT / "experiments/population_compute_scaling_v0/gate9d_learned_shared_router_v0.md"


def _load():
    spec = importlib.util.spec_from_file_location("gate9d_learned_router_contract_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load learned router contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    module = _load()
    assert module.VERSION == "gate9d-learned-shared-router-v0"
    assert module.BASE_HEAD == "6ad02bd4f0907bafa6a1d202eb157d701e26cbe8"
    assert module.POPULATION_SIZES == (9, 16, 64, 256)
    model = module.SharedRouter()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1218

    worker = torch.tensor([0, 1, 2, 3, 128], dtype=torch.long)
    query = torch.tensor([0, 1, 2, 255, 128], dtype=torch.long)
    targets = module.routing_targets(worker, query)
    assert targets.tolist() == [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ]
    logits = model(worker, query)
    assert tuple(logits.shape) == (5, 2)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())

    domain_worker, domain_query, domain_targets = module.exhaustive_router_domain(torch.device("cpu"))
    assert domain_worker.numel() == 65536
    assert domain_query.numel() == 65536
    assert tuple(domain_targets.shape) == (65536, 2)
    assert int(domain_targets[:, 0].sum()) == 256
    assert int(domain_targets[:, 1].sum()) == 1024

    source = MODULE_PATH.read_text(encoding="utf-8")
    cli = CLI_PATH.read_text(encoding="utf-8")
    wrapper = WRAPPER_PATH.read_bytes().decode("ascii")
    document = DOC_PATH.read_text(encoding="utf-8")
    assert "support_output_used_by_router\": False" in source
    assert "supervised_routing_labels_used\": True" in source
    assert "automatic_coordinate_discovery_claimed\": False" in source
    assert "zipfile.ZipFile" in cli
    assert "git-status.txt" in cli
    assert "GATE9D_LEARNED_ROUTER_WRAPPER_SMOKE" in wrapper
    assert "SUPERVISED ROUTING" in document
    for forbidden in (
        "generate_gate9_test_world(",
        "scientific_assignment_key",
        "torch.save(",
    ):
        assert forbidden not in source
        assert forbidden not in cli


verify()
