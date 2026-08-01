from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / (
    "ai_hypothesis/population_compute/gate9d_fast_diagnostic_harness.py"
)
CLI_PATH = ROOT / "scripts/run_gate9d_fast_diagnostic_harness.py"
WRAPPER_PATH = ROOT / "scripts/run_gate9d_fast_diagnostic_harness.ps1"
DOC_PATH = ROOT / (
    "experiments/population_compute_scaling_v0/"
    "gate9d_fast_diagnostic_harness_v0.md"
)


def _load_harness():
    name = "gate9d_fast_harness_contract_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D fast harness contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _classification_rows(
    *,
    lookup: tuple[bool, bool, bool],
    parity: tuple[bool, bool, bool],
    query: tuple[bool, bool, bool],
    full: tuple[bool, bool, bool],
) -> list[dict[str, object]]:
    results = []
    for variant, values in (
        ("byte_lookup_zero_init", lookup),
        ("parity_feature_linear", parity),
        ("current_query_only", query),
        ("current_full_context", full),
    ):
        for seed_index, passes in enumerate(values):
            results.append(
                {
                    "variant": variant,
                    "seed_index": seed_index,
                    "passes": passes,
                }
            )
    return results


def verify_gate9d_fast_harness_contract() -> None:
    if importlib.util.find_spec("torch") is None:
        return

    import torch
    import torch.nn.functional as F

    harness = _load_harness()
    assert harness.GATE9D_FAST_HARNESS_VERSION == (
        "gate9d-fast-diagnostic-harness-v0"
    )
    assert harness.GATE9D_FAST_HARNESS_STATUS == (
        "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
    )
    assert harness.GATE9D_FAST_HARNESS_BRANCH == (
        "agent/gate9d-fast-diagnostic-harness-v0"
    )
    assert harness.GATE9D_FROZEN_STAGE1_EXECUTION_HEAD == (
        "2e1b91d578e7bf9b4c54aa2ee1c120a9ec01b21c"
    )
    assert harness.GATE9D_FAST_VARIANTS == (
        "byte_lookup_zero_init",
        "parity_feature_linear",
        "current_query_only",
        "current_full_context",
    )
    assert harness.GATE9D_FAST_CHECKPOINT_STEPS == (
        0,
        1,
        16,
        64,
        128,
        256,
        512,
        1024,
    )
    assert harness.GATE9D_FAST_TRAIN_STEPS == 1024

    queries = torch.tensor([3, 5, 6, 7], dtype=torch.long)
    targets = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    lookup = harness.ByteLookupBaseline()
    parity = harness.ParityFeatureLinear()
    assert sum(parameter.numel() for parameter in lookup.parameters()) == 2048
    assert sum(parameter.numel() for parameter in parity.parameters()) == 2056
    assert tuple(lookup(queries).shape) == (4, 8)
    assert tuple(parity.features(queries).shape) == (4, 256)
    assert set(parity.features(queries).unique().tolist()) <= {-1.0, 1.0}
    assert bool(torch.equal(lookup(queries), torch.zeros(4, 8)))

    optimizer = torch.optim.AdamW(lookup.parameters(), lr=1.0e-3)
    loss = F.binary_cross_entropy_with_logits(
        lookup(queries), harness.target_bits(targets)
    )
    loss.backward()
    optimizer.step()
    assert harness.metrics_from_logits(lookup(queries), targets)["passes"]

    exact_logits = (harness.target_bits(targets) * 2.0 - 1.0) * 10.0
    exact = harness.metrics_from_logits(exact_logits, targets)
    assert exact == {
        "rows": 4,
        "exact_correct": 4,
        "exact_accuracy": 1.0,
        "bit_correct": 32,
        "bit_total": 32,
        "bit_accuracy": 1.0,
        "passes": True,
    }

    all_pass = (True, True, True)
    all_fail = (False, False, False)
    mixed = (True, False, True)
    assert harness.classify_fast_results(
        _classification_rows(
            lookup=all_fail,
            parity=all_pass,
            query=all_pass,
            full=all_pass,
        )
    ) == "G9D_FAST_LOOKUP_PIPELINE_FAILED"
    assert harness.classify_fast_results(
        _classification_rows(
            lookup=all_pass,
            parity=all_fail,
            query=all_pass,
            full=all_pass,
        )
    ) == "G9D_FAST_PARITY_REPRESENTATION_FAILED"
    assert harness.classify_fast_results(
        _classification_rows(
            lookup=all_pass,
            parity=all_pass,
            query=all_fail,
            full=all_fail,
        )
    ) == "G9D_FAST_CURRENT_QUERY_PATH_FAILED"
    assert harness.classify_fast_results(
        _classification_rows(
            lookup=all_pass,
            parity=all_pass,
            query=all_pass,
            full=all_fail,
        )
    ) == "G9D_FAST_SUPPORT_PATH_INTERFERENCE"
    assert harness.classify_fast_results(
        _classification_rows(
            lookup=all_pass,
            parity=all_pass,
            query=all_fail,
            full=all_pass,
        )
    ) == "G9D_FAST_SUPPORT_CONTEXT_RESCUES_QUERY_PATH"
    assert harness.classify_fast_results(
        _classification_rows(
            lookup=all_pass,
            parity=all_pass,
            query=mixed,
            full=all_fail,
        )
    ) == "G9D_FAST_CURRENT_WORKER_MIXED"

    harness_source = HARNESS_PATH.read_text(encoding="utf-8")
    cli_source = CLI_PATH.read_text(encoding="utf-8")
    wrapper_source = WRAPPER_PATH.read_bytes().decode("ascii")
    document = DOC_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "generate_gate9_test_world(",
        "scientific_assignment_key",
        "classify_diagnostic(",
        "GATE9D_COLLISION_OPERATOR_COUNTERS",
        "GATE9D_HELD_IN_OPERATOR_RANGE",
        "GATE9D_UNSEEN_EVAL_OPERATOR_RANGE",
    ):
        assert forbidden not in harness_source
        assert forbidden not in cli_source
    assert "zipfile.ZipFile" in cli_source
    assert "git-status.txt" in cli_source
    assert cli_source.index('status = _git("status", "--porcelain")') < (
        cli_source.index("harness.run_fast_diagnostic")
    )
    assert "GATE9D_FAST_HARNESS_WRAPPER_SMOKE" in wrapper_source
    assert "DEVELOPMENT-ONLY" in document
    assert "not generally linearly separable" in document


verify_gate9d_fast_harness_contract()
