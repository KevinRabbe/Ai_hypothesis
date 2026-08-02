from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = ROOT / (
    "ai_hypothesis/population_compute/gate9d_query_capacity_diagnostic.py"
)
CLI_PATH = ROOT / "scripts/run_gate9d_query_capacity_diagnostic.py"
WRAPPER_PATH = ROOT / "scripts/run_gate9d_query_capacity_diagnostic.ps1"
DOC_PATH = ROOT / (
    "experiments/population_compute_scaling_v0/"
    "gate9d_query_capacity_diagnostic_v0.md"
)


def _load_diagnostic():
    name = "gate9d_query_capacity_contract_module"
    spec = importlib.util.spec_from_file_location(name, DIAGNOSTIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D query-capacity contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _classification_rows(
    *,
    current_1024: tuple[bool, bool, bool],
    current_4096: tuple[bool, bool, bool],
    raw_32: tuple[bool, bool, bool],
    raw_64: tuple[bool, bool, bool],
    walsh_24: tuple[bool, bool, bool],
) -> list[dict[str, object]]:
    rows = []
    for variant, values in (
        ("current_query_only_1024", current_1024),
        ("current_query_only_4096", current_4096),
        ("raw_bits_tanh_32", raw_32),
        ("raw_bits_tanh_64", raw_64),
        ("walsh_tanh_24", walsh_24),
    ):
        for seed_index, passes in enumerate(values):
            rows.append(
                {
                    "variant": variant,
                    "seed_index": seed_index,
                    "passes": passes,
                }
            )
    return rows


def verify_gate9d_query_capacity_contract() -> None:
    if importlib.util.find_spec("torch") is None:
        return

    import torch
    import torch.nn.functional as F

    diagnostic = _load_diagnostic()
    assert diagnostic.GATE9D_QUERY_CAPACITY_VERSION == (
        "gate9d-query-capacity-diagnostic-v0"
    )
    assert diagnostic.GATE9D_QUERY_CAPACITY_STATUS == (
        "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
    )
    assert diagnostic.GATE9D_QUERY_CAPACITY_BRANCH == (
        "agent/gate9d-query-capacity-diagnostic-v0"
    )
    assert diagnostic.GATE9D_QUERY_CAPACITY_BASE_HEAD == (
        "84f6038dc58547718d1d1ab9df7ef11538f21fb8"
    )
    assert diagnostic.GATE9D_QUERY_CAPACITY_VARIANTS == (
        "current_query_only_1024",
        "current_query_only_4096",
        "raw_bits_tanh_32",
        "raw_bits_tanh_64",
        "walsh_tanh_24",
    )
    assert diagnostic.GATE9D_QUERY_CAPACITY_STEPS == {
        "current_query_only_1024": 1024,
        "current_query_only_4096": 4096,
        "raw_bits_tanh_32": 1024,
        "raw_bits_tanh_64": 1024,
        "walsh_tanh_24": 1024,
    }

    raw_32 = diagnostic.RawBitsTanh(32)
    raw_64 = diagnostic.RawBitsTanh(64)
    walsh = diagnostic.WalshTanh24()
    assert sum(parameter.numel() for parameter in raw_32.parameters()) == 544
    assert sum(parameter.numel() for parameter in raw_64.parameters()) == 1088
    assert sum(parameter.numel() for parameter in walsh.parameters()) == 6360

    queries = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    assert tuple(raw_32(queries).shape) == (4, 8)
    assert tuple(raw_64(queries).shape) == (4, 8)
    assert tuple(walsh.features(queries).shape) == (4, 256)
    assert set(walsh.features(queries).unique().tolist()) <= {-1.0, 1.0}

    loss = F.binary_cross_entropy_with_logits(
        raw_32(queries), diagnostic.byte_bits(targets)
    )
    assert bool(torch.isfinite(loss))
    loss.backward()
    active = diagnostic._active_gradient_elements(raw_32)
    assert 0 < active <= 544
    assert all(parameter.grad is not None for parameter in raw_32.parameters())
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in raw_32.parameters()
    )

    all_queries = torch.arange(256, dtype=torch.long)
    identity_contract = diagnostic._affine_parity_contract(
        all_queries, all_queries
    )
    assert [row["input_mask"] for row in identity_contract] == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ]
    assert [row["bias"] for row in identity_contract] == [0] * 8
    assert [row["parity_weight"] for row in identity_contract] == [1] * 8

    runtime = diagnostic._load_stage1_runtime()
    assert diagnostic._learning_rate(runtime, 1024) == 0.0001
    assert diagnostic._learning_rate(runtime, 1025) == 0.0001
    assert diagnostic._learning_rate(runtime, 4096) == 0.0001

    passed = (True, True, True)
    failed = (False, False, False)
    mixed = (True, False, False)
    assert diagnostic.classify_query_capacity(
        _classification_rows(
            current_1024=passed,
            current_4096=passed,
            raw_32=passed,
            raw_64=passed,
            walsh_24=passed,
        )
    ) == "G9D_QUERY_CAPACITY_FAILURE_NOT_REPRODUCED"
    assert diagnostic.classify_query_capacity(
        _classification_rows(
            current_1024=failed,
            current_4096=passed,
            raw_32=failed,
            raw_64=failed,
            walsh_24=failed,
        )
    ) == "G9D_QUERY_CAPACITY_TRAINING_BUDGET_LIMIT"
    assert diagnostic.classify_query_capacity(
        _classification_rows(
            current_1024=failed,
            current_4096=failed,
            raw_32=passed,
            raw_64=passed,
            walsh_24=passed,
        )
    ) == "G9D_QUERY_CAPACITY_24_UNIT_BOTTLENECK"
    assert diagnostic.classify_query_capacity(
        _classification_rows(
            current_1024=failed,
            current_4096=failed,
            raw_32=failed,
            raw_64=passed,
            walsh_24=passed,
        )
    ) == "G9D_QUERY_CAPACITY_BETWEEN_32_AND_64"
    assert diagnostic.classify_query_capacity(
        _classification_rows(
            current_1024=failed,
            current_4096=failed,
            raw_32=failed,
            raw_64=failed,
            walsh_24=passed,
        )
    ) == "G9D_QUERY_CAPACITY_RAW_BIT_PARITY_MISMATCH"
    assert diagnostic.classify_query_capacity(
        _classification_rows(
            current_1024=failed,
            current_4096=mixed,
            raw_32=passed,
            raw_64=passed,
            walsh_24=passed,
        )
    ) == "G9D_QUERY_CAPACITY_TRAINING_BUDGET_MIXED"

    source = DIAGNOSTIC_PATH.read_text(encoding="utf-8")
    cli = CLI_PATH.read_text(encoding="utf-8")
    wrapper = WRAPPER_PATH.read_bytes().decode("ascii")
    document = DOC_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "generate_gate9_test_world(",
        "scientific_assignment_key",
        "classify_diagnostic(",
        "GATE9D_COLLISION_OPERATOR_COUNTERS",
        "GATE9D_HELD_IN_OPERATOR_RANGE",
        "GATE9D_UNSEEN_EVAL_OPERATOR_RANGE",
    ):
        assert forbidden not in source
        assert forbidden not in cli
    assert "zipfile.ZipFile" in cli
    assert "git-status.txt" in cli
    assert cli.index('status = _git("status", "--porcelain")') < cli.index(
        "diagnostic.run_query_capacity_diagnostic"
    )
    assert "GATE9D_QUERY_CAPACITY_WRAPPER_SMOKE" in wrapper
    assert "DEVELOPMENT-ONLY" in document
    assert "five-input parity" in document
    assert "six-input parity" in document


verify_gate9d_query_capacity_contract()
