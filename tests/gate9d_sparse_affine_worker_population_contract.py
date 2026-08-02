from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = ROOT / (
    "ai_hypothesis/population_compute/"
    "gate9d_sparse_affine_worker_population.py"
)
CLI_PATH = ROOT / "scripts/run_gate9d_sparse_affine_worker_population.py"
WRAPPER_PATH = ROOT / "scripts/run_gate9d_sparse_affine_worker_population.ps1"
DOC_PATH = ROOT / (
    "experiments/population_compute_scaling_v0/"
    "gate9d_sparse_affine_worker_population_v0.md"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_gate9d_sparse_population_contract() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    diagnostic = _load(DIAGNOSTIC_PATH, "gate9d_sparse_population_contract_module")
    runner = _load(CLI_PATH, "gate9d_sparse_population_runner_contract")
    assert diagnostic.GATE9D_SPARSE_POPULATION_VERSION == (
        "gate9d-sparse-affine-worker-population-v0"
    )
    assert diagnostic.GATE9D_SPARSE_POPULATION_STATUS == (
        "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
    )
    assert diagnostic.GATE9D_SPARSE_POPULATION_BASE_HEAD == (
        "c0242268f2938fe1131f2aa90c87b5a48ae248f6"
    )
    assert diagnostic.GATE9D_SPARSE_POPULATION_SIZES == (9, 16, 64, 256)
    assert diagnostic.GATE9D_SPARSE_POPULATION_PARAMETER_COUNT == 0
    assert diagnostic.GATE9D_SPARSE_POPULATION_OPERATOR_COUNT == 128

    support_input_rows = []
    support_output_rows = []
    queries = []
    targets = []
    counters = []
    for counter in (
        diagnostic.GATE9D_SPARSE_POPULATION_COUNTER_START,
        diagnostic.GATE9D_SPARSE_POPULATION_COUNTER_START + 1,
    ):
        operator = diagnostic.operators.operator_from_counter(counter)
        supports = diagnostic.operators.public_support_pairs(operator)
        inputs = tuple(source for source, _ in supports)
        outputs = tuple(target for _, target in supports)
        for query in diagnostic.QUERY_VALUES:
            support_input_rows.append(inputs)
            support_output_rows.append(outputs)
            queries.append(query)
            targets.append(operator.apply(query))
            counters.append(counter)

    support_inputs = torch.tensor(support_input_rows, dtype=torch.long)
    support_outputs = torch.tensor(support_output_rows, dtype=torch.long)
    query_tensor = torch.tensor(queries, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    counter_tensor = torch.tensor(counters, dtype=torch.long)

    for population_size in diagnostic.GATE9D_SPARSE_POPULATION_SIZES:
        inputs, outputs = diagnostic.augment_population(
            support_inputs, support_outputs, counter_tensor, population_size
        )
        result = diagnostic.sparse_population_execute(inputs, outputs, query_tensor)
        assert torch.equal(result.predictions, target_tensor)
        permutation = diagnostic.deterministic_permutation(population_size)
        permuted = diagnostic.sparse_population_execute(
            inputs[:, permutation], outputs[:, permutation], query_tensor
        )
        assert torch.equal(permuted.predictions, target_tensor)
        assert result.bias_messages == target_tensor.numel()
        assert result.contribution_messages == sum(
            int(query).bit_count() for query in queries
        )
        assert result.active_worker_count <= 9

    # Prove the corrected causal subset on a small materialized bundle.
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary) / "bundle"
        summary = diagnostic.run_sparse_population_diagnostic(
            root, "a" * 40
        )
        assert summary["diagnosis"] == diagnostic.GATE9D_SPARSE_POPULATION_FAIL
        corrected = runner._correct_causal_control(root)
        assert corrected["diagnosis"] == runner.CORRECTED_PASS
        assert corrected["diagnosis_legacy_aggregate_no_bias"] == (
            diagnostic.GATE9D_SPARSE_POPULATION_FAIL
        )
        for row in corrected["rows"]:
            assert row["no_bias_odd_exact_accuracy"] == 1.0
            assert row["no_bias_even_exact_accuracy"] <= runner.CONTROL_MAX
            assert row["full_exact_accuracy"] == 1.0
            assert row["permuted_exact_accuracy"] == 1.0
            assert row["shuffled_exact_accuracy"] <= runner.CONTROL_MAX
        stored = json.loads((root / "aggregate-summary.json").read_text())
        assert stored == corrected

    source = DIAGNOSTIC_PATH.read_text(encoding="utf-8")
    cli = CLI_PATH.read_text(encoding="utf-8")
    wrapper = WRAPPER_PATH.read_bytes().decode("ascii")
    document = DOC_PATH.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "nn.Module" not in source
    assert "_correct_causal_control" in cli
    assert "even-popcount queries" in cli
    assert "diagnosis_legacy_aggregate_no_bias" in cli
    for forbidden in (
        "generate_gate9_test_world(",
        "scientific_assignment_key",
        "classify_diagnostic(",
        "torch.save(",
    ):
        assert forbidden not in source
        assert forbidden not in cli
    assert "zipfile.ZipFile" in cli
    assert "git-status.txt" in cli
    assert "GATE9D_SPARSE_POPULATION_WRAPPER_SMOKE" in wrapper
    assert "DEVELOPMENT-ONLY" in document
    assert "even-popcount" in document
    assert "It would not establish" in document


verify_gate9d_sparse_population_contract()
