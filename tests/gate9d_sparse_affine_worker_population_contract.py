from __future__ import annotations

import importlib.util
import pathlib
import sys

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


def _load_diagnostic():
    name = "gate9d_sparse_population_contract_module"
    spec = importlib.util.spec_from_file_location(name, DIAGNOSTIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D sparse population contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_gate9d_sparse_population_contract() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    diagnostic = _load_diagnostic()
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
    assert diagnostic.GATE9D_SPARSE_POPULATION_COUNTER_START == (
        (1 << 57) + 0x2000
    )

    distractors = diagnostic.distractor_inputs(247)
    assert len(distractors) == 247
    assert all(value != 0 and value & (value - 1) != 0 for value in distractors)
    assert diagnostic.distractor_output(1 << 57, 9, distractors[0]) == (
        diagnostic.distractor_output(1 << 57, 9, distractors[0])
    )

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
        assert inputs == diagnostic.SUPPORT_ORDER
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
            support_inputs,
            support_outputs,
            counter_tensor,
            population_size,
        )
        result = diagnostic.sparse_population_execute(
            inputs,
            outputs,
            query_tensor,
        )
        assert torch.equal(result.predictions, target_tensor)
        permutation = diagnostic.deterministic_permutation(population_size)
        permuted = diagnostic.sparse_population_execute(
            inputs[:, permutation],
            outputs[:, permutation],
            query_tensor,
        )
        assert torch.equal(permuted.predictions, target_tensor)
        assert result.nominal_population_size == population_size
        assert result.bias_messages == target_tensor.numel()
        assert result.contribution_messages == sum(
            int(query).bit_count() for query in queries
        )
        assert result.active_worker_count <= 9

    passing_rows = [
        {
            "population_size": size,
            "parameter_count": 0,
            "full_exact_accuracy": 1.0,
            "permuted_exact_accuracy": 1.0,
            "shuffled_exact_accuracy": 0.004,
            "no_bias_exact_accuracy": 0.004,
        }
        for size in diagnostic.GATE9D_SPARSE_POPULATION_SIZES
    ]
    assert diagnostic.classify_population(passing_rows) == (
        "G9D_SPARSE_AFFINE_POPULATION_PASSES"
    )
    failing_rows = [dict(row) for row in passing_rows]
    failing_rows[-1]["full_exact_accuracy"] = 0.999
    assert diagnostic.classify_population(failing_rows) == (
        "G9D_SPARSE_AFFINE_POPULATION_FAILED"
    )

    source = DIAGNOSTIC_PATH.read_text(encoding="utf-8")
    cli = CLI_PATH.read_text(encoding="utf-8")
    wrapper = WRAPPER_PATH.read_bytes().decode("ascii")
    document = DOC_PATH.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "nn.Module" not in source
    assert "learned_parameter_count\": 0" in source
    assert "automatic_coordinate_discovery_claimed\": False" in source
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
    assert cli.index('status = _git("status", "--porcelain")') < cli.index(
        "diagnostic.run_sparse_population_diagnostic"
    )
    assert "GATE9D_SPARSE_POPULATION_WRAPPER_SMOKE" in wrapper
    assert "DEVELOPMENT-ONLY" in document
    assert "does not establish" in document


verify_gate9d_sparse_population_contract()
