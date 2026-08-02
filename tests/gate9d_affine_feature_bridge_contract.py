from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BRIDGE_PATH = ROOT / (
    "ai_hypothesis/population_compute/gate9d_affine_feature_bridge.py"
)
CLI_PATH = ROOT / "scripts/run_gate9d_affine_feature_bridge.py"
WRAPPER_PATH = ROOT / "scripts/run_gate9d_affine_feature_bridge.ps1"
DOC_PATH = ROOT / (
    "experiments/population_compute_scaling_v0/"
    "gate9d_affine_feature_bridge_v0.md"
)


def _load_bridge():
    name = "gate9d_affine_feature_bridge_contract_module"
    spec = importlib.util.spec_from_file_location(name, BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D affine feature bridge contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_gate9d_affine_feature_bridge_contract() -> None:
    if importlib.util.find_spec("torch") is None:
        return

    import torch
    import torch.nn.functional as F

    bridge = _load_bridge()
    assert bridge.GATE9D_AFFINE_BRIDGE_VERSION == (
        "gate9d-affine-feature-bridge-v0"
    )
    assert bridge.GATE9D_AFFINE_BRIDGE_STATUS == (
        "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
    )
    assert bridge.GATE9D_AFFINE_BRIDGE_BRANCH == (
        "agent/gate9d-affine-feature-bridge-v0"
    )
    assert bridge.GATE9D_AFFINE_BRIDGE_BASE_HEAD == (
        "f9cff8e1609cfae5642f8cef2242eee74f9488c7"
    )
    assert bridge.GATE9D_AFFINE_BRIDGE_TRAIN_COUNTER_START == 1 << 57
    assert bridge.GATE9D_AFFINE_BRIDGE_TRAIN_OPERATOR_COUNT == 256
    assert bridge.GATE9D_AFFINE_BRIDGE_EVAL_COUNTER_START == (1 << 57) + 0x1000
    assert bridge.GATE9D_AFFINE_BRIDGE_EVAL_OPERATOR_COUNT == 64
    assert bridge.GATE9D_AFFINE_BRIDGE_QUERY_COUNT == 247
    assert bridge.GATE9D_AFFINE_BRIDGE_TRAIN_STEPS == 512
    assert bridge.GATE9D_AFFINE_BRIDGE_BATCH_SIZE == 512
    assert bridge.GATE9D_AFFINE_BRIDGE_CHECKPOINTS == (
        0,
        1,
        16,
        32,
        64,
        128,
        256,
        512,
    )

    model = bridge.AffineFeatureBridgeDecoder()
    assert model.learned_parameter_count() == 65
    assert sum(parameter.numel() for parameter in model.parameters()) == 65
    assert tuple(model.hidden.weight.shape) == (16, 2)
    assert tuple(model.hidden.bias.shape) == (16,)
    assert tuple(model.output.weight.shape) == (1, 16)
    assert tuple(model.output.bias.shape) == (1,)

    counters = (
        bridge.GATE9D_AFFINE_BRIDGE_TRAIN_COUNTER_START,
        bridge.GATE9D_AFFINE_BRIDGE_EVAL_COUNTER_START,
    )
    support_rows = []
    queries = []
    targets = []
    for counter in counters:
        operator = bridge.operators.operator_from_counter(counter)
        support = bridge.operators.public_support_pairs(operator)
        assert tuple(source for source, _ in support) == (
            bridge.GATE9D_AFFINE_BRIDGE_SUPPORT_ORDER
        )
        support_outputs = tuple(target for _, target in support)
        for query in (3, 5, 6, 7, 255):
            support_rows.append(support_outputs)
            queries.append(query)
            targets.append(operator.apply(query))

    support_tensor = torch.tensor(support_rows, dtype=torch.long)
    query_tensor = torch.tensor(queries, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    bias_bits, mask_bits = bridge.affine_signature(support_tensor)
    assert tuple(bias_bits.shape) == (10, 8)
    assert tuple(mask_bits.shape) == (10, 8, 8)
    assert set(bias_bits.unique().tolist()) <= {0.0, 1.0}
    assert set(mask_bits.unique().tolist()) <= {0.0, 1.0}

    features = bridge.affine_bridge_features(support_tensor, query_tensor)
    assert tuple(features.shape) == (10, 8, 2)
    assert set(features.unique().tolist()) <= {-1.0, 1.0}
    oracle = bridge.fixed_bridge_answers(support_tensor, query_tensor)
    assert torch.equal(oracle, target_tensor)

    logits = model(support_tensor, query_tensor)
    assert tuple(logits.shape) == (10, 8)
    target_bits = bridge.byte_bits(target_tensor)
    loss = F.binary_cross_entropy_with_logits(logits, target_bits)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert bridge._active_gradient_elements(model) > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    query_only = model.forward_query_only(query_tensor)
    assert tuple(query_only.shape) == (10, 8)

    assert bridge.learning_rate_at_step(1) == 0.001 / 16
    assert bridge.learning_rate_at_step(16) == 0.001
    assert bridge.learning_rate_at_step(512) == 0.0001
    try:
        bridge.learning_rate_at_step(0)
    except ValueError:
        pass
    else:
        raise AssertionError("Gate9D affine bridge admitted step zero as training")

    assert bridge.classify_seed_passes((True, True, True)) == (
        "G9D_AFFINE_FEATURE_BRIDGE_PASSES"
    )
    assert bridge.classify_seed_passes((False, False, False)) == (
        "G9D_AFFINE_FEATURE_BRIDGE_FAILED"
    )
    assert bridge.classify_seed_passes((True, False, False)) == (
        "G9D_AFFINE_FEATURE_BRIDGE_MIXED"
    )

    perfect = {"exact_accuracy": 1.0, "bit_accuracy": 1.0}
    chance = {"exact_accuracy": 1.0 / 256.0, "bit_accuracy": 0.5}
    assert bridge.seed_passes(perfect, chance, chance, perfect)
    assert not bridge.seed_passes(chance, chance, chance, perfect)

    train = set(bridge.train_counters())
    evaluation = set(bridge.evaluation_counters())
    assert len(train) == 256
    assert len(evaluation) == 64
    assert not train & evaluation
    frozen_ranges = (
        range(0, 262_144),
        range(1 << 32, (1 << 32) + 32_768),
        range(1 << 40, (1 << 40) + 4_096),
        range(1 << 48, (1 << 48) + 2_629_632),
        range(1 << 56, (1 << 56) + 0x3000),
    )
    for counter in train | evaluation:
        assert all(counter not in frozen for frozen in frozen_ranges)

    source = BRIDGE_PATH.read_text(encoding="utf-8")
    cli = CLI_PATH.read_text(encoding="utf-8")
    wrapper = WRAPPER_PATH.read_bytes().decode("ascii")
    document = DOC_PATH.read_text(encoding="utf-8")
    assert "65-parameter decoder" in source
    assert "operator_counter_visible_to_model" in source
    assert '"operator_counter_visible_to_model": False' in source
    assert '"operator_key_visible_to_model": False' in source
    assert '"per_operator_parameters": False' in source
    assert "zipfile.ZipFile" in cli
    assert "git-status.txt" in cli
    assert cli.index('status = _git("status", "--porcelain")') < cli.index(
        "bridge.run_affine_feature_bridge"
    )
    assert '"device": "cpu"' in cli
    assert "GATE9D_AFFINE_BRIDGE_WRAPPER_SMOKE" in wrapper
    assert "DEVELOPMENT-ONLY" in document
    assert "302.29" in document
    for forbidden in (
        "generate_gate9_test_world(",
        "population_runtime",
        "classify_diagnostic(",
        "torch.save(",
        "selected-checkpoint",
    ):
        assert forbidden not in source
        assert forbidden not in cli


verify_gate9d_affine_feature_bridge_contract()
