"""Exact Gate-9 train/validation episode materialization and runtime guards."""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import random
import sys
from typing import Any, Iterable

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent
_OPERATOR_PATH = _ROOT / "gate9_contextual_operator_contract.py"
_PROTOCOL_PATH = _ROOT / "gate9_contextual_training_protocol.py"


def _load(path: pathlib.Path, name: str):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate9 training dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


operators = _load(_OPERATOR_PATH, "gate9_training_data_operator_dependency")
protocol = _load(_PROTOCOL_PATH, "gate9_training_data_protocol_dependency")

GATE9_VALIDATION_SHUFFLE_OFFSET = (
    int.from_bytes(
        hashlib.sha256(
            protocol.GATE9_VALIDATION_SHUFFLE_NAMESPACE.encode("ascii")
        ).digest()[:8],
        "big",
    )
    % (protocol.GATE9_VALIDATION_EPISODES - 1)
    + 1
)


def fast_operator_material(
    counter: int, query: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    key = operators.splitmix64_bijection(counter)
    lower_bits = key & ((1 << 28) - 1)
    upper_bits = (key >> 28) & ((1 << 28) - 1)
    bias = (key >> 56) & 0xFF
    lower, upper = operators._rows_from_triangular_bits(lower_bits, upper_bits)
    matrix = operators.multiply_gf2_rows(lower, upper)
    support = tuple(
        (value, operators.apply_linear_rows(matrix, value) ^ bias)
        for value in operators.GATE9_GLOBAL_SUPPORT_ORDER
    )
    return support, operators.apply_linear_rows(matrix, query) ^ bias


def validation_shuffled_episode_index(episode_index: int) -> int:
    if not 0 <= episode_index < protocol.GATE9_VALIDATION_EPISODES:
        raise ValueError("Gate9 validation episode index lies outside frozen range")
    return (
        episode_index + GATE9_VALIDATION_SHUFFLE_OFFSET
    ) % protocol.GATE9_VALIDATION_EPISODES


def batch_arrays(
    *,
    counters: Iterable[int],
    queries: Iterable[int],
    verify_public_oracle: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counter_rows = tuple(counters)
    query_rows = tuple(queries)
    if len(counter_rows) != len(query_rows) or not counter_rows:
        raise ValueError("Gate9 batch counters/queries disagree or are empty")
    support_outputs = np.empty((len(counter_rows), 9), dtype=np.uint8)
    targets = np.empty(len(counter_rows), dtype=np.uint8)
    oracle_targets = np.empty(len(counter_rows), dtype=np.uint8)
    for index, (counter, query) in enumerate(
        zip(counter_rows, query_rows, strict=True)
    ):
        support, target = fast_operator_material(counter, query)
        support_outputs[index] = [output for _, output in support]
        targets[index] = target
        oracle_targets[index] = (
            operators.apply_public_support_oracle(
                support, query, require_novel_query=True
            )
            if verify_public_oracle
            else target
        )
    support_inputs = np.broadcast_to(
        np.asarray(operators.GATE9_GLOBAL_SUPPORT_ORDER, dtype=np.uint8),
        support_outputs.shape,
    ).copy()
    return (
        support_inputs,
        support_outputs,
        np.asarray(query_rows, dtype=np.uint8),
        targets,
        oracle_targets,
    )


def training_batch_arrays(seed: int, step: int):
    if not 1 <= step <= protocol.GATE9_TRAIN_STEPS:
        raise ValueError("Gate9 training step lies outside 1..512")
    start = (step - 1) * protocol.GATE9_TRAIN_BATCH_SIZE
    episode_indices = range(start, start + protocol.GATE9_TRAIN_BATCH_SIZE)
    ordinals = np.fromiter(
        (protocol.training_operator_ordinal(seed, index) for index in episode_indices),
        dtype=np.int64,
        count=protocol.GATE9_TRAIN_BATCH_SIZE,
    )
    counters = ordinals + protocol.GATE9_TRAIN_OPERATOR_COUNTER_START
    queries = np.fromiter(
        (protocol.training_query(seed, int(ordinal)) for ordinal in ordinals),
        dtype=np.uint8,
        count=protocol.GATE9_TRAIN_BATCH_SIZE,
    )
    arrays = batch_arrays(
        counters=(int(value) for value in counters),
        queries=(int(value) for value in queries),
        verify_public_oracle=False,
    )
    return ordinals, counters, arrays


def validation_batch_arrays(batch_index: int):
    if not 0 <= batch_index < protocol.GATE9_VALIDATION_BATCHES:
        raise ValueError("Gate9 validation batch lies outside 0..63")
    start = batch_index * protocol.GATE9_VALIDATION_BATCH_SIZE
    indices = tuple(range(start, start + protocol.GATE9_VALIDATION_BATCH_SIZE))
    ordinals = np.fromiter(
        (protocol.validation_operator_ordinal(index) for index in indices),
        dtype=np.int64,
        count=protocol.GATE9_VALIDATION_BATCH_SIZE,
    )
    counters = ordinals + protocol.GATE9_VALIDATION_OPERATOR_COUNTER_START
    queries = np.fromiter(
        (protocol.validation_query(int(ordinal)) for ordinal in ordinals),
        dtype=np.uint8,
        count=protocol.GATE9_VALIDATION_BATCH_SIZE,
    )
    arrays = batch_arrays(
        counters=(int(value) for value in counters),
        queries=(int(value) for value in queries),
        verify_public_oracle=True,
    )
    shuffled_indices = tuple(
        validation_shuffled_episode_index(index) for index in indices
    )
    shuffled_ordinals = np.fromiter(
        (protocol.validation_operator_ordinal(index) for index in shuffled_indices),
        dtype=np.int64,
        count=protocol.GATE9_VALIDATION_BATCH_SIZE,
    )
    shuffled_counters = (
        shuffled_ordinals + protocol.GATE9_VALIDATION_OPERATOR_COUNTER_START
    )
    _, shuffled_outputs, _, _, _ = batch_arrays(
        counters=(int(value) for value in shuffled_counters),
        queries=(int(value) for value in queries),
        verify_public_oracle=False,
    )
    return indices, ordinals, counters, arrays, shuffled_ordinals, shuffled_outputs


def tensor_batch(arrays, device: torch.device):
    return tuple(
        torch.as_tensor(array, dtype=torch.long, device=device) for array in arrays
    )


def target_bits(targets: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, device=targets.device, dtype=torch.long)
    return ((targets.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def configure_determinism(seed: int) -> None:
    index = protocol.GATE9_CHECKPOINT_SEEDS.index(seed)
    initialization_seed = protocol.GATE9_INITIALIZATION_SEEDS[index]
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def validate_model_state(model: torch.nn.Module) -> None:
    state = model.state_dict()
    if set(state) != set(protocol.GATE9_STATE_TENSOR_SHAPES):
        raise ValueError("Gate9 model state tensor names drifted")
    if sum(tensor.numel() for tensor in state.values()) != 19_649:
        raise ValueError("Gate9 model state parameter count drifted")
    for name, expected_shape in protocol.GATE9_STATE_TENSOR_SHAPES.items():
        tensor = state[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Gate9 model state shape drifted: {name}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"Gate9 model state dtype drifted: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Gate9 model state is non-finite: {name}")


def digest_update(digest: Any, *arrays: np.ndarray) -> None:
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
