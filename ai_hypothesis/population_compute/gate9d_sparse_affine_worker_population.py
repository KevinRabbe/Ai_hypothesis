"""Development-only sparse population execution of the affine support bridge.

Each worker owns one observed support pair. Round 1 broadcasts the output at
input zero. Round 2 allows only one-hot basis workers selected by the query to
emit an XOR contribution. Irrelevant workers remain silent. The same fixed
worker rule is reused at nominal populations 9, 16, 64, and 256.

This experiment does not claim automatic representation discovery. It tests
whether the verified affine coordinate bridge can be organized as sparse,
permutation-invariant population computation with communication independent of
nominal population size.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor

_ROOT = pathlib.Path(__file__).resolve().parent
_BRIDGE_PATH = _ROOT / "gate9d_affine_feature_bridge.py"

GATE9D_SPARSE_POPULATION_VERSION = "gate9d-sparse-affine-worker-population-v0"
GATE9D_SPARSE_POPULATION_STATUS = "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
GATE9D_SPARSE_POPULATION_BRANCH = (
    "agent/gate9d-sparse-affine-worker-population-v0"
)
GATE9D_SPARSE_POPULATION_BASE_HEAD = (
    "c0242268f2938fe1131f2aa90c87b5a48ae248f6"
)
GATE9D_SPARSE_POPULATION_COUNTER_START = (1 << 57) + 0x2000
GATE9D_SPARSE_POPULATION_OPERATOR_COUNT = 128
GATE9D_SPARSE_POPULATION_SIZES = (9, 16, 64, 256)
GATE9D_SPARSE_POPULATION_PARAMETER_COUNT = 0
GATE9D_SPARSE_POPULATION_PASS = "G9D_SPARSE_AFFINE_POPULATION_PASSES"
GATE9D_SPARSE_POPULATION_FAIL = "G9D_SPARSE_AFFINE_POPULATION_FAILED"
GATE9D_SPARSE_POPULATION_EXACT_REQUIRED = 1.0
GATE9D_SPARSE_POPULATION_CONTROL_MAX = 0.02


def _load_bridge():
    name = "gate9d_sparse_population_bridge_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D affine bridge dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge()
operators = bridge.operators
SUPPORT_ORDER = tuple(bridge.GATE9D_AFFINE_BRIDGE_SUPPORT_ORDER)
QUERY_VALUES = tuple(bridge.GATE9D_AFFINE_BRIDGE_QUERY_VALUES)

if len(SUPPORT_ORDER) != 9 or set(SUPPORT_ORDER) != {
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
}:
    raise RuntimeError("Gate9D sparse population support contract drifted")
if len(QUERY_VALUES) != 247:
    raise RuntimeError("Gate9D sparse population query domain drifted")
if min(GATE9D_SPARSE_POPULATION_SIZES) != len(SUPPORT_ORDER):
    raise RuntimeError("Gate9D sparse population minimum size drifted")


@dataclass(frozen=True)
class PopulationExecution:
    predictions: Tensor
    bias_messages: int
    contribution_messages: int
    active_worker_count: int
    nominal_population_size: int


def _validate_byte_tensor(values: Tensor, label: str, rank: int) -> None:
    if values.dtype != torch.long or values.ndim != rank:
        raise ValueError(f"Gate9D {label} must be rank-{rank} torch.long")
    if values.numel() == 0:
        raise ValueError(f"Gate9D {label} cannot be empty")
    if bool(torch.any((values < 0) | (values > 255))):
        raise ValueError(f"Gate9D {label} contains a value outside 0..255")


def byte_bits(values: Tensor) -> Tensor:
    if values.dtype != torch.long:
        raise ValueError("Gate9D byte tensors must use torch.long")
    shifts = torch.arange(8, dtype=torch.long, device=values.device)
    return ((values.unsqueeze(-1) >> shifts) & 1).to(torch.long)


def decode_bits(bits: Tensor) -> Tensor:
    if bits.dtype != torch.long or bits.ndim != 2 or bits.shape[1] != 8:
        raise ValueError("Gate9D answer bits must have shape [batch,8]")
    if bool(torch.any((bits < 0) | (bits > 1))):
        raise ValueError("Gate9D answer bits must be binary")
    weights = 1 << torch.arange(8, dtype=torch.long, device=bits.device)
    return torch.sum(bits * weights, dim=-1)


def _is_basis_value(values: Tensor) -> Tensor:
    return (values > 0) & ((values & (values - 1)) == 0)


def _basis_index(values: Tensor) -> Tensor:
    _validate_byte_tensor(values, "basis input", 1)
    if not bool(torch.all(_is_basis_value(values))):
        raise ValueError("Gate9D basis index requires one-hot byte inputs")
    result = torch.zeros_like(values)
    for index in range(8):
        result = torch.where(values == (1 << index), index, result)
    return result


def distractor_inputs(count: int) -> tuple[int, ...]:
    if type(count) is not int or count < 0:
        raise ValueError("Gate9D distractor count must be nonnegative")
    candidates = tuple(
        value
        for value in range(1, 256)
        if value & (value - 1) != 0
    )
    return tuple(candidates[index % len(candidates)] for index in range(count))


def distractor_output(counter: int, worker_slot: int, support_input: int) -> int:
    if type(counter) is not int or not 0 <= counter < (1 << 64):
        raise ValueError("Gate9D distractor counter lies outside uint64")
    if type(worker_slot) is not int or worker_slot < 0:
        raise ValueError("Gate9D distractor worker slot is invalid")
    if type(support_input) is not int or not 0 <= support_input <= 255:
        raise ValueError("Gate9D distractor support input is invalid")
    value = (
        counter * 0x9E3779B185EBCA87
        + worker_slot * 0xC2B2AE3D27D4EB4F
        + support_input * 0x165667B19E3779F9
    ) & ((1 << 64) - 1)
    return ((value >> 17) ^ (value >> 41) ^ value) & 0xFF


def augment_population(
    support_inputs: Tensor,
    support_outputs: Tensor,
    counters: Tensor,
    nominal_population_size: int,
) -> tuple[Tensor, Tensor]:
    _validate_byte_tensor(support_inputs, "support inputs", 2)
    _validate_byte_tensor(support_outputs, "support outputs", 2)
    if support_inputs.shape != support_outputs.shape:
        raise ValueError("Gate9D support input/output shapes disagree")
    if support_inputs.shape[1] != 9:
        raise ValueError("Gate9D sparse population requires nine true supports")
    if counters.dtype != torch.long or counters.ndim != 1:
        raise ValueError("Gate9D counters must be one long vector")
    if counters.shape[0] != support_inputs.shape[0]:
        raise ValueError("Gate9D counter/support batch sizes disagree")
    if nominal_population_size not in GATE9D_SPARSE_POPULATION_SIZES:
        raise ValueError("Gate9D nominal population size is not qualified")

    extra = nominal_population_size - 9
    if extra == 0:
        return support_inputs.clone(), support_outputs.clone()
    distractors = distractor_inputs(extra)
    input_extra = torch.tensor(
        distractors,
        dtype=torch.long,
        device=support_inputs.device,
    ).unsqueeze(0).expand(support_inputs.shape[0], -1)
    output_rows = []
    for counter in counters.detach().cpu().tolist():
        output_rows.append(
            [
                distractor_output(counter, 9 + index, value)
                for index, value in enumerate(distractors)
            ]
        )
    output_extra = torch.tensor(
        output_rows,
        dtype=torch.long,
        device=support_outputs.device,
    )
    return (
        torch.cat((support_inputs, input_extra), dim=1),
        torch.cat((support_outputs, output_extra), dim=1),
    )


def deterministic_permutation(population_size: int) -> Tensor:
    if population_size not in GATE9D_SPARSE_POPULATION_SIZES:
        raise ValueError("Gate9D population size is not qualified")
    order = list(range(population_size))
    order.sort(key=lambda value: ((value * 73) ^ (value << 3) ^ 0x5A))
    return torch.tensor(order, dtype=torch.long)


def sparse_population_execute(
    worker_inputs: Tensor,
    worker_outputs: Tensor,
    query: Tensor,
    *,
    use_bias_broadcast: bool = True,
) -> PopulationExecution:
    """Execute the two-round shared worker rule.

    Round 1: only the worker whose observed input is zero emits an eight-bit
    bias message. Round 2: only one-hot basis workers selected by active query
    bits emit their output delta relative to the broadcast bias. XOR reduction
    returns the answer. Non-basis distractors never emit.
    """

    _validate_byte_tensor(worker_inputs, "worker inputs", 2)
    _validate_byte_tensor(worker_outputs, "worker outputs", 2)
    _validate_byte_tensor(query, "query", 1)
    if worker_inputs.shape != worker_outputs.shape:
        raise ValueError("Gate9D worker input/output shapes disagree")
    if worker_inputs.shape[0] != query.shape[0]:
        raise ValueError("Gate9D worker/query batch sizes disagree")
    batch, population = worker_inputs.shape

    zero_mask = worker_inputs == 0
    if not bool(torch.all(torch.sum(zero_mask, dim=1) == 1)):
        raise ValueError("Gate9D population requires exactly one zero worker")
    zero_slot = torch.argmax(zero_mask.to(torch.long), dim=1)
    bias_bytes = worker_outputs.gather(1, zero_slot.unsqueeze(1)).squeeze(1)
    bias_bits = byte_bits(bias_bytes)
    if not use_bias_broadcast:
        bias_bits = torch.zeros_like(bias_bits)

    basis_mask = _is_basis_value(worker_inputs)
    basis_values = worker_inputs[basis_mask]
    basis_indices = _basis_index(basis_values)
    query_bits = byte_bits(query)
    selected = torch.zeros_like(basis_mask)
    selected[basis_mask] = query_bits.repeat_interleave(
        torch.sum(basis_mask, dim=1), dim=0
    ).gather(1, basis_indices.unsqueeze(1)).squeeze(1).to(torch.bool)

    output_bits = byte_bits(worker_outputs)
    deltas = torch.bitwise_xor(output_bits, bias_bits.unsqueeze(1))
    contributions = deltas * selected.unsqueeze(-1).to(torch.long)
    parity = torch.remainder(torch.sum(contributions, dim=1), 2)
    answer_bits = torch.bitwise_xor(parity, bias_bits)
    predictions = decode_bits(answer_bits)

    contribution_messages = int(torch.sum(selected).item())
    active_worker_count = int(
        torch.sum(torch.any(selected, dim=0)).item()
        + torch.sum(torch.any(zero_mask, dim=0)).item()
    )
    return PopulationExecution(
        predictions=predictions,
        bias_messages=batch,
        contribution_messages=contribution_messages,
        active_worker_count=active_worker_count,
        nominal_population_size=population,
    )


def materialize_evaluation() -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
    support_input_rows: list[tuple[int, ...]] = []
    support_output_rows: list[tuple[int, ...]] = []
    query_rows: list[int] = []
    target_rows: list[int] = []
    counter_rows: list[int] = []
    digest = hashlib.sha256()
    for counter in range(
        GATE9D_SPARSE_POPULATION_COUNTER_START,
        GATE9D_SPARSE_POPULATION_COUNTER_START
        + GATE9D_SPARSE_POPULATION_OPERATOR_COUNT,
    ):
        operator = operators.operator_from_counter(counter)
        supports = operators.public_support_pairs(operator)
        inputs = tuple(source for source, _ in supports)
        outputs = tuple(target for _, target in supports)
        if inputs != SUPPORT_ORDER:
            raise RuntimeError("Gate9D sparse population support order drifted")
        for query in QUERY_VALUES:
            target = operator.apply(query)
            support_input_rows.append(inputs)
            support_output_rows.append(outputs)
            query_rows.append(query)
            target_rows.append(target)
            counter_rows.append(counter)
            digest.update(counter.to_bytes(8, "little"))
            digest.update(bytes(inputs))
            digest.update(bytes(outputs))
            digest.update(bytes((query, target)))
    return (
        torch.tensor(support_input_rows, dtype=torch.long),
        torch.tensor(support_output_rows, dtype=torch.long),
        torch.tensor(query_rows, dtype=torch.long),
        torch.tensor(target_rows, dtype=torch.long),
        torch.tensor(counter_rows, dtype=torch.long),
        digest.hexdigest(),
    )


def metrics(predictions: Tensor, targets: Tensor) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.ndim != 1:
        raise ValueError("Gate9D prediction/target shapes disagree")
    rows = int(targets.numel())
    exact_correct = int(torch.sum(predictions == targets).item())
    prediction_bits = byte_bits(predictions)
    target_bits = byte_bits(targets)
    bit_correct = int(torch.sum(prediction_bits == target_bits).item())
    return {
        "rows": rows,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_correct / rows,
        "bit_correct": bit_correct,
        "bit_total": rows * 8,
        "bit_accuracy": bit_correct / (rows * 8),
    }


def classify_population(rows: Iterable[dict[str, Any]]) -> str:
    records = tuple(rows)
    if len(records) != len(GATE9D_SPARSE_POPULATION_SIZES):
        raise ValueError("Gate9D sparse population requires one row per size")
    seen: set[int] = set()
    for row in records:
        size = row.get("population_size")
        if size not in GATE9D_SPARSE_POPULATION_SIZES or size in seen:
            raise ValueError("Gate9D sparse population size rows drifted")
        seen.add(size)
        if row.get("parameter_count") != 0:
            raise ValueError("Gate9D sparse population parameter budget drifted")
        if row.get("full_exact_accuracy") != 1.0:
            return GATE9D_SPARSE_POPULATION_FAIL
        if row.get("permuted_exact_accuracy") != 1.0:
            return GATE9D_SPARSE_POPULATION_FAIL
        if row.get("shuffled_exact_accuracy", 1.0) > GATE9D_SPARSE_POPULATION_CONTROL_MAX:
            return GATE9D_SPARSE_POPULATION_FAIL
        if row.get("no_bias_exact_accuracy", 1.0) > GATE9D_SPARSE_POPULATION_CONTROL_MAX:
            return GATE9D_SPARSE_POPULATION_FAIL
    return GATE9D_SPARSE_POPULATION_PASS


def run_sparse_population_diagnostic(output_root: pathlib.Path, execution_head: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Gate9D sparse population output already exists: {output_root}")
    if len(execution_head) != 40 or any(
        character not in "0123456789abcdef" for character in execution_head
    ):
        raise ValueError("Gate9D sparse population execution head is malformed")
    output_root.mkdir(parents=True)

    support_inputs, support_outputs, queries, targets, counters, dataset_sha = (
        materialize_evaluation()
    )
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    shuffled_outputs = torch.roll(support_outputs, shifts=247, dims=0)

    for population_size in GATE9D_SPARSE_POPULATION_SIZES:
        inputs, outputs = augment_population(
            support_inputs,
            support_outputs,
            counters,
            population_size,
        )
        full = sparse_population_execute(inputs, outputs, queries)
        permutation = deterministic_permutation(population_size).to(inputs.device)
        permuted = sparse_population_execute(
            inputs[:, permutation],
            outputs[:, permutation],
            queries,
        )
        shuffled_inputs, shuffled_augmented_outputs = augment_population(
            support_inputs,
            shuffled_outputs,
            counters,
            population_size,
        )
        shuffled = sparse_population_execute(
            shuffled_inputs,
            shuffled_augmented_outputs,
            queries,
        )
        no_bias = sparse_population_execute(
            inputs,
            outputs,
            queries,
            use_bias_broadcast=False,
        )
        full_metrics = metrics(full.predictions, targets)
        permuted_metrics = metrics(permuted.predictions, targets)
        shuffled_metrics = metrics(shuffled.predictions, targets)
        no_bias_metrics = metrics(no_bias.predictions, targets)
        row = {
            "population_size": population_size,
            "parameter_count": GATE9D_SPARSE_POPULATION_PARAMETER_COUNT,
            "full_exact_accuracy": full_metrics["exact_accuracy"],
            "full_bit_accuracy": full_metrics["bit_accuracy"],
            "permuted_exact_accuracy": permuted_metrics["exact_accuracy"],
            "permuted_bit_accuracy": permuted_metrics["bit_accuracy"],
            "shuffled_exact_accuracy": shuffled_metrics["exact_accuracy"],
            "shuffled_bit_accuracy": shuffled_metrics["bit_accuracy"],
            "no_bias_exact_accuracy": no_bias_metrics["exact_accuracy"],
            "no_bias_bit_accuracy": no_bias_metrics["bit_accuracy"],
            "bias_messages": full.bias_messages,
            "contribution_messages": full.contribution_messages,
            "messages_per_episode": (
                full.bias_messages + full.contribution_messages
            ) / int(targets.numel()),
            "active_worker_count": full.active_worker_count,
        }
        rows.append(row)
        for index in range(int(targets.numel())):
            episode_rows.append(
                {
                    "population_size": population_size,
                    "episode_index": index,
                    "operator_counter": int(counters[index]),
                    "query": int(queries[index]),
                    "target": int(targets[index]),
                    "full_prediction": int(full.predictions[index]),
                    "permuted_prediction": int(permuted.predictions[index]),
                    "shuffled_prediction": int(shuffled.predictions[index]),
                    "no_bias_prediction": int(no_bias.predictions[index]),
                }
            )

    diagnosis = classify_population(rows)
    summary = {
        "status": "G9D_SPARSE_AFFINE_POPULATION_COMPLETE_DEVELOPMENT_ONLY",
        "version": GATE9D_SPARSE_POPULATION_VERSION,
        "diagnosis": diagnosis,
        "development_only": True,
        "confirmation_result": False,
        "execution_head": execution_head,
        "affine_bridge_base_head": GATE9D_SPARSE_POPULATION_BASE_HEAD,
        "operator_range": {
            "start": GATE9D_SPARSE_POPULATION_COUNTER_START,
            "count": GATE9D_SPARSE_POPULATION_OPERATOR_COUNT,
        },
        "dataset_sha256": dataset_sha,
        "episodes": int(targets.numel()),
        "population_sizes": list(GATE9D_SPARSE_POPULATION_SIZES),
        "learned_parameter_count": 0,
        "rows": rows,
        "boundaries": {
            "automatic_coordinate_discovery_claimed": False,
            "operator_counter_visible_to_population": False,
            "operator_key_visible_to_population": False,
            "frozen_result_modified": False,
            "later_diagnostic_stage_opened": False,
            "gate9_v0_science_executed": False,
            "population_confirmation_claimed": False,
        },
    }
    (output_root / "aggregate-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (output_root / "population-rows.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (output_root / "episodes.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in episode_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return summary
