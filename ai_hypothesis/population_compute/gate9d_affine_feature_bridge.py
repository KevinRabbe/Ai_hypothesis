"""Development-only Gate-9D affine feature bridge diagnostic.

This module tests whether the failed contextual worker can be repaired by a
representation that respects the frozen affine GF(2)^8 operator family.  It
uses only public zero-plus-basis support outputs and the incoming query.  It
never exposes operator counters or keys to the learned decoder.

The fixed bridge derives, per output bit, the affine bias bit and eight linear
mask bits from the public support rows.  It then supplies a parity sign and a
bias sign to one 65-parameter decoder shared across all output bits and all
operators.  The learned decoder must compose those two signs into output-bit
logits.  Shuffled-support and query-only controls remain mandatory.

This is development evidence only.  It cannot classify the frozen Gate-9D
ladder, open later stages, execute population science, or mutate prior results.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_ROOT = pathlib.Path(__file__).resolve().parent
_OPERATOR_PATH = _ROOT / "gate9_contextual_operator_contract.py"

GATE9D_AFFINE_BRIDGE_VERSION = "gate9d-affine-feature-bridge-v0"
GATE9D_AFFINE_BRIDGE_STATUS = "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
GATE9D_AFFINE_BRIDGE_BRANCH = "agent/gate9d-affine-feature-bridge-v0"
GATE9D_AFFINE_BRIDGE_BASE_HEAD = (
    "f9cff8e1609cfae5642f8cef2242eee74f9488c7"
)
GATE9D_AFFINE_BRIDGE_TRAIN_COUNTER_START = 1 << 57
GATE9D_AFFINE_BRIDGE_TRAIN_OPERATOR_COUNT = 256
GATE9D_AFFINE_BRIDGE_EVAL_COUNTER_START = (1 << 57) + 0x1000
GATE9D_AFFINE_BRIDGE_EVAL_OPERATOR_COUNT = 64
GATE9D_AFFINE_BRIDGE_INITIALIZATION_SEEDS = (910_900, 910_901, 910_902)
GATE9D_AFFINE_BRIDGE_TRAIN_STEPS = 512
GATE9D_AFFINE_BRIDGE_BATCH_SIZE = 512
GATE9D_AFFINE_BRIDGE_CHECKPOINTS = (0, 1, 16, 32, 64, 128, 256, 512)
GATE9D_AFFINE_BRIDGE_BASE_LEARNING_RATE = 1.0e-3
GATE9D_AFFINE_BRIDGE_MIN_LEARNING_RATE = 1.0e-4
GATE9D_AFFINE_BRIDGE_WARMUP_STEPS = 16
GATE9D_AFFINE_BRIDGE_ADAM_BETAS = (0.9, 0.95)
GATE9D_AFFINE_BRIDGE_ADAM_EPSILON = 1.0e-8
GATE9D_AFFINE_BRIDGE_WEIGHT_DECAY = 1.0e-4
GATE9D_AFFINE_BRIDGE_GRADIENT_CLIP_NORM = 1.0
GATE9D_AFFINE_BRIDGE_EXACT_ACCURACY_MIN = 0.995
GATE9D_AFFINE_BRIDGE_BIT_ACCURACY_MIN = 0.999
GATE9D_AFFINE_BRIDGE_CONTEXT_DELTA_MIN_STRICT = 0.50
GATE9D_AFFINE_BRIDGE_ORACLE_ACCURACY_REQUIRED = 1.0
GATE9D_AFFINE_BRIDGE_HIDDEN = 16
GATE9D_AFFINE_BRIDGE_PARAMETER_COUNT = 65
GATE9D_AFFINE_BRIDGE_PASS = "G9D_AFFINE_FEATURE_BRIDGE_PASSES"
GATE9D_AFFINE_BRIDGE_FAIL = "G9D_AFFINE_FEATURE_BRIDGE_FAILED"
GATE9D_AFFINE_BRIDGE_MIXED = "G9D_AFFINE_FEATURE_BRIDGE_MIXED"


def _load_operator_contract():
    name = "gate9d_affine_bridge_operator_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _OPERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D affine-bridge operator contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


operators = _load_operator_contract()
GATE9D_AFFINE_BRIDGE_SUPPORT_ORDER = tuple(operators.GATE9_GLOBAL_SUPPORT_ORDER)
GATE9D_AFFINE_BRIDGE_SUPPORT_INPUTS = tuple(
    operators.protocol.GATE9_SUPPORT_INPUTS
)
GATE9D_AFFINE_BRIDGE_QUERY_VALUES = tuple(
    value
    for value in range(256)
    if value not in set(GATE9D_AFFINE_BRIDGE_SUPPORT_INPUTS)
)
GATE9D_AFFINE_BRIDGE_QUERY_COUNT = len(GATE9D_AFFINE_BRIDGE_QUERY_VALUES)
GATE9D_AFFINE_BRIDGE_ZERO_SLOT = GATE9D_AFFINE_BRIDGE_SUPPORT_ORDER.index(0)
GATE9D_AFFINE_BRIDGE_BASIS_SLOTS = tuple(
    GATE9D_AFFINE_BRIDGE_SUPPORT_ORDER.index(1 << index)
    for index in range(8)
)

if GATE9D_AFFINE_BRIDGE_QUERY_COUNT != 247:
    raise RuntimeError("Gate9D affine-bridge query-domain arithmetic drifted")
if GATE9D_AFFINE_BRIDGE_CHECKPOINTS[-1] != GATE9D_AFFINE_BRIDGE_TRAIN_STEPS:
    raise RuntimeError("Gate9D affine-bridge checkpoint schedule drifted")
if (
    GATE9D_AFFINE_BRIDGE_TRAIN_COUNTER_START
    + GATE9D_AFFINE_BRIDGE_TRAIN_OPERATOR_COUNT
    > GATE9D_AFFINE_BRIDGE_EVAL_COUNTER_START
):
    raise RuntimeError("Gate9D affine-bridge train/eval counter ranges overlap")


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
    return ((values.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def decode_bytes(bit_logits: Tensor) -> Tensor:
    if bit_logits.ndim != 2 or bit_logits.shape[1] != 8:
        raise ValueError("Gate9D bit logits must have shape [batch,8]")
    if not torch.is_floating_point(bit_logits):
        raise ValueError("Gate9D bit logits must be floating point")
    weights = 1 << torch.arange(8, dtype=torch.long, device=bit_logits.device)
    return ((bit_logits >= 0).to(torch.long) * weights).sum(dim=-1)


def affine_signature(support_outputs: Tensor) -> tuple[Tensor, Tensor]:
    """Return public-support-derived bias bits and linear mask bits.

    ``bias_bits`` has shape ``[batch, output_bit]``. ``mask_bits`` has shape
    ``[batch, output_bit, input_bit]``.  No operator key, counter, or hidden
    matrix is consumed.
    """

    _validate_byte_tensor(support_outputs, "support outputs", 2)
    if support_outputs.shape[1] != 9:
        raise ValueError("Gate9D affine bridge requires exactly nine supports")
    support_bits = byte_bits(support_outputs)
    bias_bits = support_bits[:, GATE9D_AFFINE_BRIDGE_ZERO_SLOT, :]
    basis_bits = support_bits[:, GATE9D_AFFINE_BRIDGE_BASIS_SLOTS, :].permute(
        0, 2, 1
    )
    mask_bits = torch.ne(basis_bits, bias_bits.unsqueeze(-1)).to(torch.float32)
    return bias_bits, mask_bits


def affine_bridge_features(support_outputs: Tensor, query: Tensor) -> Tensor:
    """Create two signed features per output bit from public support and query."""

    _validate_byte_tensor(query, "query", 1)
    if support_outputs.shape[0] != query.shape[0]:
        raise ValueError("Gate9D affine-bridge support/query batch sizes disagree")
    if support_outputs.device != query.device:
        raise ValueError("Gate9D affine-bridge tensors must share one device")
    bias_bits, mask_bits = affine_signature(support_outputs)
    query_bits = byte_bits(query)
    parity_bits = torch.remainder(
        torch.sum(mask_bits * query_bits.unsqueeze(1), dim=-1), 2.0
    )
    parity_sign = 1.0 - 2.0 * parity_bits
    bias_sign = 1.0 - 2.0 * bias_bits
    return torch.stack((parity_sign, bias_sign), dim=-1)


def fixed_bridge_answers(support_outputs: Tensor, query: Tensor) -> Tensor:
    """Parameter-free bridge oracle used only as a representation control."""

    features = affine_bridge_features(support_outputs, query)
    parity_bits = features[..., 0] < 0
    bias_bits = features[..., 1] < 0
    answer_bits = torch.logical_xor(parity_bits, bias_bits).to(torch.long)
    weights = 1 << torch.arange(8, dtype=torch.long, device=query.device)
    return torch.sum(answer_bits * weights, dim=-1)


class AffineFeatureBridgeDecoder(nn.Module):
    """One 65-parameter decoder shared across bits, operators, and examples."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, GATE9D_AFFINE_BRIDGE_HIDDEN)
        self.output = nn.Linear(GATE9D_AFFINE_BRIDGE_HIDDEN, 1)
        if self.learned_parameter_count() != GATE9D_AFFINE_BRIDGE_PARAMETER_COUNT:
            raise RuntimeError("Gate9D affine-bridge parameter count drifted")

    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward_from_features(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1:] != (8, 2):
            raise ValueError("Gate9D affine-bridge features must have shape [batch,8,2]")
        logits = self.output(torch.tanh(self.hidden(features))).squeeze(-1)
        if logits.shape != (features.shape[0], 8):
            raise RuntimeError("Gate9D affine-bridge output shape drifted")
        return logits

    def forward(self, support_outputs: Tensor, query: Tensor) -> Tensor:
        return self.forward_from_features(
            affine_bridge_features(support_outputs, query)
        )

    def forward_query_only(self, query: Tensor) -> Tensor:
        _validate_byte_tensor(query, "query-only query", 1)
        features = torch.ones(
            query.shape[0],
            8,
            2,
            dtype=self.hidden.weight.dtype,
            device=query.device,
        )
        return self.forward_from_features(features)


def learning_rate_at_step(step: int) -> float:
    if type(step) is not int or not 1 <= step <= GATE9D_AFFINE_BRIDGE_TRAIN_STEPS:
        raise ValueError("Gate9D affine-bridge step lies outside the schedule")
    if step <= GATE9D_AFFINE_BRIDGE_WARMUP_STEPS:
        return (
            GATE9D_AFFINE_BRIDGE_BASE_LEARNING_RATE
            * step
            / GATE9D_AFFINE_BRIDGE_WARMUP_STEPS
        )
    progress = (
        step - GATE9D_AFFINE_BRIDGE_WARMUP_STEPS
    ) / (
        GATE9D_AFFINE_BRIDGE_TRAIN_STEPS
        - GATE9D_AFFINE_BRIDGE_WARMUP_STEPS
    )
    return GATE9D_AFFINE_BRIDGE_MIN_LEARNING_RATE + 0.5 * (
        GATE9D_AFFINE_BRIDGE_BASE_LEARNING_RATE
        - GATE9D_AFFINE_BRIDGE_MIN_LEARNING_RATE
    ) * (1.0 + math.cos(math.pi * progress))


def configure_determinism(seed_index: int) -> int:
    if type(seed_index) is not int or seed_index not in (0, 1, 2):
        raise ValueError("Gate9D affine-bridge seed index must be 0, 1, or 2")
    initialization_seed = GATE9D_AFFINE_BRIDGE_INITIALIZATION_SEEDS[seed_index]
    torch.manual_seed(initialization_seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return initialization_seed


def _counter_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(start, start + count))


def train_counters() -> tuple[int, ...]:
    return _counter_range(
        GATE9D_AFFINE_BRIDGE_TRAIN_COUNTER_START,
        GATE9D_AFFINE_BRIDGE_TRAIN_OPERATOR_COUNT,
    )


def evaluation_counters() -> tuple[int, ...]:
    return _counter_range(
        GATE9D_AFFINE_BRIDGE_EVAL_COUNTER_START,
        GATE9D_AFFINE_BRIDGE_EVAL_OPERATOR_COUNT,
    )


def materialize_dataset(
    counters: Iterable[int],
) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
    support_rows: list[tuple[int, ...]] = []
    query_rows: list[int] = []
    target_rows: list[int] = []
    counter_rows: list[int] = []
    digest = hashlib.sha256()
    seen: set[int] = set()
    for counter in counters:
        if type(counter) is not int or not 0 <= counter < (1 << 64):
            raise ValueError("Gate9D affine-bridge counter lies outside uint64")
        if counter in seen:
            raise ValueError("Gate9D affine-bridge operator counter repeats")
        seen.add(counter)
        operator = operators.operator_from_counter(counter)
        support = operators.public_support_pairs(operator)
        support_inputs = tuple(source for source, _ in support)
        support_outputs = tuple(target for _, target in support)
        if support_inputs != GATE9D_AFFINE_BRIDGE_SUPPORT_ORDER:
            raise RuntimeError("Gate9D affine-bridge support order drifted")
        for query in GATE9D_AFFINE_BRIDGE_QUERY_VALUES:
            target = operator.apply(query)
            support_rows.append(support_outputs)
            query_rows.append(query)
            target_rows.append(target)
            counter_rows.append(counter)
            digest.update(counter.to_bytes(8, "little"))
            digest.update(bytes(support_outputs))
            digest.update(bytes((query, target)))
    if not seen:
        raise ValueError("Gate9D affine-bridge dataset cannot be empty")
    return (
        torch.tensor(support_rows, dtype=torch.long),
        torch.tensor(query_rows, dtype=torch.long),
        torch.tensor(target_rows, dtype=torch.long),
        torch.tensor(counter_rows, dtype=torch.long),
        digest.hexdigest(),
    )


def _metrics(predictions: Tensor, targets: Tensor) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.ndim != 1:
        raise ValueError("Gate9D affine-bridge prediction/target shapes disagree")
    rows = int(targets.numel())
    exact_correct = int(torch.sum(predictions == targets).item())
    prediction_bits = byte_bits(predictions).to(torch.long)
    target_bits = byte_bits(targets).to(torch.long)
    per_bit_correct = torch.sum(prediction_bits == target_bits, dim=0).tolist()
    bit_correct = int(sum(per_bit_correct))
    return {
        "rows": rows,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_correct / rows,
        "bit_correct": bit_correct,
        "bit_total": rows * 8,
        "bit_accuracy": bit_correct / (rows * 8),
        "per_bit_correct": [int(value) for value in per_bit_correct],
        "per_bit_accuracy": [int(value) / rows for value in per_bit_correct],
    }


def _parameter_l2(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        value = parameter.detach().to(torch.float64)
        total += torch.sum(value * value)
    return float(torch.sqrt(total))


def _parameter_update_l2(model: nn.Module, initial: tuple[Tensor, ...]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter, start in zip(model.parameters(), initial, strict=True):
        difference = parameter.detach().to(torch.float64) - start
        total += torch.sum(difference * difference)
    return float(torch.sqrt(total))


def _active_gradient_elements(model: nn.Module) -> int:
    return sum(
        int(torch.count_nonzero(parameter.grad).item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _evaluate_logits(
    model: AffineFeatureBridgeDecoder,
    support_outputs: Tensor,
    queries: Tensor,
    *,
    chunk_size: int = 4096,
) -> Tensor:
    rows = []
    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        rows.append(model(support_outputs[start:stop], queries[start:stop]))
    return torch.cat(rows, dim=0)


def seed_passes(
    full: dict[str, Any],
    shuffled: dict[str, Any],
    query_only: dict[str, Any],
    oracle: dict[str, Any],
) -> bool:
    return (
        full["exact_accuracy"] >= GATE9D_AFFINE_BRIDGE_EXACT_ACCURACY_MIN
        and full["bit_accuracy"] >= GATE9D_AFFINE_BRIDGE_BIT_ACCURACY_MIN
        and oracle["exact_accuracy"]
        == GATE9D_AFFINE_BRIDGE_ORACLE_ACCURACY_REQUIRED
        and full["exact_accuracy"] - shuffled["exact_accuracy"]
        > GATE9D_AFFINE_BRIDGE_CONTEXT_DELTA_MIN_STRICT
        and full["exact_accuracy"] - query_only["exact_accuracy"]
        > GATE9D_AFFINE_BRIDGE_CONTEXT_DELTA_MIN_STRICT
    )


def classify_seed_passes(values: Iterable[bool]) -> str:
    passes = tuple(values)
    if len(passes) != 3 or any(type(value) is not bool for value in passes):
        raise ValueError("Gate9D affine bridge requires three Boolean seed results")
    if all(passes):
        return GATE9D_AFFINE_BRIDGE_PASS
    if any(passes):
        return GATE9D_AFFINE_BRIDGE_MIXED
    return GATE9D_AFFINE_BRIDGE_FAIL


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def run_affine_feature_bridge(
    *, output_root: pathlib.Path, execution_head: str
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"Gate9D affine-bridge output already exists: {output_root}"
        )
    if len(execution_head) != 40 or any(
        character not in "0123456789abcdef" for character in execution_head
    ):
        raise ValueError("Gate9D affine-bridge execution head is malformed")
    output_root.mkdir(parents=True)

    (
        train_support,
        train_queries,
        train_targets,
        train_counter_rows,
        train_dataset_sha256,
    ) = materialize_dataset(train_counters())
    (
        evaluation_support,
        evaluation_queries,
        evaluation_targets,
        evaluation_counter_rows,
        evaluation_dataset_sha256,
    ) = materialize_dataset(evaluation_counters())
    if set(train_counters()) & set(evaluation_counters()):
        raise RuntimeError("Gate9D affine-bridge train/eval counters overlap")

    fixed_predictions = fixed_bridge_answers(
        evaluation_support, evaluation_queries
    )
    fixed_metrics = _metrics(fixed_predictions, evaluation_targets)
    if fixed_metrics["exact_accuracy"] != 1.0:
        raise RuntimeError("Gate9D fixed affine feature bridge is not exact")

    curves: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    seed_results: list[bool] = []
    train_target_bits = byte_bits(train_targets)
    evaluation_target_bits = byte_bits(evaluation_targets)

    for seed_index in (0, 1, 2):
        initialization_seed = configure_determinism(seed_index)
        model = AffineFeatureBridgeDecoder()
        initial = tuple(
            parameter.detach().to(torch.float64).clone()
            for parameter in model.parameters()
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=GATE9D_AFFINE_BRIDGE_BASE_LEARNING_RATE,
            betas=GATE9D_AFFINE_BRIDGE_ADAM_BETAS,
            eps=GATE9D_AFFINE_BRIDGE_ADAM_EPSILON,
            weight_decay=GATE9D_AFFINE_BRIDGE_WEIGHT_DECAY,
        )
        sampler = torch.Generator(device="cpu")
        sampler.manual_seed(initialization_seed ^ 0xA9F17E)

        model.eval()
        with torch.no_grad():
            initial_logits = _evaluate_logits(
                model, evaluation_support, evaluation_queries
            )
            initial_loss = F.binary_cross_entropy_with_logits(
                initial_logits, evaluation_target_bits
            )
            initial_metrics = _metrics(
                decode_bytes(initial_logits), evaluation_targets
            )
        curves.append(
            {
                "seed_index": seed_index,
                "initialization_seed": initialization_seed,
                "step": 0,
                "learning_rate": None,
                "loss": float(initial_loss),
                "gradient_norm": None,
                "active_gradient_elements": 0,
                "parameter_count": model.learned_parameter_count(),
                "parameter_l2": _parameter_l2(model),
                "parameter_update_l2": 0.0,
                **initial_metrics,
            }
        )

        started = time.perf_counter()
        last_gradient_norm = math.nan
        last_active_gradients = 0
        for step in range(1, GATE9D_AFFINE_BRIDGE_TRAIN_STEPS + 1):
            indices = torch.randint(
                train_queries.shape[0],
                (GATE9D_AFFINE_BRIDGE_BATCH_SIZE,),
                generator=sampler,
            )
            learning_rate = learning_rate_at_step(step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_support[indices], train_queries[indices])
            loss = F.binary_cross_entropy_with_logits(
                logits, train_target_bits[indices]
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Gate9D affine-bridge loss became non-finite")
            loss.backward()
            last_active_gradients = _active_gradient_elements(model)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GATE9D_AFFINE_BRIDGE_GRADIENT_CLIP_NORM
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Gate9D affine-bridge gradient became non-finite")
            last_gradient_norm = float(gradient_norm)
            optimizer.step()

            if step in GATE9D_AFFINE_BRIDGE_CHECKPOINTS:
                model.eval()
                with torch.no_grad():
                    checkpoint_logits = _evaluate_logits(
                        model, evaluation_support, evaluation_queries
                    )
                    checkpoint_loss = F.binary_cross_entropy_with_logits(
                        checkpoint_logits, evaluation_target_bits
                    )
                    checkpoint_metrics = _metrics(
                        decode_bytes(checkpoint_logits), evaluation_targets
                    )
                curves.append(
                    {
                        "seed_index": seed_index,
                        "initialization_seed": initialization_seed,
                        "step": step,
                        "learning_rate": learning_rate,
                        "loss": float(checkpoint_loss),
                        "gradient_norm": last_gradient_norm,
                        "active_gradient_elements": last_active_gradients,
                        "parameter_count": model.learned_parameter_count(),
                        "parameter_l2": _parameter_l2(model),
                        "parameter_update_l2": _parameter_update_l2(
                            model, initial
                        ),
                        **checkpoint_metrics,
                    }
                )

        elapsed = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            full_logits = _evaluate_logits(
                model, evaluation_support, evaluation_queries
            )
            full_predictions = decode_bytes(full_logits)
            shuffled_support = torch.roll(
                evaluation_support,
                shifts=GATE9D_AFFINE_BRIDGE_QUERY_COUNT,
                dims=0,
            )
            shuffled_predictions = decode_bytes(
                _evaluate_logits(model, shuffled_support, evaluation_queries)
            )
            query_only_predictions = decode_bytes(
                model.forward_query_only(evaluation_queries)
            )
            oracle_predictions = fixed_bridge_answers(
                evaluation_support, evaluation_queries
            )
            final_loss = F.binary_cross_entropy_with_logits(
                full_logits, evaluation_target_bits
            )

        full_metrics = _metrics(full_predictions, evaluation_targets)
        shuffled_metrics = _metrics(shuffled_predictions, evaluation_targets)
        query_only_metrics = _metrics(query_only_predictions, evaluation_targets)
        oracle_metrics = _metrics(oracle_predictions, evaluation_targets)
        passes = seed_passes(
            full_metrics, shuffled_metrics, query_only_metrics, oracle_metrics
        )
        seed_results.append(passes)
        final_rows.append(
            {
                "seed_index": seed_index,
                "initialization_seed": initialization_seed,
                "steps": GATE9D_AFFINE_BRIDGE_TRAIN_STEPS,
                "seconds": elapsed,
                "parameter_count": model.learned_parameter_count(),
                "final_loss": float(final_loss),
                "gradient_norm_final": last_gradient_norm,
                "active_gradient_elements_final": last_active_gradients,
                "parameter_update_l2": _parameter_update_l2(model, initial),
                "full": full_metrics,
                "shuffled": shuffled_metrics,
                "query_only": query_only_metrics,
                "oracle": oracle_metrics,
                "full_minus_shuffled": (
                    full_metrics["exact_accuracy"]
                    - shuffled_metrics["exact_accuracy"]
                ),
                "full_minus_query_only": (
                    full_metrics["exact_accuracy"]
                    - query_only_metrics["exact_accuracy"]
                ),
                "passes": passes,
            }
        )

        for episode_index in range(evaluation_queries.shape[0]):
            answer = int(evaluation_targets[episode_index])
            full_prediction = int(full_predictions[episode_index])
            shuffled_prediction = int(shuffled_predictions[episode_index])
            query_only_prediction = int(query_only_predictions[episode_index])
            oracle_prediction = int(oracle_predictions[episode_index])
            evaluation_rows.append(
                {
                    "seed_index": seed_index,
                    "episode_index": episode_index,
                    "operator_counter": int(evaluation_counter_rows[episode_index]),
                    "query": int(evaluation_queries[episode_index]),
                    "answer": answer,
                    "full_prediction": full_prediction,
                    "shuffled_prediction": shuffled_prediction,
                    "query_only_prediction": query_only_prediction,
                    "oracle_prediction": oracle_prediction,
                    "full_correct": full_prediction == answer,
                    "shuffled_correct": shuffled_prediction == answer,
                    "query_only_correct": query_only_prediction == answer,
                    "oracle_correct": oracle_prediction == answer,
                    "full_bit_correct": 8
                    - (full_prediction ^ answer).bit_count(),
                    "shuffled_bit_correct": 8
                    - (shuffled_prediction ^ answer).bit_count(),
                    "query_only_bit_correct": 8
                    - (query_only_prediction ^ answer).bit_count(),
                    "oracle_bit_correct": 8
                    - (oracle_prediction ^ answer).bit_count(),
                }
            )

    diagnosis = classify_seed_passes(seed_results)
    summary = {
        "version": GATE9D_AFFINE_BRIDGE_VERSION,
        "status": "G9D_AFFINE_FEATURE_BRIDGE_COMPLETE_DEVELOPMENT_ONLY",
        "development_only": True,
        "confirmation_result": False,
        "execution_head": execution_head,
        "query_capacity_base_head": GATE9D_AFFINE_BRIDGE_BASE_HEAD,
        "learned_parameter_count": GATE9D_AFFINE_BRIDGE_PARAMETER_COUNT,
        "parameter_reduction_vs_gate9_v0": 19_649
        / GATE9D_AFFINE_BRIDGE_PARAMETER_COUNT,
        "train_operator_range": {
            "start": GATE9D_AFFINE_BRIDGE_TRAIN_COUNTER_START,
            "count": GATE9D_AFFINE_BRIDGE_TRAIN_OPERATOR_COUNT,
        },
        "evaluation_operator_range": {
            "start": GATE9D_AFFINE_BRIDGE_EVAL_COUNTER_START,
            "count": GATE9D_AFFINE_BRIDGE_EVAL_OPERATOR_COUNT,
        },
        "train_dataset_sha256": train_dataset_sha256,
        "evaluation_dataset_sha256": evaluation_dataset_sha256,
        "train_examples": int(train_queries.shape[0]),
        "evaluation_examples": int(evaluation_queries.shape[0]),
        "query_count_per_operator": GATE9D_AFFINE_BRIDGE_QUERY_COUNT,
        "support_order": list(GATE9D_AFFINE_BRIDGE_SUPPORT_ORDER),
        "seeds": [0, 1, 2],
        "initialization_seeds": list(
            GATE9D_AFFINE_BRIDGE_INITIALIZATION_SEEDS
        ),
        "fixed_bridge_oracle": fixed_metrics,
        "seed_passes": seed_results,
        "diagnosis": diagnosis,
        "boundaries": {
            "frozen_gate9d_ladder_classified": False,
            "later_diagnostic_stage_opened": False,
            "gate9_v0_science_executed": False,
            "population_execution_performed": False,
            "frozen_result_modified": False,
            "checkpoint_written": False,
            "operator_counter_visible_to_model": False,
            "operator_key_visible_to_model": False,
            "per_operator_parameters": False,
        },
    }
    _write_jsonl(output_root / "curves.jsonl", curves)
    _write_jsonl(output_root / "final-runs.jsonl", final_rows)
    _write_jsonl(output_root / "evaluation.jsonl", evaluation_rows)
    _write_json(output_root / "aggregate-summary.json", summary)
    return summary
