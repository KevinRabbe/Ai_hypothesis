"""Deterministic Gate-9 affine-operator and public-support contract.

This standard-library module implements only the frozen 64-bit counter mapping,
affine operator mechanics, nine-example public support representation, exact
public-support reconstruction, and mathematical split-disjointness audit. It
contains no graph world, model, optimizer, checkpoint, training, scientific
execution, or result-artifact surface.
"""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable

_PROTOCOL_PATH = pathlib.Path(__file__).with_name(
    "gate9_contextual_operator_induction_protocol.py"
)


def _load_protocol():
    name = "gate9_contextual_operator_contract_protocol_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen Gate9 protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()

GATE9_OPERATOR_CONTRACT_VERSION = "gate9-contextual-affine-operator-contract-v0"
GATE9_OPERATOR_CONTRACT_STATUS = (
    "GATE9_CONTEXTUAL_OPERATOR_SUPPORT_ORACLE_CONTRACT_QUALIFIED_EXECUTION_CLOSED"
)
GATE9_PROTOCOL_HEAD = "e5e20e8de6707d35f1a7a9315a5a9a67deacc9a1"

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL2 = 0x94D049BB133111EB
_SPLITMIX_INV_MUL1 = 0x96DE1B173F119089
_SPLITMIX_INV_MUL2 = 0x319642B2D24D8EC3
_SUPPORT_ORDER_NAMESPACE = "gate9-contextual-support-global-order-v0"

_LOWER_POSITIONS = tuple(
    (row, column) for row in range(1, 8) for column in range(row)
)
_UPPER_POSITIONS = tuple(
    (row, column) for row in range(7) for column in range(row + 1, 8)
)
if len(_LOWER_POSITIONS) != 28 or len(_UPPER_POSITIONS) != 28:
    raise RuntimeError("Gate9 triangular parameter geometry drifted")


def _valid_byte(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 256:
        raise ValueError(f"{label} must be an integer within 0..255")
    return value


def _valid_u64(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MASK64:
        raise ValueError(f"{label} must be an unsigned 64-bit integer")
    return value


def _undo_right_xor(value: int, shift: int) -> int:
    result = value & _MASK64
    distance = shift
    while distance < 64:
        result ^= value >> distance
        distance += shift
    return result & _MASK64


def splitmix64_bijection(counter: int) -> int:
    """Map one uint64 counter to one uint64 key through frozen SplitMix64."""

    value = (_valid_u64(counter, "Gate9 operator counter") + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_MUL1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_MUL2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def inverse_splitmix64_bijection(key: int) -> int:
    """Invert :func:`splitmix64_bijection` exactly over uint64."""

    value = _undo_right_xor(_valid_u64(key, "Gate9 operator key"), 31)
    value = (value * _SPLITMIX_INV_MUL2) & _MASK64
    value = _undo_right_xor(value, 27)
    value = (value * _SPLITMIX_INV_MUL1) & _MASK64
    value = _undo_right_xor(value, 30)
    return (value - _SPLITMIX_GAMMA) & _MASK64


def _identity_rows() -> list[int]:
    return [1 << index for index in range(8)]


def _rows_from_triangular_bits(
    lower_bits: int, upper_bits: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0 <= lower_bits < (1 << 28):
        raise ValueError("Gate9 lower-triangular bits are outside 28 bits")
    if not 0 <= upper_bits < (1 << 28):
        raise ValueError("Gate9 upper-triangular bits are outside 28 bits")
    lower = _identity_rows()
    upper = _identity_rows()
    for bit, (row, column) in enumerate(_LOWER_POSITIONS):
        if (lower_bits >> bit) & 1:
            lower[row] |= 1 << column
    for bit, (row, column) in enumerate(_UPPER_POSITIONS):
        if (upper_bits >> bit) & 1:
            upper[row] |= 1 << column
    return tuple(lower), tuple(upper)


def _triangular_bits_from_rows(
    lower_rows: tuple[int, ...], upper_rows: tuple[int, ...]
) -> tuple[int, int]:
    if len(lower_rows) != 8 or len(upper_rows) != 8:
        raise ValueError("Gate9 triangular matrices must contain eight rows")
    lower_bits = 0
    upper_bits = 0
    for row in range(8):
        expected_lower_mask = (1 << (row + 1)) - 1
        expected_upper_mask = _MASK64 ^ ((1 << row) - 1)
        if lower_rows[row] & ~0xFF or upper_rows[row] & ~0xFF:
            raise ValueError("Gate9 matrix row exceeds eight bits")
        if lower_rows[row] & ~expected_lower_mask:
            raise ValueError("Gate9 lower matrix has an above-diagonal bit")
        if upper_rows[row] & (~expected_upper_mask & 0xFF):
            raise ValueError("Gate9 upper matrix has a below-diagonal bit")
        if not ((lower_rows[row] >> row) & 1) or not ((upper_rows[row] >> row) & 1):
            raise ValueError("Gate9 triangular matrix diagonal is not unit")
    for bit, (row, column) in enumerate(_LOWER_POSITIONS):
        lower_bits |= ((lower_rows[row] >> column) & 1) << bit
    for bit, (row, column) in enumerate(_UPPER_POSITIONS):
        upper_bits |= ((upper_rows[row] >> column) & 1) << bit
    return lower_bits, upper_bits


def multiply_gf2_rows(
    left_rows: tuple[int, ...], right_rows: tuple[int, ...]
) -> tuple[int, ...]:
    if len(left_rows) != 8 or len(right_rows) != 8:
        raise ValueError("Gate9 GF(2) matrices must contain eight rows")
    output: list[int] = []
    for left_row in left_rows:
        if not 0 <= left_row < 256:
            raise ValueError("Gate9 left matrix row exceeds eight bits")
        result = 0
        for index in range(8):
            if (left_row >> index) & 1:
                result ^= right_rows[index]
        output.append(result)
    if any(not 0 <= row < 256 for row in output):
        raise ValueError("Gate9 product matrix row exceeds eight bits")
    return tuple(output)


def apply_linear_rows(rows: tuple[int, ...], value: int) -> int:
    _valid_byte(value, "Gate9 affine input")
    if len(rows) != 8 or any(not 0 <= row < 256 for row in rows):
        raise ValueError("Gate9 linear map must contain eight byte rows")
    output = 0
    for row_index, row in enumerate(rows):
        parity = (row & value).bit_count() & 1
        output |= parity << row_index
    return output


def decompose_unit_lu(matrix_rows: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Recover the unique unit-lower/unit-upper factorization over GF(2)."""

    if len(matrix_rows) != 8 or any(not 0 <= row < 256 for row in matrix_rows):
        raise ValueError("Gate9 affine matrix must contain eight byte rows")
    lower = _identity_rows()
    upper = _identity_rows()
    for pivot in range(8):
        for column in range(pivot, 8):
            accumulated = 0
            for prior in range(pivot):
                accumulated ^= (
                    ((lower[pivot] >> prior) & 1)
                    & ((upper[prior] >> column) & 1)
                )
            value = ((matrix_rows[pivot] >> column) & 1) ^ accumulated
            if column == pivot:
                if value != 1:
                    raise ValueError(
                        "Gate9 affine matrix lacks a unit LU factorization"
                    )
            elif value:
                upper[pivot] |= 1 << column
        for row in range(pivot + 1, 8):
            accumulated = 0
            for prior in range(pivot):
                accumulated ^= (
                    ((lower[row] >> prior) & 1)
                    & ((upper[prior] >> pivot) & 1)
                )
            value = ((matrix_rows[row] >> pivot) & 1) ^ accumulated
            if value:
                lower[row] |= 1 << pivot
    lower_tuple = tuple(lower)
    upper_tuple = tuple(upper)
    if multiply_gf2_rows(lower_tuple, upper_tuple) != matrix_rows:
        raise ValueError("Gate9 unit LU reconstruction disagrees with matrix")
    return lower_tuple, upper_tuple


@dataclass(frozen=True, slots=True)
class Gate9AffineOperator:
    key: int
    lower_rows: tuple[int, ...]
    upper_rows: tuple[int, ...]
    matrix_rows: tuple[int, ...]
    bias: int

    def validate(self) -> None:
        _valid_u64(self.key, "Gate9 operator key")
        _valid_byte(self.bias, "Gate9 operator bias")
        lower_bits, upper_bits = _triangular_bits_from_rows(
            self.lower_rows, self.upper_rows
        )
        expected_key = lower_bits | (upper_bits << 28) | (self.bias << 56)
        if expected_key != self.key:
            raise ValueError("Gate9 operator key disagrees with triangular factors")
        if multiply_gf2_rows(self.lower_rows, self.upper_rows) != self.matrix_rows:
            raise ValueError("Gate9 operator matrix disagrees with triangular factors")
        recovered_lower, recovered_upper = decompose_unit_lu(self.matrix_rows)
        if recovered_lower != self.lower_rows or recovered_upper != self.upper_rows:
            raise ValueError("Gate9 operator unit LU factorization is not unique")

    def apply(self, value: int) -> int:
        self.validate()
        return apply_linear_rows(self.matrix_rows, value) ^ self.bias

    def to_public_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "lower_rows": list(self.lower_rows),
            "upper_rows": list(self.upper_rows),
            "matrix_rows": list(self.matrix_rows),
            "bias": self.bias,
        }


def operator_from_key(key: int) -> Gate9AffineOperator:
    key = _valid_u64(key, "Gate9 operator key")
    lower_bits = key & ((1 << 28) - 1)
    upper_bits = (key >> 28) & ((1 << 28) - 1)
    bias = (key >> 56) & 0xFF
    lower, upper = _rows_from_triangular_bits(lower_bits, upper_bits)
    operator = Gate9AffineOperator(
        key=key,
        lower_rows=lower,
        upper_rows=upper,
        matrix_rows=multiply_gf2_rows(lower, upper),
        bias=bias,
    )
    operator.validate()
    return operator


def operator_from_counter(counter: int) -> Gate9AffineOperator:
    return operator_from_key(splitmix64_bijection(counter))


def _global_support_order() -> tuple[int, ...]:
    return tuple(
        sorted(
            protocol.GATE9_SUPPORT_INPUTS,
            key=lambda value: hashlib.sha256(
                f"{_SUPPORT_ORDER_NAMESPACE}:{value}".encode("ascii")
            ).digest(),
        )
    )


GATE9_GLOBAL_SUPPORT_ORDER = _global_support_order()
if set(GATE9_GLOBAL_SUPPORT_ORDER) != set(protocol.GATE9_SUPPORT_INPUTS):
    raise RuntimeError("Gate9 global support ordering drifted")


def public_support_pairs(operator: Gate9AffineOperator) -> tuple[tuple[int, int], ...]:
    operator.validate()
    return tuple((value, operator.apply(value)) for value in GATE9_GLOBAL_SUPPORT_ORDER)


def _normalize_support_pairs(
    pairs: Iterable[tuple[int, int]],
) -> dict[int, int]:
    materialized = tuple(pairs)
    if len(materialized) != protocol.GATE9_SUPPORT_EXAMPLES:
        raise ValueError("Gate9 public support must contain exactly nine pairs")
    mapping: dict[int, int] = {}
    for pair in materialized:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Gate9 public support row must be one input/output tuple")
        source = _valid_byte(pair[0], "Gate9 support input")
        target = _valid_byte(pair[1], "Gate9 support output")
        if source in mapping:
            raise ValueError("Gate9 public support contains a duplicate input")
        mapping[source] = target
    if set(mapping) != set(protocol.GATE9_SUPPORT_INPUTS):
        raise ValueError("Gate9 public support input basis drifted")
    return mapping


def reconstruct_operator_from_support(
    pairs: Iterable[tuple[int, int]],
) -> Gate9AffineOperator:
    mapping = _normalize_support_pairs(pairs)
    bias = mapping[0]
    columns = tuple(mapping[1 << index] ^ bias for index in range(8))
    matrix_rows = tuple(
        sum(((columns[column] >> row) & 1) << column for column in range(8))
        for row in range(8)
    )
    lower, upper = decompose_unit_lu(matrix_rows)
    lower_bits, upper_bits = _triangular_bits_from_rows(lower, upper)
    key = lower_bits | (upper_bits << 28) | (bias << 56)
    operator = Gate9AffineOperator(
        key=key,
        lower_rows=lower,
        upper_rows=upper,
        matrix_rows=matrix_rows,
        bias=bias,
    )
    operator.validate()
    expected_pairs = dict(public_support_pairs(operator))
    if expected_pairs != mapping:
        raise ValueError("Gate9 reconstructed operator disagrees with public support")
    return operator


def apply_public_support_oracle(
    pairs: Iterable[tuple[int, int]],
    query: int,
    *,
    require_novel_query: bool = True,
) -> int:
    query = _valid_byte(query, "Gate9 oracle query")
    if require_novel_query and query in protocol.GATE9_SUPPORT_INPUTS:
        raise ValueError("Gate9 oracle query must lie outside the support basis")
    return reconstruct_operator_from_support(pairs).apply(query)


@dataclass(frozen=True, slots=True)
class Gate9CounterRange:
    name: str
    start: int
    count: int

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Gate9 counter range requires a name")
        _valid_u64(self.start, f"Gate9 {self.name} range start")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("Gate9 counter range count must be positive")
        if self.start + self.count - 1 > _MASK64:
            raise ValueError("Gate9 counter range exceeds uint64")

    @property
    def stop(self) -> int:
        self.validate()
        return self.start + self.count

    def intersection_size(self, other: "Gate9CounterRange") -> int:
        self.validate()
        other.validate()
        return max(0, min(self.stop, other.stop) - max(self.start, other.start))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


GATE9_COUNTER_RANGES = (
    Gate9CounterRange(
        "train",
        protocol.GATE9_TRAIN_OPERATOR_COUNTER_START,
        protocol.GATE9_TRAIN_OPERATOR_COUNT,
    ),
    Gate9CounterRange(
        "validation",
        protocol.GATE9_VALIDATION_OPERATOR_COUNTER_START,
        protocol.GATE9_VALIDATION_OPERATOR_COUNT,
    ),
    Gate9CounterRange(
        "local_test",
        protocol.GATE9_LOCAL_TEST_OPERATOR_COUNTER_START,
        protocol.GATE9_LOCAL_TEST_OPERATOR_COUNT,
    ),
    Gate9CounterRange(
        "graph_test",
        protocol.GATE9_GRAPH_TEST_OPERATOR_COUNTER_START,
        protocol.GATE9_GRAPH_TEST_OPERATOR_COUNT,
    ),
)


def gate9_operator_split_audit() -> dict[str, Any]:
    for row in GATE9_COUNTER_RANGES:
        row.validate()
    intersections = {
        f"{left.name}_x_{right.name}": left.intersection_size(right)
        for index, left in enumerate(GATE9_COUNTER_RANGES)
        for right in GATE9_COUNTER_RANGES[index + 1 :]
    }
    if any(intersections.values()):
        raise ValueError("Gate9 frozen counter ranges overlap")
    boundary_counters = tuple(
        value
        for row in GATE9_COUNTER_RANGES
        for value in (row.start, row.stop - 1)
    )
    for counter in boundary_counters:
        key = splitmix64_bijection(counter)
        if inverse_splitmix64_bijection(key) != counter:
            raise ValueError("Gate9 SplitMix64 boundary round-trip failed")
        if reconstruct_operator_from_support(
            public_support_pairs(operator_from_counter(counter))
        ).key != key:
            raise ValueError("Gate9 boundary support reconstruction failed")
    return {
        "version": GATE9_OPERATOR_CONTRACT_VERSION,
        "status": GATE9_OPERATOR_CONTRACT_STATUS,
        "protocol_head": GATE9_PROTOCOL_HEAD,
        "operator_family": protocol.GATE9_OPERATOR_FAMILY,
        "operator_family_size": protocol.GATE9_OPERATOR_FAMILY_SIZE,
        "counter_mapping": protocol.GATE9_OPERATOR_COUNTER_PERMUTATION,
        "counter_mapping_bijective": True,
        "operator_mapping_injective": True,
        "ranges": [row.to_dict() for row in GATE9_COUNTER_RANGES],
        "intersections": intersections,
        "support_inputs": list(protocol.GATE9_SUPPORT_INPUTS),
        "support_order": list(GATE9_GLOBAL_SUPPORT_ORDER),
        "support_order_operator_independent": True,
        "operator_key_visible_to_model": False,
        "operator_generation_admitted": True,
        "public_support_generation_admitted": True,
        "public_support_oracle_admitted": True,
        "graph_world_generation_admitted": False,
        "architecture_admitted": False,
        "training_admitted": False,
        "checkpoint_loading_admitted": False,
        "scientific_test_admitted": False,
        "result_classification_admitted": False,
    }
