"""Deterministic unseen operator worlds for Post-Training Learning L0."""
from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import math
from typing import Literal

VERSION = "population-language-post-training-learning-l0-protocol-v0"
MODEL_INITIALIZATION_SEEDS = (120100, 120101, 120102)
CALIBRATION_WORLD_SEEDS = (210100, 210101, 210102)
FINAL_WORLD_SEEDS = (220100, 220101, 220102)

OPERATOR_TOKENS = ("dax", "wug", "zorp", "kiki", "blicket", "toma", "fep", "luma")
VALUE_TOKENS = (
    "red", "blue", "green", "yellow", "orange", "purple", "white", "black",
    "triangle", "circle", "square", "star", "hexagon", "oval", "diamond", "cross",
)
DOMAIN_SIZE = len(VALUE_TOKENS)
ODD_MULTIPLIERS = tuple(range(1, DOMAIN_SIZE, 2))

ADAPTATION_EXAMPLES = 64
DIRECT_HOLDOUT_EXAMPLES = 64
CALIBRATION_EXAMPLES = 512
VALIDATION_EXAMPLES = 2_048
TEST_EXAMPLES = 8_192
RETENTION_TEST_EPISODES = 16_384
ADAPTATION_INPUTS_PER_OPERATOR = 8

Split = Literal["adaptation", "direct_holdout", "calibration", "validation", "test"]
SPLIT_COUNTS: dict[Split, int] = {
    "adaptation": ADAPTATION_EXAMPLES,
    "direct_holdout": DIRECT_HOLDOUT_EXAMPLES,
    "calibration": CALIBRATION_EXAMPLES,
    "validation": VALIDATION_EXAMPLES,
    "test": TEST_EXAMPLES,
}
SPLIT_DEPTH: dict[Split, int] = {
    "adaptation": 1,
    "direct_holdout": 1,
    "calibration": 2,
    "validation": 3,
    "test": 4,
}
FINAL_SPLITS: tuple[Split, ...] = ("adaptation", "direct_holdout", "validation", "test")
CALIBRATION_SPLITS: tuple[Split, ...] = ("adaptation", "direct_holdout", "calibration")


def _digest(*parts: object) -> bytes:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


@dataclass(frozen=True)
class OperatorRule:
    token: str
    multiplier: int
    offset: int

    def apply(self, value: int) -> int:
        if type(value) is not int or not 0 <= value < DOMAIN_SIZE:
            raise ValueError("operator input lies outside the value domain")
        return (self.multiplier * value + self.offset) % DOMAIN_SIZE


@dataclass(frozen=True)
class AdaptationWorld:
    seed: int
    operators: tuple[OperatorRule, ...]

    def operator(self, token: str) -> OperatorRule:
        for rule in self.operators:
            if rule.token == token:
                return rule
        raise ValueError(f"unknown operator token: {token}")


@dataclass(frozen=True)
class LearningExample:
    split: Split
    ordinal: int
    world_seed: int
    operators: tuple[str, ...]
    input_value: int
    output_value: int
    tokens: tuple[str, ...]


@functools.lru_cache(maxsize=None)
def make_world(seed: int) -> AdaptationWorld:
    if type(seed) is not int or seed < 0:
        raise ValueError("world seed must be a nonnegative integer")
    used: set[tuple[int, int]] = set()
    rules: list[OperatorRule] = []
    for operator_index, token in enumerate(OPERATOR_TOKENS):
        for attempt in range(10_000):
            digest = _digest(VERSION, seed, "operator", operator_index, attempt)
            pair = (
                ODD_MULTIPLIERS[digest[0] % len(ODD_MULTIPLIERS)],
                int.from_bytes(digest[1:3], "little") % DOMAIN_SIZE,
            )
            if pair in used:
                continue
            used.add(pair)
            rules.append(OperatorRule(token, *pair))
            break
        else:
            raise RuntimeError("could not construct a unique adaptation world")
    return AdaptationWorld(seed, tuple(rules))


def apply_chain(world: AdaptationWorld, operators: tuple[str, ...], value: int) -> int:
    if not operators:
        raise ValueError("operator chain cannot be empty")
    result = value
    for token in reversed(operators):
        result = world.operator(token).apply(result)
    return result


def _validate_world_role(split: Split, world_seed: int) -> None:
    if split in ("adaptation", "direct_holdout"):
        allowed = set(CALIBRATION_WORLD_SEEDS) | set(FINAL_WORLD_SEEDS)
    elif split == "calibration":
        allowed = set(CALIBRATION_WORLD_SEEDS)
    else:
        allowed = set(FINAL_WORLD_SEEDS)
    if world_seed not in allowed:
        raise ValueError(f"world seed {world_seed} is not allowed for split {split}")


@functools.lru_cache(maxsize=None)
def _direct_input_order(world_seed: int, operator_index: int) -> tuple[int, ...]:
    return tuple(sorted(
        range(DOMAIN_SIZE),
        key=lambda value: _digest(VERSION, world_seed, "direct-order", operator_index, value),
    ))


def _decode_composition(index: int, depth: int) -> tuple[tuple[int, ...], int]:
    total = DOMAIN_SIZE * len(OPERATOR_TOKENS) ** depth
    if type(index) is not int or not 0 <= index < total:
        raise ValueError("composition index lies outside the split space")
    input_value, remaining = index % DOMAIN_SIZE, index // DOMAIN_SIZE
    reversed_indices: list[int] = []
    for _ in range(depth):
        reversed_indices.append(remaining % len(OPERATOR_TOKENS))
        remaining //= len(OPERATOR_TOKENS)
    return tuple(reversed(reversed_indices)), input_value


def _coprime_multiplier(total: int, split: Split, world_seed: int) -> int:
    candidate = int.from_bytes(_digest(VERSION, world_seed, split, "multiplier")[:8], "little") % total
    candidate = max(candidate, 1)
    while math.gcd(candidate, total) != 1:
        candidate += 1
        if candidate >= total:
            candidate = 1
    return candidate


def make_example(split: Split, ordinal: int, world_seed: int) -> LearningExample:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"unknown post-training-learning split: {split}")
    _validate_world_role(split, world_seed)
    if type(ordinal) is not int or not 0 <= ordinal < SPLIT_COUNTS[split]:
        raise ValueError("example ordinal lies outside the locked split")

    world = make_world(world_seed)
    depth = SPLIT_DEPTH[split]
    if split in ("adaptation", "direct_holdout"):
        operator_index, position = divmod(ordinal, ADAPTATION_INPUTS_PER_OPERATOR)
        order = _direct_input_order(world_seed, operator_index)
        offset = 0 if split == "adaptation" else ADAPTATION_INPUTS_PER_OPERATOR
        operator_indices = (operator_index,)
        input_value = order[position + offset]
    else:
        total = DOMAIN_SIZE * len(OPERATOR_TOKENS) ** depth
        multiplier = _coprime_multiplier(total, split, world_seed)
        shift = int.from_bytes(_digest(VERSION, world_seed, split, "shift")[:8], "little") % total
        operator_indices, input_value = _decode_composition(
            (ordinal * multiplier + shift) % total,
            depth,
        )

    operators = tuple(OPERATOR_TOKENS[index] for index in operator_indices)
    output_value = apply_chain(world, operators, input_value)
    tokens = (
        "<bos>", "<query>", *operators, VALUE_TOKENS[input_value],
        "<answer>", VALUE_TOKENS[output_value], "<eos>",
    )
    return LearningExample(split, ordinal, world_seed, operators, input_value, output_value, tokens)


@functools.lru_cache(maxsize=None)
def split_fingerprint(split: Split, count: int, world_seed: int) -> str:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"unknown post-training-learning split: {split}")
    _validate_world_role(split, world_seed)
    if type(count) is not int or not 0 < count <= SPLIT_COUNTS[split]:
        raise ValueError("fingerprint count lies outside the locked split")
    digest = hashlib.sha256()
    for ordinal in range(count):
        example = make_example(split, ordinal, world_seed)
        digest.update(ordinal.to_bytes(8, "little"))
        digest.update(b"\0".join(token.encode("utf-8") for token in example.tokens))
        digest.update(b"\n")
    return digest.hexdigest()


def calibration_world_fingerprints() -> dict[str, dict[str, str]]:
    return {
        str(seed): {
            split: split_fingerprint(split, SPLIT_COUNTS[split], seed)
            for split in CALIBRATION_SPLITS
        }
        for seed in CALIBRATION_WORLD_SEEDS
    }


def final_world_fingerprints(seed: int) -> dict[str, str]:
    if seed not in FINAL_WORLD_SEEDS:
        raise ValueError("final-world fingerprint requires a preregistered final seed")
    return {
        split: split_fingerprint(split, SPLIT_COUNTS[split], seed)
        for split in FINAL_SPLITS
    }
