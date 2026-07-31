"""Canonical Gate-8 encoders and pre-tokenization budget contract.

This stage serializes already-admitted public worlds. It loads no tokenizer or
model, performs no training or inference, and opens no scientific test path.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

GATE8_ENCODER_VERSION = "gate8-distributed-transformation-encoder-contract-v0"
GATE8_ENCODER_WORLD_CONTRACT_HEAD = "722c646eacfd05c51fb9d1e8887fe1620d53672c"
GATE8_ENCODER_STATUS = (
    "GATE8_CANONICAL_ENCODER_AND_PRETOKENIZATION_BOUND_ADMITTED_"
    "TOKENIZER_MODEL_AND_EXECUTION_CLOSED"
)

GATE8_REFERENCE_MODEL_ID = "google/gemma-3-1b-it"
GATE8_REFERENCE_MAX_INPUT_TOKENS = 24_576
GATE8_REFERENCE_CONTENT_ASCII_BYTE_LIMIT = 20_000
GATE8_REFERENCE_TEMPLATE_AND_SPECIAL_TOKEN_RESERVE = 4_576
GATE8_REFERENCE_DEMONSTRATION_COUNT = 8
GATE8_REFERENCE_DEMONSTRATION_POPULATION = 32
GATE8_REFERENCE_DEMONSTRATION_DEPTH = 4
GATE8_REFERENCE_DEMONSTRATION_SEED = 0
GATE8_REFERENCE_DEMONSTRATION_INDICES = tuple(range(8))
GATE8_NODE_ALIAS_WIDTH = 2
GATE8_BASE36_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GATE8_ANSWER_PATTERN = re.compile(r"^[0-9A-F]$")

GATE8_TRANSFORM_HEX = (
    "D56AFE8271C93B40",
    "6C832E9074ADF B15".replace(" ", ""),
    "7DE5B20613CF98A4",
    "9D37C68025FB1AE4",
    "C19AE4562FB730D8",
    "8674132D5C0FA9EB",
    "436E95F02DC81A7B",
    "AE263709F4DC851B",
)
GATE8_TASK_RULE = (
    "Start at root R with symbol S. Follow the unique directed path from R to "
    "target Q. Apply each edge transform in path order. Output exactly one "
    "uppercase hexadecimal symbol."
)


def _read_attr(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise TypeError(f"Gate8 encoder input is missing {name}")
    return getattr(value, name)


def _base36_fixed(value: int, width: int = GATE8_NODE_ALIAS_WIDTH) -> str:
    if value < 0:
        raise ValueError("Gate8 alias index must be non-negative")
    digits = []
    current = value
    if current == 0:
        digits.append("0")
    while current:
        current, remainder = divmod(current, len(GATE8_BASE36_ALPHABET))
        digits.append(GATE8_BASE36_ALPHABET[remainder])
    result = "".join(reversed(digits)).rjust(width, "0")
    if len(result) != width:
        raise ValueError("Gate8 alias width is insufficient for this graph")
    return result


def _public_world(value: Any) -> Any:
    return getattr(value, "public", value)


def _truth(value: Any) -> Any:
    if not hasattr(value, "truth"):
        raise TypeError("Gate8 demonstration is missing its public truth")
    return value.truth


def _validate_public_world(world: Any) -> None:
    validator = getattr(world, "validate", None)
    if callable(validator):
        validator()
    population = int(_read_attr(world, "population"))
    depth = int(_read_attr(world, "depth"))
    workers = tuple(_read_attr(world, "workers"))
    query = _read_attr(world, "query")
    if population not in (32, 64, 128, 256, 512, 1_024):
        raise ValueError("Gate8 encoder population is outside the frozen ladder")
    if depth not in (4, 8, 16, 32, 64, 128) or 8 * depth > population:
        raise ValueError("Gate8 encoder condition is outside the frozen matrix")
    if len(workers) != population:
        raise ValueError("Gate8 encoder requires exactly one worker per edge")
    if tuple(int(_read_attr(worker, "worker_index")) for worker in workers) != tuple(
        range(population)
    ):
        raise ValueError("Gate8 encoder requires canonical worker-index order")
    if not 0 <= int(_read_attr(query, "root_symbol")) < 16:
        raise ValueError("Gate8 encoder root symbol is outside 0..15")


def gate8_node_aliases(world_like: Any) -> dict[str, str]:
    world = _public_world(world_like)
    _validate_public_world(world)
    query = _read_attr(world, "query")
    labels = {
        str(_read_attr(query, "root_node")),
        str(_read_attr(query, "target_node")),
    }
    for worker in tuple(_read_attr(world, "workers")):
        labels.add(str(_read_attr(worker, "source_node")))
        labels.add(str(_read_attr(worker, "target_node")))
    population = int(_read_attr(world, "population"))
    if len(labels) != population + 1:
        raise ValueError("Gate8 encoder expected population + 1 unique nodes")
    if len(labels) > len(GATE8_BASE36_ALPHABET) ** GATE8_NODE_ALIAS_WIDTH:
        raise ValueError("Gate8 alias namespace is too small")
    return {
        label: _base36_fixed(index)
        for index, label in enumerate(sorted(labels))
    }


def _transform_table_lines() -> tuple[str, ...]:
    if len(GATE8_TRANSFORM_HEX) != 8:
        raise RuntimeError("Gate8 encoder transform table changed")
    for encoded in GATE8_TRANSFORM_HEX:
        if len(encoded) != 16 or not re.fullmatch(r"[0-9A-F]{16}", encoded):
            raise RuntimeError("Gate8 encoder transform table is malformed")
        if len(set(encoded)) != 16:
            raise RuntimeError("Gate8 encoder transform table is not bijective")
    return tuple(f"T{index}={encoded}" for index, encoded in enumerate(GATE8_TRANSFORM_HEX))


def encode_gate8_public_graph(world_like: Any) -> str:
    world = _public_world(world_like)
    _validate_public_world(world)
    aliases = gate8_node_aliases(world)
    query = _read_attr(world, "query")
    root = aliases[str(_read_attr(query, "root_node"))]
    target = aliases[str(_read_attr(query, "target_node"))]
    symbol = format(int(_read_attr(query, "root_symbol")), "X")
    population = int(_read_attr(world, "population"))
    depth = int(_read_attr(world, "depth"))
    lines = [f"Q|P={population}|D={depth}|R={root}|Z={target}|S={symbol}"]
    for worker in tuple(_read_attr(world, "workers")):
        source = aliases[str(_read_attr(worker, "source_node"))]
        destination = aliases[str(_read_attr(worker, "target_node"))]
        transform_id = int(_read_attr(worker, "transform_id"))
        if not 0 <= transform_id < 8:
            raise ValueError("Gate8 encoder transform id is outside 0..7")
        lines.append(f"{source}>{destination}:{transform_id}")
    result = "\n".join(lines)
    result.encode("ascii")
    return result


def encode_gate8_worker_observation(world_like: Any, worker_index: int) -> str:
    world = _public_world(world_like)
    _validate_public_world(world)
    population = int(_read_attr(world, "population"))
    if not 0 <= worker_index < population:
        raise ValueError("Gate8 worker observation index is outside the population")
    aliases = gate8_node_aliases(world)
    query = _read_attr(world, "query")
    worker = tuple(_read_attr(world, "workers"))[worker_index]
    root = aliases[str(_read_attr(query, "root_node"))]
    target = aliases[str(_read_attr(query, "target_node"))]
    source = aliases[str(_read_attr(worker, "source_node"))]
    destination = aliases[str(_read_attr(worker, "target_node"))]
    transform_id = int(_read_attr(worker, "transform_id"))
    symbol = format(int(_read_attr(query, "root_symbol")), "X")
    depth = int(_read_attr(world, "depth"))
    observation = (
        f"G8W0|P={population}|D={depth}|R={root}|Z={target}|S={symbol}|"
        f"I={_base36_fixed(worker_index)}|E={source}>{destination}:{transform_id}"
    )
    observation.encode("ascii")
    return observation


def _validate_demonstrations(demonstrations: Iterable[Any]) -> tuple[Any, ...]:
    rows = tuple(demonstrations)
    if len(rows) != GATE8_REFERENCE_DEMONSTRATION_COUNT:
        raise ValueError("Gate8 reference prompt requires exactly eight demonstrations")
    observed_indices = []
    for generated in rows:
        public = _public_world(generated)
        _validate_public_world(public)
        if str(_read_attr(public, "split")) != "demonstration":
            raise ValueError("Gate8 reference demonstrations require the demonstration split")
        if int(_read_attr(public, "seed")) != GATE8_REFERENCE_DEMONSTRATION_SEED:
            raise ValueError("Gate8 reference demonstration seed changed")
        if (
            int(_read_attr(public, "population")),
            int(_read_attr(public, "depth")),
        ) != (
            GATE8_REFERENCE_DEMONSTRATION_POPULATION,
            GATE8_REFERENCE_DEMONSTRATION_DEPTH,
        ):
            raise ValueError("Gate8 reference demonstration condition changed")
        observed_indices.append(int(_read_attr(public, "world_index")))
        truth = _truth(generated)
        answer = int(_read_attr(truth, "answer_symbol"))
        if not 0 <= answer < 16:
            raise ValueError("Gate8 demonstration answer is outside 0..15")
    if tuple(observed_indices) != GATE8_REFERENCE_DEMONSTRATION_INDICES:
        raise ValueError("Gate8 reference demonstrations must be ordered 0..7")
    return rows


def encode_gate8_reference_prompt(target_world_like: Any, demonstrations: Iterable[Any]) -> str:
    target = _public_world(target_world_like)
    _validate_public_world(target)
    if str(_read_attr(target, "split")) not in ("contract", "test"):
        raise ValueError("Gate8 reference target must use contract or scientific-test split")
    demos = _validate_demonstrations(demonstrations)
    lines = [
        "G8V0",
        "A=0123456789ABCDEF",
        *_transform_table_lines(),
        "RULE=" + GATE8_TASK_RULE,
    ]
    for index, generated in enumerate(demos):
        lines.append(f"D{index}")
        lines.extend(encode_gate8_public_graph(generated).splitlines())
        answer = int(_read_attr(_truth(generated), "answer_symbol"))
        lines.append("Y=" + format(answer, "X"))
    lines.append("X")
    lines.extend(encode_gate8_public_graph(target).splitlines())
    lines.append("Y=")
    prompt = "\n".join(lines)
    validate_gate8_reference_prompt_budget(prompt)
    return prompt


@dataclass(frozen=True, slots=True)
class Gate8PromptBudget:
    ascii_bytes: int
    content_byte_limit: int
    template_and_special_token_reserve: int
    max_input_tokens: int
    exact_tokenizer_bound_proven: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_gate8_reference_prompt_budget(prompt: str) -> Gate8PromptBudget:
    try:
        encoded = prompt.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Gate8 canonical prompt must remain ASCII-only") from exc
    byte_count = len(encoded)
    if byte_count > GATE8_REFERENCE_CONTENT_ASCII_BYTE_LIMIT:
        raise ValueError("Gate8 canonical prompt exceeds the frozen ASCII byte limit")
    if (
        GATE8_REFERENCE_CONTENT_ASCII_BYTE_LIMIT
        + GATE8_REFERENCE_TEMPLATE_AND_SPECIAL_TOKEN_RESERVE
        != GATE8_REFERENCE_MAX_INPUT_TOKENS
    ):
        raise RuntimeError("Gate8 pre-tokenization budget does not close exactly")
    return Gate8PromptBudget(
        ascii_bytes=byte_count,
        content_byte_limit=GATE8_REFERENCE_CONTENT_ASCII_BYTE_LIMIT,
        template_and_special_token_reserve=(
            GATE8_REFERENCE_TEMPLATE_AND_SPECIAL_TOKEN_RESERVE
        ),
        max_input_tokens=GATE8_REFERENCE_MAX_INPUT_TOKENS,
        exact_tokenizer_bound_proven=False,
    )


def gate8_reference_prompt_sha256(prompt: str) -> str:
    validate_gate8_reference_prompt_budget(prompt)
    return hashlib.sha256(prompt.encode("ascii")).hexdigest()


def parse_gate8_reference_answer(text: str) -> int:
    normalized = text.strip().upper()
    if not GATE8_ANSWER_PATTERN.fullmatch(normalized):
        raise ValueError("Gate8 reference answer must be exactly one hexadecimal symbol")
    return int(normalized, 16)


def gate8_encoder_contract_plan() -> dict[str, Any]:
    return {
        "version": GATE8_ENCODER_VERSION,
        "world_contract_head": GATE8_ENCODER_WORLD_CONTRACT_HEAD,
        "scientific_status": GATE8_ENCODER_STATUS,
        "encoder_admitted": True,
        "tokenizer_bound": False,
        "model_bound": False,
        "training_admitted": False,
        "baseline_execution_admitted": False,
        "scientific_execution_admitted": False,
        "reference_model_id": GATE8_REFERENCE_MODEL_ID,
        "reference_max_input_tokens": GATE8_REFERENCE_MAX_INPUT_TOKENS,
        "content_ascii_byte_limit": GATE8_REFERENCE_CONTENT_ASCII_BYTE_LIMIT,
        "template_and_special_token_reserve": (
            GATE8_REFERENCE_TEMPLATE_AND_SPECIAL_TOKEN_RESERVE
        ),
        "demonstration_count": GATE8_REFERENCE_DEMONSTRATION_COUNT,
        "demonstration_condition": [
            GATE8_REFERENCE_DEMONSTRATION_POPULATION,
            GATE8_REFERENCE_DEMONSTRATION_DEPTH,
        ],
        "demonstration_seed": GATE8_REFERENCE_DEMONSTRATION_SEED,
        "demonstration_indices": list(GATE8_REFERENCE_DEMONSTRATION_INDICES),
        "node_alias_width": GATE8_NODE_ALIAS_WIDTH,
        "exact_tokenizer_verification_required_before_baseline_execution": True,
    }


_transform_table_lines()
