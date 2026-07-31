"""Frozen Gate-8 v1 Gemma reference-execution contract.

This module defines immutable identities, row schemas, ordering, and execution
boundaries for the 1B reference phase. It imports no model framework, loads no
snapshot, generates no benchmark world, and performs no inference.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

GATE8_V1_GEMMA_REFERENCE_EXECUTION_VERSION = (
    "gate8-v1-gemma-reference-execution-v0"
)
GATE8_V1_GEMMA_REFERENCE_STATUS = (
    "GATE8_V1_GEMMA_REFERENCE_EXECUTION_ADMITTED_JOINT_CLASSIFICATION_CLOSED"
)

GATE8_V1_POPULATION_RESULT_HEAD = "14636d219781381853f81036b96c691b7e6997ee"
GATE8_V1_POPULATION_SUMMARY_SHA256 = (
    "6d30d773f11c1155df3346128385da9231610ea05e95937e5acccb5529fca3fe"
)
GATE8_V1_POPULATION_PER_WORLD_SHA256 = (
    "45e36bda230440d4fa2342183154b474473498df51917e147a37e0baa81c3323"
)
GATE8_V1_SCIENTIFIC_PROTOCOL_HEAD = "6bb89111a47713bea0a23bb1cae662ed5ec56b42"
GATE8_V1_WORLD_CONTRACT_HEAD = "722c646eacfd05c51fb9d1e8887fe1620d53672c"
GATE8_V1_ENCODER_HEAD = "9882256ae0152bc266dc4d96cab3bbeb0c4ef95b"
GATE8_V1_TOKENIZER_RESULT_HEAD = "c7f5260189ef9ac1a1beb73596446316631090c7"
GATE8_V1_TOKENIZER_RESULT_SHA256 = (
    "c8d6adb733cadbbd251d91d35f9d224e255705dac49ba144655717f9f4ab7b8d"
)
GATE8_V1_TOKENIZER_MANIFEST_SHA256 = (
    "21de192eb57c0759fbf2236fae2252e5319696b71689ada1471b74a9f1315a88"
)
GATE8_V1_WEIGHT_RESULT_HEAD = "8237732aecbec083c66668de9fae132e0cc4c1f9"
GATE8_V1_WEIGHT_RESULT_SHA256 = (
    "c554b66068b04ade24e77bb561fb7fff148fc3fd9a6316e011f710b0f320c10d"
)
GATE8_V1_WEIGHT_MANIFEST_SHA256 = (
    "99ae54872115207c1e703a7c63fb66a1f4c145741958e63354bcf074310ae51c"
)

GATE8_V1_REFERENCE_REPO_ID = "google/gemma-3-1b-it"
GATE8_V1_REFERENCE_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
GATE8_V1_REFERENCE_PARAMETER_COUNT = 999_885_952
GATE8_V1_REFERENCE_DTYPE = "bfloat16"
GATE8_V1_REFERENCE_MAX_INPUT_TOKENS = 24_576
GATE8_V1_REFERENCE_MAX_NEW_TOKENS = 64
GATE8_V1_REFERENCE_DEMONSTRATIONS = 8
GATE8_V1_REFERENCE_DECODING = "greedy_temperature_0"
GATE8_V1_REFERENCE_BATCH_SIZE = 1
GATE8_V1_REFERENCE_ATTENTION_IMPLEMENTATION = "sdpa"

GATE8_V1_TEST_SPLIT = "test"
GATE8_V1_TEST_SEED = 0
GATE8_V1_TEST_WORLD_START = 0
GATE8_V1_TEST_WORLD_END_INCLUSIVE = 511
GATE8_V1_WORLDS_PER_CONDITION = 512
GATE8_V1_POPULATIONS = (32, 64, 128, 256, 512, 1_024)
GATE8_V1_DEPTHS = (4, 8, 16, 32, 64, 128)
GATE8_V1_VALID_CONDITIONS = tuple(
    (population, depth)
    for population in GATE8_V1_POPULATIONS
    for depth in GATE8_V1_DEPTHS
    if 8 * depth <= population
)
GATE8_V1_CONDITION_COUNT = 21
GATE8_V1_REFERENCE_ROWS = GATE8_V1_CONDITION_COUNT * GATE8_V1_WORLDS_PER_CONDITION
GATE8_V1_BOOTSTRAP_SAMPLES = 20_000
GATE8_V1_BOOTSTRAP_CONFIDENCE = 0.95
GATE8_V1_BOOTSTRAP_NAMESPACE = "gate8-v1-gemma-reference-bootstrap-v0"
GATE8_V1_REFERENCE_POOLING = "equal_weight_across_21_conditions"

GATE8_V1_REQUIRED_SOFTWARE = {
    "python": "3.11.9",
    "torch": "2.9.1+cu130",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "numpy": "2.3.5",
    "huggingface-hub": "0.36.2",
}
GATE8_V1_REQUIRED_TOKENIZER_FILE_SHA256 = {
    "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
    "config.json": "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e",
    "special_tokens_map.json": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
    "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
    "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    "tokenizer_config.json": "bfe25c2735e395407beb78456ea9a6984a1f00d8c16fa04a8b75f2a614cf53e1",
}
GATE8_V1_REQUIRED_MODEL_FILE_SHA256 = {
    "config.json": "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e",
    "generation_config.json": "fd9324becc53c4be610db39e13a613006f09fd6ef71a95fb6320dc33157490a3",
    "model.safetensors": "3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6",
}
GATE8_V1_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"
)
GATE8_V1_ANSWER_PATTERN = re.compile(r"^[0-9A-F]$")
GATE8_V1_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GATE8_V1_WORLD_ID_PATTERN = re.compile(r"^g8_[0-9a-f]{24}$")


def gate8_v1_reference_sequence(
    population: int, depth: int, world_index: int
) -> int:
    try:
        condition_index = GATE8_V1_VALID_CONDITIONS.index((population, depth))
    except ValueError as exc:
        raise ValueError("Gate8 v1 reference condition is outside the frozen matrix") from exc
    if not GATE8_V1_TEST_WORLD_START <= world_index <= GATE8_V1_TEST_WORLD_END_INCLUSIVE:
        raise ValueError("Gate8 v1 reference world index is outside 0..511")
    return condition_index * GATE8_V1_WORLDS_PER_CONDITION + world_index


def parse_gate8_v1_reference_answer(text: str) -> int:
    if not isinstance(text, str):
        raise TypeError("Gate8 v1 reference output must be text")
    normalized = text.strip().upper()
    if not GATE8_V1_ANSWER_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Gate8 v1 reference answer must be exactly one hexadecimal symbol"
        )
    return int(normalized, 16)


def canonical_json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def canonical_matrix_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_line(row).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Gate8V1ReferencePromptRow:
    sequence: int
    population: int
    depth: int
    world_index: int
    world_id: str
    prompt_sha256: str
    ascii_bytes: int
    input_tokens: int
    answer_symbol: int

    def validate(self) -> None:
        if self.sequence != gate8_v1_reference_sequence(
            self.population, self.depth, self.world_index
        ):
            raise ValueError("Gate8 v1 reference prompt sequence drifted")
        if not GATE8_V1_WORLD_ID_PATTERN.fullmatch(self.world_id):
            raise ValueError("Gate8 v1 reference world ID is malformed")
        if not GATE8_V1_SHA256_PATTERN.fullmatch(self.prompt_sha256):
            raise ValueError("Gate8 v1 reference prompt hash is malformed")
        if not 0 < self.ascii_bytes <= 20_000:
            raise ValueError("Gate8 v1 reference prompt exceeds its ASCII envelope")
        if not 0 < self.input_tokens <= GATE8_V1_REFERENCE_MAX_INPUT_TOKENS:
            raise ValueError("Gate8 v1 reference prompt exceeds its token budget")
        if not 0 <= self.answer_symbol < 16:
            raise ValueError("Gate8 v1 reference oracle answer is outside 0..15")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate8V1ReferenceResultRow:
    sequence: int
    population: int
    depth: int
    world_index: int
    world_id: str
    prompt_sha256: str
    ascii_bytes: int
    input_tokens: int
    answer_symbol: int
    generated_text: str
    output_token_ids: tuple[int, ...]
    predicted_symbol: int | None
    parse_status: str
    correct: bool
    wall_seconds: float
    peak_device_bytes: int

    def validate(self) -> None:
        prompt = Gate8V1ReferencePromptRow(
            sequence=self.sequence,
            population=self.population,
            depth=self.depth,
            world_index=self.world_index,
            world_id=self.world_id,
            prompt_sha256=self.prompt_sha256,
            ascii_bytes=self.ascii_bytes,
            input_tokens=self.input_tokens,
            answer_symbol=self.answer_symbol,
        )
        prompt.validate()
        if self.parse_status not in ("valid", "invalid"):
            raise ValueError("Gate8 v1 reference parse status is unknown")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.output_token_ids
        ):
            raise ValueError("Gate8 v1 reference output token IDs are invalid")
        if len(self.output_token_ids) > GATE8_V1_REFERENCE_MAX_NEW_TOKENS:
            raise ValueError("Gate8 v1 reference output exceeded 64 tokens")
        if self.wall_seconds < 0.0 or self.peak_device_bytes < 0:
            raise ValueError("Gate8 v1 reference resource accounting is negative")
        if self.parse_status == "valid":
            parsed = parse_gate8_v1_reference_answer(self.generated_text)
            if self.predicted_symbol != parsed:
                raise ValueError("Gate8 v1 reference prediction disagrees with output")
        elif self.predicted_symbol is not None:
            raise ValueError("Gate8 v1 invalid output exposed a prediction")
        if self.correct != (
            self.predicted_symbol is not None
            and self.predicted_symbol == self.answer_symbol
        ):
            raise ValueError("Gate8 v1 reference correctness is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["output_token_ids"] = list(self.output_token_ids)
        return payload


@dataclass(frozen=True, slots=True)
class Gate8V1ReferenceConditionMetric:
    population: int
    depth: int
    accuracy: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    valid_parse_rate: float
    correct: int
    valid_outputs: int
    worlds: int
    maximum_input_tokens: int
    mean_input_tokens: float
    mean_output_tokens: float
    mean_wall_seconds: float
    peak_device_bytes: int
    correctness_vector_sha256: str

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_V1_VALID_CONDITIONS:
            raise ValueError("Gate8 v1 reference metric condition drifted")
        if self.worlds != GATE8_V1_WORLDS_PER_CONDITION:
            raise ValueError("Gate8 v1 reference metric world count drifted")
        if not 0 <= self.correct <= self.valid_outputs <= self.worlds:
            raise ValueError("Gate8 v1 reference metric count is invalid")
        for value in (
            self.accuracy,
            self.bootstrap_ci_low,
            self.bootstrap_ci_high,
            self.valid_parse_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Gate8 v1 reference metric rate is outside 0..1")
        if self.bootstrap_ci_low > self.bootstrap_ci_high:
            raise ValueError("Gate8 v1 reference metric interval is reversed")
        if abs(self.accuracy - self.correct / self.worlds) > 1e-12:
            raise ValueError("Gate8 v1 reference metric accuracy is inconsistent")
        if abs(self.valid_parse_rate - self.valid_outputs / self.worlds) > 1e-12:
            raise ValueError("Gate8 v1 reference parse rate is inconsistent")
        if not 0 < self.maximum_input_tokens <= GATE8_V1_REFERENCE_MAX_INPUT_TOKENS:
            raise ValueError("Gate8 v1 reference maximum token count is invalid")
        if self.mean_input_tokens <= 0.0 or self.mean_output_tokens < 0.0:
            raise ValueError("Gate8 v1 reference token means are invalid")
        if self.mean_wall_seconds < 0.0 or self.peak_device_bytes < 0:
            raise ValueError("Gate8 v1 reference resources are invalid")
        if not GATE8_V1_SHA256_PATTERN.fullmatch(self.correctness_vector_sha256):
            raise ValueError("Gate8 v1 reference correctness hash is malformed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def validate_gate8_v1_prompt_matrix(
    rows: Iterable[Gate8V1ReferencePromptRow],
) -> tuple[Gate8V1ReferencePromptRow, ...]:
    result = tuple(rows)
    if len(result) != GATE8_V1_REFERENCE_ROWS:
        raise ValueError("Gate8 v1 reference prompt matrix must contain 10,752 rows")
    if tuple(row.sequence for row in result) != tuple(range(GATE8_V1_REFERENCE_ROWS)):
        raise ValueError("Gate8 v1 reference prompt matrix sequence is not contiguous")
    world_ids: set[str] = set()
    for row in result:
        row.validate()
        if row.world_id in world_ids:
            raise ValueError("Gate8 v1 reference prompt matrix repeats a world ID")
        world_ids.add(row.world_id)
    return result


def validate_gate8_v1_result_prefix(
    rows: Iterable[Gate8V1ReferenceResultRow],
    prompts: tuple[Gate8V1ReferencePromptRow, ...],
) -> tuple[Gate8V1ReferenceResultRow, ...]:
    result = tuple(rows)
    if len(result) > len(prompts):
        raise ValueError("Gate8 v1 reference result prefix exceeds the prompt matrix")
    for sequence, row in enumerate(result):
        row.validate()
        if row.sequence != sequence:
            raise ValueError("Gate8 v1 reference result prefix is not contiguous")
        expected = prompts[sequence]
        observed_identity = (
            row.population,
            row.depth,
            row.world_index,
            row.world_id,
            row.prompt_sha256,
            row.ascii_bytes,
            row.input_tokens,
            row.answer_symbol,
        )
        expected_identity = (
            expected.population,
            expected.depth,
            expected.world_index,
            expected.world_id,
            expected.prompt_sha256,
            expected.ascii_bytes,
            expected.input_tokens,
            expected.answer_symbol,
        )
        if observed_identity != expected_identity:
            raise ValueError("Gate8 v1 reference result prefix identity drifted")
    return result


def gate8_v1_gemma_reference_execution_plan() -> dict[str, Any]:
    if len(GATE8_V1_VALID_CONDITIONS) != GATE8_V1_CONDITION_COUNT:
        raise RuntimeError("Gate8 v1 reference condition count changed")
    if GATE8_V1_REFERENCE_ROWS != 10_752:
        raise RuntimeError("Gate8 v1 reference row count changed")
    return {
        "version": GATE8_V1_GEMMA_REFERENCE_EXECUTION_VERSION,
        "scientific_status": GATE8_V1_GEMMA_REFERENCE_STATUS,
        "population_result_head": GATE8_V1_POPULATION_RESULT_HEAD,
        "population_summary_sha256": GATE8_V1_POPULATION_SUMMARY_SHA256,
        "population_per_world_sha256": GATE8_V1_POPULATION_PER_WORLD_SHA256,
        "scientific_protocol_head": GATE8_V1_SCIENTIFIC_PROTOCOL_HEAD,
        "world_contract_head": GATE8_V1_WORLD_CONTRACT_HEAD,
        "encoder_head": GATE8_V1_ENCODER_HEAD,
        "tokenizer_result_head": GATE8_V1_TOKENIZER_RESULT_HEAD,
        "weight_result_head": GATE8_V1_WEIGHT_RESULT_HEAD,
        "reference_repo_id": GATE8_V1_REFERENCE_REPO_ID,
        "reference_revision": GATE8_V1_REFERENCE_REVISION,
        "reference_parameter_count": GATE8_V1_REFERENCE_PARAMETER_COUNT,
        "reference_dtype": GATE8_V1_REFERENCE_DTYPE,
        "reference_max_input_tokens": GATE8_V1_REFERENCE_MAX_INPUT_TOKENS,
        "reference_max_new_tokens": GATE8_V1_REFERENCE_MAX_NEW_TOKENS,
        "reference_demonstrations": GATE8_V1_REFERENCE_DEMONSTRATIONS,
        "reference_decoding": GATE8_V1_REFERENCE_DECODING,
        "reference_batch_size": GATE8_V1_REFERENCE_BATCH_SIZE,
        "attention_implementation": GATE8_V1_REFERENCE_ATTENTION_IMPLEMENTATION,
        "test_split": GATE8_V1_TEST_SPLIT,
        "test_seed": GATE8_V1_TEST_SEED,
        "world_index_start": GATE8_V1_TEST_WORLD_START,
        "world_index_end_inclusive": GATE8_V1_TEST_WORLD_END_INCLUSIVE,
        "worlds_per_condition": GATE8_V1_WORLDS_PER_CONDITION,
        "conditions": [list(row) for row in GATE8_V1_VALID_CONDITIONS],
        "condition_count": GATE8_V1_CONDITION_COUNT,
        "reference_rows": GATE8_V1_REFERENCE_ROWS,
        "bootstrap_samples": GATE8_V1_BOOTSTRAP_SAMPLES,
        "bootstrap_confidence": GATE8_V1_BOOTSTRAP_CONFIDENCE,
        "bootstrap_namespace": GATE8_V1_BOOTSTRAP_NAMESPACE,
        "reference_pooling": GATE8_V1_REFERENCE_POOLING,
        "required_software": dict(GATE8_V1_REQUIRED_SOFTWARE),
        "required_tokenizer_file_sha256": dict(
            GATE8_V1_REQUIRED_TOKENIZER_FILE_SHA256
        ),
        "required_model_file_sha256": dict(GATE8_V1_REQUIRED_MODEL_FILE_SHA256),
        "chat_template_sha256": GATE8_V1_CHAT_TEMPLATE_SHA256,
        "prompt_index_completed_before_model_load": True,
        "transactional_resume": True,
        "training_admitted": False,
        "population_execution_admitted": False,
        "joint_reference_comparison_admitted": False,
        "final_classifier_admitted": False,
    }
