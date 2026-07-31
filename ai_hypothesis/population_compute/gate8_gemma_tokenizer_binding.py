"""Gate-8 exact Gemma tokenizer binding and token-count contracts.

The module handles tokenizer files and token counts only. It never imports or
loads a causal language model and never performs generation or training.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

GATE8_TOKENIZER_BINDING_VERSION = "gate8-gemma-tokenizer-binding-v0"
GATE8_TOKENIZER_BINDING_ENCODER_HEAD = "9882256ae0152bc266dc4d96cab3bbeb0c4ef95b"
GATE8_TOKENIZER_BINDING_STATUS = (
    "GATE8_EXACT_GEMMA_TOKENIZER_BINDING_ADMITTED_MODEL_AND_INFERENCE_CLOSED"
)
GATE8_GEMMA_REPO_ID = "google/gemma-3-1b-it"
GATE8_GEMMA_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
GATE8_GEMMA_MAX_INPUT_TOKENS = 24_576
GATE8_GEMMA_MAX_OUTPUT_TOKENS = 64
GATE8_GEMMA_REQUIRED_TOKENIZER_FILES = (
    "added_tokens.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
GATE8_GEMMA_FORBIDDEN_MODEL_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
)
GATE8_GEMMA_FORBIDDEN_MODEL_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "generation_config.json",
)
GATE8_VALID_CONDITIONS = tuple(
    (population, depth)
    for population in (32, 64, 128, 256, 512, 1_024)
    for depth in (4, 8, 16, 32, 64, 128)
    if 8 * depth <= population
)
GATE8_FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
GATE8_FILE_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate8_gemma_revision(revision: str) -> None:
    if revision != GATE8_GEMMA_REVISION:
        raise ValueError("Gate8 Gemma revision changed from the frozen immutable commit")
    if not GATE8_FULL_SHA_PATTERN.fullmatch(revision):
        raise ValueError("Gate8 Gemma revision must be one full lowercase commit SHA")


def _visible_snapshot_files(snapshot_root: Path) -> tuple[Path, ...]:
    root = snapshot_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Gate8 tokenizer snapshot directory does not exist: {root}")
    return tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and ".cache" not in path.relative_to(root).parts
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def validate_gate8_tokenizer_snapshot(snapshot_root: Path) -> dict[str, str]:
    root = snapshot_root.resolve()
    files = _visible_snapshot_files(root)
    relative = tuple(path.relative_to(root).as_posix() for path in files)
    if relative != GATE8_GEMMA_REQUIRED_TOKENIZER_FILES:
        missing = sorted(set(GATE8_GEMMA_REQUIRED_TOKENIZER_FILES) - set(relative))
        extra = sorted(set(relative) - set(GATE8_GEMMA_REQUIRED_TOKENIZER_FILES))
        raise ValueError(
            f"Gate8 tokenizer snapshot file set changed; missing={missing}, extra={extra}"
        )
    for path in files:
        name = path.name.lower()
        if name in GATE8_GEMMA_FORBIDDEN_MODEL_FILENAMES or name.endswith(
            GATE8_GEMMA_FORBIDDEN_MODEL_SUFFIXES
        ):
            raise ValueError("Gate8 tokenizer snapshot contains a forbidden model file")
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in files
    }
    if tuple(hashes) != GATE8_GEMMA_REQUIRED_TOKENIZER_FILES:
        raise RuntimeError("Gate8 tokenizer hash order changed")
    if any(not GATE8_FILE_SHA_PATTERN.fullmatch(value) for value in hashes.values()):
        raise RuntimeError("Gate8 tokenizer file hash is malformed")
    return hashes


def gate8_chat_template_sha256(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("Gate8 tokenizer has no non-empty chat template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def gate8_prompt_token_ids(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Gate8 prompt must be non-empty text")
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    if not isinstance(encoded, dict) and not hasattr(encoded, "__getitem__"):
        raise TypeError("Gate8 chat template did not return a token mapping")
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1:
            raise ValueError("Gate8 tokenizer returned more than one prompt row")
        input_ids = input_ids[0]
    result = tuple(int(token_id) for token_id in input_ids)
    if not result:
        raise ValueError("Gate8 tokenizer returned an empty input")
    if any(token_id < 0 for token_id in result):
        raise ValueError("Gate8 tokenizer returned a negative token id")
    return result


@dataclass(frozen=True, slots=True)
class Gate8TokenizerConditionCount:
    population: int
    depth: int
    prompt_sha256: str
    ascii_bytes: int
    input_tokens: int

    def validate(self) -> None:
        if (self.population, self.depth) not in GATE8_VALID_CONDITIONS:
            raise ValueError("Gate8 tokenizer row is outside the frozen condition matrix")
        if not GATE8_FILE_SHA_PATTERN.fullmatch(self.prompt_sha256):
            raise ValueError("Gate8 tokenizer prompt hash is malformed")
        if self.ascii_bytes <= 0 or self.ascii_bytes > 20_000:
            raise ValueError("Gate8 tokenizer row violates the frozen ASCII envelope")
        if self.input_tokens <= 0 or self.input_tokens > GATE8_GEMMA_MAX_INPUT_TOKENS:
            raise ValueError("Gate8 tokenizer row violates the frozen input-token limit")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def validate_gate8_tokenizer_condition_matrix(
    rows: Iterable[Gate8TokenizerConditionCount],
) -> tuple[Gate8TokenizerConditionCount, ...]:
    result = tuple(rows)
    observed = tuple((row.population, row.depth) for row in result)
    if observed != GATE8_VALID_CONDITIONS:
        raise ValueError("Gate8 tokenizer rows must cover the exact population-major matrix")
    for row in result:
        row.validate()
    return result


def gate8_maximum_token_condition(
    rows: Iterable[Gate8TokenizerConditionCount],
) -> Gate8TokenizerConditionCount:
    matrix = validate_gate8_tokenizer_condition_matrix(rows)
    return max(matrix, key=lambda row: (row.input_tokens, row.ascii_bytes, row.population, row.depth))


@dataclass(frozen=True, slots=True)
class Gate8TokenizerBindingSummary:
    repo_id: str
    revision: str
    tokenizer_class: str
    transformers_version: str
    tokenizers_version: str
    huggingface_hub_version: str
    chat_template_sha256: str
    tokenizer_file_sha256: dict[str, str]
    conditions: tuple[Gate8TokenizerConditionCount, ...]

    def validate(self) -> None:
        if self.repo_id != GATE8_GEMMA_REPO_ID:
            raise ValueError("Gate8 tokenizer repository changed")
        validate_gate8_gemma_revision(self.revision)
        if not self.tokenizer_class:
            raise ValueError("Gate8 tokenizer class is missing")
        for version in (
            self.transformers_version,
            self.tokenizers_version,
            self.huggingface_hub_version,
        ):
            if not version:
                raise ValueError("Gate8 tokenizer software identity is incomplete")
        if not GATE8_FILE_SHA_PATTERN.fullmatch(self.chat_template_sha256):
            raise ValueError("Gate8 chat-template hash is malformed")
        if tuple(self.tokenizer_file_sha256) != GATE8_GEMMA_REQUIRED_TOKENIZER_FILES:
            raise ValueError("Gate8 tokenizer file-hash matrix changed")
        if any(
            not GATE8_FILE_SHA_PATTERN.fullmatch(value)
            for value in self.tokenizer_file_sha256.values()
        ):
            raise ValueError("Gate8 tokenizer file hash is malformed")
        validate_gate8_tokenizer_condition_matrix(self.conditions)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        maximum = gate8_maximum_token_condition(self.conditions)
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "tokenizer_class": self.tokenizer_class,
            "transformers_version": self.transformers_version,
            "tokenizers_version": self.tokenizers_version,
            "huggingface_hub_version": self.huggingface_hub_version,
            "chat_template_sha256": self.chat_template_sha256,
            "tokenizer_file_sha256": dict(self.tokenizer_file_sha256),
            "conditions": [row.to_dict() for row in self.conditions],
            "maximum_condition": maximum.to_dict(),
            "maximum_input_tokens": maximum.input_tokens,
            "frozen_input_token_limit": GATE8_GEMMA_MAX_INPUT_TOKENS,
            "tokenizer_bound": True,
            "model_bound": False,
            "model_weights_downloaded": False,
            "inference_performed": False,
            "scientific_test_worlds_generated": False,
        }


def gate8_tokenizer_binding_plan() -> dict[str, Any]:
    return {
        "version": GATE8_TOKENIZER_BINDING_VERSION,
        "encoder_head": GATE8_TOKENIZER_BINDING_ENCODER_HEAD,
        "scientific_status": GATE8_TOKENIZER_BINDING_STATUS,
        "repo_id": GATE8_GEMMA_REPO_ID,
        "revision": GATE8_GEMMA_REVISION,
        "required_tokenizer_files": list(GATE8_GEMMA_REQUIRED_TOKENIZER_FILES),
        "maximum_input_tokens": GATE8_GEMMA_MAX_INPUT_TOKENS,
        "maximum_output_tokens": GATE8_GEMMA_MAX_OUTPUT_TOKENS,
        "tokenizer_binding_admitted": True,
        "model_binding_admitted": False,
        "model_weights_downloaded": False,
        "training_admitted": False,
        "inference_admitted": False,
        "scientific_test_worlds_admitted": False,
    }
