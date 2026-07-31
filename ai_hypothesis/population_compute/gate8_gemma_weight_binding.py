"""Gate-8 exact Gemma model-file binding contracts.

This module validates downloaded files and inspects a Safetensors header using
only the Python standard library. It never imports a model framework, loads a
model or tokenizer, performs inference or training, or generates benchmark
worlds.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


GATE8_GEMMA_WEIGHT_BINDING_VERSION = "gate8-gemma-weight-binding-v0"
GATE8_GEMMA_WEIGHT_BINDING_PROTOCOL_HEAD = (
    "6bb89111a47713bea0a23bb1cae662ed5ec56b42"
)
GATE8_GEMMA_WEIGHT_BINDING_TOKENIZER_RESULT_HEAD = (
    "c7f5260189ef9ac1a1beb73596446316631090c7"
)
GATE8_GEMMA_WEIGHT_BINDING_STATUS = (
    "GATE8_EXACT_GEMMA_MODEL_FILE_BINDING_ADMITTED_INFERENCE_AND_TEST_CLOSED"
)
GATE8_GEMMA_REPO_ID = "google/gemma-3-1b-it"
GATE8_GEMMA_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
GATE8_GEMMA_REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
)
GATE8_GEMMA_CONFIG_SHA256 = (
    "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e"
)
GATE8_GEMMA_TOKENIZER_RESULT_SHA256 = (
    "c8d6adb733cadbbd251d91d35f9d224e255705dac49ba144655717f9f4ab7b8d"
)
GATE8_GEMMA_TOKENIZER_MANIFEST_SHA256 = (
    "21de192eb57c0759fbf2236fae2252e5319696b71689ada1471b74a9f1315a88"
)
GATE8_GEMMA_EXPECTED_ARCHITECTURES = ("Gemma3ForCausalLM",)
GATE8_GEMMA_EXPECTED_MODEL_TYPE = "gemma3_text"
GATE8_GEMMA_EXPECTED_TORCH_DTYPE = "bfloat16"
GATE8_GEMMA_EXPECTED_HIDDEN_SIZE = 1_152
GATE8_GEMMA_EXPECTED_INTERMEDIATE_SIZE = 6_912
GATE8_GEMMA_EXPECTED_LAYERS = 26
GATE8_GEMMA_EXPECTED_ATTENTION_HEADS = 4
GATE8_GEMMA_EXPECTED_KEY_VALUE_HEADS = 1
GATE8_GEMMA_EXPECTED_MAX_POSITION_EMBEDDINGS = 32_768
GATE8_GEMMA_EXPECTED_VOCAB_SIZE = 262_144
GATE8_GEMMA_PARAMETER_COUNT_MINIMUM = 900_000_000
GATE8_GEMMA_PARAMETER_COUNT_MAXIMUM = 1_100_000_000
GATE8_GEMMA_MAX_HEADER_BYTES = 128 * 1024 * 1024
GATE8_FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
GATE8_FILE_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")

GATE8_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate8_gemma_revision(revision: str) -> None:
    if revision != GATE8_GEMMA_REVISION:
        raise ValueError("Gate8 Gemma revision changed from the frozen commit")
    if not GATE8_FULL_SHA_PATTERN.fullmatch(revision):
        raise ValueError("Gate8 Gemma revision must be one full lowercase commit SHA")


def _visible_snapshot_files(snapshot_root: Path) -> tuple[Path, ...]:
    root = snapshot_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Gate8 Gemma model snapshot directory does not exist: {root}"
        )
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


def validate_gate8_gemma_model_snapshot(
    snapshot_root: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    root = snapshot_root.resolve()
    files = _visible_snapshot_files(root)
    relative = tuple(path.relative_to(root).as_posix() for path in files)
    if relative != GATE8_GEMMA_REQUIRED_MODEL_FILES:
        missing = sorted(set(GATE8_GEMMA_REQUIRED_MODEL_FILES) - set(relative))
        extra = sorted(set(relative) - set(GATE8_GEMMA_REQUIRED_MODEL_FILES))
        raise ValueError(
            "Gate8 Gemma model snapshot file set changed; "
            f"missing={missing}, extra={extra}"
        )
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in files
    }
    sizes = {
        path.relative_to(root).as_posix(): path.stat().st_size for path in files
    }
    if tuple(hashes) != GATE8_GEMMA_REQUIRED_MODEL_FILES:
        raise RuntimeError("Gate8 Gemma model hash order changed")
    if tuple(sizes) != GATE8_GEMMA_REQUIRED_MODEL_FILES:
        raise RuntimeError("Gate8 Gemma model size order changed")
    if any(not GATE8_FILE_SHA_PATTERN.fullmatch(value) for value in hashes.values()):
        raise RuntimeError("Gate8 Gemma model file hash is malformed")
    if hashes["config.json"] != GATE8_GEMMA_CONFIG_SHA256:
        raise ValueError(
            "Gate8 Gemma config hash disagrees with the qualified tokenizer binding"
        )
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("Gate8 Gemma model snapshot contains an empty file")
    return hashes, sizes


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Gate8 JSON root must be an object: {path.name}")
    return payload


def validate_gate8_gemma_config(config_path: Path) -> dict[str, Any]:
    config = _load_json_object(config_path)
    expected: dict[str, Any] = {
        "architectures": list(GATE8_GEMMA_EXPECTED_ARCHITECTURES),
        "model_type": GATE8_GEMMA_EXPECTED_MODEL_TYPE,
        "torch_dtype": GATE8_GEMMA_EXPECTED_TORCH_DTYPE,
        "hidden_size": GATE8_GEMMA_EXPECTED_HIDDEN_SIZE,
        "intermediate_size": GATE8_GEMMA_EXPECTED_INTERMEDIATE_SIZE,
        "num_hidden_layers": GATE8_GEMMA_EXPECTED_LAYERS,
        "num_attention_heads": GATE8_GEMMA_EXPECTED_ATTENTION_HEADS,
        "num_key_value_heads": GATE8_GEMMA_EXPECTED_KEY_VALUE_HEADS,
        "max_position_embeddings": GATE8_GEMMA_EXPECTED_MAX_POSITION_EMBEDDINGS,
        "vocab_size": GATE8_GEMMA_EXPECTED_VOCAB_SIZE,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Gate8 Gemma config semantics changed: {mismatches}")
    return expected


def validate_gate8_gemma_generation_config(
    generation_config_path: Path,
) -> dict[str, Any]:
    config = _load_json_object(generation_config_path)
    required_integer_fields = ("bos_token_id", "pad_token_id")
    for field in required_integer_fields:
        value = config.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"Gate8 Gemma generation config has invalid {field}"
            )
    eos = config.get("eos_token_id")
    if isinstance(eos, int) and not isinstance(eos, bool):
        eos_ids = (eos,)
    elif isinstance(eos, list) and eos and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in eos
    ):
        eos_ids = tuple(eos)
    else:
        raise ValueError("Gate8 Gemma generation config has invalid eos_token_id")
    return {
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": list(eos_ids),
        "pad_token_id": config["pad_token_id"],
        "source_do_sample": config.get("do_sample"),
        "source_top_k": config.get("top_k"),
        "source_top_p": config.get("top_p"),
        "source_cache_implementation": config.get("cache_implementation"),
        "scientific_decoding_override": "greedy_temperature_0",
        "scientific_max_new_tokens": 64,
    }


def _parse_json_bytes(payload: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate Safetensors header key: {key}")
            result[key] = value
        return result

    try:
        decoded = payload.decode("utf-8")
        parsed = json.loads(decoded, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Gate8 Safetensors header is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Gate8 Safetensors header root must be an object")
    return parsed


@dataclass(frozen=True, slots=True)
class Gate8SafetensorsTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    parameters: int
    storage_bytes: int

    def validate(self) -> None:
        if not self.name or self.name == "__metadata__":
            raise ValueError("Gate8 Safetensors tensor name is invalid")
        if self.dtype not in GATE8_SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"Gate8 Safetensors dtype is unsupported: {self.dtype}")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.shape
        ):
            raise ValueError("Gate8 Safetensors shape is invalid")
        if self.data_start < 0 or self.data_end < self.data_start:
            raise ValueError("Gate8 Safetensors data offsets are invalid")
        expected_parameters = math.prod(self.shape) if self.shape else 1
        if self.parameters != expected_parameters:
            raise ValueError("Gate8 Safetensors parameter count is inconsistent")
        expected_bytes = (
            expected_parameters * GATE8_SAFETENSORS_DTYPE_BYTES[self.dtype]
        )
        if self.storage_bytes != expected_bytes:
            raise ValueError("Gate8 Safetensors storage size is inconsistent")
        if self.data_end - self.data_start != self.storage_bytes:
            raise ValueError("Gate8 Safetensors data span is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True, slots=True)
class Gate8SafetensorsSummary:
    file_size: int
    header_bytes: int
    data_bytes: int
    tensor_count: int
    parameter_count: int
    storage_bytes: int
    dtype_parameter_counts: dict[str, int]
    dtype_storage_bytes: dict[str, int]
    metadata: dict[str, str]
    tensors: tuple[Gate8SafetensorsTensor, ...]

    def validate(self) -> None:
        if self.file_size != 8 + self.header_bytes + self.data_bytes:
            raise ValueError("Gate8 Safetensors file-size accounting is inconsistent")
        if not 2 <= self.header_bytes <= GATE8_GEMMA_MAX_HEADER_BYTES:
            raise ValueError("Gate8 Safetensors header size is outside the contract")
        if self.tensor_count != len(self.tensors) or self.tensor_count <= 0:
            raise ValueError("Gate8 Safetensors tensor count is inconsistent")
        for tensor in self.tensors:
            tensor.validate()
        if tuple(sorted(self.tensors, key=lambda row: row.data_start)) != self.tensors:
            raise ValueError("Gate8 Safetensors tensors are not data-offset ordered")
        cursor = 0
        for tensor in self.tensors:
            if tensor.data_start != cursor:
                raise ValueError("Gate8 Safetensors data layout has a gap or overlap")
            cursor = tensor.data_end
        if cursor != self.data_bytes:
            raise ValueError("Gate8 Safetensors data layout does not cover the file")
        if self.parameter_count != sum(row.parameters for row in self.tensors):
            raise ValueError("Gate8 Safetensors total parameter count is inconsistent")
        if self.storage_bytes != sum(row.storage_bytes for row in self.tensors):
            raise ValueError("Gate8 Safetensors total storage bytes are inconsistent")
        if self.storage_bytes != self.data_bytes:
            raise ValueError("Gate8 Safetensors declared tensor bytes do not cover data")
        observed_parameters: dict[str, int] = {}
        observed_bytes: dict[str, int] = {}
        for tensor in self.tensors:
            observed_parameters[tensor.dtype] = (
                observed_parameters.get(tensor.dtype, 0) + tensor.parameters
            )
            observed_bytes[tensor.dtype] = (
                observed_bytes.get(tensor.dtype, 0) + tensor.storage_bytes
            )
        if self.dtype_parameter_counts != dict(sorted(observed_parameters.items())):
            raise ValueError("Gate8 Safetensors dtype parameter ledger is inconsistent")
        if self.dtype_storage_bytes != dict(sorted(observed_bytes.items())):
            raise ValueError("Gate8 Safetensors dtype byte ledger is inconsistent")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.metadata.items()):
            raise ValueError("Gate8 Safetensors metadata must contain string pairs")

    def validate_gemma_weight_contract(self) -> None:
        self.validate()
        if not (
            GATE8_GEMMA_PARAMETER_COUNT_MINIMUM
            <= self.parameter_count
            <= GATE8_GEMMA_PARAMETER_COUNT_MAXIMUM
        ):
            raise ValueError("Gate8 Gemma parameter count is outside the frozen 1B class")
        if set(self.dtype_parameter_counts) != {"BF16"}:
            raise ValueError("Gate8 Gemma weights are not exclusively BF16")

    def to_dict(self, *, include_tensor_ledger: bool = True) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "file_size": self.file_size,
            "header_bytes": self.header_bytes,
            "data_bytes": self.data_bytes,
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "storage_bytes": self.storage_bytes,
            "dtype_parameter_counts": dict(self.dtype_parameter_counts),
            "dtype_storage_bytes": dict(self.dtype_storage_bytes),
            "metadata": dict(self.metadata),
        }
        if include_tensor_ledger:
            payload["tensors"] = [tensor.to_dict() for tensor in self.tensors]
        return payload


def inspect_gate8_safetensors(path: Path) -> Gate8SafetensorsSummary:
    file_size = path.stat().st_size
    if file_size < 10:
        raise ValueError("Gate8 Safetensors file is too small")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Gate8 Safetensors header length is truncated")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_bytes <= GATE8_GEMMA_MAX_HEADER_BYTES:
            raise ValueError("Gate8 Safetensors header length is outside the contract")
        if 8 + header_bytes > file_size:
            raise ValueError("Gate8 Safetensors header extends beyond the file")
        header = _parse_json_bytes(handle.read(header_bytes))
    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise ValueError("Gate8 Safetensors metadata is malformed")
    tensors: list[Gate8SafetensorsTensor] = []
    for name, descriptor in header.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError("Gate8 Safetensors tensor descriptor is malformed")
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(
                f"Gate8 Safetensors tensor descriptor keys changed: {name}"
            )
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if not isinstance(dtype, str):
            raise ValueError("Gate8 Safetensors tensor dtype is malformed")
        if not isinstance(shape, list):
            raise ValueError("Gate8 Safetensors tensor shape is malformed")
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in offsets
            )
        ):
            raise ValueError("Gate8 Safetensors tensor data offsets are malformed")
        shape_tuple = tuple(shape)
        parameters = math.prod(shape_tuple) if shape_tuple else 1
        bytes_per_element = GATE8_SAFETENSORS_DTYPE_BYTES.get(dtype)
        if bytes_per_element is None:
            raise ValueError(f"Gate8 Safetensors dtype is unsupported: {dtype}")
        tensor = Gate8SafetensorsTensor(
            name=name,
            dtype=dtype,
            shape=shape_tuple,
            data_start=offsets[0],
            data_end=offsets[1],
            parameters=parameters,
            storage_bytes=parameters * bytes_per_element,
        )
        tensor.validate()
        tensors.append(tensor)
    ordered = tuple(sorted(tensors, key=lambda row: (row.data_start, row.name)))
    dtype_parameters: dict[str, int] = {}
    dtype_bytes: dict[str, int] = {}
    for tensor in ordered:
        dtype_parameters[tensor.dtype] = (
            dtype_parameters.get(tensor.dtype, 0) + tensor.parameters
        )
        dtype_bytes[tensor.dtype] = (
            dtype_bytes.get(tensor.dtype, 0) + tensor.storage_bytes
        )
    summary = Gate8SafetensorsSummary(
        file_size=file_size,
        header_bytes=header_bytes,
        data_bytes=file_size - 8 - header_bytes,
        tensor_count=len(ordered),
        parameter_count=sum(row.parameters for row in ordered),
        storage_bytes=sum(row.storage_bytes for row in ordered),
        dtype_parameter_counts=dict(sorted(dtype_parameters.items())),
        dtype_storage_bytes=dict(sorted(dtype_bytes.items())),
        metadata=dict(sorted(raw_metadata.items())),
        tensors=ordered,
    )
    summary.validate()
    return summary


@dataclass(frozen=True, slots=True)
class Gate8GemmaWeightBindingSummary:
    repo_id: str
    revision: str
    huggingface_hub_version: str
    file_sha256: dict[str, str]
    file_sizes: dict[str, int]
    config_semantics: dict[str, Any]
    generation_config_semantics: dict[str, Any]
    safetensors: Gate8SafetensorsSummary

    def validate(self) -> None:
        if self.repo_id != GATE8_GEMMA_REPO_ID:
            raise ValueError("Gate8 Gemma weight repository changed")
        validate_gate8_gemma_revision(self.revision)
        if not self.huggingface_hub_version:
            raise ValueError("Gate8 Hugging Face Hub package identity is missing")
        if tuple(self.file_sha256) != GATE8_GEMMA_REQUIRED_MODEL_FILES:
            raise ValueError("Gate8 Gemma model file-hash ledger changed")
        if tuple(self.file_sizes) != GATE8_GEMMA_REQUIRED_MODEL_FILES:
            raise ValueError("Gate8 Gemma model file-size ledger changed")
        if self.file_sha256["config.json"] != GATE8_GEMMA_CONFIG_SHA256:
            raise ValueError("Gate8 Gemma config hash changed")
        if any(
            not GATE8_FILE_SHA_PATTERN.fullmatch(value)
            for value in self.file_sha256.values()
        ):
            raise ValueError("Gate8 Gemma model file hash is malformed")
        if any(value <= 0 for value in self.file_sizes.values()):
            raise ValueError("Gate8 Gemma model file size is invalid")
        if self.file_sizes["model.safetensors"] != self.safetensors.file_size:
            raise ValueError("Gate8 Gemma Safetensors file size disagrees with ledger")
        self.safetensors.validate_gemma_weight_contract()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "huggingface_hub_version": self.huggingface_hub_version,
            "file_sha256": dict(self.file_sha256),
            "file_sizes": dict(self.file_sizes),
            "config_semantics": dict(self.config_semantics),
            "generation_config_semantics": dict(
                self.generation_config_semantics
            ),
            "safetensors": self.safetensors.to_dict(),
            "tokenizer_result_sha256": GATE8_GEMMA_TOKENIZER_RESULT_SHA256,
            "tokenizer_manifest_sha256": GATE8_GEMMA_TOKENIZER_MANIFEST_SHA256,
            "model_file_binding_complete": True,
            "model_files_downloaded": True,
            "model_instantiated": False,
            "tokenizer_loaded": False,
            "training_performed": False,
            "inference_performed": False,
            "scientific_test_worlds_generated": False,
        }


def validate_gate8_model_binding_file_matrix(
    rows: Iterable[str],
) -> tuple[str, ...]:
    result = tuple(rows)
    if result != GATE8_GEMMA_REQUIRED_MODEL_FILES:
        raise ValueError("Gate8 Gemma model binding file matrix changed")
    return result


def gate8_gemma_weight_binding_plan() -> dict[str, Any]:
    validate_gate8_gemma_revision(GATE8_GEMMA_REVISION)
    validate_gate8_model_binding_file_matrix(GATE8_GEMMA_REQUIRED_MODEL_FILES)
    return {
        "version": GATE8_GEMMA_WEIGHT_BINDING_VERSION,
        "scientific_protocol_head": GATE8_GEMMA_WEIGHT_BINDING_PROTOCOL_HEAD,
        "tokenizer_result_head": GATE8_GEMMA_WEIGHT_BINDING_TOKENIZER_RESULT_HEAD,
        "scientific_status": GATE8_GEMMA_WEIGHT_BINDING_STATUS,
        "repo_id": GATE8_GEMMA_REPO_ID,
        "revision": GATE8_GEMMA_REVISION,
        "required_model_files": list(GATE8_GEMMA_REQUIRED_MODEL_FILES),
        "qualified_config_sha256": GATE8_GEMMA_CONFIG_SHA256,
        "tokenizer_result_sha256": GATE8_GEMMA_TOKENIZER_RESULT_SHA256,
        "tokenizer_manifest_sha256": GATE8_GEMMA_TOKENIZER_MANIFEST_SHA256,
        "parameter_count_minimum": GATE8_GEMMA_PARAMETER_COUNT_MINIMUM,
        "parameter_count_maximum": GATE8_GEMMA_PARAMETER_COUNT_MAXIMUM,
        "required_weight_dtype": "BF16",
        "model_file_binding_admitted": True,
        "model_file_download_admitted": True,
        "model_instantiation_admitted": False,
        "tokenizer_loading_admitted": False,
        "training_admitted": False,
        "inference_admitted": False,
        "scientific_test_worlds_admitted": False,
    }
