"""Fresh-process persistence probe for Post-Training Learning L0.

The harness proves that a separately started Python process can load one exact
base checkpoint plus one exact tensor-only adapter artifact and produce neural
outputs after all parent-process model state has been excluded.

It intentionally accepts no labels and performs no calibration or final-world
evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import secrets
import stat
import subprocess
import sys
from typing import Any, Sequence

import torch
from torch import Tensor

from . import l0_protocol
from . import post_training_learning_l0_adapter as adapter
from . import post_training_learning_l0_checkpoint as checkpoint
from . import post_training_learning_l0_execution as execution
from . import post_training_learning_l0_world as world

VERSION = "population-language-post-training-learning-l0-fresh-process-v0"
BRANCH = "agent/population-language-post-training-learning-l0-fresh-process-v0"
STATUS = "FRESH_PROCESS_HARNESS_ONLY_NO_CALIBRATION_OR_FINAL_RESULT"
SOURCE_CHECKPOINT_CONTRACT_HEAD = "0b43d2cfedcaaf92a9905750ba3cac809645bebd"

REQUEST_MAX_BYTES = 32 * 1024
RESULT_MAX_BYTES = 64 * 1024
MAX_PROBE_BATCH = 8
MAX_PROBE_SEQUENCE = 8
MAX_PROBE_WORKERS = 256
FRESH_PROCESS_COMPLETE = "POST_TRAINING_LEARNING_L0_FRESH_PROCESS_PROBE_COMPLETE"

REQUEST_KEYS = (
    "version",
    "request_nonce",
    "parent_pid",
    "checkpoint_path",
    "checkpoint_seed",
    "checkpoint_file_sha256",
    "checkpoint_canonical_sha256",
    "adapter_path",
    "adapter_file_sha256",
    "adapter_rank",
    "worker_count",
    "input_ids",
    "result_path",
)
RESULT_KEYS = (
    "version",
    "status",
    "source_checkpoint_contract_head",
    "request_nonce",
    "request_sha256",
    "parent_pid",
    "child_pid",
    "child_start_nonce",
    "checkpoint_seed",
    "checkpoint_file_sha256",
    "checkpoint_canonical_sha256_before",
    "checkpoint_canonical_sha256_after",
    "adapter_file_sha256",
    "adapter_rank",
    "worker_count",
    "input_sha256",
    "final_argmax_token_ids",
    "logits_sha256",
)


@dataclass(frozen=True)
class FreshProcessRequestRecord:
    path: str
    sha256: str
    request_nonce: str
    result_path: str


@dataclass(frozen=True)
class FreshProcessProbeResult:
    request_sha256: str
    parent_pid: int
    child_pid: int
    child_start_nonce: str
    checkpoint_seed: int
    checkpoint_file_sha256: str
    checkpoint_canonical_sha256: str
    adapter_file_sha256: str
    adapter_rank: int
    worker_count: int
    input_sha256: str
    final_argmax_token_ids: tuple[int, ...]
    logits_sha256: str


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains a duplicate key")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_create_once(path: pathlib.Path, value: object, maximum: int) -> str:
    payload = _canonical_json_bytes(value)
    if len(payload) > maximum:
        raise ValueError("JSON artifact exceeds its byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _read_json_regular(
    path: pathlib.Path,
    maximum: int,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    if not isinstance(path, pathlib.Path):
        raise TypeError("JSON artifact path must be pathlib.Path")
    if path.is_symlink():
        raise ValueError("JSON artifact path must not be a symbolic link")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError("JSON artifact path is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("JSON artifact path is not a regular file")
    if not 0 < metadata.st_size <= maximum:
        raise ValueError("JSON artifact size is outside the contract")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if opened.st_size != metadata.st_size:
                raise ValueError("JSON artifact changed before reading")
            payload = handle.read(maximum + 1)
    except OSError as error:
        raise ValueError("JSON artifact could not be read") from error
    if len(payload) != metadata.st_size or len(payload) > maximum:
        raise ValueError("JSON artifact changed while reading or exceeded its bound")
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ValueError("expected JSON SHA-256 is invalid")
        if observed != expected_sha256:
            raise ValueError("JSON artifact SHA-256 mismatch")
    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=_plain_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JSON artifact is not valid canonical UTF-8 JSON") from error
    if type(decoded) is not dict:
        raise TypeError("JSON artifact must contain a plain object")
    return decoded, observed


def _validate_absolute_path(value: object, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"{label} must be a bounded path string")
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


def _validate_probe_inputs(value: object) -> list[list[int]]:
    if type(value) is not list or not 1 <= len(value) <= MAX_PROBE_BATCH:
        raise ValueError("probe input batch lies outside the contract")
    ids = l0_protocol.TOKEN_TO_ID
    operator_ids = {ids[token] for token in world.OPERATOR_TOKENS}
    value_ids = {ids[token] for token in world.VALUE_TOKENS}
    rows: list[list[int]] = []
    width: int | None = None
    for raw_row in value:
        if type(raw_row) is not list:
            raise TypeError("probe input row must be a list")
        if not 5 <= len(raw_row) <= MAX_PROBE_SEQUENCE:
            raise ValueError("probe input sequence lies outside the adaptation contract")
        row: list[int] = []
        for token_id in raw_row:
            if type(token_id) is not int or not 0 <= token_id < len(l0_protocol.VOCABULARY):
                raise ValueError("probe input token lies outside the base vocabulary")
            row.append(token_id)
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("probe input rows must have equal length")
        if (
            row[0] != ids["<bos>"]
            or row[1] != ids["<query>"]
            or row[-1] != ids["<answer>"]
            or row[-2] not in value_ids
            or any(token_id not in operator_ids for token_id in row[2:-2])
        ):
            raise ValueError("probe input does not activate the adaptation path")
        rows.append(row)
    return rows


def validate_request(value: object) -> dict[str, Any]:
    if type(value) is not dict or tuple(value) != REQUEST_KEYS:
        raise ValueError("fresh-process request keys or order drifted")
    if value["version"] != VERSION:
        raise ValueError("fresh-process request version drifted")
    if not _is_sha256(value["request_nonce"]):
        raise ValueError("fresh-process request nonce is invalid")
    if type(value["parent_pid"]) is not int or value["parent_pid"] <= 0:
        raise ValueError("fresh-process parent PID is invalid")
    checkpoint_path = _validate_absolute_path(value["checkpoint_path"], "checkpoint path")
    adapter_path = _validate_absolute_path(value["adapter_path"], "adapter path")
    result_path = _validate_absolute_path(value["result_path"], "result path")
    if len({checkpoint_path, adapter_path, result_path}) != 3:
        raise ValueError("checkpoint, adapter, and result paths must be distinct")
    if value["checkpoint_seed"] not in world.MODEL_INITIALIZATION_SEEDS:
        raise ValueError("fresh-process checkpoint seed is outside the preregistered set")
    if not _is_sha256(value["checkpoint_file_sha256"]):
        raise ValueError("fresh-process checkpoint file SHA-256 is invalid")
    if not _is_sha256(value["checkpoint_canonical_sha256"]):
        raise ValueError("fresh-process canonical checkpoint SHA-256 is invalid")
    if not _is_sha256(value["adapter_file_sha256"]):
        raise ValueError("fresh-process adapter SHA-256 is invalid")
    if value["adapter_rank"] not in adapter.SUPPORTED_RANKS:
        raise ValueError("fresh-process adapter rank is unsupported")
    if (
        type(value["worker_count"]) is not int
        or not 1 <= value["worker_count"] <= MAX_PROBE_WORKERS
    ):
        raise ValueError("fresh-process worker count lies outside the contract")
    rows = _validate_probe_inputs(value["input_ids"])
    validated = dict(value)
    validated["input_ids"] = rows
    return validated


def _input_sha256(input_ids: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(map(list, input_ids)))).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_fresh_process_request_create_once(
    path: pathlib.Path,
    *,
    checkpoint_path: pathlib.Path,
    checkpoint_seed: int,
    checkpoint_file_sha256: str,
    checkpoint_canonical_sha256: str,
    adapter_path: pathlib.Path,
    adapter_file_sha256: str,
    adapter_rank: int,
    worker_count: int,
    input_ids: Sequence[Sequence[int]],
    result_path: pathlib.Path,
) -> FreshProcessRequestRecord:
    if not all(
        isinstance(item, pathlib.Path)
        for item in (path, checkpoint_path, adapter_path, result_path)
    ):
        raise TypeError("fresh-process paths must be pathlib.Path")
    request_path = path.absolute()
    checkpoint_absolute = checkpoint_path.absolute()
    adapter_absolute = adapter_path.absolute()
    result_absolute = result_path.absolute()
    nonce = secrets.token_hex(32)
    request = {
        "version": VERSION,
        "request_nonce": nonce,
        "parent_pid": os.getpid(),
        "checkpoint_path": str(checkpoint_absolute),
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "checkpoint_canonical_sha256": checkpoint_canonical_sha256,
        "adapter_path": str(adapter_absolute),
        "adapter_file_sha256": adapter_file_sha256,
        "adapter_rank": adapter_rank,
        "worker_count": worker_count,
        "input_ids": [list(row) for row in input_ids],
        "result_path": str(result_absolute),
    }
    validate_request(request)
    if result_absolute.exists() or result_absolute.is_symlink():
        raise FileExistsError("fresh-process result path already exists")
    request_sha256 = _write_json_create_once(request_path, request, REQUEST_MAX_BYTES)
    return FreshProcessRequestRecord(
        path=str(request_path),
        sha256=request_sha256,
        request_nonce=nonce,
        result_path=str(result_absolute),
    )


def _execute_child(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    parent_pid = request["parent_pid"]
    child_pid = os.getpid()
    if child_pid == parent_pid:
        raise RuntimeError("fresh-process child PID matches the parent PID")

    loaded = checkpoint.load_reference_checkpoint(
        pathlib.Path(request["checkpoint_path"]),
        expected_seed=request["checkpoint_seed"],
        expected_file_sha256=request["checkpoint_file_sha256"],
        expected_canonical_sha256=request["checkpoint_canonical_sha256"],
    )
    base = loaded.model
    canonical_before = execution.canonical_base_state_sha256(base)
    adaptation_state = execution.load_adaptation_artifact(
        pathlib.Path(request["adapter_path"]),
        expected_sha256=request["adapter_file_sha256"],
    )
    observed_rank, _ = execution.validate_adaptation_state(adaptation_state)
    if observed_rank != request["adapter_rank"]:
        raise ValueError("fresh-process adapter rank does not match the request")

    adapted = adapter.BoundedPopulationAdapter(
        base,
        model_seed=request["checkpoint_seed"],
        config=adapter.AdapterConfig(rank=request["adapter_rank"]),
    )
    adapted.load_adaptation_state_dict(adaptation_state)
    input_tensor = torch.tensor(request["input_ids"], dtype=torch.long)
    with torch.no_grad():
        logits = adapted(input_tensor, worker_count=request["worker_count"])
    canonical_after = execution.canonical_base_state_sha256(base)
    if canonical_after != canonical_before:
        raise RuntimeError("fresh-process probe mutated the immutable base checkpoint")

    final_argmax = torch.argmax(logits[:, -1, :], dim=-1).to(torch.int64).tolist()
    return {
        "version": VERSION,
        "status": FRESH_PROCESS_COMPLETE,
        "source_checkpoint_contract_head": SOURCE_CHECKPOINT_CONTRACT_HEAD,
        "request_nonce": request["request_nonce"],
        "request_sha256": request_sha256,
        "parent_pid": parent_pid,
        "child_pid": child_pid,
        "child_start_nonce": secrets.token_hex(32),
        "checkpoint_seed": request["checkpoint_seed"],
        "checkpoint_file_sha256": loaded.file_sha256,
        "checkpoint_canonical_sha256_before": canonical_before,
        "checkpoint_canonical_sha256_after": canonical_after,
        "adapter_file_sha256": request["adapter_file_sha256"],
        "adapter_rank": observed_rank,
        "worker_count": request["worker_count"],
        "input_sha256": _input_sha256(request["input_ids"]),
        "final_argmax_token_ids": final_argmax,
        "logits_sha256": _tensor_sha256(logits),
    }


def _validate_result(
    value: object,
    *,
    request: dict[str, Any],
    request_sha256: str,
) -> FreshProcessProbeResult:
    if type(value) is not dict or tuple(value) != RESULT_KEYS:
        raise ValueError("fresh-process result keys or order drifted")
    if value["version"] != VERSION or value["status"] != FRESH_PROCESS_COMPLETE:
        raise ValueError("fresh-process result status or version drifted")
    if value["source_checkpoint_contract_head"] != SOURCE_CHECKPOINT_CONTRACT_HEAD:
        raise ValueError("fresh-process source checkpoint head drifted")
    if value["request_nonce"] != request["request_nonce"]:
        raise ValueError("fresh-process result nonce drifted")
    if value["request_sha256"] != request_sha256:
        raise ValueError("fresh-process result request SHA-256 drifted")
    if value["parent_pid"] != request["parent_pid"]:
        raise ValueError("fresh-process result parent PID drifted")
    if (
        type(value["child_pid"]) is not int
        or value["child_pid"] <= 0
        or value["child_pid"] == value["parent_pid"]
    ):
        raise ValueError("fresh-process child PID is invalid")
    if not _is_sha256(value["child_start_nonce"]):
        raise ValueError("fresh-process child start nonce is invalid")
    if value["checkpoint_seed"] != request["checkpoint_seed"]:
        raise ValueError("fresh-process result checkpoint seed drifted")
    if value["checkpoint_file_sha256"] != request["checkpoint_file_sha256"]:
        raise ValueError("fresh-process result checkpoint file SHA-256 drifted")
    if (
        value["checkpoint_canonical_sha256_before"]
        != request["checkpoint_canonical_sha256"]
        or value["checkpoint_canonical_sha256_after"]
        != request["checkpoint_canonical_sha256"]
    ):
        raise ValueError("fresh-process result canonical checkpoint SHA-256 drifted")
    if value["adapter_file_sha256"] != request["adapter_file_sha256"]:
        raise ValueError("fresh-process result adapter SHA-256 drifted")
    if value["adapter_rank"] != request["adapter_rank"]:
        raise ValueError("fresh-process result adapter rank drifted")
    if value["worker_count"] != request["worker_count"]:
        raise ValueError("fresh-process result worker count drifted")
    expected_input_sha256 = _input_sha256(request["input_ids"])
    if value["input_sha256"] != expected_input_sha256:
        raise ValueError("fresh-process result input SHA-256 drifted")
    final_ids = value["final_argmax_token_ids"]
    if (
        type(final_ids) is not list
        or len(final_ids) != len(request["input_ids"])
        or any(type(token_id) is not int or not 0 <= token_id < len(l0_protocol.VOCABULARY) for token_id in final_ids)
    ):
        raise ValueError("fresh-process final argmax token IDs are invalid")
    if not _is_sha256(value["logits_sha256"]):
        raise ValueError("fresh-process logits SHA-256 is invalid")
    return FreshProcessProbeResult(
        request_sha256=request_sha256,
        parent_pid=value["parent_pid"],
        child_pid=value["child_pid"],
        child_start_nonce=value["child_start_nonce"],
        checkpoint_seed=value["checkpoint_seed"],
        checkpoint_file_sha256=value["checkpoint_file_sha256"],
        checkpoint_canonical_sha256=value["checkpoint_canonical_sha256_after"],
        adapter_file_sha256=value["adapter_file_sha256"],
        adapter_rank=value["adapter_rank"],
        worker_count=value["worker_count"],
        input_sha256=value["input_sha256"],
        final_argmax_token_ids=tuple(final_ids),
        logits_sha256=value["logits_sha256"],
    )


def run_fresh_process_probe(
    request_path: pathlib.Path,
    *,
    expected_request_sha256: str,
    timeout_seconds: int = 180,
) -> FreshProcessProbeResult:
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise ValueError("fresh-process timeout lies outside the contract")
    request, observed_request_sha256 = _read_json_regular(
        request_path,
        REQUEST_MAX_BYTES,
        expected_sha256=expected_request_sha256,
    )
    request = validate_request(request)
    if request["parent_pid"] != os.getpid():
        raise ValueError("fresh-process request was not created by the current parent")
    result_path = pathlib.Path(request["result_path"])
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("fresh-process result path already exists")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, "-m", __name__, "--child", str(request_path.absolute())],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=environment,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(
            f"fresh-process child failed with exit code {completed.returncode}: {diagnostic}"
        )
    result, _ = _read_json_regular(result_path, RESULT_MAX_BYTES)
    return _validate_result(
        result,
        request=request,
        request_sha256=observed_request_sha256,
    )


def child_main(request_path: pathlib.Path) -> int:
    request, request_sha256 = _read_json_regular(request_path, REQUEST_MAX_BYTES)
    request = validate_request(request)
    result_path = pathlib.Path(request["result_path"])
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("fresh-process child result path already exists")
    result = _execute_child(request, request_sha256)
    if tuple(result) != RESULT_KEYS:
        raise RuntimeError("fresh-process child result keys drifted")
    _write_json_create_once(result_path, result, RESULT_MAX_BYTES)
    return 0


def validate_fresh_process_contract() -> dict[str, object]:
    checks = {
        "request_keys_are_exact": len(REQUEST_KEYS) == 13,
        "result_keys_are_exact": len(RESULT_KEYS) == 19,
        "request_is_bounded": REQUEST_MAX_BYTES == 32 * 1024,
        "result_is_bounded": RESULT_MAX_BYTES == 64 * 1024,
        "probe_batch_is_bounded": MAX_PROBE_BATCH == 8,
        "probe_sequence_is_adaptation_bounded": MAX_PROBE_SEQUENCE == 8,
        "probe_workers_are_bounded": MAX_PROBE_WORKERS == 256,
        "source_checkpoint_head_is_pinned": SOURCE_CHECKPOINT_CONTRACT_HEAD
        == "0b43d2cfedcaaf92a9905750ba3cac809645bebd",
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "source_checkpoint_contract_head": SOURCE_CHECKPOINT_CONTRACT_HEAD,
        "request_keys": list(REQUEST_KEYS),
        "result_keys": list(RESULT_KEYS),
        "request_max_bytes": REQUEST_MAX_BYTES,
        "result_max_bytes": RESULT_MAX_BYTES,
        "checks": checks,
        "valid": all(checks.values()),
    }


def _main(arguments: Sequence[str]) -> int:
    if len(arguments) != 2 or arguments[0] != "--child":
        raise SystemExit("usage: python -m ...post_training_learning_l0_fresh_process --child REQUEST.json")
    return child_main(pathlib.Path(arguments[1]))


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
