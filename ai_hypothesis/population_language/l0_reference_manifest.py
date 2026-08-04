"""Strict, read-only verifier for completed Population Language L0 outputs.

The verifier accepts one explicitly named output directory after training has
finished. It does not discover outputs, start training, or access any
Post-Training Learning calibration/final world.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import stat
from typing import Any

from . import l0_protocol as l0_protocol
from . import l0_reference_training as training
from . import post_training_learning_l0_checkpoint as checkpoint

VERSION = "population-language-l0-reference-manifest-verifier-v0"
BRANCH = "agent/population-language-l0-reference-manifest-verifier-v0"
STATUS = "REFERENCE_MANIFEST_VERIFIER_ONLY_NO_ACTIVE_OUTPUT_ACCESS"
SOURCE_FRESH_PROCESS_HEAD = "f0cf83d1be0426fda976f08a379ab040be53ba89"

JSON_MAX_BYTES = 16 * 1024 * 1024
CHECKPOINT_MAX_BYTES = checkpoint.REFERENCE_CHECKPOINT_MAX_BYTES
EXPECTED_FILE_COUNT = 17


@dataclass(frozen=True)
class PopulationCheckpointManifest:
    seed: int
    path: str
    file_bytes: int
    file_sha256: str
    canonical_state_sha256: str


@dataclass(frozen=True)
class ReferenceOutputManifest:
    root: str
    summary_sha256: str
    execution_head: str
    diagnosis: str
    population_scaling_conclusion: str
    post_training_base_eligible: bool
    population_checkpoints: tuple[PopulationCheckpointManifest, ...]


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
            raise ValueError("reference JSON contains a duplicate key")
        result[key] = value
    return result


def _read_regular_bytes(path: pathlib.Path, maximum: int) -> tuple[bytes, str]:
    if path.is_symlink():
        raise ValueError(f"reference output path is a symbolic link: {path}")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError(f"reference output path is unavailable: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"reference output path is not a regular file: {path}")
    if not 0 < metadata.st_size <= maximum:
        raise ValueError(f"reference output file size is outside the contract: {path}")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if opened.st_size != metadata.st_size:
                raise ValueError(f"reference output changed before reading: {path}")
            payload = handle.read(maximum + 1)
    except OSError as error:
        raise ValueError(f"reference output could not be read: {path}") from error
    if len(payload) != metadata.st_size or len(payload) > maximum:
        raise ValueError(f"reference output changed while reading: {path}")
    return payload, hashlib.sha256(payload).hexdigest()


def _read_json(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    payload, digest = _read_regular_bytes(path, JSON_MAX_BYTES)
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_plain_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"reference output is not valid UTF-8 JSON: {path}") from error
    if type(value) is not dict:
        raise TypeError(f"reference JSON root must be a plain object: {path}")
    return value, digest


def expected_relative_files() -> tuple[str, ...]:
    files = ["run-start.json", "summary.json"]
    for seed in l0_protocol.INITIALIZATION_SEEDS:
        files.append(f"seed-{seed}.json")
        for model in ("transformer", "population"):
            files.append(f"progress/{model}-seed-{seed}.json")
            files.append(f"checkpoints/{model}-seed-{seed}.pt")
    return tuple(sorted(files))


def _inventory(root: pathlib.Path) -> tuple[str, ...]:
    observed: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"reference output contains a symbolic link: {path}")
        if path.is_file():
            observed.append(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError(f"reference output contains an unsupported entry: {path}")
    return tuple(sorted(observed))


def _validate_execution_head(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError("reference execution head is malformed")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("reference execution head is malformed")
    return value


def _validate_contract(value: object) -> dict[str, int]:
    if type(value) is not dict:
        raise TypeError("reference training contract must be a plain object")
    microbatch = value.get("microbatch")
    evaluation_microbatch = value.get("evaluation_microbatch")
    if type(microbatch) is not int or type(evaluation_microbatch) is not int:
        raise ValueError("reference training contract microbatches are invalid")
    expected = training.validate_contract(microbatch, evaluation_microbatch).__dict__
    if value != expected:
        raise ValueError("reference training contract drifted")
    return value


def _expected_fingerprints() -> dict[str, str]:
    return {
        split: l0_protocol.dataset_fingerprint(split, 256)
        for split in ("train", "validation", "test")
    }


def _validate_start(
    value: dict[str, Any],
    *,
    execution_head: str,
    contract: dict[str, int],
    fingerprints: dict[str, str],
) -> None:
    if (
        value.get("status") != training.STATUS
        or value.get("phase") != "TRAINING"
        or value.get("version") != training.VERSION
        or value.get("branch") != training.BRANCH
        or value.get("base_head") != training.BASE_HEAD
        or value.get("execution_head") != execution_head
        or value.get("training_schedule_sha256") != training.training_schedule_sha256()
        or value.get("contract") != contract
        or value.get("seeds") != list(l0_protocol.INITIALIZATION_SEEDS)
        or value.get("dataset_fingerprints_first_256") != fingerprints
    ):
        raise ValueError("reference run-start contract drifted")
    for field in ("dataset_cache_build_seconds", "dataset_cache_resident_bytes"):
        number = value.get(field)
        if not isinstance(number, (int, float)) or float(number) <= 0:
            raise ValueError(f"reference run-start {field} is invalid")


def _validate_summary_header(
    value: dict[str, Any],
    *,
    execution_head: str,
    contract: dict[str, int],
    fingerprints: dict[str, str],
) -> None:
    if (
        value.get("status") != training.STATUS
        or value.get("version") != training.VERSION
        or value.get("branch") != training.BRANCH
        or value.get("base_head") != training.BASE_HEAD
        or value.get("execution_head") != execution_head
        or value.get("training_schedule_sha256") != training.training_schedule_sha256()
        or value.get("contract") != contract
        or value.get("dataset_fingerprints_first_256") != fingerprints
    ):
        raise ValueError("reference summary header drifted")
    for field in ("dataset_cache_build_seconds", "dataset_cache_resident_bytes"):
        number = value.get(field)
        if not isinstance(number, (int, float)) or float(number) <= 0:
            raise ValueError(f"reference summary {field} is invalid")
    cuda = value.get("cuda")
    if type(cuda) is not dict:
        raise ValueError("reference summary CUDA evidence is missing")
    required_cuda = (
        "device_name",
        "device_capability",
        "total_memory_bytes",
        "torch_version",
        "cuda_version",
        "bf16_supported",
    )
    if (
        any(field not in cuda for field in required_cuda)
        or cuda.get("bf16_supported") is not True
    ):
        raise ValueError("reference summary CUDA evidence drifted")


def _expected_boundaries() -> dict[str, bool]:
    return {
        "full_next_token_objective_only": True,
        "answer_span_training_weighted": False,
        "fixed_final_checkpoint_used": True,
        "test_used_for_checkpoint_selection": False,
        "population_trained_only_at_32_workers": True,
        "same_population_checkpoint_used_at_all_worker_counts": True,
        "worker_specific_learned_parameters_used": False,
        "gate9_evidence_modified": False,
    }


def _canonicalize_seed_rows(seed_rows: list[object]) -> list[dict[str, Any]]:
    """Restore the locked numeric worker order after sort_keys JSON serialization."""
    expected_worker_keys = [str(worker) for worker in l0_protocol.EVAL_WORKERS]
    normalized: list[dict[str, Any]] = []
    for seed_row in seed_rows:
        if type(seed_row) is not dict:
            raise TypeError("reference seed row must be a plain object")
        copied = dict(seed_row)
        population = copied.get("population")
        if type(population) is not dict:
            raise TypeError("reference population row must be a plain object")
        population_copy = dict(population)
        for field in ("validation_by_workers", "test_by_workers"):
            worker_rows = population_copy.get(field)
            if type(worker_rows) is not dict:
                raise TypeError(f"reference {field} must be a plain object")
            if set(worker_rows) != set(expected_worker_keys):
                raise ValueError(f"reference {field} worker keys drifted")
            population_copy[field] = {
                key: worker_rows[key] for key in expected_worker_keys
            }
        copied["population"] = population_copy
        normalized.append(copied)
    return normalized


def _verify_progress(
    path: pathlib.Path,
    *,
    model: str,
    seed: int,
    trained: dict[str, Any],
) -> None:
    progress, _ = _read_json(path)
    if tuple(progress) != (
        "canonical_checkpoint_sha256",
        "checkpoint_file_sha256",
        "curves",
        "last_completed_optimizer_step",
        "model",
        "seed",
        "status",
        "version",
    ):
        raise ValueError("reference COMPLETE progress keys drifted")
    if (
        progress["status"] != "COMPLETE"
        or progress["version"] != training.VERSION
        or progress["model"] != model
        or progress["seed"] != seed
        or progress["last_completed_optimizer_step"] != training.OPTIMIZER_STEPS
        or progress["canonical_checkpoint_sha256"]
        != trained.get("canonical_checkpoint_sha256")
        or progress["checkpoint_file_sha256"]
        != trained.get("checkpoint_file_sha256")
        or progress["curves"] != trained.get("curves")
    ):
        raise ValueError("reference COMPLETE progress evidence drifted")


def _verify_checkpoint_file(
    root: pathlib.Path,
    *,
    model: str,
    seed: int,
    trained: dict[str, Any],
) -> tuple[pathlib.Path, int, str]:
    expected_relative = f"checkpoints/{model}-seed-{seed}.pt"
    if trained.get("checkpoint_file") != expected_relative:
        raise ValueError("reference checkpoint relative path drifted")
    expected_file_hash = trained.get("checkpoint_file_sha256")
    expected_canonical_hash = trained.get("canonical_checkpoint_sha256")
    if not _is_sha256(expected_file_hash) or not _is_sha256(expected_canonical_hash):
        raise ValueError("reference checkpoint hashes are malformed")
    path = root / expected_relative
    payload, observed_hash = _read_regular_bytes(path, CHECKPOINT_MAX_BYTES)
    if observed_hash != expected_file_hash:
        raise ValueError("reference checkpoint file SHA-256 mismatch")
    return path, len(payload), observed_hash


def verify_reference_output(
    root: pathlib.Path,
    *,
    expected_execution_head: str,
) -> ReferenceOutputManifest:
    """Verify one completed, explicitly named reference output directory."""
    if not isinstance(root, pathlib.Path):
        raise TypeError("reference output root must be pathlib.Path")
    if not root.is_absolute():
        raise ValueError("reference output root must be absolute")
    if root.is_symlink():
        raise ValueError("reference output root must not be a symbolic link")
    try:
        metadata = root.stat()
    except OSError as error:
        raise ValueError("reference output root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("reference output root is not a directory")

    execution_head = _validate_execution_head(expected_execution_head)
    expected_inventory = expected_relative_files()
    if len(expected_inventory) != EXPECTED_FILE_COUNT:
        raise RuntimeError("reference expected file count drifted")
    initial_inventory = _inventory(root)
    if initial_inventory != expected_inventory:
        raise ValueError("reference output file inventory drifted")

    summary, summary_sha256 = _read_json(root / "summary.json")
    contract = _validate_contract(summary.get("contract"))
    fingerprints = _expected_fingerprints()
    _validate_summary_header(
        summary,
        execution_head=execution_head,
        contract=contract,
        fingerprints=fingerprints,
    )
    start, _ = _read_json(root / "run-start.json")
    _validate_start(
        start,
        execution_head=execution_head,
        contract=contract,
        fingerprints=fingerprints,
    )
    if (
        start.get("dataset_cache_build_seconds")
        != summary.get("dataset_cache_build_seconds")
    ):
        raise ValueError("reference dataset cache build time drifted between artifacts")
    if (
        start.get("dataset_cache_resident_bytes")
        != summary.get("dataset_cache_resident_bytes")
    ):
        raise ValueError("reference dataset cache bytes drifted between artifacts")

    seed_rows = summary.get("seed_rows")
    if type(seed_rows) is not list:
        raise TypeError("reference summary seed_rows must be a list")
    normalized_seed_rows = _canonicalize_seed_rows(seed_rows)
    diagnosis = training.classify(normalized_seed_rows)
    if summary.get("diagnosis") != diagnosis:
        raise ValueError("reference summary diagnosis does not match recomputation")
    scaling = training.population_scaling_summary(normalized_seed_rows)
    if summary.get("population_scaling") != scaling:
        raise ValueError(
            "reference population scaling summary does not match recomputation"
        )
    if summary.get("boundaries") != _expected_boundaries():
        raise ValueError("reference scientific boundaries drifted")

    population_records: list[PopulationCheckpointManifest] = []
    for expected_seed, seed_row in zip(l0_protocol.INITIALIZATION_SEEDS, seed_rows):
        if type(seed_row) is not dict or seed_row.get("seed") != expected_seed:
            raise ValueError("reference seed row order or identity drifted")
        seed_file, _ = _read_json(root / f"seed-{expected_seed}.json")
        if seed_file != seed_row:
            raise ValueError("reference per-seed artifact differs from summary")
        for model in ("transformer", "population"):
            trained = seed_row.get(model)
            if type(trained) is not dict:
                raise TypeError("reference trained-model row must be a plain object")
            _verify_progress(
                root / "progress" / f"{model}-seed-{expected_seed}.json",
                model=model,
                seed=expected_seed,
                trained=trained,
            )
            checkpoint_path, file_bytes, observed_hash = _verify_checkpoint_file(
                root,
                model=model,
                seed=expected_seed,
                trained=trained,
            )
            if model == "population":
                loaded = checkpoint.load_reference_checkpoint(
                    checkpoint_path,
                    expected_seed=expected_seed,
                    expected_file_sha256=observed_hash,
                    expected_canonical_sha256=trained[
                        "canonical_checkpoint_sha256"
                    ],
                )
                if (
                    loaded.seed != expected_seed
                    or loaded.file_sha256 != observed_hash
                    or loaded.canonical_state_sha256
                    != trained["canonical_checkpoint_sha256"]
                ):
                    raise ValueError("strict population checkpoint evidence drifted")
                population_records.append(
                    PopulationCheckpointManifest(
                        seed=expected_seed,
                        path=str(checkpoint_path),
                        file_bytes=file_bytes,
                        file_sha256=observed_hash,
                        canonical_state_sha256=trained[
                            "canonical_checkpoint_sha256"
                        ],
                    )
                )
                del loaded

    if _inventory(root) != initial_inventory:
        raise ValueError("reference output inventory changed during verification")
    return ReferenceOutputManifest(
        root=str(root),
        summary_sha256=summary_sha256,
        execution_head=execution_head,
        diagnosis=diagnosis,
        population_scaling_conclusion=scaling["conclusion"],
        post_training_base_eligible=diagnosis == training.PASS,
        population_checkpoints=tuple(population_records),
    )


def validate_reference_manifest_contract() -> dict[str, object]:
    expected = expected_relative_files()
    checks = {
        "source_fresh_process_head_is_pinned": SOURCE_FRESH_PROCESS_HEAD
        == "f0cf83d1be0426fda976f08a379ab040be53ba89",
        "json_bound_is_exact": JSON_MAX_BYTES == 16 * 1024 * 1024,
        "checkpoint_bound_matches_loader": CHECKPOINT_MAX_BYTES
        == checkpoint.REFERENCE_CHECKPOINT_MAX_BYTES,
        "expected_file_count_is_exact": len(expected) == EXPECTED_FILE_COUNT == 17,
        "all_three_seed_files_are_present": all(
            f"seed-{seed}.json" in expected
            for seed in l0_protocol.INITIALIZATION_SEEDS
        ),
        "all_population_checkpoints_are_present": all(
            f"checkpoints/population-seed-{seed}.pt" in expected
            for seed in l0_protocol.INITIALIZATION_SEEDS
        ),
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "source_fresh_process_head": SOURCE_FRESH_PROCESS_HEAD,
        "expected_relative_files": list(expected),
        "json_max_bytes": JSON_MAX_BYTES,
        "checkpoint_max_bytes": CHECKPOINT_MAX_BYTES,
        "checks": checks,
        "valid": all(checks.values()),
    }
