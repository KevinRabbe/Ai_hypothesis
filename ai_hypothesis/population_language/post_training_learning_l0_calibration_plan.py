"""Immutable, non-authorizing calibration plan for Post-Training Learning L0.

The plan binds a previously verified reference manifest to the frozen
calibration grid and seed pairing. It contains no final-world material and
cannot authorize calibration or final execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import stat
from typing import Any

from . import l0_reference_manifest as reference_manifest
from . import l0_reference_training as reference_training
from . import post_training_learning_l0_calibration as calibration
from . import post_training_learning_l0_world as world

VERSION = "population-language-post-training-learning-l0-calibration-plan-v0"
BRANCH = "agent/population-language-post-training-learning-l0-calibration-plan-v0"
STATUS = "CALIBRATION_PLAN_ONLY_NO_EXECUTION_AUTHORIZATION"
SOURCE_REFERENCE_MANIFEST_HEAD = "4bb09762948fd83eb7a7ea2beb5b1f8ecdbd450d"

PLAN_MAX_BYTES = 256 * 1024

PLAN_KEYS = (
    "version",
    "status",
    "source_reference_manifest_head",
    "reference_output_root",
    "reference_summary_sha256",
    "reference_execution_head",
    "reference_diagnosis",
    "population_scaling_conclusion",
    "population_checkpoints",
    "calibration_grid_sha256",
    "calibration_candidates",
    "calibration_pairs",
    "calibration_world_fingerprints",
    "expected_result_rows",
    "expected_result_keys",
    "result_root",
    "calibration_authorized",
    "final_execution_authorized",
)
CHECKPOINT_KEYS = (
    "model_seed",
    "path",
    "file_bytes",
    "file_sha256",
    "canonical_state_sha256",
)
CANDIDATE_KEYS = (
    "candidate_id",
    "rank",
    "learning_rate",
    "updates",
    "trainable_parameters",
    "persisted_fp32_bytes",
    "example_presentations",
)
PAIR_KEYS = (
    "model_seed",
    "calibration_world_seed",
)


@dataclass(frozen=True)
class CalibrationPlanArtifact:
    path: str
    bytes: int
    sha256: str


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("calibration plan JSON contains a duplicate key")
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


def _absolute_path_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"{label} must be a bounded path string")
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return str(path)


def _checkpoint_rows(
    manifest: reference_manifest.ReferenceOutputManifest,
) -> list[dict[str, object]]:
    expected_seeds = list(world.MODEL_INITIALIZATION_SEEDS)
    records = list(manifest.population_checkpoints)
    if [record.seed for record in records] != expected_seeds:
        raise ValueError("reference manifest population checkpoints are incomplete or reordered")
    rows: list[dict[str, object]] = []
    for record in records:
        row = {
            "model_seed": record.seed,
            "path": record.path,
            "file_bytes": record.file_bytes,
            "file_sha256": record.file_sha256,
            "canonical_state_sha256": record.canonical_state_sha256,
        }
        _validate_checkpoint_row(row, record.seed)
        rows.append(row)
    return rows


def _candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate.identifier,
            "rank": candidate.rank,
            "learning_rate": candidate.learning_rate,
            "updates": candidate.updates,
            "trainable_parameters": candidate.trainable_parameters,
            "persisted_fp32_bytes": candidate.persisted_fp32_bytes,
            "example_presentations": candidate.example_presentations,
        }
        for candidate in calibration.candidate_grid()
    ]


def _pair_rows() -> list[dict[str, int]]:
    return [
        {
            "model_seed": model_seed,
            "calibration_world_seed": calibration_world_seed,
        }
        for model_seed, calibration_world_seed in calibration.CALIBRATION_PAIRS
    ]


def _result_keys() -> list[list[object]]:
    return [list(key) for key in calibration.expected_result_keys()]


def build_calibration_plan(
    manifest: reference_manifest.ReferenceOutputManifest,
    *,
    result_root: pathlib.Path,
) -> dict[str, object]:
    if not isinstance(manifest, reference_manifest.ReferenceOutputManifest):
        raise TypeError("calibration plan requires a verified reference manifest")
    if not isinstance(result_root, pathlib.Path):
        raise TypeError("calibration result root must be pathlib.Path")
    if not result_root.is_absolute():
        raise ValueError("calibration result root must be absolute")
    if manifest.diagnosis != reference_training.PASS:
        raise ValueError("reference manifest diagnosis is not valid")
    if manifest.post_training_base_eligible is not True:
        raise ValueError("reference manifest is not Post-Training base eligible")
    if pathlib.Path(manifest.root) == result_root:
        raise ValueError("calibration result root must differ from the reference output root")

    plan: dict[str, object] = {
        "version": VERSION,
        "status": STATUS,
        "source_reference_manifest_head": SOURCE_REFERENCE_MANIFEST_HEAD,
        "reference_output_root": manifest.root,
        "reference_summary_sha256": manifest.summary_sha256,
        "reference_execution_head": manifest.execution_head,
        "reference_diagnosis": manifest.diagnosis,
        "population_scaling_conclusion": manifest.population_scaling_conclusion,
        "population_checkpoints": _checkpoint_rows(manifest),
        "calibration_grid_sha256": calibration.GRID_SHA256,
        "calibration_candidates": _candidate_rows(),
        "calibration_pairs": _pair_rows(),
        "calibration_world_fingerprints": world.calibration_world_fingerprints(),
        "expected_result_rows": calibration.EXPECTED_RESULT_ROWS,
        "expected_result_keys": _result_keys(),
        "result_root": str(result_root),
        "calibration_authorized": False,
        "final_execution_authorized": False,
    }
    validate_calibration_plan(plan)
    return plan


def _validate_checkpoint_row(row: object, expected_seed: int) -> None:
    if type(row) is not dict or tuple(row) != CHECKPOINT_KEYS:
        raise ValueError("calibration checkpoint row keys or order drifted")
    if row["model_seed"] != expected_seed:
        raise ValueError("calibration checkpoint model seed drifted")
    _absolute_path_string(row["path"], "calibration checkpoint path")
    if (
        type(row["file_bytes"]) is not int
        or not 0 < row["file_bytes"] <= reference_manifest.CHECKPOINT_MAX_BYTES
    ):
        raise ValueError("calibration checkpoint byte count is invalid")
    if not _is_sha256(row["file_sha256"]):
        raise ValueError("calibration checkpoint file SHA-256 is invalid")
    if not _is_sha256(row["canonical_state_sha256"]):
        raise ValueError("calibration checkpoint canonical SHA-256 is invalid")


def _validate_candidate_rows(value: object) -> None:
    expected = _candidate_rows()
    if type(value) is not list or value != expected:
        raise ValueError("calibration candidate grid drifted")
    if any(type(row) is not dict or tuple(row) != CANDIDATE_KEYS for row in value):
        raise ValueError("calibration candidate row keys or order drifted")


def _validate_pair_rows(value: object) -> None:
    expected = _pair_rows()
    if type(value) is not list or value != expected:
        raise ValueError("calibration seed pairs drifted")
    if any(type(row) is not dict or tuple(row) != PAIR_KEYS for row in value):
        raise ValueError("calibration pair row keys or order drifted")


def validate_calibration_plan(value: object) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != PLAN_KEYS:
        raise ValueError("calibration plan keys or order drifted")
    if value["version"] != VERSION or value["status"] != STATUS:
        raise ValueError("calibration plan version or status drifted")
    if value["source_reference_manifest_head"] != SOURCE_REFERENCE_MANIFEST_HEAD:
        raise ValueError("calibration plan source manifest head drifted")
    if not _is_git_sha(value["reference_execution_head"]):
        raise ValueError("calibration plan reference execution head is invalid")
    if not _is_sha256(value["reference_summary_sha256"]):
        raise ValueError("calibration plan reference summary SHA-256 is invalid")
    reference_root = _absolute_path_string(
        value["reference_output_root"], "reference output root"
    )
    result_root = _absolute_path_string(value["result_root"], "calibration result root")
    if reference_root == result_root:
        raise ValueError("calibration and reference roots must differ")
    if value["reference_diagnosis"] != reference_training.PASS:
        raise ValueError("calibration plan reference diagnosis is not valid")
    if value["population_scaling_conclusion"] not in (
        reference_training.SCALING_SUPPORTS,
        reference_training.SCALING_DOES_NOT_SUPPORT,
    ):
        raise ValueError("calibration plan scaling conclusion is invalid")

    checkpoints = value["population_checkpoints"]
    if type(checkpoints) is not list or len(checkpoints) != len(
        world.MODEL_INITIALIZATION_SEEDS
    ):
        raise ValueError("calibration plan checkpoint count drifted")
    for row, expected_seed in zip(
        checkpoints, world.MODEL_INITIALIZATION_SEEDS, strict=True
    ):
        _validate_checkpoint_row(row, expected_seed)
    paths = [row["path"] for row in checkpoints]
    file_hashes = [row["file_sha256"] for row in checkpoints]
    canonical_hashes = [row["canonical_state_sha256"] for row in checkpoints]
    if len(set(paths)) != len(paths):
        raise ValueError("calibration plan checkpoint paths are not unique")
    if len(set(file_hashes)) != len(file_hashes):
        raise ValueError("calibration plan checkpoint file hashes are not unique")
    if len(set(canonical_hashes)) != len(canonical_hashes):
        raise ValueError("calibration plan canonical checkpoint hashes are not unique")

    if value["calibration_grid_sha256"] != calibration.GRID_SHA256:
        raise ValueError("calibration plan grid SHA-256 drifted")
    _validate_candidate_rows(value["calibration_candidates"])
    _validate_pair_rows(value["calibration_pairs"])
    if value["calibration_world_fingerprints"] != world.calibration_world_fingerprints():
        raise ValueError("calibration plan world fingerprints drifted")
    if value["expected_result_rows"] != calibration.EXPECTED_RESULT_ROWS:
        raise ValueError("calibration plan expected result count drifted")
    if value["expected_result_keys"] != _result_keys():
        raise ValueError("calibration plan expected result keys drifted")
    if value["calibration_authorized"] is not False:
        raise ValueError("calibration plan must not authorize calibration")
    if value["final_execution_authorized"] is not False:
        raise ValueError("calibration plan must not authorize final execution")

    checks = {
        "source_manifest_head_is_pinned": SOURCE_REFERENCE_MANIFEST_HEAD
        == "4bb09762948fd83eb7a7ea2beb5b1f8ecdbd450d",
        "all_three_population_checkpoints_bound": len(checkpoints) == 3,
        "candidate_grid_is_exact": len(value["calibration_candidates"])
        == calibration.CANDIDATE_COUNT,
        "calibration_pairs_are_exact": len(value["calibration_pairs"])
        == calibration.CALIBRATION_PAIR_COUNT,
        "result_rows_are_exact": len(value["expected_result_keys"])
        == calibration.EXPECTED_RESULT_ROWS,
        "calibration_is_not_authorized": value["calibration_authorized"] is False,
        "final_execution_is_not_authorized": value["final_execution_authorized"]
        is False,
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "checks": checks,
        "valid": all(checks.values()),
    }


def save_calibration_plan_create_once(
    path: pathlib.Path,
    plan: dict[str, object],
) -> CalibrationPlanArtifact:
    if not isinstance(path, pathlib.Path):
        raise TypeError("calibration plan path must be pathlib.Path")
    validate_calibration_plan(plan)
    payload = _canonical_json_bytes(plan)
    if not 0 < len(payload) <= PLAN_MAX_BYTES:
        raise ValueError("calibration plan size lies outside the contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return CalibrationPlanArtifact(
        path=str(path),
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_calibration_plan(
    path: pathlib.Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    if not isinstance(path, pathlib.Path):
        raise TypeError("calibration plan path must be pathlib.Path")
    if not _is_sha256(expected_sha256):
        raise ValueError("expected calibration plan SHA-256 is invalid")
    if path.is_symlink():
        raise ValueError("calibration plan path must not be a symbolic link")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError("calibration plan path is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("calibration plan path is not a regular file")
    if not 0 < metadata.st_size <= PLAN_MAX_BYTES:
        raise ValueError("calibration plan file size lies outside the contract")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if opened.st_size != metadata.st_size:
                raise ValueError("calibration plan changed before reading")
            payload = handle.read(PLAN_MAX_BYTES + 1)
    except OSError as error:
        raise ValueError("calibration plan could not be read") from error
    if len(payload) != metadata.st_size or len(payload) > PLAN_MAX_BYTES:
        raise ValueError("calibration plan changed while reading")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError("calibration plan SHA-256 mismatch")
    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=_plain_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("calibration plan is not valid UTF-8 JSON") from error
    validate_calibration_plan(decoded)
    return decoded


def validate_calibration_plan_contract() -> dict[str, object]:
    checks = {
        "source_manifest_head_is_exact": SOURCE_REFERENCE_MANIFEST_HEAD
        == "4bb09762948fd83eb7a7ea2beb5b1f8ecdbd450d",
        "plan_size_is_bounded": PLAN_MAX_BYTES == 256 * 1024,
        "plan_keys_are_exact": len(PLAN_KEYS) == 18,
        "checkpoint_keys_are_exact": len(CHECKPOINT_KEYS) == 5,
        "candidate_keys_are_exact": len(CANDIDATE_KEYS) == 7,
        "pair_keys_are_exact": len(PAIR_KEYS) == 2,
        "candidate_count_is_exact": len(_candidate_rows()) == 48,
        "pair_count_is_exact": len(_pair_rows()) == 3,
        "result_key_count_is_exact": len(_result_keys()) == 144,
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "source_reference_manifest_head": SOURCE_REFERENCE_MANIFEST_HEAD,
        "plan_max_bytes": PLAN_MAX_BYTES,
        "checks": checks,
        "valid": all(checks.values()),
    }
