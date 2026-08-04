"""Authorization-gated calibration orchestration and independent result verification.

This module prepares and verifies the complete Post-Training Learning L0
calibration workload. It never creates an authorization, executes a model,
loads a checkpoint, or accesses a final world.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import stat
from typing import Any, Mapping, Sequence

from . import post_training_learning_l0_calibration as calibration
from . import post_training_learning_l0_calibration_plan as calibration_plan
from . import post_training_learning_l0_protocol as protocol
from . import post_training_learning_l0_world as world

VERSION = "population-language-post-training-learning-l0-calibration-orchestrator-v0"
BRANCH = "agent/population-language-post-training-learning-l0-calibration-orchestrator-v0"
STATUS = "CALIBRATION_ORCHESTRATION_PREPARATION_ONLY_NO_MODEL_EXECUTION"
SOURCE_CALIBRATION_PLAN_HEAD = "780c19c0c8e63dafe6e7c74bbfe3d579129e53fe"

AUTHORIZATION_VERSION = "population-language-post-training-learning-l0-calibration-authorization-v0"
AUTHORIZATION_SCOPE = "POST_TRAINING_LEARNING_L0_CALIBRATION_ONLY"
AUTHORIZATION_ACKNOWLEDGEMENT = (
    "I explicitly authorize the frozen 144-row calibration only; "
    "final-world execution remains unauthorized."
)
RUN_MANIFEST_STATUS = "POST_TRAINING_LEARNING_L0_CALIBRATION_RUN_MANIFEST_READY"
RESULT_BUNDLE_STATUS = "POST_TRAINING_LEARNING_L0_CALIBRATION_RESULT_BUNDLE_COMPLETE"
VERIFICATION_STATUS = "POST_TRAINING_LEARNING_L0_CALIBRATION_RESULT_VERIFIED"

AUTHORIZATION_MAX_BYTES = 32 * 1024
RUN_MANIFEST_MAX_BYTES = 2 * 1024 * 1024
RESULT_BUNDLE_MAX_BYTES = 16 * 1024 * 1024
VERIFICATION_MAX_BYTES = 64 * 1024

SCHEDULE_SHA256_BY_UPDATES = {
    32: "391e3cedb1290c5956cd0d8b72fea240054f20914b64f81317966c70173ac81d",
    64: "0df432ac0bbfde71a84a199118d041467371b9c47f21c547d3aa06ebeced42ca",
    128: "77d166ee7e7fcb579acd16b3295ab56e9f42aed37ed4fa884fbca388461a7bed",
    256: "f8dbafd553ab4bca6d3d6b977a3cb8bc939adb18a86a46e9837d3c5bb9dd8958",
}

AUTHORIZATION_KEYS = (
    "version",
    "scope",
    "source_calibration_plan_head",
    "calibration_plan_sha256",
    "reference_summary_sha256",
    "result_root",
    "authorization_id",
    "operator_acknowledgement",
    "calibration_authorized",
    "final_execution_authorized",
)
RUN_MANIFEST_KEYS = (
    "version",
    "status",
    "source_calibration_plan_head",
    "calibration_plan_sha256",
    "authorization_sha256",
    "reference_summary_sha256",
    "reference_execution_head",
    "result_root",
    "expected_result_rows",
    "work_items",
    "calibration_authorized",
    "final_execution_authorized",
)
WORK_ITEM_KEYS = (
    "ordinal",
    "candidate_id",
    "rank",
    "learning_rate",
    "updates",
    "model_seed",
    "calibration_world_seed",
    "checkpoint_path",
    "checkpoint_file_sha256",
    "checkpoint_canonical_sha256",
    "adapter_initialization_seed",
    "adaptation_schedule_sha256",
    "adapter_artifact_path",
    "fresh_process_request_path",
    "fresh_process_result_path",
    "result_row_path",
    "expected_result_key",
    "work_item_sha256",
)
RESULT_RECORD_KEYS = (
    "ordinal",
    "work_item_sha256",
    "result",
)
RESULT_BUNDLE_KEYS = (
    "version",
    "status",
    "source_calibration_plan_head",
    "calibration_plan_sha256",
    "run_manifest_sha256",
    "reference_summary_sha256",
    "result_root",
    "result_records",
    "calibration_complete",
    "final_execution_authorized",
)
VERIFICATION_KEYS = (
    "version",
    "status",
    "source_calibration_plan_head",
    "calibration_plan_sha256",
    "run_manifest_sha256",
    "result_bundle_sha256",
    "reference_summary_sha256",
    "run_classification",
    "conclusion",
    "qualified_candidate_count",
    "selected_candidate_id",
    "selected_rank",
    "selected_learning_rate",
    "selected_updates",
    "selected_trainable_adaptation_parameters",
    "selected_persisted_adaptation_bytes",
    "minimum_direct_gain",
    "mean_direct_gain",
    "minimum_composition_gain",
    "mean_composition_gain",
    "final_execution_eligibility",
    "separate_explicit_authorization_still_required",
    "calibration_was_authorized",
    "final_execution_authorized",
)


@dataclass(frozen=True)
class JsonArtifact:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CalibrationVerification:
    run_classification: str
    conclusion: str
    selected_candidate_id: str | None
    final_execution_eligibility: str
    report: dict[str, object]


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
            raise ValueError("JSON artifact contains a duplicate key")
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


def _require_canonical_sha256(value: object, expected_sha256: str, label: str) -> None:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} SHA-256 is invalid")
    observed = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"{label} canonical SHA-256 mismatch")


def _absolute_path(value: object, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"{label} must be a bounded path string")
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


def _read_json(
    path: pathlib.Path,
    maximum: int,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(path, pathlib.Path):
        raise TypeError("JSON artifact path must be pathlib.Path")
    if not _is_sha256(expected_sha256):
        raise ValueError("expected JSON artifact SHA-256 is invalid")
    if path.is_symlink():
        raise ValueError("JSON artifact must not be a symbolic link")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError("JSON artifact is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("JSON artifact is not a regular file")
    if not 0 < metadata.st_size <= maximum:
        raise ValueError("JSON artifact size lies outside the contract")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if opened.st_size != metadata.st_size:
                raise ValueError("JSON artifact changed before reading")
            payload = handle.read(maximum + 1)
    except OSError as error:
        raise ValueError("JSON artifact could not be read") from error
    if len(payload) != metadata.st_size or len(payload) > maximum:
        raise ValueError("JSON artifact changed while reading")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("JSON artifact SHA-256 mismatch")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_plain_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JSON artifact is not valid UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError("JSON artifact must contain a plain object")
    return value, observed_sha256


def _save_json_create_once(
    path: pathlib.Path,
    value: object,
    maximum: int,
) -> JsonArtifact:
    if not isinstance(path, pathlib.Path):
        raise TypeError("JSON artifact path must be pathlib.Path")
    payload = _canonical_json_bytes(value)
    if not 0 < len(payload) <= maximum:
        raise ValueError("JSON artifact size lies outside the contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return JsonArtifact(
        path=str(path),
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def validate_calibration_authorization(
    value: object,
    *,
    plan: Mapping[str, object],
    plan_sha256: str,
) -> dict[str, object]:
    calibration_plan.validate_calibration_plan(plan)
    _require_canonical_sha256(plan, plan_sha256, "calibration plan")
    if type(value) is not dict or tuple(value) != AUTHORIZATION_KEYS:
        raise ValueError("calibration authorization keys or order drifted")
    if value["version"] != AUTHORIZATION_VERSION:
        raise ValueError("calibration authorization version drifted")
    if value["scope"] != AUTHORIZATION_SCOPE:
        raise ValueError("calibration authorization scope drifted")
    if value["source_calibration_plan_head"] != SOURCE_CALIBRATION_PLAN_HEAD:
        raise ValueError("calibration authorization source plan head drifted")
    if value["calibration_plan_sha256"] != plan_sha256:
        raise ValueError("calibration authorization plan SHA-256 drifted")
    if value["reference_summary_sha256"] != plan["reference_summary_sha256"]:
        raise ValueError("calibration authorization reference summary drifted")
    if value["result_root"] != plan["result_root"]:
        raise ValueError("calibration authorization result root drifted")
    if not _is_sha256(value["authorization_id"]):
        raise ValueError("calibration authorization ID is invalid")
    if value["operator_acknowledgement"] != AUTHORIZATION_ACKNOWLEDGEMENT:
        raise ValueError("calibration authorization acknowledgement drifted")
    if value["calibration_authorized"] is not True:
        raise ValueError("calibration authorization does not explicitly authorize calibration")
    if value["final_execution_authorized"] is not False:
        raise ValueError("calibration authorization must not authorize final execution")
    return dict(value)


def load_calibration_authorization(
    path: pathlib.Path,
    *,
    expected_sha256: str,
    plan: Mapping[str, object],
    plan_sha256: str,
) -> dict[str, object]:
    value, _ = _read_json(
        path,
        AUTHORIZATION_MAX_BYTES,
        expected_sha256=expected_sha256,
    )
    _require_canonical_sha256(
        value,
        expected_sha256,
        "calibration authorization",
    )
    return validate_calibration_authorization(
        value,
        plan=plan,
        plan_sha256=plan_sha256,
    )


def _adaptation_schedule_sha256(updates: int) -> str:
    if updates not in calibration.UPDATE_COUNTS:
        raise ValueError("adaptation update count lies outside the locked grid")
    digest = hashlib.sha256()
    for update_index in range(updates):
        start = update_index * calibration.MICROBATCH_SIZE
        for offset in range(calibration.MICROBATCH_SIZE):
            ordinal = (start + offset) % world.ADAPTATION_EXAMPLES
            digest.update(ordinal.to_bytes(2, "little"))
    observed = digest.hexdigest()
    if observed != SCHEDULE_SHA256_BY_UPDATES[updates]:
        raise RuntimeError("adaptation schedule SHA-256 drifted")
    return observed


def _work_item_payload(row: Mapping[str, object]) -> dict[str, object]:
    return {key: row[key] for key in WORK_ITEM_KEYS[:-1]}


def _work_item_sha256(row: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(_work_item_payload(row))).hexdigest()


def _safe_component(value: str) -> str:
    if (
        not value
        or len(value) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in value)
    ):
        raise ValueError("work-item path component is invalid")
    return value


def _work_items(plan: Mapping[str, object]) -> list[dict[str, object]]:
    candidates = plan["calibration_candidates"]
    pairs = plan["calibration_pairs"]
    checkpoints = plan["population_checkpoints"]
    if type(candidates) is not list or type(pairs) is not list or type(checkpoints) is not list:
        raise TypeError("calibration plan lists are malformed")
    checkpoint_by_seed = {row["model_seed"]: row for row in checkpoints}
    root = _absolute_path(plan["result_root"], "calibration result root")
    items: list[dict[str, object]] = []
    ordinal = 0
    for candidate in candidates:
        if type(candidate) is not dict:
            raise TypeError("calibration candidate row must be a plain object")
        candidate_id = _safe_component(str(candidate["candidate_id"]))
        for pair in pairs:
            if type(pair) is not dict:
                raise TypeError("calibration pair row must be a plain object")
            model_seed = pair["model_seed"]
            calibration_world_seed = pair["calibration_world_seed"]
            checkpoint = checkpoint_by_seed.get(model_seed)
            if type(checkpoint) is not dict:
                raise ValueError("calibration checkpoint is missing for a model seed")
            stem = (
                f"{ordinal:03d}-{candidate_id}-"
                f"m{model_seed}-w{calibration_world_seed}"
            )
            row: dict[str, object] = {
                "ordinal": ordinal,
                "candidate_id": candidate["candidate_id"],
                "rank": candidate["rank"],
                "learning_rate": candidate["learning_rate"],
                "updates": candidate["updates"],
                "model_seed": model_seed,
                "calibration_world_seed": calibration_world_seed,
                "checkpoint_path": checkpoint["path"],
                "checkpoint_file_sha256": checkpoint["file_sha256"],
                "checkpoint_canonical_sha256": checkpoint["canonical_state_sha256"],
                "adapter_initialization_seed": protocol.adapter_initialization_seed(model_seed),
                "adaptation_schedule_sha256": _adaptation_schedule_sha256(
                    int(candidate["updates"])
                ),
                "adapter_artifact_path": str(root / "adapters" / f"{stem}.bin"),
                "fresh_process_request_path": str(
                    root / "fresh-process" / "requests" / f"{stem}.json"
                ),
                "fresh_process_result_path": str(
                    root / "fresh-process" / "results" / f"{stem}.json"
                ),
                "result_row_path": str(root / "rows" / f"{stem}.json"),
                "expected_result_key": [
                    candidate["candidate_id"],
                    model_seed,
                    calibration_world_seed,
                ],
            }
            row["work_item_sha256"] = _work_item_sha256(row)
            items.append(row)
            ordinal += 1
    return items


def validate_run_manifest(
    value: object,
    *,
    plan: Mapping[str, object],
    plan_sha256: str,
    authorization_sha256: str,
) -> dict[str, object]:
    calibration_plan.validate_calibration_plan(plan)
    _require_canonical_sha256(plan, plan_sha256, "calibration plan")
    if not _is_sha256(authorization_sha256):
        raise ValueError("run-manifest authorization SHA-256 is invalid")
    if type(value) is not dict or tuple(value) != RUN_MANIFEST_KEYS:
        raise ValueError("calibration run-manifest keys or order drifted")
    if value["version"] != VERSION or value["status"] != RUN_MANIFEST_STATUS:
        raise ValueError("calibration run-manifest version or status drifted")
    if value["source_calibration_plan_head"] != SOURCE_CALIBRATION_PLAN_HEAD:
        raise ValueError("calibration run-manifest source plan head drifted")
    if value["calibration_plan_sha256"] != plan_sha256:
        raise ValueError("calibration run-manifest plan SHA-256 drifted")
    if value["authorization_sha256"] != authorization_sha256:
        raise ValueError("calibration run-manifest authorization SHA-256 drifted")
    if value["reference_summary_sha256"] != plan["reference_summary_sha256"]:
        raise ValueError("calibration run-manifest reference summary drifted")
    if value["reference_execution_head"] != plan["reference_execution_head"]:
        raise ValueError("calibration run-manifest reference execution head drifted")
    if value["result_root"] != plan["result_root"]:
        raise ValueError("calibration run-manifest result root drifted")
    if value["expected_result_rows"] != calibration.EXPECTED_RESULT_ROWS:
        raise ValueError("calibration run-manifest result count drifted")
    expected_items = _work_items(plan)
    if value["work_items"] != expected_items:
        raise ValueError("calibration run-manifest work items drifted")
    if any(
        type(row) is not dict
        or tuple(row) != WORK_ITEM_KEYS
        or row["work_item_sha256"] != _work_item_sha256(row)
        for row in value["work_items"]
    ):
        raise ValueError("calibration work-item schema or SHA-256 drifted")
    if value["calibration_authorized"] is not True:
        raise ValueError("calibration run-manifest is not authorized")
    if value["final_execution_authorized"] is not False:
        raise ValueError("calibration run-manifest must not authorize final execution")
    return dict(value)


def build_run_manifest(
    plan: Mapping[str, object],
    *,
    plan_sha256: str,
    authorization: Mapping[str, object],
    authorization_sha256: str,
) -> dict[str, object]:
    calibration_plan.validate_calibration_plan(plan)
    validate_calibration_authorization(
        authorization,
        plan=plan,
        plan_sha256=plan_sha256,
    )
    _require_canonical_sha256(
        authorization,
        authorization_sha256,
        "calibration authorization",
    )
    manifest: dict[str, object] = {
        "version": VERSION,
        "status": RUN_MANIFEST_STATUS,
        "source_calibration_plan_head": SOURCE_CALIBRATION_PLAN_HEAD,
        "calibration_plan_sha256": plan_sha256,
        "authorization_sha256": authorization_sha256,
        "reference_summary_sha256": plan["reference_summary_sha256"],
        "reference_execution_head": plan["reference_execution_head"],
        "result_root": plan["result_root"],
        "expected_result_rows": calibration.EXPECTED_RESULT_ROWS,
        "work_items": _work_items(plan),
        "calibration_authorized": True,
        "final_execution_authorized": False,
    }
    return validate_run_manifest(
        manifest,
        plan=plan,
        plan_sha256=plan_sha256,
        authorization_sha256=authorization_sha256,
    )


def save_run_manifest_create_once(
    path: pathlib.Path,
    manifest: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    plan_sha256: str,
    authorization_sha256: str,
) -> JsonArtifact:
    validate_run_manifest(
        manifest,
        plan=plan,
        plan_sha256=plan_sha256,
        authorization_sha256=authorization_sha256,
    )
    return _save_json_create_once(path, manifest, RUN_MANIFEST_MAX_BYTES)


def load_run_manifest(
    path: pathlib.Path,
    *,
    expected_sha256: str,
    plan: Mapping[str, object],
    plan_sha256: str,
    authorization_sha256: str,
) -> dict[str, object]:
    value, _ = _read_json(
        path,
        RUN_MANIFEST_MAX_BYTES,
        expected_sha256=expected_sha256,
    )
    return validate_run_manifest(
        value,
        plan=plan,
        plan_sha256=plan_sha256,
        authorization_sha256=authorization_sha256,
    )


def _bounded_result_row(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("calibration result row must be a plain object")
    payload = _canonical_json_bytes(value)
    if not 0 < len(payload) <= 64 * 1024:
        raise ValueError("calibration result row size lies outside the contract")
    return dict(value)


def build_result_bundle(
    plan: Mapping[str, object],
    *,
    plan_sha256: str,
    run_manifest: Mapping[str, object],
    run_manifest_sha256: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    calibration_plan.validate_calibration_plan(plan)
    _require_canonical_sha256(
        run_manifest,
        run_manifest_sha256,
        "calibration run manifest",
    )
    authorization_sha256 = run_manifest.get("authorization_sha256")
    if not _is_sha256(authorization_sha256):
        raise ValueError("run-manifest authorization SHA-256 is invalid")
    validate_run_manifest(
        run_manifest,
        plan=plan,
        plan_sha256=plan_sha256,
        authorization_sha256=authorization_sha256,
    )
    if len(rows) != calibration.EXPECTED_RESULT_ROWS:
        raise ValueError("calibration result bundle requires exactly 144 rows")
    work_items = run_manifest["work_items"]
    records: list[dict[str, object]] = []
    for ordinal, (work_item, raw_row) in enumerate(
        zip(work_items, rows, strict=True)
    ):
        row = _bounded_result_row(raw_row)
        observed_key = [
            row.get("candidate_id"),
            row.get("model_seed"),
            row.get("calibration_world_seed"),
        ]
        if observed_key != work_item["expected_result_key"]:
            raise ValueError("calibration result row order or identity drifted")
        records.append(
            {
                "ordinal": ordinal,
                "work_item_sha256": work_item["work_item_sha256"],
                "result": row,
            }
        )
    bundle: dict[str, object] = {
        "version": VERSION,
        "status": RESULT_BUNDLE_STATUS,
        "source_calibration_plan_head": SOURCE_CALIBRATION_PLAN_HEAD,
        "calibration_plan_sha256": plan_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "reference_summary_sha256": plan["reference_summary_sha256"],
        "result_root": plan["result_root"],
        "result_records": records,
        "calibration_complete": True,
        "final_execution_authorized": False,
    }
    return validate_result_bundle(
        bundle,
        plan=plan,
        plan_sha256=plan_sha256,
        run_manifest=run_manifest,
        run_manifest_sha256=run_manifest_sha256,
    )


def validate_result_bundle(
    value: object,
    *,
    plan: Mapping[str, object],
    plan_sha256: str,
    run_manifest: Mapping[str, object],
    run_manifest_sha256: str,
) -> dict[str, object]:
    calibration_plan.validate_calibration_plan(plan)
    _require_canonical_sha256(plan, plan_sha256, "calibration plan")
    _require_canonical_sha256(
        run_manifest,
        run_manifest_sha256,
        "calibration run manifest",
    )
    authorization_sha256 = run_manifest.get("authorization_sha256")
    if not _is_sha256(authorization_sha256):
        raise ValueError("run-manifest authorization SHA-256 is invalid")
    validate_run_manifest(
        run_manifest,
        plan=plan,
        plan_sha256=plan_sha256,
        authorization_sha256=authorization_sha256,
    )
    if type(value) is not dict or tuple(value) != RESULT_BUNDLE_KEYS:
        raise ValueError("calibration result-bundle keys or order drifted")
    if value["version"] != VERSION or value["status"] != RESULT_BUNDLE_STATUS:
        raise ValueError("calibration result-bundle version or status drifted")
    if value["source_calibration_plan_head"] != SOURCE_CALIBRATION_PLAN_HEAD:
        raise ValueError("calibration result-bundle source plan head drifted")
    if value["calibration_plan_sha256"] != plan_sha256:
        raise ValueError("calibration result-bundle plan SHA-256 drifted")
    if value["run_manifest_sha256"] != run_manifest_sha256:
        raise ValueError("calibration result-bundle manifest SHA-256 drifted")
    if value["reference_summary_sha256"] != plan["reference_summary_sha256"]:
        raise ValueError("calibration result-bundle reference summary drifted")
    if value["result_root"] != plan["result_root"]:
        raise ValueError("calibration result-bundle result root drifted")
    if value["calibration_complete"] is not True:
        raise ValueError("calibration result-bundle is incomplete")
    if value["final_execution_authorized"] is not False:
        raise ValueError("calibration result-bundle must not authorize final execution")
    records = value["result_records"]
    work_items = run_manifest["work_items"]
    if type(records) is not list or len(records) != calibration.EXPECTED_RESULT_ROWS:
        raise ValueError("calibration result-record count drifted")
    for ordinal, (record, work_item) in enumerate(zip(records, work_items, strict=True)):
        if type(record) is not dict or tuple(record) != RESULT_RECORD_KEYS:
            raise ValueError("calibration result-record keys or order drifted")
        if record["ordinal"] != ordinal:
            raise ValueError("calibration result-record ordinal drifted")
        if record["work_item_sha256"] != work_item["work_item_sha256"]:
            raise ValueError("calibration result-record work-item SHA-256 drifted")
        row = _bounded_result_row(record["result"])
        if [
            row.get("candidate_id"),
            row.get("model_seed"),
            row.get("calibration_world_seed"),
        ] != work_item["expected_result_key"]:
            raise ValueError("calibration result-record identity drifted")
    return dict(value)


def save_result_bundle_create_once(
    path: pathlib.Path,
    bundle: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    plan_sha256: str,
    run_manifest: Mapping[str, object],
    run_manifest_sha256: str,
) -> JsonArtifact:
    validate_result_bundle(
        bundle,
        plan=plan,
        plan_sha256=plan_sha256,
        run_manifest=run_manifest,
        run_manifest_sha256=run_manifest_sha256,
    )
    return _save_json_create_once(path, bundle, RESULT_BUNDLE_MAX_BYTES)


def load_result_bundle(
    path: pathlib.Path,
    *,
    expected_sha256: str,
    plan: Mapping[str, object],
    plan_sha256: str,
    run_manifest: Mapping[str, object],
    run_manifest_sha256: str,
) -> dict[str, object]:
    value, _ = _read_json(
        path,
        RESULT_BUNDLE_MAX_BYTES,
        expected_sha256=expected_sha256,
    )
    return validate_result_bundle(
        value,
        plan=plan,
        plan_sha256=plan_sha256,
        run_manifest=run_manifest,
        run_manifest_sha256=run_manifest_sha256,
    )


def _normalized_verification(
    outcome: Mapping[str, object],
    *,
    plan_sha256: str,
    run_manifest_sha256: str,
    result_bundle_sha256: str,
    reference_summary_sha256: str,
) -> dict[str, object]:
    selected = outcome.get("selected_candidate_id")
    valid_selection = (
        outcome.get("run_classification") == calibration.CALIBRATION_RUN_VALID
        and outcome.get("conclusion") == calibration.CALIBRATION_SELECTS_CANDIDATE
        and isinstance(selected, str)
    )
    report: dict[str, object] = {
        "version": VERSION,
        "status": VERIFICATION_STATUS,
        "source_calibration_plan_head": SOURCE_CALIBRATION_PLAN_HEAD,
        "calibration_plan_sha256": plan_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "result_bundle_sha256": result_bundle_sha256,
        "reference_summary_sha256": reference_summary_sha256,
        "run_classification": outcome.get("run_classification"),
        "conclusion": outcome.get("conclusion"),
        "qualified_candidate_count": outcome.get("qualified_candidate_count", 0),
        "selected_candidate_id": selected,
        "selected_rank": outcome.get("selected_rank"),
        "selected_learning_rate": outcome.get("selected_learning_rate"),
        "selected_updates": outcome.get("selected_updates"),
        "selected_trainable_adaptation_parameters": outcome.get(
            "selected_trainable_adaptation_parameters"
        ),
        "selected_persisted_adaptation_bytes": outcome.get(
            "selected_persisted_adaptation_bytes"
        ),
        "minimum_direct_gain": outcome.get("minimum_direct_gain"),
        "mean_direct_gain": outcome.get("mean_direct_gain"),
        "minimum_composition_gain": outcome.get("minimum_composition_gain"),
        "mean_composition_gain": outcome.get("mean_composition_gain"),
        "final_execution_eligibility": outcome.get(
            "final_execution_eligibility",
            calibration.FINAL_EXECUTION_NOT_ELIGIBLE,
        ),
        "separate_explicit_authorization_still_required": bool(valid_selection),
        "calibration_was_authorized": True,
        "final_execution_authorized": False,
    }
    if tuple(report) != VERIFICATION_KEYS:
        raise RuntimeError("calibration verification report keys drifted")
    return report


def verify_result_bundle(
    plan: Mapping[str, object],
    *,
    plan_sha256: str,
    run_manifest: Mapping[str, object],
    run_manifest_sha256: str,
    result_bundle: Mapping[str, object],
    result_bundle_sha256: str,
) -> CalibrationVerification:
    _require_canonical_sha256(
        result_bundle,
        result_bundle_sha256,
        "calibration result bundle",
    )
    validate_result_bundle(
        result_bundle,
        plan=plan,
        plan_sha256=plan_sha256,
        run_manifest=run_manifest,
        run_manifest_sha256=run_manifest_sha256,
    )
    rows = [record["result"] for record in result_bundle["result_records"]]
    checkpoint_by_seed = {
        row["model_seed"]: row for row in plan["population_checkpoints"]
    }
    checkpoint_binding_valid = all(
        row.get("base_checkpoint_sha256")
        == checkpoint_by_seed[row.get("model_seed")]["canonical_state_sha256"]
        and row.get("base_checkpoint_after_sha256")
        == checkpoint_by_seed[row.get("model_seed")]["canonical_state_sha256"]
        for row in rows
        if row.get("model_seed") in checkpoint_by_seed
    ) and all(row.get("model_seed") in checkpoint_by_seed for row in rows)
    if checkpoint_binding_valid:
        outcome = calibration.evaluate_calibration(rows)
    else:
        outcome = {
            "run_classification": calibration.CALIBRATION_RUN_INVALID,
            "conclusion": calibration.INVALID_CALIBRATION_NO_SELECTION,
            "final_execution_eligibility": calibration.FINAL_EXECUTION_NOT_ELIGIBLE,
        }
    report = _normalized_verification(
        outcome,
        plan_sha256=plan_sha256,
        run_manifest_sha256=run_manifest_sha256,
        result_bundle_sha256=result_bundle_sha256,
        reference_summary_sha256=str(plan["reference_summary_sha256"]),
    )
    return CalibrationVerification(
        run_classification=str(report["run_classification"]),
        conclusion=str(report["conclusion"]),
        selected_candidate_id=(
            str(report["selected_candidate_id"])
            if report["selected_candidate_id"] is not None
            else None
        ),
        final_execution_eligibility=str(report["final_execution_eligibility"]),
        report=report,
    )


def save_verification_create_once(
    path: pathlib.Path,
    verification: CalibrationVerification,
) -> JsonArtifact:
    if not isinstance(verification, CalibrationVerification):
        raise TypeError("verification must be CalibrationVerification")
    if tuple(verification.report) != VERIFICATION_KEYS:
        raise ValueError("verification report schema drifted")
    if verification.report["final_execution_authorized"] is not False:
        raise ValueError("verification must not authorize final execution")
    return _save_json_create_once(path, verification.report, VERIFICATION_MAX_BYTES)


def validate_calibration_orchestrator_contract() -> dict[str, object]:
    schedule_hashes = {
        updates: _adaptation_schedule_sha256(updates)
        for updates in calibration.UPDATE_COUNTS
    }
    checks = {
        "source_calibration_plan_head_is_pinned": SOURCE_CALIBRATION_PLAN_HEAD
        == "780c19c0c8e63dafe6e7c74bbfe3d579129e53fe",
        "authorization_is_calibration_only": AUTHORIZATION_SCOPE
        == "POST_TRAINING_LEARNING_L0_CALIBRATION_ONLY",
        "authorization_does_not_cover_final_execution": "final-world"
        not in AUTHORIZATION_SCOPE.lower(),
        "expected_result_rows_are_exact": calibration.EXPECTED_RESULT_ROWS == 144,
        "all_schedule_hashes_are_exact": schedule_hashes == SCHEDULE_SHA256_BY_UPDATES,
        "authorization_size_is_bounded": AUTHORIZATION_MAX_BYTES == 32 * 1024,
        "run_manifest_size_is_bounded": RUN_MANIFEST_MAX_BYTES == 2 * 1024 * 1024,
        "result_bundle_size_is_bounded": RESULT_BUNDLE_MAX_BYTES == 16 * 1024 * 1024,
        "verification_size_is_bounded": VERIFICATION_MAX_BYTES == 64 * 1024,
        "authorization_keys_are_exact": len(AUTHORIZATION_KEYS) == 10,
        "run_manifest_keys_are_exact": len(RUN_MANIFEST_KEYS) == 12,
        "work_item_keys_are_exact": len(WORK_ITEM_KEYS) == 18,
        "result_record_keys_are_exact": len(RESULT_RECORD_KEYS) == 3,
        "result_bundle_keys_are_exact": len(RESULT_BUNDLE_KEYS) == 10,
        "verification_keys_are_exact": len(VERIFICATION_KEYS) == 24,
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "source_calibration_plan_head": SOURCE_CALIBRATION_PLAN_HEAD,
        "expected_result_rows": calibration.EXPECTED_RESULT_ROWS,
        "schedule_sha256_by_updates": schedule_hashes,
        "checks": checks,
        "valid": all(checks.values()),
    }
