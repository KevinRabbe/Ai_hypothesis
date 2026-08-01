"""Independent read-only audit of one completed Gate-9 training seed."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Any, Iterator

import torch

AUDIT_VERSION = "gate9-contextual-training-seed-audit-v0"
EXPECTED_EXECUTION_BRANCH = "agent/gate9-contextual-training-execution-v0"
EXPECTED_EXECUTION_HEAD = "bdc1af9bc65b94b01ae3946977686bd90158786f"
EXPECTED_TRAINING_PROTOCOL_HEAD = "1228c19cbf85da4ab738c3355c58f946cd6a965c"
EXPECTED_ARCHITECTURE_HEAD = "c689cc3f38f6f642916ee1a702d7de7bd0e43b"
EXPECTED_EXPERIMENT_VERSION = "gate9-contextual-training-execution-v0"
EXPECTED_SEED_STATUS = "G9_CONTEXTUAL_TRAINING_SEED_COMPLETE"
EXPECTED_SOFTWARE = {
    "python": "3.11.9",
    "torch": "2.9.1+cu130",
    "numpy": "2.3.5",
}
SUPPORT_INPUTS = frozenset((0, 1, 2, 4, 8, 16, 32, 64, 128))
TRAIN_EPISODES = 262_144
TRAIN_STEPS = 512
TRAIN_BATCH_SIZE = 512
VALIDATION_EPISODES = 32_768
VALIDATION_COUNTER_START = 1 << 32
VALIDATION_COUNTER_STOP = VALIDATION_COUNTER_START + VALIDATION_EPISODES
LEARNED_PARAMETER_COUNT = 19_649
STATE_TENSOR_COUNT = 17
BASE_LR = 1.0e-3
MIN_LR = 1.0e-4
WARMUP_STEPS = 16
EXACT_ACCURACY_MIN = 0.995
BIT_ACCURACY_MIN = 0.999
CONTEXT_DELTA_MIN = 0.50
STATE_TENSOR_SHAPES = {
    "support_slot_modulation": (9, 24),
    "output_scale": (),
    "pair_projection.weight": (48, 16),
    "pair_projection.bias": (48,),
    "query_projection.weight": (48, 8),
    "query_projection.bias": (48,),
    "support_attention.in_proj_weight": (144, 48),
    "support_attention.in_proj_bias": (144,),
    "support_attention.out_proj.weight": (48, 48),
    "support_attention.out_proj.bias": (48,),
    "support_ff_in.weight": (64, 48),
    "support_ff_in.bias": (64,),
    "support_ff_out.weight": (48, 64),
    "support_ff_out.bias": (48,),
    "query_support_fusion.weight": (24, 96),
    "query_support_fusion.bias": (24,),
    "output_bit_head.weight": (8, 24),
}


class AuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"{label} is not text")
    _require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not lowercase SHA-256",
    )
    return value


def read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON artifact: {path}") from error


def validate_exact_mapping(
    actual: Any,
    expected: dict[str, Any],
    label: str,
) -> None:
    """Validate one JSON object field-by-field with exact Python types."""

    _require(isinstance(actual, dict), f"{label} is not an object")
    actual_keys = set(actual)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    _require(
        not missing and not extra,
        f"{label} field set drifted: missing={missing!r}, extra={extra!r}",
    )
    for field in sorted(expected):
        observed = actual[field]
        required = expected[field]
        _require(
            type(observed) is type(required),
            f"{label} field type drifted: {field}: "
            f"observed={type(observed).__name__}, "
            f"expected={type(required).__name__}",
        )
        _require(
            observed == required,
            f"{label} field value drifted: {field}: "
            f"observed={observed!r}, expected={required!r}",
        )


def iter_jsonl(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                _require(
                    line.endswith("\n"),
                    f"JSONL row lacks newline: {path}:{line_number}",
                )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AuditError(
                        f"invalid JSONL row: {path}:{line_number}"
                    ) from error
                _require(
                    isinstance(row, dict),
                    f"JSONL row is not an object: {path}:{line_number}",
                )
                yield row
    except OSError as error:
        raise AuditError(f"could not read JSONL artifact: {path}") from error


def learning_rate_at_step(step: int) -> float:
    _require(1 <= step <= TRAIN_STEPS, "training step lies outside 1..512")
    if step <= WARMUP_STEPS:
        return BASE_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TRAIN_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (BASE_LR - MIN_LR) * cosine


def verify_manifest(root: pathlib.Path) -> dict[str, str]:
    manifest_path = root / "manifest.sha256"
    _require(manifest_path.is_file(), "manifest.sha256 is missing")
    lines = manifest_path.read_text(encoding="ascii").splitlines()
    parsed: list[tuple[str, str]] = []
    for line in lines:
        parts = line.split("  ", 1)
        _require(len(parts) == 2, "manifest row is malformed")
        digest, relative = parts
        _valid_sha256(digest, f"manifest digest for {relative}")
        _require(relative != "manifest.sha256", "manifest includes itself")
        parsed.append((relative, digest))
    relative_paths = [relative for relative, _ in parsed]
    _require(
        relative_paths == sorted(relative_paths),
        "manifest paths are not sorted",
    )
    _require(
        len(relative_paths) == len(set(relative_paths)),
        "manifest contains a duplicate path",
    )
    entries: dict[str, str] = {}
    for relative, digest in parsed:
        path = root / pathlib.PurePosixPath(relative)
        _require(path.is_file(), f"manifest artifact is missing: {relative}")
        _require(
            sha256_file(path) == digest,
            f"manifest digest mismatch: {relative}",
        )
        entries[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    _require(set(entries) == actual, "manifest file set does not match artifact root")
    return entries


def audit_training_ledger(path: pathlib.Path, seed: int) -> dict[str, Any]:
    rows = 0
    final_loss: float | None = None
    batch_hashes: set[str] = set()
    query_hashes: set[str] = set()
    for expected_step, row in enumerate(iter_jsonl(path), 1):
        rows += 1
        _require(row.get("seed") == seed, "training ledger seed drifted")
        _require(row.get("step") == expected_step, "training steps are not contiguous")
        _require(
            row.get("episodes_seen") == expected_step * TRAIN_BATCH_SIZE,
            "training episodes_seen drifted",
        )
        observed_lr = row.get("learning_rate")
        _require(isinstance(observed_lr, (int, float)), "learning rate is invalid")
        _require(
            math.isclose(
                float(observed_lr),
                learning_rate_at_step(expected_step),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"learning rate drifted at step {expected_step}",
        )
        loss = row.get("loss")
        gradient = row.get("pre_clip_gradient_norm")
        wall = row.get("wall_seconds")
        _require(
            isinstance(loss, (int, float))
            and math.isfinite(float(loss))
            and float(loss) >= 0.0,
            "training loss is invalid",
        )
        _require(
            isinstance(gradient, (int, float))
            and math.isfinite(float(gradient))
            and float(gradient) >= 0.0,
            "gradient norm is invalid",
        )
        _require(
            isinstance(wall, (int, float))
            and math.isfinite(float(wall))
            and float(wall) >= 0.0,
            "training wall time is invalid",
        )
        batch_hash = _valid_sha256(
            row.get("batch_operator_ordinal_sha256"),
            "training batch ordinal hash",
        )
        query_hash = _valid_sha256(
            row.get("batch_query_sha256"),
            "training batch query hash",
        )
        _require(batch_hash not in batch_hashes, "training batch hash repeated")
        batch_hashes.add(batch_hash)
        query_hashes.add(query_hash)
        final_loss = float(loss)
    _require(rows == TRAIN_STEPS, f"training ledger has {rows} rows, expected 512")
    _require(final_loss is not None, "training ledger is empty")
    return {
        "rows": rows,
        "episodes": rows * TRAIN_BATCH_SIZE,
        "final_loss": final_loss,
        "unique_batch_hashes": len(batch_hashes),
        "unique_query_hashes": len(query_hashes),
    }


def audit_validation_ledger(path: pathlib.Path) -> dict[str, Any]:
    seen_ordinals = bytearray(VALIDATION_EPISODES)
    seen_shuffled = bytearray(VALIDATION_EPISODES)
    full_correct = 0
    shuffled_correct = 0
    query_correct = 0
    oracle_correct = 0
    bit_correct = 0
    rows = 0
    for expected_index, row in enumerate(iter_jsonl(path)):
        rows += 1
        _require(
            row.get("episode_index") == expected_index,
            "validation episode indices are not contiguous",
        )
        ordinal = row.get("operator_ordinal")
        counter = row.get("operator_counter")
        shuffled_ordinal = row.get("shuffled_context_operator_ordinal")
        _require(
            isinstance(ordinal, int) and 0 <= ordinal < VALIDATION_EPISODES,
            "validation operator ordinal is invalid",
        )
        _require(not seen_ordinals[ordinal], "validation operator ordinal repeated")
        seen_ordinals[ordinal] = 1
        _require(
            counter == VALIDATION_COUNTER_START + ordinal,
            "validation counter/ordinal identity drifted",
        )
        _require(
            VALIDATION_COUNTER_START <= counter < VALIDATION_COUNTER_STOP,
            "validation counter escaped frozen range",
        )
        _require(
            isinstance(shuffled_ordinal, int)
            and 0 <= shuffled_ordinal < VALIDATION_EPISODES
            and shuffled_ordinal != ordinal,
            "shuffled-context ordinal is not a valid derangement",
        )
        _require(
            not seen_shuffled[shuffled_ordinal],
            "shuffled-context operator ordinal repeated",
        )
        seen_shuffled[shuffled_ordinal] = 1
        values = {
            "query": row.get("query"),
            "answer": row.get("answer"),
            "full": row.get("full_prediction"),
            "shuffled": row.get("shuffled_context_prediction"),
            "query_only": row.get("query_only_prediction"),
            "oracle": row.get("oracle_prediction"),
        }
        for label, value in values.items():
            _require(
                isinstance(value, int) and 0 <= value <= 255,
                f"validation {label} byte is invalid",
            )
        _require(values["query"] not in SUPPORT_INPUTS, "validation query is a support input")
        flags = {
            "full_correct": values["full"] == values["answer"],
            "shuffled_context_correct": values["shuffled"] == values["answer"],
            "query_only_correct": values["query_only"] == values["answer"],
            "oracle_correct": values["oracle"] == values["answer"],
        }
        for field, expected in flags.items():
            _require(row.get(field) is expected, f"validation {field} drifted")
        full_correct += int(flags["full_correct"])
        shuffled_correct += int(flags["shuffled_context_correct"])
        query_correct += int(flags["query_only_correct"])
        oracle_correct += int(flags["oracle_correct"])
        bit_correct += 8 - ((values["full"] ^ values["answer"]).bit_count())
    _require(
        rows == VALIDATION_EPISODES,
        f"validation ledger has {rows} rows, expected 32768",
    )
    _require(sum(seen_ordinals) == VALIDATION_EPISODES, "validation coverage incomplete")
    _require(sum(seen_shuffled) == VALIDATION_EPISODES, "shuffled-context coverage incomplete")
    _require(oracle_correct == VALIDATION_EPISODES, "oracle accuracy is not exactly 1.0")
    return {
        "rows": rows,
        "unique_operator_ordinals": sum(seen_ordinals),
        "unique_shuffled_operator_ordinals": sum(seen_shuffled),
        "full_correct": full_correct,
        "shuffled_correct": shuffled_correct,
        "query_only_correct": query_correct,
        "oracle_correct": oracle_correct,
        "exact_accuracy": full_correct / rows,
        "bit_accuracy": bit_correct / (rows * 8),
        "shuffled_accuracy": shuffled_correct / rows,
        "query_only_accuracy": query_correct / rows,
        "oracle_accuracy": oracle_correct / rows,
    }


def audit_checkpoint(path: pathlib.Path, seed: int) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise AuditError("selected checkpoint could not be loaded safely") from error
    _require(isinstance(payload, dict), "checkpoint payload is not a mapping")
    required = {
        "experiment_version",
        "architecture_head",
        "training_protocol_head",
        "seed",
        "initialization_seed",
        "step",
        "train_episodes",
        "learned_parameter_count",
        "tensor_count",
        "state_dict",
    }
    _require(set(payload) == required, "checkpoint field set drifted")
    _require(payload["experiment_version"] == EXPECTED_EXPERIMENT_VERSION, "checkpoint experiment drifted")
    _require(payload["architecture_head"] == EXPECTED_ARCHITECTURE_HEAD, "checkpoint architecture drifted")
    _require(payload["training_protocol_head"] == EXPECTED_TRAINING_PROTOCOL_HEAD, "checkpoint protocol drifted")
    _require(payload["seed"] == seed, "checkpoint seed drifted")
    _require(payload["initialization_seed"] == 900_900 + seed, "checkpoint initialization seed drifted")
    _require(payload["step"] == TRAIN_STEPS, "checkpoint is not fixed final step")
    _require(payload["train_episodes"] == TRAIN_EPISODES, "checkpoint training coverage drifted")
    _require(payload["learned_parameter_count"] == LEARNED_PARAMETER_COUNT, "checkpoint parameter count drifted")
    _require(payload["tensor_count"] == STATE_TENSOR_COUNT, "checkpoint tensor count drifted")
    state = payload["state_dict"]
    _require(isinstance(state, dict), "checkpoint state_dict is invalid")
    _require(set(state) == set(STATE_TENSOR_SHAPES), "checkpoint tensor names drifted")
    parameters = 0
    for name, shape in STATE_TENSOR_SHAPES.items():
        tensor = state[name]
        _require(isinstance(tensor, torch.Tensor), f"checkpoint tensor is invalid: {name}")
        _require(tuple(tensor.shape) == shape, f"checkpoint tensor shape drifted: {name}")
        _require(tensor.dtype == torch.float32, f"checkpoint tensor dtype drifted: {name}")
        _require(bool(torch.isfinite(tensor).all()), f"checkpoint tensor is non-finite: {name}")
        parameters += tensor.numel()
    _require(parameters == LEARNED_PARAMETER_COUNT, "checkpoint state parameter total drifted")
    return {
        "sha256": sha256_file(path),
        "tensor_count": len(state),
        "learned_parameter_count": parameters,
        "all_finite_float32": True,
    }


def audit_seed_artifact(
    root: pathlib.Path,
    *,
    seed: int,
    expected_summary_sha256: str | None = None,
    expected_validation_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    _require(root.is_dir(), f"artifact root is missing: {root}")
    _require(seed in (0, 1, 2), "seed is outside 0..2")
    seed_root = root / f"seed-{seed}"
    paths = {
        "run_config": root / "run-config.json",
        "git_head": root / "git-head.txt",
        "git_status": root / "git-status.txt",
        "manifest": root / "manifest.sha256",
        "summary": seed_root / "summary.json",
        "training": seed_root / "train-steps.jsonl",
        "validation": seed_root / "validation-per-episode.jsonl",
        "checkpoint": seed_root / "selected-checkpoint.pt",
    }
    for label, path in paths.items():
        _require(path.is_file(), f"required {label} artifact is missing: {path}")

    manifest_entries = verify_manifest(root)
    manifest_hash = sha256_file(paths["manifest"])
    if expected_manifest_sha256 is not None:
        _require(
            manifest_hash == expected_manifest_sha256.lower(),
            "manifest identity differs from terminal",
        )

    head = paths["git_head"].read_text(encoding="ascii").strip()
    status = paths["git_status"].read_text(encoding="utf-8")
    _require(head == EXPECTED_EXECUTION_HEAD, "execution Git head drifted")
    _require(status == "", "execution working tree was not clean")

    config = read_json(paths["run_config"])
    expected_config = {
        "experiment_version": EXPECTED_EXPERIMENT_VERSION,
        "execution_head": EXPECTED_EXECUTION_HEAD,
        "branch": EXPECTED_EXECUTION_BRANCH,
        "seed": seed,
        "training_protocol_head": EXPECTED_TRAINING_PROTOCOL_HEAD,
        "architecture_head": EXPECTED_ARCHITECTURE_HEAD,
        "python": EXPECTED_SOFTWARE["python"],
        "torch": EXPECTED_SOFTWARE["torch"],
        "numpy": EXPECTED_SOFTWARE["numpy"],
        "output_root": str(root),
        "local_test_operator_access": False,
        "graph_test_operator_access": False,
        "scientific_assignment_key_access": False,
    }
    validate_exact_mapping(config, expected_config, "run-config")

    summary_hash = sha256_file(paths["summary"])
    validation_hash = sha256_file(paths["validation"])
    checkpoint_hash = sha256_file(paths["checkpoint"])
    for observed, expected, label in (
        (summary_hash, expected_summary_sha256, "summary"),
        (validation_hash, expected_validation_sha256, "validation ledger"),
        (checkpoint_hash, expected_checkpoint_sha256, "checkpoint"),
    ):
        if expected is not None:
            _require(
                observed == expected.lower(),
                f"{label} identity differs from terminal",
            )

    summary = read_json(paths["summary"])
    _require(isinstance(summary, dict), "summary is not an object")
    for field, expected in (
        ("experiment_version", EXPECTED_EXPERIMENT_VERSION),
        ("scientific_status", EXPECTED_SEED_STATUS),
        ("execution_head", EXPECTED_EXECUTION_HEAD),
        ("training_protocol_head", EXPECTED_TRAINING_PROTOCOL_HEAD),
        ("architecture_head", EXPECTED_ARCHITECTURE_HEAD),
        ("seed", seed),
    ):
        _require(summary.get(field) == expected, f"summary {field} drifted")
    expected_boundaries = {
        "training_performed": True,
        "validation_performed": True,
        "checkpoint_serialized": True,
        "local_test_operator_accessed": False,
        "graph_test_operator_accessed": False,
        "scientific_assignment_key_accessed": False,
        "scientific_test_generated": False,
        "scientific_execution_performed": False,
        "result_classification_performed": False,
    }
    _require(summary.get("boundaries") == expected_boundaries, "summary boundary evidence drifted")

    training = audit_training_ledger(paths["training"], seed)
    validation = audit_validation_ledger(paths["validation"])
    checkpoint = audit_checkpoint(paths["checkpoint"], seed)
    evidence = summary.get("validation_evidence")
    _require(isinstance(evidence, dict), "validation evidence is missing")
    expected_evidence = {
        "seed": seed,
        "initialization_seed": 900_900 + seed,
        "checkpoint_step": TRAIN_STEPS,
        "train_episodes": TRAIN_EPISODES,
        "unique_train_operators": TRAIN_EPISODES,
        "validation_episodes": VALIDATION_EPISODES,
        "unique_validation_operators": VALIDATION_EPISODES,
        "learned_parameter_count": LEARNED_PARAMETER_COUNT,
        "tensor_count": STATE_TENSOR_COUNT,
        "checkpoint_sha256": checkpoint_hash,
        "parameters_finite": True,
        "final_train_loss": training["final_loss"],
        "validation_exact_accuracy": validation["exact_accuracy"],
        "validation_bit_accuracy": validation["bit_accuracy"],
        "shuffled_context_accuracy": validation["shuffled_accuracy"],
        "query_only_accuracy": validation["query_only_accuracy"],
        "oracle_accuracy": validation["oracle_accuracy"],
    }
    for field, expected in expected_evidence.items():
        observed = evidence.get(field)
        if isinstance(expected, float):
            _require(
                isinstance(observed, (int, float))
                and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=0.0),
                f"summary validation evidence drifted: {field}",
            )
        else:
            _require(observed == expected, f"summary validation evidence drifted: {field}")

    admission_passes = (
        validation["exact_accuracy"] >= EXACT_ACCURACY_MIN
        and validation["bit_accuracy"] >= BIT_ACCURACY_MIN
        and validation["exact_accuracy"] - validation["shuffled_accuracy"] > CONTEXT_DELTA_MIN
        and validation["exact_accuracy"] - validation["query_only_accuracy"] > CONTEXT_DELTA_MIN
        and validation["oracle_accuracy"] == 1.0
    )
    _require(evidence.get("admission_passes") is admission_passes, "admission flag drifted")
    artifacts = summary.get("artifacts")
    _require(isinstance(artifacts, dict), "summary artifact map is missing")
    _require(
        artifacts.get("selected_checkpoint_sha256") == checkpoint_hash,
        "summary checkpoint identity drifted",
    )

    required_manifest_bindings = {
        "git-head.txt": paths["git_head"],
        "git-status.txt": paths["git_status"],
        "run-config.json": paths["run_config"],
        f"seed-{seed}/summary.json": paths["summary"],
        f"seed-{seed}/train-steps.jsonl": paths["training"],
        f"seed-{seed}/validation-per-episode.jsonl": paths["validation"],
        f"seed-{seed}/selected-checkpoint.pt": paths["checkpoint"],
    }
    for relative, path in required_manifest_bindings.items():
        _require(
            manifest_entries.get(relative) == sha256_file(path),
            f"manifest binding drifted: {relative}",
        )

    outcome = (
        "G9_CONTEXTUAL_SEED_CHECKPOINT_ADMITTED"
        if admission_passes
        else "G9_CONTEXTUAL_SEED_CHECKPOINT_ADMISSION_FAILED"
    )
    return {
        "audit_version": AUDIT_VERSION,
        "status": "G9_CONTEXTUAL_SEED_AUDIT_COMPLETE",
        "seed": seed,
        "artifact_root": str(root),
        "execution_head": head,
        "seed_outcome": outcome,
        "all_seed_admission_still_possible": admission_passes,
        "scientific_test_generation_allowed": False,
        "training": training,
        "validation": validation,
        "checkpoint": checkpoint,
        "artifact_sha256": {
            "summary": summary_hash,
            "validation_ledger": validation_hash,
            "checkpoint": checkpoint_hash,
            "manifest": manifest_hash,
        },
        "manifest_entries": len(manifest_entries),
        "source_artifact_modified": False,
    }
