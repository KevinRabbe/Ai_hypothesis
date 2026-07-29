"""Independent structural/scientific auditor for frozen Gate-2 confirmation artifacts.

A scientifically negative confirmation is a valid artifact.  Structural/provenance corruption is
reported separately as an invalid audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROTOCOL = "gate2-persistent-state-confirmation-v0"
EXPERIMENT_VERSION = "gate2-persistent-state-confirmation-v0"
TRAINING_SEEDS = (3, 4, 5)
ENTITY_COUNTS = (16, 64, 256)
WIDTHS = {16: (1, 4, 16), 64: (1, 4, 16, 64), 256: (1, 4, 16, 64, 256)}
MODES = ("stable_persistent", "reshuffled_locality", "reset_state")
WORLD_COUNT = 512
EVAL_BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 2_000
CONFIRMATION_WORLD_START = 2 << 30
EXPECTED_CONDITION_COUNT = 36
EXPECTED_PAIRED_COUNT = 33


@dataclass(frozen=True, slots=True)
class ConfirmationAudit:
    artifact_valid: bool
    capability_confirmation_passed: bool | None
    seed_passes: dict[int, bool]
    primary_ci_lows: dict[int, dict[str, float]]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _primary_key(row: dict[str, Any]) -> str | None:
    comparison = row.get("comparison")
    c = row.get("entity_count")
    w = row.get("treatment_width")
    if comparison == "stable_width_vs_width1" and (c, w) == (64, 64):
        return "c64_w64_vs_w1"
    if comparison == "stable_width_vs_width1" and (c, w) == (256, 256):
        return "c256_w256_vs_w1"
    if comparison == "stable_vs_reshuffled" and (c, w) == (256, 256):
        return "c256_stable_vs_reshuffled"
    if comparison == "stable_vs_reset" and (c, w) == (256, 256):
        return "c256_stable_vs_reset"
    return None


def audit_confirmation_root(root: Path) -> ConfirmationAudit:
    root = root.resolve()
    errors: list[str] = []
    seed_passes: dict[int, bool] = {}
    primary_ci_lows: dict[int, dict[str, float]] = {}

    suite_path = root / "confirmation-suite.json"
    config_path = root / "run-config.json"
    if not suite_path.is_file():
        errors.append("missing confirmation-suite.json")
    if not config_path.is_file():
        errors.append("missing run-config.json")
    if errors:
        return ConfirmationAudit(False, None, seed_passes, primary_ci_lows, tuple(errors))

    try:
        suite = _load_json(suite_path)
        config = _load_json(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ConfirmationAudit(False, None, seed_passes, primary_ci_lows, (str(exc),))

    expected_config = {
        "protocol": PROTOCOL,
        "scientific_status": "FROZEN_CONFIRMATION",
        "training_seeds": [3, 4, 5],
        "steps": 1000,
        "training_batch_size": 32,
        "state_width": 64,
        "query_width": 24,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "evaluation_world_count": WORLD_COUNT,
        "evaluation_batch_size": EVAL_BATCH_SIZE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "device": "cuda",
        "idle_machine_attested": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(f"run-config {key!r} expected {expected!r}, got {config.get(key)!r}")

    if suite.get("protocol") != PROTOCOL:
        errors.append("suite protocol mismatch")
    if suite.get("confirmation_training_seeds") != [3, 4, 5]:
        errors.append("suite confirmation training seeds must be exactly 3/4/5")
    if suite.get("gate2_overall_verdict") != "NOT_ASSIGNED_UNTIL_RESOURCE_PROTOCOL_COMPLETE":
        errors.append("suite assigned an overall Gate-2 verdict before resource completion")

    suite_seed_rows = suite.get("seeds")
    if not isinstance(suite_seed_rows, list) or len(suite_seed_rows) != 3:
        errors.append("suite must contain exactly three seed summaries")
        suite_seed_rows = []
    suite_by_seed = {
        row.get("training_seed"): row
        for row in suite_seed_rows
        if isinstance(row, dict) and isinstance(row.get("training_seed"), int)
    }
    if set(suite_by_seed) != set(TRAINING_SEEDS):
        errors.append("suite seed summaries must be exactly seeds 3/4/5")

    for seed in TRAINING_SEEDS:
        seed_root = root / f"seed_{seed}"
        result_path = seed_root / "gate2-confirmation.json"
        checkpoint_path = seed_root / "gate2-confirmation-checkpoint.pt"
        runtime_path = seed_root / "runtime.json"
        for path in (result_path, checkpoint_path, runtime_path):
            if not path.is_file():
                errors.append(f"seed {seed}: missing {path.name}")
        if not result_path.is_file() or not checkpoint_path.is_file() or not runtime_path.is_file():
            continue

        try:
            result = _load_json(result_path)
            runtime = _load_json(runtime_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"seed {seed}: {exc}")
            continue

        if result.get("experiment_version") != EXPERIMENT_VERSION:
            errors.append(f"seed {seed}: experiment version mismatch")
        if result.get("evaluation_split") != "confirmation" or result.get("confirmation_opened") is not True:
            errors.append(f"seed {seed}: artifact is not an opened confirmation result")
        if result.get("evaluation_world_count") != WORLD_COUNT:
            errors.append(f"seed {seed}: confirmation world count mismatch")
        if result.get("evaluation_batch_size") != EVAL_BATCH_SIZE:
            errors.append(f"seed {seed}: evaluation batch size mismatch")
        if result.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
            errors.append(f"seed {seed}: bootstrap sample count mismatch")
        if result.get("scientific_status") != "CONFIRMATION_SEED_RESULT":
            errors.append(f"seed {seed}: scientific_status mismatch")
        if result.get("gate2_verdict") != "NOT_ASSIGNED_UNTIL_ALL_SEEDS_AND_RESOURCE_PROTOCOL_COMPLETE":
            errors.append(f"seed {seed}: result assigned a premature Gate-2 verdict")

        training = result.get("training", {})
        training_config = result.get("training_config", {})
        model_config = training_config.get("model", {}) if isinstance(training_config, dict) else {}
        expected_training = {
            "training_seed": seed,
            "steps": 1000,
            "examples_seen": 32000,
            "stable_training_condition_count": 12,
        }
        for key, expected in expected_training.items():
            if training.get(key) != expected:
                errors.append(f"seed {seed}: training {key} mismatch")
        for key, expected in {
            "steps": 1000,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
        }.items():
            if training_config.get(key) != expected:
                errors.append(f"seed {seed}: training_config {key} mismatch")
        if model_config != {"state_width": 64, "query_width": 24}:
            errors.append(f"seed {seed}: model config mismatch")

        fingerprint = training.get("parameter_fingerprint")
        parameter_count = training.get("learned_parameter_count")
        if not isinstance(fingerprint, str) or not fingerprint:
            errors.append(f"seed {seed}: missing parameter fingerprint")
        if not isinstance(parameter_count, int) or parameter_count <= 0:
            errors.append(f"seed {seed}: invalid parameter count")

        checkpoint_hash = _sha256(checkpoint_path)
        if result.get("checkpoint_sha256") != checkpoint_hash:
            errors.append(f"seed {seed}: checkpoint SHA-256 mismatch")
        if runtime.get("checkpoint_sha256") != checkpoint_hash:
            errors.append(f"seed {seed}: runtime checkpoint SHA-256 mismatch")
        if runtime.get("scientific_status") != "FROZEN_GATE2_CONFIRMATION_SEED":
            errors.append(f"seed {seed}: runtime scientific_status mismatch")

        suite_row = suite_by_seed.get(seed)
        if suite_row:
            if suite_row.get("result_sha256") != _sha256(result_path):
                errors.append(f"seed {seed}: suite result SHA-256 mismatch")
            if suite_row.get("checkpoint_sha256") != checkpoint_hash:
                errors.append(f"seed {seed}: suite checkpoint SHA-256 mismatch")
            if suite_row.get("parameter_fingerprint") != fingerprint:
                errors.append(f"seed {seed}: suite fingerprint mismatch")

        conditions = result.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITION_COUNT:
            errors.append(f"seed {seed}: expected {EXPECTED_CONDITION_COUNT} conditions")
            conditions = []
        condition_index: dict[tuple[int, int, str], dict[str, Any]] = {}
        for row in conditions:
            if not isinstance(row, dict):
                errors.append(f"seed {seed}: non-object condition row")
                continue
            key = (row.get("entity_count"), row.get("width"), row.get("mode"))
            if key in condition_index:
                errors.append(f"seed {seed}: duplicate condition {key}")
            condition_index[key] = row

        expected_keys = {
            (c, w, mode)
            for c in ENTITY_COUNTS
            for w in WIDTHS[c]
            for mode in MODES
        }
        if set(condition_index) != expected_keys:
            errors.append(f"seed {seed}: condition matrix is not canonical")

        for c in ENTITY_COUNTS:
            expected_world_seeds = list(range(CONFIRMATION_WORLD_START, CONFIRMATION_WORLD_START + WORLD_COUNT))
            reference_worlds: list[int] | None = None
            for w in WIDTHS[c]:
                for mode in MODES:
                    row = condition_index.get((c, w, mode))
                    if not row:
                        continue
                    if row.get("world_count") != WORLD_COUNT:
                        errors.append(f"seed {seed}: C{c}/W{w}/{mode} world_count mismatch")
                    worlds = row.get("world_seeds")
                    if worlds != expected_world_seeds:
                        errors.append(f"seed {seed}: C{c}/W{w}/{mode} confirmation world seeds mismatch")
                    if reference_worlds is None:
                        reference_worlds = worlds
                    elif worlds != reference_worlds:
                        errors.append(f"seed {seed}: C{c} world ordering differs across conditions")
                    if row.get("learned_updates_per_world") != 8 * c:
                        errors.append(f"seed {seed}: C{c}/W{w}/{mode} learned work mismatch")
                    if row.get("inspected_entities_per_world") != c:
                        errors.append(f"seed {seed}: C{c}/W{w}/{mode} entity coverage mismatch")
                    if row.get("inspected_observations_per_world") != 8 * c:
                        errors.append(f"seed {seed}: C{c}/W{w}/{mode} observation count mismatch")
                    if row.get("collision_load") != c // w:
                        errors.append(f"seed {seed}: C{c}/W{w}/{mode} collision load mismatch")
                    if row.get("learned_parameter_count") != parameter_count:
                        errors.append(f"seed {seed}: condition parameter count mismatch")
                    if row.get("parameter_fingerprint") != fingerprint:
                        errors.append(f"seed {seed}: condition fingerprint mismatch")
                    solved = row.get("solved_by_world")
                    if not isinstance(solved, list) or len(solved) != WORLD_COUNT:
                        errors.append(f"seed {seed}: C{c}/W{w}/{mode} solved vector mismatch")

        paired = result.get("paired_summaries")
        if not isinstance(paired, list) or len(paired) != EXPECTED_PAIRED_COUNT:
            errors.append(f"seed {seed}: expected {EXPECTED_PAIRED_COUNT} paired summaries")
            paired = []

        # Width-1 stable vs reshuffled must be an exact identity in every entity tier.
        for c in ENTITY_COUNTS:
            identities = [
                row for row in paired
                if row.get("comparison") == "stable_vs_reshuffled"
                and row.get("entity_count") == c
                and row.get("treatment_width") == 1
            ]
            if len(identities) != 1:
                errors.append(f"seed {seed}: missing width-1 identity at C{c}")
                continue
            row = identities[0]
            if not (
                row.get("exact_solve_delta") == 0.0
                and row.get("treatment_only") == 0
                and row.get("reference_only") == 0
                and row.get("bootstrap_ci_low") == 0.0
                and row.get("bootstrap_ci_high") == 0.0
            ):
                errors.append(f"seed {seed}: width-1 identity failed at C{c}")

        primaries: dict[str, dict[str, Any]] = {}
        for row in paired:
            key = _primary_key(row)
            if key is not None:
                if key in primaries:
                    errors.append(f"seed {seed}: duplicate primary comparison {key}")
                primaries[key] = row
        expected_primary_keys = {
            "c64_w64_vs_w1",
            "c256_w256_vs_w1",
            "c256_stable_vs_reshuffled",
            "c256_stable_vs_reset",
        }
        if set(primaries) != expected_primary_keys:
            errors.append(f"seed {seed}: primary comparison set mismatch")

        primary_result_rows = result.get("primary_comparisons")
        if not isinstance(primary_result_rows, list) or len(primary_result_rows) != 4:
            errors.append(f"seed {seed}: result primary_comparisons must contain four rows")
            primary_result_rows = []
        declared_by_key = {_primary_key(row): row for row in primary_result_rows if isinstance(row, dict)}

        lows: dict[str, float] = {}
        recomputed_seed_pass = True
        for key in sorted(expected_primary_keys):
            row = primaries.get(key)
            if row is None:
                recomputed_seed_pass = False
                continue
            low = row.get("bootstrap_ci_low")
            high = row.get("bootstrap_ci_high")
            delta = row.get("exact_solve_delta")
            if not all(isinstance(value, (int, float)) for value in (low, high, delta)):
                errors.append(f"seed {seed}: malformed primary values for {key}")
                recomputed_seed_pass = False
                continue
            lows[key] = float(low)
            passed = float(low) > 0.0
            recomputed_seed_pass = recomputed_seed_pass and passed
            declared = declared_by_key.get(key)
            if declared is None:
                errors.append(f"seed {seed}: missing declared primary row {key}")
            else:
                if declared.get("bootstrap_ci_low") != low or declared.get("bootstrap_ci_high") != high:
                    errors.append(f"seed {seed}: declared primary CI differs from paired summary for {key}")
                if declared.get("exact_solve_delta") != delta:
                    errors.append(f"seed {seed}: declared primary delta differs for {key}")
                if declared.get("passed") is not passed:
                    errors.append(f"seed {seed}: declared primary pass differs for {key}")

        primary_ci_lows[seed] = lows
        recomputed_seed_pass = recomputed_seed_pass and result.get("width1_identity_passed") is True
        seed_passes[seed] = recomputed_seed_pass
        if result.get("seed_passed") is not recomputed_seed_pass:
            errors.append(f"seed {seed}: declared seed_passed differs from independent recomputation")
        if runtime.get("seed_passed") is not recomputed_seed_pass:
            errors.append(f"seed {seed}: runtime seed_passed differs from independent recomputation")
        if suite_row and suite_row.get("seed_passed") is not recomputed_seed_pass:
            errors.append(f"seed {seed}: suite seed_passed differs from independent recomputation")

    if errors:
        return ConfirmationAudit(False, None, seed_passes, primary_ci_lows, tuple(errors))

    capability_passed = len(seed_passes) == 3 and all(seed_passes.get(seed, False) for seed in TRAINING_SEEDS)
    if suite.get("capability_confirmation_passed") is not capability_passed:
        errors.append("suite capability_confirmation_passed differs from independent recomputation")
        return ConfirmationAudit(False, None, seed_passes, primary_ci_lows, tuple(errors))

    return ConfirmationAudit(True, capability_passed, seed_passes, primary_ci_lows, ())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Gate-2 confirmation_v0 output root")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = audit_confirmation_root(args.root)
    payload = audit.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if audit.artifact_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
