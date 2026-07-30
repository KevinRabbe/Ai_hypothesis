"""Independent auditor for the fresh low-scale Gate-7 scale-neutral transition bridge artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_VERSION = "gate7-scale-neutral-transition-bridge-v0"
TRANSITION_CHECKPOINTS = {
    0: {
        "sha256": "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
        "fingerprint": "0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa",
        "training_seed": 0,
    },
    1: {
        "sha256": "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
        "fingerprint": "b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46",
        "training_seed": 1,
    },
    2: {
        "sha256": "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
        "fingerprint": "1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb",
        "training_seed": 2,
    },
}
ORIGINAL_CHECKPOINTS = {
    0: {
        "sha256": "e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590",
        "fingerprint": "e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc",
    },
    1: {
        "sha256": "8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989",
        "fingerprint": "2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c",
    },
    2: {
        "sha256": "103826bbd9451b965eced5134d1674cb8e893f5d3b378cb828312ffdb6fc9a37",
        "fingerprint": "8afaf956f200f41ea914eafdd1b5f151dd303cae1552165c74e12bb8c945af02",
    },
}
TRAINING_GIT_HEAD = "07307650b2bbbfaa09b80e40caa4419ecdda2947"
WORLD_COUNT = 256
BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 2_000
DEPTH = 10
HINT_RELIABILITY = 0.70
POPULATIONS = (128, 256)
NONINFERIORITY_MARGIN = 0.05
PARAMETER_COUNT = 19_649
STAGE_A_SLOTS = 255
STAGE_B_SLOTS = 128
TOTAL_UPDATES = 6_128
TRANSITION_VERSION = "gate7-scale-neutral-scorer-transition-v0"
EXPECTED_CONDITION_COUNT = 21
EXPECTED_PAIR_COUNT = 15


@dataclass(frozen=True, slots=True)
class Gate7TransitionBridgeAudit:
    artifact_valid: bool
    scientific_status: str
    transition_outcome: str | None
    primary_deltas: dict[str, float]
    primary_ci_lows: dict[str, float]
    primary_ci_highs: dict[str, float]
    descriptive_deltas: dict[str, float]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-7 transition bridge result must be one JSON object")
    return payload


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _runtime_seed(world_index: int) -> int:
    return _seed_from_parts("gate7-scale-neutral-transition-bridge-runtime", world_index, DEPTH)


def _bootstrap_ci(
    differences: tuple[int, ...], *, checkpoint: int, population: int, comparison: str
) -> tuple[float, float]:
    rng = random.Random(
        _seed_from_parts(
            "gate7-scale-neutral-transition-bridge-bootstrap",
            checkpoint,
            population,
            comparison,
        )
    )
    estimates = sorted(
        sum(differences[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))],
    )


def _float_equal(expected: float, observed: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _classify(lows: dict[str, float]) -> str:
    required = []
    for checkpoint in (0, 1, 2):
        required.extend(
            (
                lows[f"t{checkpoint}_n128_k16_vs_hash"] > 0.0,
                lows[f"t{checkpoint}_n256_k16_vs_hash"] > 0.0,
                lows[f"t{checkpoint}_n128_k16_vs_global"] > -NONINFERIORITY_MARGIN,
                lows[f"t{checkpoint}_n256_transition_global_vs_original_global"]
                > -NONINFERIORITY_MARGIN,
            )
        )
    if all(required):
        return "GATE7_SCALE_NEUTRAL_TRANSITION_QUALIFIED"
    return "GATE7_SCALE_NEUTRAL_TRANSITION_NOT_QUALIFIED"


def _expected_pairs(checkpoint: int) -> dict[str, tuple[tuple[str, int, str], tuple[str, int, str]]]:
    return {
        f"t{checkpoint}_n128_k16_vs_hash": (
            ("transition", 128, "bounded_score_k16"),
            ("transition", 128, "bounded_hash_k16"),
        ),
        f"t{checkpoint}_n256_k16_vs_hash": (
            ("transition", 256, "bounded_score_k16"),
            ("transition", 256, "bounded_hash_k16"),
        ),
        f"t{checkpoint}_n128_k16_vs_global": (
            ("transition", 128, "bounded_score_k16"),
            ("transition", 128, "global_score"),
        ),
        f"t{checkpoint}_n256_transition_global_vs_original_global": (
            ("transition", 256, "global_score"),
            ("original", 256, "global_score"),
        ),
        f"t{checkpoint}_n256_k16_vs_global": (
            ("transition", 256, "bounded_score_k16"),
            ("transition", 256, "global_score"),
        ),
    }


def audit_gate7_scale_neutral_transition_bridge(path: Path) -> Gate7TransitionBridgeAudit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate7TransitionBridgeAudit(False, "INVALID_ARTIFACT", None, {}, {}, {}, {}, (str(exc),))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected experiment version")
    if payload.get("scientific_status") != "FRESH_LOW_SCALE_TRANSITION_BRIDGE_EVIDENCE":
        errors.append("unexpected scientific status")
    if payload.get("training_performed") is not False:
        errors.append("bridge runner must perform no training")
    if payload.get("checkpoint_selection_performed") is not False:
        errors.append("bridge runner must not select checkpoints")
    if payload.get("high_scale_gate7_opened") is not False:
        errors.append("high-scale Gate-7 must remain closed")
    if payload.get("world_count") != WORLD_COUNT:
        errors.append("world count differs from frozen 256")
    if payload.get("evaluation_batch_size") != BATCH_SIZE:
        errors.append("batch size differs from frozen 64")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("bootstrap sample count differs from frozen 2000")
    if payload.get("depth") != DEPTH:
        errors.append("bridge depth differs from frozen 10")
    if not _float_equal(HINT_RELIABILITY, payload.get("hint_reliability")):
        errors.append("hint reliability differs from frozen 0.70")
    if payload.get("populations") != list(POPULATIONS):
        errors.append("bridge populations differ from N128/N256")
    if not _float_equal(NONINFERIORITY_MARGIN, payload.get("noninferiority_margin")):
        errors.append("non-inferiority margin differs from frozen 0.05")

    transition_rows = payload.get("transition_checkpoints")
    if not isinstance(transition_rows, list) or len(transition_rows) != 3:
        errors.append("artifact must identify exactly three transition checkpoints")
        transition_rows = []
    seen_transition: set[int] = set()
    for row in transition_rows:
        if not isinstance(row, dict) or row.get("checkpoint_index") not in TRANSITION_CHECKPOINTS:
            errors.append("invalid transition checkpoint identity")
            continue
        checkpoint = int(row["checkpoint_index"])
        if checkpoint in seen_transition:
            errors.append(f"duplicate transition checkpoint {checkpoint}")
            continue
        seen_transition.add(checkpoint)
        expected = TRANSITION_CHECKPOINTS[checkpoint]
        if str(row.get("checkpoint_sha256", "")).lower() != expected["sha256"]:
            errors.append(f"transition checkpoint {checkpoint} SHA256 mismatch")
        if row.get("parameter_fingerprint") != expected["fingerprint"]:
            errors.append(f"transition checkpoint {checkpoint} fingerprint mismatch")
        if row.get("training_seed") != expected["training_seed"]:
            errors.append(f"transition checkpoint {checkpoint} seed mismatch")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"transition checkpoint {checkpoint} parameter count mismatch")
        if row.get("transition_version") != TRANSITION_VERSION:
            errors.append(f"transition checkpoint {checkpoint} version mismatch")
        if row.get("training_git_head") != TRAINING_GIT_HEAD:
            errors.append(f"transition checkpoint {checkpoint} training head mismatch")
    if seen_transition != set(TRANSITION_CHECKPOINTS):
        errors.append("transition checkpoint identity set is incomplete")

    original_rows = payload.get("original_checkpoints")
    if not isinstance(original_rows, list) or len(original_rows) != 3:
        errors.append("artifact must identify exactly three original checkpoints")
        original_rows = []
    seen_original: set[int] = set()
    for row in original_rows:
        if not isinstance(row, dict) or row.get("checkpoint_index") not in ORIGINAL_CHECKPOINTS:
            errors.append("invalid original checkpoint identity")
            continue
        checkpoint = int(row["checkpoint_index"])
        if checkpoint in seen_original:
            errors.append(f"duplicate original checkpoint {checkpoint}")
            continue
        seen_original.add(checkpoint)
        expected = ORIGINAL_CHECKPOINTS[checkpoint]
        if str(row.get("checkpoint_sha256", "")).lower() != expected["sha256"]:
            errors.append(f"original checkpoint {checkpoint} SHA256 mismatch")
        if row.get("parameter_fingerprint") != expected["fingerprint"]:
            errors.append(f"original checkpoint {checkpoint} fingerprint mismatch")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"original checkpoint {checkpoint} parameter count mismatch")
    if seen_original != set(ORIGINAL_CHECKPOINTS):
        errors.append("original checkpoint identity set is incomplete")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITION_COUNT:
        errors.append("result must contain exactly 21 frozen conditions")
        conditions = []
    index: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    expected_world_indices = list(range(WORLD_COUNT))
    expected_runtime_seeds = [_runtime_seed(index) for index in range(WORLD_COUNT)]
    for row in conditions:
        if not isinstance(row, dict):
            errors.append("condition is not an object")
            continue
        key = (
            row.get("checkpoint_index"),
            row.get("checkpoint_family"),
            row.get("population_size"),
            row.get("mode"),
        )
        if key in index:
            errors.append(f"duplicate condition {key}")
            continue
        index[key] = row
        checkpoint, family, population, mode = key
        expected_condition = (
            checkpoint in (0, 1, 2)
            and (
                family == "transition"
                and population in POPULATIONS
                and mode in ("global_score", "bounded_score_k16", "bounded_hash_k16")
                or family == "original" and population == 256 and mode == "global_score"
            )
        )
        if not expected_condition:
            errors.append(f"unexpected condition {key}")
            continue
        if row.get("world_indices") != expected_world_indices:
            errors.append(f"condition {key} world indices differ from frozen namespace")
        if row.get("runtime_seeds") != expected_runtime_seeds:
            errors.append(f"condition {key} runtime seeds differ from frozen namespace")
        covered = row.get("covered_by_world")
        if not isinstance(covered, list) or len(covered) != WORLD_COUNT or any(type(x) is not bool for x in covered):
            errors.append(f"condition {key} has invalid coverage vector")
            continue
        expected_rate = sum(int(value) for value in covered) / WORLD_COUNT
        if not _float_equal(expected_rate, row.get("coverage_rate")):
            errors.append(f"condition {key} coverage rate does not match its vector")
        if row.get("stage_a_parent_slots") != STAGE_A_SLOTS:
            errors.append(f"condition {key} Stage-A work mismatch")
        if row.get("stage_b_parent_slots") != STAGE_B_SLOTS:
            errors.append(f"condition {key} Stage-B work mismatch")
        if row.get("total_learned_updates_per_world") != TOTAL_UPDATES:
            errors.append(f"condition {key} learned-work mismatch")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"condition {key} parameter count mismatch")
        expected_fingerprint = (
            TRANSITION_CHECKPOINTS[int(checkpoint)]["fingerprint"]
            if family == "transition"
            else ORIGINAL_CHECKPOINTS[int(checkpoint)]["fingerprint"]
        )
        if row.get("parameter_fingerprint") != expected_fingerprint:
            errors.append(f"condition {key} parameter fingerprint mismatch")

    expected_keys = {
        (checkpoint, "transition", population, mode)
        for checkpoint in (0, 1, 2)
        for population in POPULATIONS
        for mode in ("global_score", "bounded_score_k16", "bounded_hash_k16")
    } | {(checkpoint, "original", 256, "global_score") for checkpoint in (0, 1, 2)}
    if set(index) != expected_keys:
        errors.append("condition matrix differs from the frozen 21-cell bridge")

    pair_rows = payload.get("paired_summaries")
    if not isinstance(pair_rows, list) or len(pair_rows) != EXPECTED_PAIR_COUNT:
        errors.append("result must contain exactly 15 paired summaries")
        pair_rows = []
    pair_index: dict[str, dict[str, Any]] = {}
    primary_deltas: dict[str, float] = {}
    primary_lows: dict[str, float] = {}
    primary_highs: dict[str, float] = {}
    descriptive_deltas: dict[str, float] = {}
    for row in pair_rows:
        if not isinstance(row, dict) or not isinstance(row.get("comparison"), str):
            errors.append("invalid paired summary")
            continue
        comparison = str(row["comparison"])
        if comparison in pair_index:
            errors.append(f"duplicate paired summary {comparison}")
            continue
        pair_index[comparison] = row

    expected_pair_names: set[str] = set()
    for checkpoint in (0, 1, 2):
        for comparison, (treatment_key, reference_key) in _expected_pairs(checkpoint).items():
            expected_pair_names.add(comparison)
            row = pair_index.get(comparison)
            if row is None:
                errors.append(f"missing paired summary {comparison}")
                continue
            treatment = index.get((checkpoint, *treatment_key))
            reference = index.get((checkpoint, *reference_key))
            if treatment is None or reference is None:
                errors.append(f"paired summary {comparison} references missing conditions")
                continue
            a = treatment.get("covered_by_world")
            b = reference.get("covered_by_world")
            if not isinstance(a, list) or not isinstance(b, list):
                errors.append(f"paired summary {comparison} has unavailable coverage vectors")
                continue
            differences = tuple(int(bool(x)) - int(bool(y)) for x, y in zip(a, b, strict=True))
            delta = sum(differences) / WORLD_COUNT
            low, high = _bootstrap_ci(
                differences,
                checkpoint=checkpoint,
                population=int(treatment_key[1]),
                comparison=comparison,
            )
            if row.get("checkpoint_index") != checkpoint:
                errors.append(f"paired summary {comparison} checkpoint mismatch")
            if row.get("population_size") != treatment_key[1]:
                errors.append(f"paired summary {comparison} population mismatch")
            if row.get("treatment_family") != treatment_key[0] or row.get("reference_family") != reference_key[0]:
                errors.append(f"paired summary {comparison} family mismatch")
            if row.get("treatment_mode") != treatment_key[2] or row.get("reference_mode") != reference_key[2]:
                errors.append(f"paired summary {comparison} mode mismatch")
            if not _float_equal(delta, row.get("coverage_delta")):
                errors.append(f"paired summary {comparison} delta mismatch")
            if not _float_equal(low, row.get("bootstrap_ci_low")):
                errors.append(f"paired summary {comparison} CI low mismatch")
            if not _float_equal(high, row.get("bootstrap_ci_high")):
                errors.append(f"paired summary {comparison} CI high mismatch")
            if comparison.endswith("_n256_k16_vs_global"):
                descriptive_deltas[comparison] = delta
            else:
                primary_deltas[comparison] = delta
                primary_lows[comparison] = low
                primary_highs[comparison] = high
    if set(pair_index) != expected_pair_names:
        errors.append("paired-summary set differs from the frozen bridge")

    expected_outcome = None
    if len(primary_lows) == 12:
        expected_outcome = _classify(primary_lows)
        if payload.get("transition_outcome") != expected_outcome:
            errors.append("transition outcome does not match the twelve frozen primary criteria")
    else:
        errors.append("could not reconstruct all twelve primary criteria")
    stored_lows = payload.get("primary_ci_lows")
    if stored_lows != primary_lows:
        errors.append("stored primary CI lows differ from independent recomputation")

    status = str(payload.get("scientific_status", "INVALID_ARTIFACT"))
    return Gate7TransitionBridgeAudit(
        artifact_valid=not errors,
        scientific_status=status if not errors else "INVALID_ARTIFACT",
        transition_outcome=expected_outcome if not errors else None,
        primary_deltas=primary_deltas,
        primary_ci_lows=primary_lows,
        primary_ci_highs=primary_highs,
        descriptive_deltas=descriptive_deltas,
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_gate7_scale_neutral_transition_bridge(args.result)
    args.output.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
