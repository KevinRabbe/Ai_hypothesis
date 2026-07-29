"""Independent auditor for Gate-3 v2 ambiguity-frontier development artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_VERSION = "gate3-v2-ambiguity-frontier-v0"
CHECKPOINTS = {
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
TIERS = ("A60", "A55")
RELIABILITY = {"A60": 0.60, "A55": 0.55}
CONDITIONS = (
    (1, "stable_reserve"),
    (16, "stable_reserve"),
    (64, "stable_reserve"),
    (256, "stable_reserve"),
    (256, "collapsed_diversity"),
    (256, "reshuffled_continuity"),
)
PAIR_SPECS = (
    ("stable_l256_vs_l64", 256, "stable_reserve", 64, "stable_reserve"),
    ("stable_l64_vs_l16", 64, "stable_reserve", 16, "stable_reserve"),
    ("stable_l256_vs_l1", 256, "stable_reserve", 1, "stable_reserve"),
    ("stable_l256_vs_collapsed", 256, "stable_reserve", 256, "collapsed_diversity"),
    ("stable_l256_vs_reshuffled", 256, "stable_reserve", 256, "reshuffled_continuity"),
)
WORLD_COUNT = 256
BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 2_000
DEPTH = 10
SEARCH_ROUNDS = 256
TOTAL_UPDATES = 4_096
PARAMETER_COUNT = 19_649
EXPECTED_CONDITIONS = 36
EXPECTED_PAIRS = 30


@dataclass(frozen=True, slots=True)
class Gate3V2FrontierAudit:
    artifact_valid: bool
    scientific_status: str
    directional_outcome: str | None
    frontier_deltas: dict[str, float]
    frontier_ci_lows: dict[str, float]
    frontier_ci_highs: dict[str, float]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-3 v2 result must be one JSON object")
    return payload


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _runtime_seed(world_index: int, tier: str) -> int:
    return _seed_from_parts("gate3-v2-frontier-development-runtime", tier, world_index, DEPTH)


def _bootstrap_ci(differences: tuple[int, ...], *, checkpoint: int, tier: str, comparison: str) -> tuple[float, float]:
    rng = random.Random(_seed_from_parts("gate3-v2-frontier-bootstrap", checkpoint, tier, comparison))
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return (
        estimates[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))],
        estimates[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))],
    )


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("comparison"),
        row.get("checkpoint_index"),
        row.get("tier"),
        row.get("treatment_capacity"),
        row.get("treatment_mode"),
        row.get("reference_capacity"),
        row.get("reference_mode"),
    )


def _recompute_pair(
    *,
    comparison: str,
    checkpoint: int,
    tier: str,
    treatment: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    if treatment.get("world_indices") != reference.get("world_indices"):
        raise ValueError("paired conditions use different world indices")
    a = treatment.get("covered_by_world")
    b = reference.get("covered_by_world")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != WORLD_COUNT or len(b) != WORLD_COUNT:
        raise ValueError("paired coverage vectors must each contain 256 worlds")
    pairs = tuple(zip(a, b, strict=True))
    treatment_only = sum(int(bool(x) and not bool(y)) for x, y in pairs)
    reference_only = sum(int(bool(y) and not bool(x)) for x, y in pairs)
    both = sum(int(bool(x) and bool(y)) for x, y in pairs)
    neither = WORLD_COUNT - treatment_only - reference_only - both
    differences = tuple(int(bool(x)) - int(bool(y)) for x, y in pairs)
    low, high = _bootstrap_ci(differences, checkpoint=checkpoint, tier=tier, comparison=comparison)
    return {
        "comparison": comparison,
        "checkpoint_index": checkpoint,
        "tier": tier,
        "treatment_capacity": treatment["reserve_capacity"],
        "treatment_mode": treatment["mode"],
        "reference_capacity": reference["reserve_capacity"],
        "reference_mode": reference["mode"],
        "world_count": WORLD_COUNT,
        "treatment_only": treatment_only,
        "reference_only": reference_only,
        "both_covered": both,
        "neither_covered": neither,
        "coverage_delta": sum(differences) / WORLD_COUNT,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }


def _float_equal(expected: float, observed: Any) -> bool:
    try:
        return math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _metric_key(checkpoint: int, tier: str, comparison: str) -> str:
    return f"c{checkpoint}_{tier.lower()}_{comparison}"


def _classify(lows: dict[str, float], deltas: dict[str, float]) -> str:
    control_keys = [
        _metric_key(checkpoint, tier, comparison)
        for checkpoint in CHECKPOINTS
        for tier in TIERS
        for comparison in ("stable_l256_vs_collapsed", "stable_l256_vs_reshuffled")
    ]
    if any(lows[key] <= 0.0 for key in control_keys):
        return "V2_F4_MECHANISM_DEGRADES_UNDER_AMBIGUITY"

    frontier = {
        tier: [
            _metric_key(checkpoint, tier, "stable_l256_vs_l64")
            for checkpoint in CHECKPOINTS
        ]
        for tier in TIERS
    }
    a60_all = all(lows[key] > 0.0 for key in frontier["A60"])
    a55_all = all(lows[key] > 0.0 for key in frontier["A55"])

    if a60_all and a55_all:
        return "V2_F2_ROBUST_BEYOND_L64_EXTENSION"
    if a55_all and not a60_all:
        return "V2_F1_EXTENSION_AT_A55_ONLY"

    if all(lows[key] <= 0.0 for tier in TIERS for key in frontier[tier]):
        return "V2_F0_NO_BEYOND_L64_EXTENSION"

    for tier in TIERS:
        tier_keys = frontier[tier]
        significant = [lows[key] > 0.0 for key in tier_keys]
        signs = [math.copysign(1.0, deltas[key]) if deltas[key] != 0.0 else 0.0 for key in tier_keys]
        if len(set(significant)) > 1 or len(set(signs)) > 1:
            return "V2_F3_CHECKPOINT_SENSITIVE_FRONTIER"

    return "V2_MIXED_FRONTIER_PATTERN"


def audit_gate3_v2_frontier(path: Path) -> Gate3V2FrontierAudit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate3V2FrontierAudit(False, "INVALID_ARTIFACT", None, {}, {}, {}, (str(exc),))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected experiment version")
    if payload.get("scientific_status") != "DEVELOPMENT_ONLY_NO_GATE_VERDICT":
        errors.append("unexpected scientific status")
    if payload.get("confirmation_opened") is not False:
        errors.append("confirmation must remain closed")
    if payload.get("scientific_decision") != "DEVELOPMENT_ONLY_NOT_ASSIGNED":
        errors.append("development artifact attempted to assign a Gate verdict")
    if payload.get("world_count_per_tier") != WORLD_COUNT:
        errors.append("world count differs from frozen 256")
    if payload.get("evaluation_batch_size") != BATCH_SIZE:
        errors.append("evaluation batch size differs from frozen 64")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("bootstrap sample count differs from frozen 2000")
    if payload.get("depth") != DEPTH or payload.get("search_rounds") != SEARCH_ROUNDS:
        errors.append("depth/search rounds differ from frozen frontier workload")
    if payload.get("total_learned_updates_per_world") != TOTAL_UPDATES:
        errors.append("top-level learned-work total differs from 4096")
    if payload.get("hint_reliability") != RELIABILITY:
        errors.append("ambiguity tiers differ from frozen A60/A55 values")

    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        errors.append("artifact must identify exactly three frozen checkpoints")
        checkpoints = []
    checkpoint_index: dict[int, dict[str, Any]] = {}
    for row in checkpoints:
        if not isinstance(row, dict) or row.get("checkpoint_index") not in CHECKPOINTS:
            errors.append("invalid checkpoint identity row")
            continue
        index = int(row["checkpoint_index"])
        if index in checkpoint_index:
            errors.append(f"duplicate checkpoint identity {index}")
            continue
        checkpoint_index[index] = row
        expected = CHECKPOINTS[index]
        if str(row.get("checkpoint_sha256", "")).lower() != expected["sha256"]:
            errors.append(f"checkpoint {index} SHA256 differs from frozen identity")
        if row.get("parameter_fingerprint") != expected["fingerprint"]:
            errors.append(f"checkpoint {index} parameter fingerprint differs from frozen identity")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"checkpoint {index} parameter count differs from 19,649")
    if set(checkpoint_index) != set(CHECKPOINTS):
        errors.append("checkpoint identity set is incomplete")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITIONS:
        errors.append("result must contain exactly 36 conditions")
        conditions = []
    index: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    expected_world_indices = list(range(WORLD_COUNT))
    for row in conditions:
        if not isinstance(row, dict):
            errors.append("condition is not an object")
            continue
        key = (
            row.get("checkpoint_index"),
            row.get("tier"),
            row.get("reserve_capacity"),
            row.get("mode"),
        )
        if key in index:
            errors.append(f"duplicate condition {key}")
            continue
        index[key] = row
        checkpoint, tier, capacity, mode = key
        if checkpoint not in CHECKPOINTS or tier not in TIERS or (capacity, mode) not in CONDITIONS:
            errors.append(f"unexpected condition {key}")
            continue
        if row.get("world_count") != WORLD_COUNT or row.get("world_indices") != expected_world_indices:
            errors.append(f"condition {key} does not use frozen world indices")
        expected_runtime = [_runtime_seed(world_index, tier) for world_index in range(WORLD_COUNT)]
        if row.get("runtime_seeds") != expected_runtime:
            errors.append(f"condition {key} runtime seeds differ from frozen namespace")
        covered = row.get("covered_by_world")
        if not isinstance(covered, list) or len(covered) != WORLD_COUNT:
            errors.append(f"condition {key} has invalid coverage vector")
        else:
            rate = sum(int(bool(value)) for value in covered) / WORLD_COUNT
            if not _float_equal(rate, row.get("coverage_rate")):
                errors.append(f"condition {key} coverage rate differs from raw vector")
        if row.get("total_learned_updates_per_world") != TOTAL_UPDATES:
            errors.append(f"condition {key} violates frozen learned-work total")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"condition {key} parameter count differs from 19,649")
        if checkpoint in CHECKPOINTS and row.get("parameter_fingerprint") != CHECKPOINTS[checkpoint]["fingerprint"]:
            errors.append(f"condition {key} uses wrong checkpoint fingerprint")

        vector_fields = (
            "productive_rounds_by_world",
            "sink_rounds_by_world",
            "generated_terminal_count_by_world",
            "unique_generated_terminal_count_by_world",
            "max_reserve_population_by_world",
            "mean_reserve_population_by_world",
            "fraction_rounds_at_capacity_by_world",
            "reached_capacity_by_world",
        )
        if any(not isinstance(row.get(field), list) or len(row[field]) != WORLD_COUNT for field in vector_fields):
            errors.append(f"condition {key} has invalid telemetry vector lengths")
        else:
            productive = row["productive_rounds_by_world"]
            sink = row["sink_rounds_by_world"]
            if any(int(p) + int(s) != SEARCH_ROUNDS for p, s in zip(productive, sink, strict=True)):
                errors.append(f"condition {key} productive/sink rounds violate fixed budget")

    expected_keys = {
        (checkpoint, tier, capacity, mode)
        for checkpoint in CHECKPOINTS
        for tier in TIERS
        for capacity, mode in CONDITIONS
    }
    if set(index) != expected_keys:
        errors.append("condition matrix is incomplete or contains extras")

    # Collapsed L256 is one logical schedulable hypothesis and must remain behaviorally identical to
    # stable L1 for the same checkpoint/tier/worlds.
    for checkpoint in CHECKPOINTS:
        for tier in TIERS:
            l1 = index.get((checkpoint, tier, 1, "stable_reserve"))
            collapsed = index.get((checkpoint, tier, 256, "collapsed_diversity"))
            if l1 is None or collapsed is None:
                continue
            for field in (
                "covered_by_world",
                "productive_rounds_by_world",
                "sink_rounds_by_world",
                "generated_terminal_count_by_world",
                "unique_generated_terminal_count_by_world",
            ):
                if l1.get(field) != collapsed.get(field):
                    errors.append(f"checkpoint {checkpoint} {tier} collapsed logical-one identity failed for {field}")

    stored_pairs = payload.get("paired_summaries")
    if not isinstance(stored_pairs, list) or len(stored_pairs) != EXPECTED_PAIRS:
        errors.append("result must contain exactly 30 paired summaries")
        stored_pairs = []
    stored_index = {_pair_key(row): row for row in stored_pairs if isinstance(row, dict)}

    recomputed: list[dict[str, Any]] = []
    if set(index) == expected_keys:
        for checkpoint in CHECKPOINTS:
            for tier in TIERS:
                for comparison, tc, tm, rc, rm in PAIR_SPECS:
                    try:
                        expected = _recompute_pair(
                            comparison=comparison,
                            checkpoint=checkpoint,
                            tier=tier,
                            treatment=index[(checkpoint, tier, tc, tm)],
                            reference=index[(checkpoint, tier, rc, rm)],
                        )
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"could not recompute pair {checkpoint}/{tier}/{comparison}: {exc}")
                        continue
                    recomputed.append(expected)
                    observed = stored_index.get(_pair_key(expected))
                    if observed is None:
                        errors.append(f"missing paired summary {_pair_key(expected)}")
                        continue
                    for field in (
                        "world_count",
                        "treatment_only",
                        "reference_only",
                        "both_covered",
                        "neither_covered",
                    ):
                        if observed.get(field) != expected[field]:
                            errors.append(f"paired {_pair_key(expected)} {field} differs from recomputation")
                    for field in ("coverage_delta", "bootstrap_ci_low", "bootstrap_ci_high"):
                        if not _float_equal(float(expected[field]), observed.get(field)):
                            errors.append(f"paired {_pair_key(expected)} {field} differs from recomputation")

    deltas: dict[str, float] = {}
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for row in recomputed:
        key = _metric_key(row["checkpoint_index"], row["tier"], row["comparison"])
        deltas[key] = float(row["coverage_delta"])
        lows[key] = float(row["bootstrap_ci_low"])
        highs[key] = float(row["bootstrap_ci_high"])
    if len(deltas) != EXPECTED_PAIRS:
        errors.append("paired reconstruction is incomplete")

    if errors:
        return Gate3V2FrontierAudit(
            False,
            "INVALID_DEVELOPMENT_ARTIFACT",
            None,
            deltas,
            lows,
            highs,
            tuple(errors),
        )
    return Gate3V2FrontierAudit(
        True,
        "DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        _classify(lows, deltas),
        deltas,
        lows,
        highs,
        (),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_gate3_v2_frontier(args.result)
    text = json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if audit.artifact_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
