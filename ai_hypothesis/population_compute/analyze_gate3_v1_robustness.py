"""Independent auditor for Gate-3 v1 robustness development seeds 1 and 2.

The seed-0 analyzer remains immutable. This post-seed0 auditor enforces the same frozen mechanics,
worlds, paired statistics and structural identities while permitting only the precommitted robustness
training seeds 1 and 2. It cannot assign a Gate-3 verdict.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .analyze_gate3_v1_development import (
    BOOTSTRAP_SAMPLES,
    CAPACITIES,
    DEPTHS,
    EVAL_BATCH_SIZE,
    EXPECTED_CONDITIONS,
    EXPECTED_PAIRED,
    EXPERIMENT_VERSION,
    MODES,
    PARAMETER_COUNT,
    SEARCH_ROUNDS,
    TOTAL_UPDATES,
    WORLD_COUNT,
    WORLD_START,
    _classify,
    _float_equal,
    _pair_key,
    _pair_specs,
    _primary_key,
    _recompute_pair,
)


ROBUSTNESS_SEEDS = (1, 2)


@dataclass(frozen=True, slots=True)
class Gate3V1RobustnessAudit:
    artifact_valid: bool
    scientific_status: str
    training_seed: int | None
    directional_outcome: str | None
    primary_deltas: dict[str, float]
    primary_ci_lows: dict[str, float]
    primary_ci_highs: dict[str, float]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_gate3_v1_robustness(path: Path) -> Gate3V1RobustnessAudit:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        return Gate3V1RobustnessAudit(False, "INVALID_ARTIFACT", None, None, {}, {}, {}, (str(exc),))
    if not isinstance(payload, dict):
        return Gate3V1RobustnessAudit(False, "INVALID_ARTIFACT", None, None, {}, {}, {}, ("result must be one JSON object",))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected experiment version")
    if payload.get("evaluation_split") != "development":
        errors.append("robustness result is not development split")
    if payload.get("confirmation_opened") is not False:
        errors.append("confirmation must remain closed")
    if payload.get("scientific_decision") != "DEVELOPMENT_ONLY_NOT_ASSIGNED":
        errors.append("robustness artifact attempted to assign a scientific verdict")
    if payload.get("evaluation_world_count") != WORLD_COUNT:
        errors.append("world count differs from frozen 256")
    if payload.get("evaluation_batch_size") != EVAL_BATCH_SIZE:
        errors.append("evaluation batch differs from frozen 64")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("bootstrap count differs from frozen 2000")
    if not _float_equal(0.70, payload.get("hint_reliability")):
        errors.append("hint reliability differs from frozen 0.70")
    if payload.get("recurrent_updates_per_child") != 8:
        errors.append("per-child recurrent refinement differs from frozen eight")

    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    config = payload.get("training_config") if isinstance(payload.get("training_config"), dict) else {}
    training_seed = training.get("training_seed")
    if training_seed not in ROBUSTNESS_SEEDS:
        errors.append("robustness training seed must be exactly 1 or 2")
    if training.get("steps") != 1200 or training.get("examples_seen") != 1200 * 256:
        errors.append("training steps/examples differ from frozen recipe")
    if training.get("learned_parameter_count") != PARAMETER_COUNT:
        errors.append("learned parameter count differs from 19,649")
    fingerprint = training.get("parameter_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        errors.append("parameter fingerprint is invalid")
    expected_config = {
        "steps": 1200,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "model": {"input_projection_width": 32, "state_width": 64},
    }
    if config != expected_config:
        errors.append("training config differs from frozen seed-0 recipe")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITIONS:
        errors.append("result must contain exactly 36 conditions")
        conditions = []
    index: dict[tuple[int, int, str], dict[str, Any]] = {}
    expected_worlds = list(range(WORLD_START, WORLD_START + WORLD_COUNT))
    for row in conditions:
        if not isinstance(row, dict):
            errors.append("condition is not an object")
            continue
        key = (row.get("depth"), row.get("reserve_capacity"), row.get("mode"))
        if key in index:
            errors.append(f"duplicate condition {key}")
            continue
        index[key] = row
        depth, capacity, mode = key
        if depth not in DEPTHS or capacity not in CAPACITIES.get(depth, ()) or mode not in MODES:
            errors.append(f"unexpected condition {key}")
            continue
        if row.get("world_count") != WORLD_COUNT or row.get("world_seeds") != expected_worlds:
            errors.append(f"condition {key} does not use frozen development worlds")
        covered = row.get("covered_by_world")
        if not isinstance(covered, list) or len(covered) != WORLD_COUNT:
            errors.append(f"condition {key} has invalid coverage vector")
        else:
            rate = sum(int(bool(value)) for value in covered) / WORLD_COUNT
            if not _float_equal(rate, row.get("coverage_rate")):
                errors.append(f"condition {key} coverage rate differs from raw vector")
        if row.get("total_learned_updates_per_world") != TOTAL_UPDATES[depth]:
            errors.append(f"condition {key} violates frozen learned-work total")
        if row.get("learned_parameter_count") != PARAMETER_COUNT or row.get("parameter_fingerprint") != fingerprint:
            errors.append(f"condition {key} checkpoint identity differs from training")
        productive = row.get("productive_rounds_by_world")
        sink = row.get("sink_rounds_by_world")
        fractions = row.get("productive_work_fraction_by_world")
        terminals = row.get("generated_terminal_count_by_world")
        unique_terminals = row.get("unique_generated_terminal_count_by_world")
        vectors = (productive, sink, fractions, terminals, unique_terminals)
        if any(not isinstance(vector, list) or len(vector) != WORLD_COUNT for vector in vectors):
            errors.append(f"condition {key} has invalid telemetry vector lengths")
        elif any(int(p) + int(s) != SEARCH_ROUNDS[depth] for p, s in zip(productive, sink, strict=True)):
            errors.append(f"condition {key} productive/sink rounds do not sum to frozen budget")

    expected_keys = {
        (depth, capacity, mode)
        for depth in DEPTHS
        for capacity in CAPACITIES[depth]
        for mode in MODES
    }
    if set(index) != expected_keys:
        errors.append("condition matrix is incomplete or contains extras")

    for depth in DEPTHS:
        l1 = index.get((depth, 1, "stable_reserve"))
        if l1 is None:
            continue
        for mode in ("collapsed_diversity", "reshuffled_continuity"):
            other = index.get((depth, 1, mode))
            if other is not None:
                for field in (
                    "covered_by_world",
                    "generated_terminal_count_by_world",
                    "unique_generated_terminal_count_by_world",
                    "productive_rounds_by_world",
                    "sink_rounds_by_world",
                ):
                    if l1.get(field) != other.get(field):
                        errors.append(f"S{depth} L1 identity failed for {mode}/{field}")
        for capacity in CAPACITIES[depth]:
            collapsed = index.get((depth, capacity, "collapsed_diversity"))
            if collapsed is not None:
                for field in (
                    "covered_by_world",
                    "generated_terminal_count_by_world",
                    "unique_generated_terminal_count_by_world",
                    "productive_rounds_by_world",
                    "sink_rounds_by_world",
                ):
                    if l1.get(field) != collapsed.get(field):
                        errors.append(f"S{depth} collapsed L{capacity} logical-one identity failed for {field}")

    stored_pairs = payload.get("paired_summaries")
    if not isinstance(stored_pairs, list) or len(stored_pairs) != EXPECTED_PAIRED:
        errors.append("result must contain exactly 42 paired summaries")
        stored_pairs = []
    stored_index = {_pair_key(row): row for row in stored_pairs if isinstance(row, dict)}

    recomputed: list[dict[str, Any]] = []
    if set(index) == expected_keys:
        for spec in _pair_specs():
            try:
                expected = _recompute_pair(spec, index)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"could not recompute pair {spec}: {exc}")
                continue
            recomputed.append(expected)
            observed = stored_index.get(_pair_key(expected))
            if observed is None:
                errors.append(f"missing paired summary {_pair_key(expected)}")
                continue
            for field in ("world_count", "treatment_only", "reference_only", "both_covered", "neither_covered"):
                if observed.get(field) != expected[field]:
                    errors.append(f"paired {_pair_key(expected)} {field} differs from recomputation")
            for field in ("coverage_delta", "bootstrap_ci_low", "bootstrap_ci_high"):
                if not _float_equal(float(expected[field]), observed.get(field)):
                    errors.append(f"paired {_pair_key(expected)} {field} differs from recomputation")

    deltas: dict[str, float] = {}
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for row in recomputed:
        key = _primary_key(row)
        if key is not None:
            deltas[key] = float(row["coverage_delta"])
            lows[key] = float(row["bootstrap_ci_low"])
            highs[key] = float(row["bootstrap_ci_high"])
    expected_primary = {
        "s8_l64_vs_l1",
        "s10_l256_vs_l1",
        "s10_l256_vs_l64",
        "s10_stable_vs_collapsed",
        "s10_stable_vs_reshuffled",
    }
    if set(deltas) != expected_primary:
        errors.append("primary reconstruction is incomplete")

    if errors:
        return Gate3V1RobustnessAudit(
            False,
            "INVALID_ROBUSTNESS_ARTIFACT",
            int(training_seed) if isinstance(training_seed, int) else None,
            None,
            deltas,
            lows,
            highs,
            tuple(errors),
        )
    return Gate3V1RobustnessAudit(
        True,
        "ROBUSTNESS_DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        int(training_seed),
        _classify(deltas),
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
    audit = audit_gate3_v1_robustness(args.result)
    text = json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if audit.artifact_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
