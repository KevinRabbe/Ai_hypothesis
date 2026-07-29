"""Independent auditor/analyzer for frozen Gate-3 development-only evidence.

The analyzer reconstructs paired statistics from raw per-world outcomes. A scientifically weak or
negative development result is valid evidence; only provenance/mechanical inconsistency makes the
artifact invalid. This module never assigns a Gate-3 confirmation verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_VERSION = "gate3-hypothesis-population-development-v0"
DEPTHS = (4, 6, 8)
WIDTHS = {4: (1, 4, 16), 6: (1, 4, 16, 64), 8: (1, 4, 16, 64, 256)}
MODES = ("stable_diverse", "collapsed_diversity", "reshuffled_continuity")
WORLD_START = 1 << 30
WORLD_COUNT = 256
EVAL_BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 2_000
PARAMETER_COUNT = 19_873
SCORE_QUANTIZATION = 1e-3
WORK_BY_DEPTH = {4: 128, 6: 768, 8: 4096}
OBSERVATIONS_BY_DEPTH = {4: 8, 6: 12, 8: 16}
EXPECTED_CONDITIONS = 36
EXPECTED_PAIRED = 42


@dataclass(frozen=True, slots=True)
class Gate3DevelopmentAudit:
    artifact_valid: bool
    scientific_status: str
    directional_outcome: str | None
    primary_deltas: dict[str, float]
    primary_ci_lows: dict[str, float]
    primary_ci_highs: dict[str, float]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-3 development result must contain one JSON object")
    return payload


def _pair_specs() -> tuple[tuple[str, int, int, str, int, str], ...]:
    specs: list[tuple[str, int, int, str, int, str]] = []
    for depth in DEPTHS:
        widths = WIDTHS[depth]
        for width in widths[1:]:
            specs.append(("stable_width_vs_width1", depth, width, "stable_diverse", 1, "stable_diverse"))
        for previous, width in zip(widths[:-1], widths[1:], strict=True):
            specs.append(("stable_width_vs_previous", depth, width, "stable_diverse", previous, "stable_diverse"))
        for width in widths:
            specs.append(("stable_vs_collapsed", depth, width, "stable_diverse", width, "collapsed_diversity"))
            specs.append(("stable_vs_reshuffled", depth, width, "stable_diverse", width, "reshuffled_continuity"))
    assert len(specs) == EXPECTED_PAIRED
    return tuple(specs)


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("comparison"),
        row.get("depth"),
        row.get("treatment_width"),
        row.get("treatment_mode"),
        row.get("reference_width"),
        row.get("reference_mode"),
    )


def _bootstrap_seed(spec: tuple[object, ...]) -> int:
    digest = hashlib.sha256(
        ("gate3-bootstrap-v0:" + ":".join(str(value) for value in spec)).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_ci(differences: tuple[int, ...], *, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates: list[float] = []
    count = len(differences)
    for _ in range(BOOTSTRAP_SAMPLES):
        estimates.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
    estimates.sort()
    low_index = int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))
    high_index = int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))
    return estimates[low_index], estimates[high_index]


def _recompute_pair(
    spec: tuple[str, int, int, str, int, str],
    index: dict[tuple[int, int, str], dict[str, Any]],
) -> dict[str, Any]:
    comparison, depth, treatment_width, treatment_mode, reference_width, reference_mode = spec
    treatment = index[(depth, treatment_width, treatment_mode)]
    reference = index[(depth, reference_width, reference_mode)]
    if treatment["world_seeds"] != reference["world_seeds"]:
        raise ValueError("paired Gate-3 conditions do not use identical world ordering")
    treatment_solved = treatment["solved_by_world"]
    reference_solved = reference["solved_by_world"]
    if len(treatment_solved) != WORLD_COUNT or len(reference_solved) != WORLD_COUNT:
        raise ValueError("paired Gate-3 solved vectors do not contain 256 worlds")
    pairs = tuple(zip(treatment_solved, reference_solved, strict=True))
    treatment_only = sum(int(bool(a) and not bool(b)) for a, b in pairs)
    reference_only = sum(int(bool(b) and not bool(a)) for a, b in pairs)
    both = sum(int(bool(a) and bool(b)) for a, b in pairs)
    neither = len(pairs) - treatment_only - reference_only - both
    differences = tuple(int(bool(a)) - int(bool(b)) for a, b in pairs)
    delta = sum(differences) / len(differences)
    seed_spec = (
        comparison,
        depth,
        treatment_width,
        treatment_mode,
        reference_width,
        reference_mode,
    )
    ci_low, ci_high = _bootstrap_ci(differences, seed=_bootstrap_seed(seed_spec))
    return {
        "comparison": comparison,
        "depth": depth,
        "treatment_width": treatment_width,
        "reference_width": reference_width,
        "treatment_mode": treatment_mode,
        "reference_mode": reference_mode,
        "world_count": WORLD_COUNT,
        "treatment_only": treatment_only,
        "reference_only": reference_only,
        "both_solved": both,
        "neither_solved": neither,
        "exact_solve_delta": delta,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }


def _compare_float(label: str, expected: float, observed: Any, errors: list[str]) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        errors.append(f"{label}: expected finite float, got {observed!r}")
        return
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        errors.append(f"{label}: expected {expected}, got {value}")


def _compare_pair(expected: dict[str, Any], observed: dict[str, Any], errors: list[str]) -> None:
    key = _pair_key(expected)
    for field in (
        "comparison",
        "depth",
        "treatment_width",
        "reference_width",
        "treatment_mode",
        "reference_mode",
        "world_count",
        "treatment_only",
        "reference_only",
        "both_solved",
        "neither_solved",
    ):
        if observed.get(field) != expected[field]:
            errors.append(f"paired {key} field {field} differs from raw-world recomputation")
    for field in ("exact_solve_delta", "bootstrap_ci_low", "bootstrap_ci_high"):
        _compare_float(
            f"paired {key} {field}",
            float(expected[field]),
            observed.get(field),
            errors,
        )


def _primary_key(row: dict[str, Any]) -> str | None:
    key = _pair_key(row)
    mapping = {
        ("stable_width_vs_width1", 6, 64, "stable_diverse", 1, "stable_diverse"): "h6_w64_vs_w1",
        ("stable_width_vs_width1", 8, 256, "stable_diverse", 1, "stable_diverse"): "h8_w256_vs_w1",
        ("stable_width_vs_previous", 8, 256, "stable_diverse", 64, "stable_diverse"): "h8_w256_vs_w64",
        ("stable_vs_collapsed", 8, 256, "stable_diverse", 256, "collapsed_diversity"): "h8_stable_vs_collapsed",
        ("stable_vs_reshuffled", 8, 256, "stable_diverse", 256, "reshuffled_continuity"): "h8_stable_vs_reshuffled",
    }
    return mapping.get(key)


def _outcome(primary: dict[str, float]) -> str:
    w6 = primary["h6_w64_vs_w1"]
    w1 = primary["h8_w256_vs_w1"]
    w64 = primary["h8_w256_vs_w64"]
    collapsed = primary["h8_stable_vs_collapsed"]
    reshuffled = primary["h8_stable_vs_reshuffled"]
    if all(value > 0.0 for value in (w6, w1, w64, collapsed, reshuffled)):
        return "D_CLEAN_DIRECTIONAL_PATTERN"
    if w1 <= 0.0 or w64 <= 0.0:
        return "A_NO_OR_INCOMPLETE_BREADTH_EFFECT"
    if collapsed <= 0.0 or reshuffled <= 0.0:
        return "B_WIDTH_EFFECT_WITHOUT_CONTROL_SEPARATION"
    if w1 > 0.0 and w64 <= 0.0 and collapsed > 0.0 and reshuffled > 0.0:
        return "C_DIVERSITY_MATTERS_BUT_WIDTH_SATURATES_EARLY"
    return "MIXED_DEVELOPMENT_PATTERN"


def audit_gate3_development(path: Path) -> Gate3DevelopmentAudit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001 - auditor reports malformed evidence
        return Gate3DevelopmentAudit(False, "INVALID_ARTIFACT", None, {}, {}, {}, (str(exc),))

    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        errors.append("unexpected Gate-3 development experiment version")
    if payload.get("evaluation_split") != "development":
        errors.append("Gate-3 development result is not on the development split")
    if payload.get("confirmation_opened") is not False:
        errors.append("Gate-3 confirmation must remain closed")
    if payload.get("scientific_decision") != "DEVELOPMENT_ONLY_NOT_ASSIGNED":
        errors.append("Gate-3 development artifact must remain development-only")
    if payload.get("evaluation_world_count") != WORLD_COUNT:
        errors.append("Gate-3 development world count differs from frozen 256")
    if payload.get("evaluation_batch_size") != EVAL_BATCH_SIZE:
        errors.append("Gate-3 evaluation batch size differs from frozen 64")
    if payload.get("bootstrap_samples") != BOOTSTRAP_SAMPLES:
        errors.append("Gate-3 bootstrap count differs from frozen 2000")
    if not math.isclose(float(payload.get("hint_reliability", float("nan"))), 0.70, rel_tol=0.0, abs_tol=0.0):
        errors.append("Gate-3 hint reliability differs from frozen 0.70")

    training = payload.get("training")
    if not isinstance(training, dict):
        errors.append("Gate-3 development result is missing training summary")
        training = {}
    if training.get("training_seed") != 0:
        errors.append("first admitted Gate-3 development result must use training seed 0")
    if training.get("steps") != 1_200 or training.get("examples_seen") != 1_200 * 128:
        errors.append("Gate-3 training steps/examples differ from frozen recipe")
    if training.get("learned_parameter_count") != PARAMETER_COUNT:
        errors.append("Gate-3 learned parameter count differs from 19,873")
    fingerprint = training.get("parameter_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        errors.append("Gate-3 training fingerprint is invalid")

    config = payload.get("training_config")
    if not isinstance(config, dict):
        errors.append("Gate-3 development result is missing training config")
        config = {}
    frozen_config = {
        "steps": 1_200,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "model": {"input_projection_width": 32, "state_width": 64},
    }
    if config != frozen_config:
        errors.append("Gate-3 training config differs from frozen development recipe")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITIONS:
        errors.append("Gate-3 development result must contain exactly 36 conditions")
        conditions = []

    index: dict[tuple[int, int, str], dict[str, Any]] = {}
    expected_worlds = list(range(WORLD_START, WORLD_START + WORLD_COUNT))
    for row in conditions:
        if not isinstance(row, dict):
            errors.append("Gate-3 condition is not an object")
            continue
        key = (row.get("depth"), row.get("width"), row.get("mode"))
        if key in index:
            errors.append(f"duplicate Gate-3 condition {key}")
            continue
        index[key] = row
        depth, width, mode = key
        if depth not in DEPTHS or width not in WIDTHS.get(depth, ()) or mode not in MODES:
            errors.append(f"unexpected Gate-3 condition {key}")
            continue
        if row.get("world_count") != WORLD_COUNT:
            errors.append(f"condition {key} world_count differs from 256")
        if row.get("world_seeds") != expected_worlds:
            errors.append(f"condition {key} development world seeds differ from frozen domain")
        solved = row.get("solved_by_world")
        bits = row.get("bit_accuracy_by_world")
        if not isinstance(solved, list) or len(solved) != WORLD_COUNT:
            errors.append(f"condition {key} solved vector is invalid")
        if not isinstance(bits, list) or len(bits) != WORLD_COUNT:
            errors.append(f"condition {key} bit-accuracy vector is invalid")
        if isinstance(solved, list) and len(solved) == WORLD_COUNT:
            _compare_float(
                f"condition {key} exact solve rate",
                sum(int(bool(value)) for value in solved) / WORLD_COUNT,
                row.get("exact_solve_rate"),
                errors,
            )
        if isinstance(bits, list) and len(bits) == WORLD_COUNT:
            _compare_float(
                f"condition {key} bit accuracy",
                sum(float(value) for value in bits) / WORLD_COUNT,
                row.get("bit_accuracy"),
                errors,
            )
        if row.get("learned_updates_per_world") != WORK_BY_DEPTH[depth]:
            errors.append(f"condition {key} learned-work identity failed")
        if row.get("unique_world_observations_per_world") != OBSERVATIONS_BY_DEPTH[depth]:
            errors.append(f"condition {key} information identity failed")
        if row.get("learned_parameter_count") != PARAMETER_COUNT:
            errors.append(f"condition {key} parameter count differs from frozen model")
        if row.get("parameter_fingerprint") != fingerprint:
            errors.append(f"condition {key} checkpoint fingerprint differs from training")

    expected_keys = {
        (depth, width, mode)
        for depth in DEPTHS
        for width in WIDTHS[depth]
        for mode in MODES
    }
    if set(index) != expected_keys:
        errors.append("Gate-3 development condition matrix is incomplete or contains extras")

    # Width-one controls are semantically identical by construction.
    for depth in DEPTHS:
        try:
            stable = index[(depth, 1, "stable_diverse")]
            for control in ("collapsed_diversity", "reshuffled_continuity"):
                other = index[(depth, 1, control)]
                for field in (
                    "solved_by_world",
                    "bit_accuracy_by_world",
                    "correct_candidate_survival_rate_by_phase",
                    "mean_unique_candidates_by_phase",
                ):
                    if stable.get(field) != other.get(field):
                        errors.append(f"H{depth} W1 stable/{control} identity failed for {field}")
        except KeyError:
            pass

    # At maximum stable width every hidden path is still physically present after branching.
    for depth in DEPTHS:
        try:
            row = index[(depth, 1 << depth, "stable_diverse")]
            survival = row.get("correct_candidate_survival_rate_by_phase")
            if not isinstance(survival, list) or len(survival) != 2 * depth:
                errors.append(f"H{depth} maximum-width survival telemetry is invalid")
            elif not math.isclose(float(survival[depth - 1]), 1.0, rel_tol=0.0, abs_tol=0.0):
                errors.append(f"H{depth} maximum stable width did not retain all hypotheses through branching")
        except KeyError:
            pass

    stored_pairs = payload.get("paired_summaries")
    if not isinstance(stored_pairs, list) or len(stored_pairs) != EXPECTED_PAIRED:
        errors.append("Gate-3 result must contain exactly 42 paired summaries")
        stored_pairs = []
    stored_index = {_pair_key(row): row for row in stored_pairs if isinstance(row, dict)}

    recomputed: list[dict[str, Any]] = []
    if set(index) == expected_keys:
        for spec in _pair_specs():
            try:
                row = _recompute_pair(spec, index)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"could not recompute paired {spec}: {exc}")
                continue
            recomputed.append(row)
            observed = stored_index.get(_pair_key(row))
            if observed is None:
                errors.append(f"missing stored paired summary {_pair_key(row)}")
            else:
                _compare_pair(row, observed, errors)

    primary_deltas: dict[str, float] = {}
    primary_lows: dict[str, float] = {}
    primary_highs: dict[str, float] = {}
    for row in recomputed:
        key = _primary_key(row)
        if key is None:
            continue
        primary_deltas[key] = float(row["exact_solve_delta"])
        primary_lows[key] = float(row["bootstrap_ci_low"])
        primary_highs[key] = float(row["bootstrap_ci_high"])

    expected_primary = {
        "h6_w64_vs_w1",
        "h8_w256_vs_w1",
        "h8_w256_vs_w64",
        "h8_stable_vs_collapsed",
        "h8_stable_vs_reshuffled",
    }
    if set(primary_deltas) != expected_primary:
        errors.append("Gate-3 primary comparison reconstruction is incomplete")

    if errors:
        return Gate3DevelopmentAudit(
            artifact_valid=False,
            scientific_status="INVALID_DEVELOPMENT_ARTIFACT",
            directional_outcome=None,
            primary_deltas=primary_deltas,
            primary_ci_lows=primary_lows,
            primary_ci_highs=primary_highs,
            errors=tuple(errors),
        )

    return Gate3DevelopmentAudit(
        artifact_valid=True,
        scientific_status="DEVELOPMENT_ONLY_NO_GATE_VERDICT",
        directional_outcome=_outcome(primary_deltas),
        primary_deltas=primary_deltas,
        primary_ci_lows=primary_lows,
        primary_ci_highs=primary_highs,
        errors=(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_gate3_development(args.result)
    text = json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if audit.artifact_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
