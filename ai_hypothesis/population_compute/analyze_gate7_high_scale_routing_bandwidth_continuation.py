"""Independent auditor for Gate-7 post-confirmation continuation artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import gate7_high_scale_routing_bandwidth_continuation_audit_spec as spec
from .gate7_high_scale_routing_bandwidth_continuation_audit_tier import validate_tier


@dataclass(frozen=True, slots=True)
class Gate7ContinuationAudit:
    artifact_valid: bool
    scientific_status: str
    campaign_outcome: str | None
    completed_populations: tuple[int, ...]
    resource_frontier_population: int | None
    k_required_by_population: dict[int, int | None]
    passing_k_by_population: dict[int, tuple[int, ...]]
    tier_outcomes: dict[int, str | None]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Gate-7 continuation result must be one JSON object")
    return payload


def _validate_checkpoint_rows(payload: dict[str, Any], errors: list[str]) -> None:
    rows = payload.get("transition_checkpoints")
    if not isinstance(rows, list) or len(rows) != 3:
        errors.append("artifact must bind exactly three transition checkpoints")
        return
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("checkpoint_index") not in spec.CHECKPOINTS:
            errors.append("invalid transition checkpoint identity")
            continue
        checkpoint = int(row["checkpoint_index"])
        if checkpoint in seen:
            errors.append(f"duplicate checkpoint {checkpoint}")
            continue
        seen.add(checkpoint)
        expected = spec.CHECKPOINTS[checkpoint]
        if str(row.get("checkpoint_sha256", "")).lower() != expected["sha256"]:
            errors.append(f"checkpoint {checkpoint} SHA mismatch")
        if row.get("parameter_fingerprint") != expected["fingerprint"]:
            errors.append(f"checkpoint {checkpoint} fingerprint mismatch")
        if row.get("training_seed") != expected["training_seed"]:
            errors.append(f"checkpoint {checkpoint} training seed mismatch")
        if row.get("learned_parameter_count") != spec.PARAMETER_COUNT:
            errors.append(f"checkpoint {checkpoint} parameter count mismatch")
        if row.get("transition_version") != spec.TRANSITION_VERSION:
            errors.append(f"checkpoint {checkpoint} transition version mismatch")
        if row.get("training_git_head") != spec.TRAINING_GIT_HEAD:
            errors.append(f"checkpoint {checkpoint} training head mismatch")
    if seen != set(spec.CHECKPOINT_INDICES):
        errors.append("checkpoint identity set is incomplete")


def _validate_top_level(payload: dict[str, Any], errors: list[str]) -> None:
    exact = {
        "experiment_version": spec.EXPERIMENT_VERSION,
        "scientific_status": spec.SCIENTIFIC_STATUS,
        "continuation_protocol_head": spec.PROTOCOL_HEAD,
        "confirmation_execution_head": spec.CONFIRMATION_EXECUTION_HEAD,
        "confirmation_result_head": spec.CONFIRMATION_RESULT_HEAD,
        "confirmation_result_sha256": spec.CONFIRMATION_RESULT_SHA256,
        "confirmation_audit_sha256": spec.CONFIRMATION_AUDIT_SHA256,
        "confirmation_manifest_sha256": spec.CONFIRMATION_MANIFEST_SHA256,
        "confirmation_outcome": spec.CONFIRMATION_OUTCOME,
        "confirmed_n8192_passing_k": list(spec.CONFIRMED_N8192_PASSING_K),
        "confirmed_n8192_k_required": spec.CONFIRMED_N8192_K_REQUIRED,
        "populations": list(spec.POPULATIONS),
        "k_ladder": list(spec.K_LADDER),
        "world_count": spec.WORLD_COUNT,
        "world_count_per_checkpoint_population": spec.WORLD_COUNT,
        "evaluation_batch_size": spec.BATCH_SIZE,
        "bootstrap_samples": spec.BOOTSTRAP_SAMPLES,
        "stage_b_parent_slots": spec.STAGE_B_SLOTS,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            errors.append(f"{key} mismatch")
    if payload.get("execution_admitted") is not True:
        errors.append("continuation execution was not admitted")
    if payload.get("continuation_opened") is not True:
        errors.append("continuation artifact must record continuation_opened=true")
    if payload.get("second_confirmation_opened") is not False:
        errors.append("second confirmation must remain closed")
    if payload.get("second_continuation_opened") is not False:
        errors.append("second continuation must remain closed")
    if payload.get("training_performed") is not False:
        errors.append("continuation must perform no training")
    if payload.get("checkpoint_selection_performed") is not False:
        errors.append("continuation must perform no checkpoint selection")
    if not spec.float_equal(spec.HINT_RELIABILITY, payload.get("hint_reliability")):
        errors.append("continuation hint reliability changed")
    if not spec.float_equal(
        spec.NONINFERIORITY_MARGIN,
        payload.get("noninferiority_margin"),
    ):
        errors.append("continuation non-inferiority margin changed")
    for flag in ("compiler_enabled", "cuda_graphs_enabled", "mixed_precision_enabled"):
        if payload.get(flag) is not False:
            errors.append(f"{flag} must remain false")


def audit_gate7_high_scale_routing_bandwidth_continuation(
    path: Path,
) -> Gate7ContinuationAudit:
    errors: list[str] = []
    try:
        payload = _load(path)
    except Exception as exc:  # noqa: BLE001
        return Gate7ContinuationAudit(
            artifact_valid=False,
            scientific_status="INVALID_ARTIFACT",
            campaign_outcome=None,
            completed_populations=(),
            resource_frontier_population=None,
            k_required_by_population={},
            passing_k_by_population={},
            tier_outcomes={},
            errors=(str(exc),),
        )

    _validate_top_level(payload, errors)
    _validate_checkpoint_rows(payload, errors)

    completed_raw = payload.get("completed_populations")
    completed = (
        tuple(completed_raw)
        if isinstance(completed_raw, list) and all(type(value) is int for value in completed_raw)
        else ()
    )
    if not isinstance(completed_raw, list):
        errors.append("completed_populations must be a list")
    try:
        if completed != spec.POPULATIONS[: len(completed)]:
            raise ValueError("completed populations are not a contiguous prefix")
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    resource_population = payload.get("resource_frontier_population")
    if resource_population is not None and resource_population not in spec.POPULATIONS:
        errors.append("resource frontier population is outside the frozen ladder")
    resource_error = payload.get("resource_error")
    if resource_population is None:
        if resource_error is not None:
            errors.append("resource_error must be null without a resource frontier")
    elif not isinstance(resource_error, str) or not resource_error:
        errors.append("resource frontier must preserve a non-empty resource_error")

    tiers = payload.get("tiers")
    expected_tier_populations = list(completed)
    if (
        not isinstance(tiers, list)
        or [row.get("population") for row in tiers if isinstance(row, dict)]
        != expected_tier_populations
    ):
        errors.append("tier list must match the exact completed-population prefix")
        tiers = []

    k_required_by_population: dict[int, int | None] = {}
    passing_k_by_population: dict[int, tuple[int, ...]] = {}
    tier_outcomes: dict[int, str | None] = {}
    audited_populations: list[int] = []
    for tier in tiers:
        if not isinstance(tier, dict):
            errors.append("tier is not an object")
            continue
        try:
            summary = validate_tier(tier, errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tier audit raised unexpectedly: {exc}")
            continue
        if summary is None:
            continue
        population = int(summary["population"])
        audited_populations.append(population)
        k_required_by_population[population] = summary["smallest_passing_k"]
        passing_k_by_population[population] = tuple(summary["passing_k"])
        tier_outcomes[population] = summary["outcome"]

    if tuple(audited_populations) != completed:
        errors.append("audited populations differ from completed populations")

    observed_campaign = payload.get("campaign_outcome")
    try:
        expected_campaign = spec.classify_campaign(
            completed_populations=completed,
            resource_frontier_population=resource_population,
        )
    except Exception as exc:  # noqa: BLE001
        expected_campaign = None
        errors.append(f"campaign classification failed: {exc}")
    if observed_campaign not in spec.VALID_CAMPAIGN_OUTCOMES:
        errors.append("unknown continuation campaign outcome")
    if expected_campaign is not None and observed_campaign != expected_campaign:
        errors.append("continuation campaign outcome mismatch")

    return Gate7ContinuationAudit(
        artifact_valid=not errors,
        scientific_status=(spec.SCIENTIFIC_STATUS if not errors else "INVALID_ARTIFACT"),
        campaign_outcome=observed_campaign if isinstance(observed_campaign, str) else None,
        completed_populations=tuple(audited_populations),
        resource_frontier_population=(
            int(resource_population) if type(resource_population) is int else None
        ),
        k_required_by_population=k_required_by_population,
        passing_k_by_population=passing_k_by_population,
        tier_outcomes=tier_outcomes,
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_gate7_high_scale_routing_bandwidth_continuation(args.artifact)
    args.output.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
