"""Read-only experiment artifact indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import default_adapters
from .adapters.base import ArtifactContext, ResultAdapter
from .artifacts import ArtifactCandidate, discover_json_artifacts, make_artifact_id
from .errors import ResultsDirectoryAccessError
from .schemas import (
    ArtifactRefV1,
    DashboardStatusV1,
    ExperimentGroupV1,
    ExperimentRunV1,
    IndexErrorV1,
)


@dataclass(frozen=True, slots=True)
class DashboardIndexSnapshot:
    status: DashboardStatusV1
    experiments_by_id: dict[str, ExperimentRunV1] = field(default_factory=dict)
    experiment_order: list[str] = field(default_factory=list)
    groups_by_id: dict[str, ExperimentGroupV1] = field(default_factory=dict)
    index_errors_by_id: dict[str, IndexErrorV1] = field(default_factory=dict)

    @property
    def experiments(self) -> list[ExperimentRunV1]:
        return [self.experiments_by_id[item] for item in self.experiment_order]

    @property
    def index_errors(self) -> list[IndexErrorV1]:
        return list(self.index_errors_by_id.values())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_error(
    *,
    artifact: ArtifactCandidate,
    phase: str,
    message: str,
    adapter_id: str | None = None,
    recovered_identity: dict[str, Any] | None = None,
) -> IndexErrorV1:
    error_id = f"index_error_{make_artifact_id(artifact.artifact_ref + phase + message)}"
    return IndexErrorV1(
        error_id=error_id,
        artifact=ArtifactRefV1(
            artifact_id=artifact.artifact_id,
            artifact_ref=artifact.artifact_ref,
        ),
        phase=phase,  # type: ignore[arg-type]
        severity="ERROR",
        message=message,
        adapter_id=adapter_id,
        recovered_identity=recovered_identity,
    )


def _dump_scientific(run: ExperimentRunV1) -> dict[str, Any]:
    data = run.model_dump() if hasattr(run, "model_dump") else run.dict()
    data.pop("provenance", None)
    return data


def _apply_duplicate_policy(
    runs: list[ExperimentRunV1],
    errors: list[IndexErrorV1],
) -> tuple[dict[str, ExperimentRunV1], list[str], list[IndexErrorV1]]:
    by_id: dict[str, ExperimentRunV1] = {}
    order: list[str] = []
    signatures: dict[str, dict[str, Any]] = {}

    for run in runs:
        run_id = run.identity.experiment_id
        signature = _dump_scientific(run)
        if run_id not in by_id:
            by_id[run_id] = run
            signatures[run_id] = signature
            order.append(run_id)
            continue

        current = by_id[run_id]
        if signatures[run_id] == signature:
            refs = list(current.provenance.duplicate_artifact_refs)
            refs.append(run.provenance.artifact.artifact_ref)
            current.provenance.duplicate_artifact_count += 1
            current.provenance.duplicate_artifact_refs = refs
            current.warnings.append(
                f"{current.provenance.duplicate_artifact_count} duplicate artifact(s) detected."
            )
            by_id[run_id] = current
            continue

        errors.append(
            IndexErrorV1(
                error_id=f"index_error_duplicate_conflict_{run_id}",
                artifact=run.provenance.artifact,
                phase="NORMALIZATION",
                severity="ERROR",
                message=(
                    "Multiple artifacts normalized to the same experiment id with "
                    "conflicting scientific content."
                ),
                adapter_id=run.provenance.adapter_id,
                recovered_identity={
                    "experiment_id": run_id,
                    "experiment_name": run.identity.experiment_name,
                },
            )
        )
        if run_id in by_id:
            del by_id[run_id]
            order = [item for item in order if item != run_id]

    return by_id, order, errors


class DashboardIndexer:
    def __init__(self, adapters: list[ResultAdapter] | None = None) -> None:
        self.adapters = adapters or default_adapters()

    def build(self, results_dir: Path) -> DashboardIndexSnapshot:
        indexed_at = _now()
        runs: list[ExperimentRunV1] = []
        groups: list[ExperimentGroupV1] = []
        errors: list[IndexErrorV1] = []

        try:
            directory_status, artifacts = discover_json_artifacts(results_dir)
        except ResultsDirectoryAccessError as exc:
            directory_status = "PRESENT"
            artifacts = []
            synthetic = ArtifactCandidate(
                path=results_dir,
                artifact_id=make_artifact_id(str(results_dir)),
                artifact_ref="<results-dir>",
            )
            errors.append(
                _safe_error(
                    artifact=synthetic,
                    phase="DISCOVERY",
                    message=str(exc),
                )
            )

        for artifact in artifacts:
            try:
                payload = json.loads(artifact.path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(
                    _safe_error(
                        artifact=artifact,
                        phase="CLASSIFICATION",
                        message=f"JSON parse failed: {exc}",
                    )
                )
                continue
            if not isinstance(payload, dict):
                errors.append(
                    _safe_error(
                        artifact=artifact,
                        phase="VALIDATION",
                        message="JSON artifact root must be an object.",
                    )
                )
                continue

            context = ArtifactContext(
                path=artifact.path,
                artifact_id=artifact.artifact_id,
                artifact_ref=artifact.artifact_ref,
                indexed_at=indexed_at,
            )
            matches = [
                (adapter, adapter.can_handle(payload, context))
                for adapter in self.adapters
            ]
            matches = [(adapter, match) for adapter, match in matches if match.matched]
            if not matches:
                continue
            matches.sort(key=lambda item: item[1].confidence, reverse=True)
            if len(matches) > 1 and matches[0][1].confidence == matches[1][1].confidence:
                errors.append(
                    _safe_error(
                        artifact=artifact,
                        phase="CLASSIFICATION",
                        message="Multiple adapters matched with equal confidence.",
                    )
                )
                continue

            adapter = matches[0][0]
            try:
                result = adapter.normalize(payload, context)
                runs.extend(result.runs)
                groups.extend(result.groups)
                errors.extend(result.errors)
            except Exception as exc:
                errors.append(
                    _safe_error(
                        artifact=artifact,
                        phase="NORMALIZATION",
                        message=str(exc),
                        adapter_id=adapter.adapter_id,
                    )
                )

        experiments_by_id, experiment_order, errors = _apply_duplicate_policy(runs, errors)
        groups_by_id = {group.group_id: group for group in groups}
        errors_by_id = {error.error_id: error for error in errors}
        status = DashboardStatusV1(
            service_status="DEGRADED" if errors else "HEALTHY",
            index_status="READY_WITH_ERRORS" if errors else "READY",
            result_directory_status=directory_status,  # type: ignore[arg-type]
            indexed_experiment_count=len(experiments_by_id),
            indexing_error_count=len(errors_by_id),
            group_count=len(groups_by_id),
            complete_group_count=sum(
                1
                for group in groups_by_id.values()
                if group.seed_coverage.status == "COMPLETE"
            ),
            population_experiment_count=sum(
                1
                for run in experiments_by_id.values()
                if run.identity.experiment_type == "population"
            ),
            last_indexed_at=indexed_at,
        )
        return DashboardIndexSnapshot(
            status=status,
            experiments_by_id=experiments_by_id,
            experiment_order=experiment_order,
            groups_by_id=groups_by_id,
            index_errors_by_id=errors_by_id,
        )
