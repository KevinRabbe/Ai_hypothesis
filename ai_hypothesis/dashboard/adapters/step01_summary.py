"""Adapter for Step 1 sweep and confirmation summary artifacts."""

from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any

from ..schemas import ExperimentGroupV1, SeedCoverageV1
from .base import AdapterMatch, ArtifactContext, NormalizationResult


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


class Step01SummaryAdapter:
    adapter_id = "step01-summary-v1"

    def can_handle(self, payload: dict[str, Any], context: ArtifactContext) -> AdapterMatch:
        if payload.get("execution_mode") == "strictly_sequential" and (
            "runs" in payload or "experiments" in payload
        ):
            return AdapterMatch(True, confidence=50)
        return AdapterMatch(False)

    def normalize(
        self,
        payload: dict[str, Any],
        context: ArtifactContext,
    ) -> NormalizationResult:
        raw_runs = payload.get("runs", payload.get("experiments", []))
        if not isinstance(raw_runs, list):
            raise ValueError("Step 1 summary run list must be an array")
        expected_seeds = payload.get("requested_seeds")
        expected = [int(seed) for seed in expected_seeds] if isinstance(expected_seeds, list) else None
        groups: list[ExperimentGroupV1] = []
        by_size: dict[str, list[dict[str, Any]]] = {}
        for run in raw_runs:
            if isinstance(run, dict):
                if "seed" not in run and expected is None:
                    continue
                by_size.setdefault(str(run.get("size_label", run.get("experiment_name"))), []).append(run)

        for label, runs in sorted(by_size.items()):
            observed = sorted(
                int(run["seed"]) for run in runs if isinstance(run.get("seed"), int)
            )
            missing = sorted(set(expected or []) - set(observed)) if expected is not None else None
            if expected is None:
                coverage = "SEED_COVERAGE_UNKNOWN"
            else:
                coverage = "COMPLETE" if not missing else "PARTIAL"
            values = [float(run["test_accuracy"]) for run in runs if "test_accuracy" in run]
            group_id = f"group_{_fingerprint({'adapter': self.adapter_id, 'artifact': context.artifact_ref, 'label': label})}"
            groups.append(
                ExperimentGroupV1(
                    group_id=group_id,
                    group_label=f"Step 1 {label}",
                    grouping_dimensions={
                        "research_step": "Step 1",
                        "experiment_type": "single_worker",
                        "benchmark_version": "step01-v0",
                        "size_label": label,
                    },
                    seed_coverage=SeedCoverageV1(
                        status=coverage,  # type: ignore[arg-type]
                        expected_seeds=expected,
                        observed_seeds=observed,
                        missing_seeds=missing,
                        source="summary" if expected is not None else "unknown",
                    ),
                    experiment_ids=[],
                    aggregate_metrics={
                        "test_accuracy_mean": statistics.mean(values) if values else None,
                        "test_accuracy_sample_std": (
                            statistics.stdev(values) if len(values) > 1 else None
                        ),
                        "test_accuracy_min": min(values) if values else None,
                        "test_accuracy_max": max(values) if values else None,
                        "count": len(runs),
                    },
                )
            )
        return NormalizationResult(groups=groups)
