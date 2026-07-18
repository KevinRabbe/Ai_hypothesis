"""Adapter for Step 2 population runtime v0 result artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..schemas import (
    ArchitectureIdentityV1,
    ArtifactRefV1,
    ExperimentIdentityV1,
    ExperimentProvenanceV1,
    ExperimentRunV1,
    PerformanceMetricsV1,
    PopulationMetricsV1,
    PopulationTaskMetricsV1,
    QualityMetricsV1,
    TrainingSummaryV1,
)
from .base import AdapterMatch, ArtifactContext, NormalizationResult


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


class Step02PopulationAdapter:
    adapter_id = "step02-population-v1"

    def can_handle(self, payload: dict[str, Any], context: ArtifactContext) -> AdapterMatch:
        if payload.get("runtime_version") == "step02-population-runtime-v0":
            return AdapterMatch(True, confidence=100)
        return AdapterMatch(False)

    def normalize(
        self,
        payload: dict[str, Any],
        context: ArtifactContext,
    ) -> NormalizationResult:
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Step 2 population result requires metrics")
        unit_config = metrics.get("unit_config")
        if not isinstance(unit_config, dict):
            raise ValueError("Step 2 population metrics require unit_config")

        population_width = metrics.get("population_width")
        worker_parameter_count = None
        total_learned_parameters = None
        worker_status = "NOT_AVAILABLE"
        total_status = "NOT_AVAILABLE"

        identity_payload = {
            "research_step": "step02",
            "experiment_type": "population",
            "runtime_version": payload.get("runtime_version"),
            "evidence_contract_version": payload.get("evidence_contract_version"),
            "split": payload.get("split"),
            "backend": payload.get("backend"),
            "count": payload.get("count"),
            "batch_size": payload.get("batch_size"),
            "unit_config": unit_config,
            "population_width": population_width,
            "aggregation_config": payload.get("aggregation_config"),
            "checkpoints": payload.get("checkpoints"),
        }
        experiment_id = f"run_{_fingerprint(identity_payload)}"
        architecture_id = f"arch_{_fingerprint({'step': 'step02', 'unit_config': unit_config, 'population_width': population_width})}"

        single_worker = metrics.get("single_worker_accuracy") or {}
        worker_values = single_worker.get("values") if isinstance(single_worker, dict) else None
        by_task_payload = metrics.get("by_task")
        by_task = None
        if isinstance(by_task_payload, dict):
            by_task = {
                str(task): PopulationTaskMetricsV1(**task_metrics)
                for task, task_metrics in by_task_payload.items()
                if isinstance(task_metrics, dict)
            }

        population_metrics = PopulationMetricsV1(
            population_width=population_width,
            execution_backend=metrics.get("execution_backend", payload.get("backend")),
            evidence_reducer_accuracy=metrics.get("evidence_reducer_accuracy"),
            majority_vote_accuracy=metrics.get("majority_vote_accuracy"),
            mean_logit_accuracy=metrics.get("mean_logit_accuracy"),
            mean_probability_accuracy=metrics.get("mean_probability_accuracy"),
            oracle_any_correct_coverage=metrics.get("oracle_any_correct_coverage"),
            all_wrong_rate=metrics.get("all_wrong_rate"),
            minority_rescue_opportunity_rate=metrics.get(
                "minority_rescue_opportunity_rate"
            ),
            minority_rescue_rate=metrics.get("minority_rescue_rate"),
            minority_suppression_rate=metrics.get("minority_suppression_rate"),
            majority_harm_rate=metrics.get("majority_harm_rate"),
            evidence_utilization_gap=metrics.get("evidence_utilization_gap"),
            mean_disagreement_entropy=metrics.get("mean_disagreement_entropy"),
            mean_population_uncertainty=metrics.get("mean_population_uncertainty"),
            mean_invalid_label_mass=metrics.get("mean_invalid_label_mass"),
            individual_worker_accuracies=worker_values if isinstance(worker_values, list) else None,
            by_task=by_task,
        )

        quality = QualityMetricsV1(
            split=payload.get("split", "unknown"),
            count=metrics.get("count", payload.get("count")),
            accuracy=metrics.get("evidence_reducer_accuracy"),
            macro_task_accuracy=None,
            invalid_output_rate=metrics.get("mean_invalid_label_mass"),
            by_task={
                key: value.evidence_accuracy
                for key, value in (by_task or {}).items()
                if value.evidence_accuracy is not None
            }
            if by_task
            else None,
        )

        run = ExperimentRunV1(
            identity=ExperimentIdentityV1(
                experiment_id=experiment_id,
                experiment_name=f"step02_population_{population_width or 'unknown'}w",
                research_step="Step 2",
                experiment_type="population",
                benchmark_version="step01-v0",
                architecture_version=None,
                run_variant=payload.get("split"),
            ),
            architecture=ArchitectureIdentityV1(
                architecture_id=architecture_id,
                architecture_version=None,
                model_type="population",
                parameter_count=None,
                parameter_count_status="NOT_APPLICABLE",
                worker_parameter_count=worker_parameter_count,
                worker_parameter_count_status=worker_status,  # type: ignore[arg-type]
                population_width=population_width,
                population_width_status=(
                    "AVAILABLE" if population_width is not None else "NOT_AVAILABLE"
                ),
                total_learned_parameters=total_learned_parameters,
                total_learned_parameters_status=total_status,  # type: ignore[arg-type]
                config_fingerprint=_fingerprint(unit_config),
                display_label=f"Population {population_width or '?'} workers",
            ),
            training=TrainingSummaryV1(
                seed=None,
                train_count=None,
                validation_count=payload.get("count")
                if payload.get("split") == "validation"
                else None,
                test_count=payload.get("count") if payload.get("split") == "test" else None,
                batch_size=payload.get("batch_size"),
                best_validation_status="NOT_APPLICABLE",
                latest_validation_status="NOT_APPLICABLE",
            ),
            quality={
                "validation": quality if payload.get("split") == "validation" else None,
                "test": quality if payload.get("split") == "test" else None,
            },
            performance=PerformanceMetricsV1(telemetry_status="NOT_AVAILABLE"),
            population_metrics=population_metrics,
            runtime_version=payload.get("runtime_version"),
            evidence_contract_version=payload.get("evidence_contract_version"),
            aggregation_config=payload.get("aggregation_config"),
            provenance=ExperimentProvenanceV1(
                git_revision=None,
                device=payload.get("device"),
                artifact=ArtifactRefV1(
                    artifact_id=context.artifact_id,
                    artifact_ref=context.artifact_ref,
                ),
                checkpoint_ref=None,
                indexed_at=context.indexed_at,
                adapter_id=self.adapter_id,
                raw_schema_version=payload.get("runtime_version"),
            ),
            status="COMPLETE",
            warnings=[
                "Step 2 runtime v0 result does not record git revision, worker parameter count, telemetry, or per-sample evidence."
            ],
        )
        return NormalizationResult(runs=[run])
