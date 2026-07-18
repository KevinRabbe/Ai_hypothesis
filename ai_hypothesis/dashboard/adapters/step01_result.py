"""Adapter for Step 1 training ``result.json`` artifacts."""

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
    PerformanceInferencePointV1,
    PerformanceMetricsV1,
    QualityMetricsV1,
    TrainingHistoryPointV1,
    TrainingSummaryV1,
)
from .base import AdapterMatch, ArtifactContext, NormalizationResult


STEP01_THRESHOLDS = (0.90, 0.92, 0.93, 0.9382)


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _run_id(parts: dict[str, Any]) -> str:
    return f"run_{_fingerprint(parts)}"


def _quality(split: str, payload: dict[str, Any] | None) -> QualityMetricsV1 | None:
    if not isinstance(payload, dict):
        return None
    return QualityMetricsV1(
        split=split,  # type: ignore[arg-type]
        count=payload.get("count"),
        loss=payload.get("loss"),
        accuracy=payload.get("accuracy"),
        macro_task_accuracy=payload.get("macro_task_accuracy"),
        uncertainty_precision=payload.get("uncertainty_precision"),
        uncertainty_recall=payload.get("uncertainty_recall"),
        invalid_output_rate=payload.get("invalid_output_rate"),
        by_task=payload.get("by_task"),
        by_difficulty=payload.get("by_difficulty"),
        by_task_difficulty=payload.get("by_task_difficulty"),
    )


class Step01ResultAdapter:
    adapter_id = "step01-result-v1"

    def can_handle(self, payload: dict[str, Any], context: ArtifactContext) -> AdapterMatch:
        if (
            payload.get("benchmark_version") == "step01-v0"
            and "validation_history" in payload
            and "test" in payload
        ):
            return AdapterMatch(True, confidence=100)
        return AdapterMatch(False)

    def normalize(
        self,
        payload: dict[str, Any],
        context: ArtifactContext,
    ) -> NormalizationResult:
        train_config = payload.get("train_config")
        unit_config = payload.get("unit_config")
        if not isinstance(train_config, dict) or not isinstance(unit_config, dict):
            raise ValueError("Step 1 result requires train_config and unit_config")

        parameter_count = int(payload["parameter_count"])
        seed = train_config.get("seed")
        best_step = payload.get("best_step")
        history = payload.get("validation_history") or []
        if not isinstance(history, list):
            raise ValueError("Step 1 validation_history must be a list")

        training_history: list[TrainingHistoryPointV1] = []
        best_validation: QualityMetricsV1 | None = None
        latest_validation: QualityMetricsV1 | None = None
        threshold_steps: dict[str, int | None] = {
            f"{threshold:.4f}": None for threshold in STEP01_THRESHOLDS
        }

        for record in history:
            if not isinstance(record, dict):
                continue
            validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
            loss = record.get("train_loss") if isinstance(record.get("train_loss"), dict) else {}
            step = int(record["step"])
            training_history.append(
                TrainingHistoryPointV1(
                    step=step,
                    train_total_loss=loss.get("total_loss"),
                    train_label_loss=loss.get("label_loss"),
                    train_uncertainty_loss=loss.get("uncertainty_loss"),
                    validation_accuracy=validation.get("accuracy"),
                    validation_macro_task_accuracy=validation.get("macro_task_accuracy"),
                    validation_invalid_output_rate=validation.get("invalid_output_rate"),
                )
            )
            score = validation.get("macro_task_accuracy")
            if score is not None:
                for threshold in STEP01_THRESHOLDS:
                    key = f"{threshold:.4f}"
                    if threshold_steps[key] is None and float(score) >= threshold:
                        threshold_steps[key] = step
            quality = _quality("validation", validation)
            if step == best_step:
                best_validation = quality
            latest_validation = quality

        config_hash = _fingerprint(unit_config)
        identity_payload = {
            "research_step": "step01",
            "experiment_type": "single_worker",
            "benchmark_version": payload.get("benchmark_version"),
            "experiment_name": payload.get("experiment_name"),
            "architecture_version": payload.get("architecture_version"),
            "unit_config": unit_config,
            "parameter_count": parameter_count,
            "train_count": train_config.get("train_count"),
            "validation_count": train_config.get("validation_count"),
            "test_count": train_config.get("test_count"),
            "max_training_steps": train_config.get("max_training_steps"),
            "seed": seed,
        }
        experiment_id = _run_id(identity_payload)
        architecture_id = f"arch_{_fingerprint({'step': 'step01', 'unit_config': unit_config, 'parameter_count': parameter_count})}"

        output_dir = train_config.get("output_dir")
        checkpoint_ref = None
        if isinstance(output_dir, str) and context.artifact_ref.endswith("result.json"):
            checkpoint_ref = context.artifact_ref.removesuffix("result.json") + "best.pt"

        inference = []
        for item in (payload.get("inference") or {}).values():
            if isinstance(item, dict):
                inference.append(
                    PerformanceInferencePointV1(
                        batch_width=int(item["batch_width"]),
                        batch_latency_ms=item.get("batch_latency_ms"),
                        unit_evaluations_per_second=item.get(
                            "unit_evaluations_per_second"
                        ),
                    )
                )

        run = ExperimentRunV1(
            identity=ExperimentIdentityV1(
                experiment_id=experiment_id,
                experiment_name=str(payload.get("experiment_name")),
                research_step="Step 1",
                experiment_type="single_worker",
                benchmark_version=payload.get("benchmark_version"),
                architecture_version=payload.get("architecture_version"),
            ),
            architecture=ArchitectureIdentityV1(
                architecture_id=architecture_id,
                architecture_version=payload.get("architecture_version"),
                model_type="single_worker",
                parameter_count=parameter_count,
                parameter_count_status="AVAILABLE",
                worker_parameter_count=None,
                worker_parameter_count_status="NOT_APPLICABLE",
                population_width=None,
                population_width_status="NOT_APPLICABLE",
                total_learned_parameters=None,
                total_learned_parameters_status="NOT_APPLICABLE",
                config_fingerprint=config_hash,
                display_label=f"{parameter_count:,} params",
            ),
            training=TrainingSummaryV1(
                seed=seed,
                train_count=train_config.get("train_count"),
                validation_count=train_config.get("validation_count"),
                test_count=train_config.get("test_count"),
                batch_size=train_config.get("batch_size"),
                max_training_steps=train_config.get("max_training_steps"),
                eval_interval=train_config.get("eval_interval"),
                best_step=best_step,
                best_validation_score=payload.get("best_validation_score"),
                best_validation=best_validation,
                best_validation_status=(
                    "AVAILABLE" if best_validation is not None else "NOT_AVAILABLE"
                ),
                latest_validation=latest_validation,
                latest_validation_status=(
                    "AVAILABLE" if latest_validation is not None else "NOT_AVAILABLE"
                ),
                training_duration_seconds=payload.get("training_duration_seconds"),
                thresholds=list(STEP01_THRESHOLDS),
                threshold_steps=threshold_steps,
            ),
            training_history=training_history,
            quality={
                "validation": best_validation,
                "test": _quality("test", payload.get("test")),
            },
            performance=PerformanceMetricsV1(
                training_duration_seconds=payload.get("training_duration_seconds"),
                checkpoint_size_bytes=payload.get("checkpoint_size_bytes"),
                inference=inference,
                telemetry_status="NOT_AVAILABLE",
            ),
            provenance=ExperimentProvenanceV1(
                git_revision=payload.get("git_revision"),
                device=payload.get("device"),
                artifact=ArtifactRefV1(
                    artifact_id=context.artifact_id,
                    artifact_ref=context.artifact_ref,
                ),
                checkpoint_ref=checkpoint_ref,
                indexed_at=context.indexed_at,
                adapter_id=self.adapter_id,
                raw_schema_version=payload.get("benchmark_version"),
            ),
            status="COMPLETE",
            warnings=[],
        )
        return NormalizationResult(runs=[run])
