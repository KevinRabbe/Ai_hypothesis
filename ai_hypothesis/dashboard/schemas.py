"""Versioned normalized schemas for the dashboard API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Availability = Literal[
    "AVAILABLE",
    "NOT_AVAILABLE",
    "NOT_APPLICABLE",
    "UNKNOWN",
    "INVALID",
]
ExperimentType = Literal["single_worker", "population", "dense", "unknown"]


class ArtifactRefV1(BaseModel):
    schema_version: Literal["dashboard-artifact-ref-v1"] = "dashboard-artifact-ref-v1"
    artifact_id: str
    artifact_ref: str


class ExperimentIdentityV1(BaseModel):
    schema_version: Literal["dashboard-experiment-identity-v1"] = (
        "dashboard-experiment-identity-v1"
    )
    experiment_id: str
    experiment_name: str
    research_step: str
    experiment_type: ExperimentType
    benchmark_version: str | None = None
    architecture_version: str | None = None
    run_variant: str | None = None


class ArchitectureIdentityV1(BaseModel):
    schema_version: Literal["dashboard-architecture-identity-v1"] = (
        "dashboard-architecture-identity-v1"
    )
    architecture_id: str
    architecture_version: str | None = None
    model_type: ExperimentType
    parameter_count: int | None = None
    parameter_count_status: Availability = "UNKNOWN"
    worker_parameter_count: int | None = None
    worker_parameter_count_status: Availability = "UNKNOWN"
    population_width: int | None = None
    population_width_status: Availability = "UNKNOWN"
    total_learned_parameters: int | None = None
    total_learned_parameters_status: Availability = "UNKNOWN"
    config_fingerprint: str | None = None
    display_label: str


class QualityMetricsV1(BaseModel):
    schema_version: Literal["dashboard-quality-metrics-v1"] = (
        "dashboard-quality-metrics-v1"
    )
    split: Literal["validation", "test", "unknown"]
    count: int | None = None
    loss: float | None = None
    accuracy: float | None = None
    macro_task_accuracy: float | None = None
    uncertainty_precision: float | None = None
    uncertainty_recall: float | None = None
    invalid_output_rate: float | None = None
    by_task: dict[str, float] | None = None
    by_difficulty: dict[str, float] | None = None
    by_task_difficulty: dict[str, float] | None = None


class TrainingHistoryPointV1(BaseModel):
    step: int
    train_total_loss: float | None = None
    train_label_loss: float | None = None
    train_uncertainty_loss: float | None = None
    validation_accuracy: float | None = None
    validation_macro_task_accuracy: float | None = None
    validation_invalid_output_rate: float | None = None


class TrainingSummaryV1(BaseModel):
    schema_version: Literal["dashboard-training-summary-v1"] = (
        "dashboard-training-summary-v1"
    )
    seed: int | None = None
    train_count: int | None = None
    validation_count: int | None = None
    test_count: int | None = None
    batch_size: int | None = None
    max_training_steps: int | None = None
    eval_interval: int | None = None
    best_step: int | None = None
    best_validation_score: float | None = None
    best_validation: QualityMetricsV1 | None = None
    best_validation_status: Availability = "UNKNOWN"
    latest_validation: QualityMetricsV1 | None = None
    latest_validation_status: Availability = "UNKNOWN"
    training_duration_seconds: float | None = None
    thresholds: list[float] = Field(default_factory=list)
    threshold_steps: dict[str, int | None] = Field(default_factory=dict)


class PerformanceInferencePointV1(BaseModel):
    batch_width: int
    batch_latency_ms: float | None = None
    unit_evaluations_per_second: float | None = None


class PerformanceMetricsV1(BaseModel):
    schema_version: Literal["dashboard-performance-metrics-v1"] = (
        "dashboard-performance-metrics-v1"
    )
    training_duration_seconds: float | None = None
    checkpoint_size_bytes: int | None = None
    inference: list[PerformanceInferencePointV1] = Field(default_factory=list)
    telemetry_status: Availability = "NOT_AVAILABLE"


class PopulationTaskMetricsV1(BaseModel):
    count: int | None = None
    evidence_accuracy: float | None = None
    majority_vote_accuracy: float | None = None
    mean_logit_accuracy: float | None = None
    mean_probability_accuracy: float | None = None
    oracle_any_correct_coverage: float | None = None


class PopulationMetricsV1(BaseModel):
    schema_version: Literal["dashboard-population-metrics-v1"] = (
        "dashboard-population-metrics-v1"
    )
    population_width: int | None = None
    execution_backend: str | None = None
    evidence_reducer_accuracy: float | None = None
    majority_vote_accuracy: float | None = None
    mean_logit_accuracy: float | None = None
    mean_probability_accuracy: float | None = None
    oracle_any_correct_coverage: float | None = None
    all_wrong_rate: float | None = None
    minority_rescue_opportunity_rate: float | None = None
    minority_rescue_rate: float | None = None
    minority_suppression_rate: float | None = None
    majority_harm_rate: float | None = None
    evidence_utilization_gap: float | None = None
    mean_disagreement_entropy: float | None = None
    mean_population_uncertainty: float | None = None
    mean_invalid_label_mass: float | None = None
    individual_worker_accuracies: list[float] | None = None
    by_task: dict[str, PopulationTaskMetricsV1] | None = None


class ExperimentProvenanceV1(BaseModel):
    schema_version: Literal["dashboard-provenance-v1"] = "dashboard-provenance-v1"
    git_revision: str | None = None
    device: str | None = None
    artifact: ArtifactRefV1
    checkpoint_ref: str | None = None
    indexed_at: str
    adapter_id: str
    raw_schema_version: str | None = None
    duplicate_artifact_count: int = 0
    duplicate_artifact_refs: list[str] = Field(default_factory=list)


class ExperimentRunV1(BaseModel):
    schema_version: Literal["dashboard-experiment-run-v1"] = (
        "dashboard-experiment-run-v1"
    )
    identity: ExperimentIdentityV1
    architecture: ArchitectureIdentityV1
    training: TrainingSummaryV1
    training_history: list[TrainingHistoryPointV1] = Field(default_factory=list)
    quality: dict[Literal["validation", "test"], QualityMetricsV1 | None]
    performance: PerformanceMetricsV1
    population_metrics: PopulationMetricsV1 | None = None
    runtime_version: str | None = None
    evidence_contract_version: str | None = None
    aggregation_config: dict[str, Any] | None = None
    provenance: ExperimentProvenanceV1
    status: Literal["COMPLETE", "PARTIAL", "INVALID"]
    warnings: list[str] = Field(default_factory=list)


class SeedCoverageV1(BaseModel):
    status: Literal["COMPLETE", "PARTIAL", "SEED_COVERAGE_UNKNOWN"]
    expected_seeds: list[int] | None = None
    observed_seeds: list[int] = Field(default_factory=list)
    missing_seeds: list[int] | None = None
    source: Literal[
        "summary",
        "manifest",
        "runner_adapter",
        "normalized_metadata",
        "unknown",
    ] = "unknown"


class ExperimentGroupV1(BaseModel):
    schema_version: Literal["dashboard-experiment-group-v1"] = (
        "dashboard-experiment-group-v1"
    )
    group_id: str
    group_label: str
    grouping_dimensions: dict[str, Any]
    seed_coverage: SeedCoverageV1
    experiment_ids: list[str]
    aggregate_metrics: dict[str, float | int | None] = Field(default_factory=dict)


class IndexErrorV1(BaseModel):
    schema_version: Literal["dashboard-index-error-v1"] = "dashboard-index-error-v1"
    error_id: str
    artifact: ArtifactRefV1
    phase: Literal["DISCOVERY", "CLASSIFICATION", "VALIDATION", "NORMALIZATION"]
    severity: Literal["WARNING", "ERROR"]
    message: str
    adapter_id: str | None = None
    recovered_identity: dict[str, Any] | None = None


class DashboardStatusV1(BaseModel):
    schema_version: Literal["dashboard-status-v1"] = "dashboard-status-v1"
    service_status: Literal["HEALTHY", "DEGRADED"]
    index_status: Literal["NOT_STARTED", "INDEXING", "READY", "READY_WITH_ERRORS"]
    result_directory_status: Literal["ABSENT", "EMPTY", "PRESENT"]
    indexed_experiment_count: int
    indexing_error_count: int
    group_count: int
    complete_group_count: int
    population_experiment_count: int
    last_indexed_at: str | None = None


class HealthResponseV1(BaseModel):
    schema_version: Literal["dashboard-health-v1"] = "dashboard-health-v1"
    status: Literal["healthy"] = "healthy"
    api_version: Literal["v1"] = "v1"


class StatusResponseV1(BaseModel):
    schema_version: Literal["dashboard-status-response-v1"] = (
        "dashboard-status-response-v1"
    )
    status: DashboardStatusV1


class ExperimentListResponseV1(BaseModel):
    schema_version: Literal["dashboard-experiment-list-response-v1"] = (
        "dashboard-experiment-list-response-v1"
    )
    items: list[ExperimentRunV1]
    total: int
    limit: int
    offset: int


class ExperimentDetailResponseV1(BaseModel):
    schema_version: Literal["dashboard-experiment-detail-response-v1"] = (
        "dashboard-experiment-detail-response-v1"
    )
    experiment: ExperimentRunV1


class IndexErrorListResponseV1(BaseModel):
    schema_version: Literal["dashboard-index-error-list-response-v1"] = (
        "dashboard-index-error-list-response-v1"
    )
    items: list[IndexErrorV1]
    total: int


class ReindexResponseV1(BaseModel):
    schema_version: Literal["dashboard-reindex-response-v1"] = (
        "dashboard-reindex-response-v1"
    )
    status: DashboardStatusV1


class ApiErrorV1(BaseModel):
    schema_version: Literal["dashboard-error-v1"] = "dashboard-error-v1"
    code: str
    message: str
