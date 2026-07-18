export type ExperimentType = "single_worker" | "population" | "dense" | "unknown";
export type Availability =
  | "AVAILABLE"
  | "NOT_AVAILABLE"
  | "NOT_APPLICABLE"
  | "UNKNOWN"
  | "INVALID";

export interface DashboardStatusV1 {
  schema_version: "dashboard-status-v1";
  service_status: "HEALTHY" | "DEGRADED";
  index_status: "NOT_STARTED" | "INDEXING" | "READY" | "READY_WITH_ERRORS";
  result_directory_status: "ABSENT" | "EMPTY" | "PRESENT";
  indexed_experiment_count: number;
  indexing_error_count: number;
  group_count: number;
  complete_group_count: number;
  population_experiment_count: number;
  last_indexed_at: string | null;
}

export interface ArtifactRefV1 {
  schema_version: "dashboard-artifact-ref-v1";
  artifact_id: string;
  artifact_ref: string;
}

export interface ExperimentRunV1 {
  schema_version: "dashboard-experiment-run-v1";
  identity: {
    schema_version: "dashboard-experiment-identity-v1";
    experiment_id: string;
    experiment_name: string;
    research_step: string;
    experiment_type: ExperimentType;
    benchmark_version: string | null;
    architecture_version: string | null;
    run_variant?: string | null;
  };
  architecture: {
    schema_version: "dashboard-architecture-identity-v1";
    architecture_id?: string;
    architecture_version?: string | null;
    model_type?: ExperimentType;
    config_fingerprint?: string | null;
    display_label: string;
    parameter_count: number | null;
    parameter_count_status: Availability;
    worker_parameter_count: number | null;
    worker_parameter_count_status: Availability;
    population_width: number | null;
    population_width_status: Availability;
    total_learned_parameters: number | null;
    total_learned_parameters_status: Availability;
  };
  training: {
    schema_version: "dashboard-training-summary-v1";
    seed: number | null;
    best_step: number | null;
    best_validation_score: number | null;
    best_validation: QualityMetricsV1 | null;
    best_validation_status: Availability;
    latest_validation: QualityMetricsV1 | null;
    latest_validation_status: Availability;
    training_duration_seconds: number | null;
    threshold_steps: Record<string, number | null>;
  };
  quality: {
    validation: QualityMetricsV1 | null;
    test: QualityMetricsV1 | null;
  };
  performance: {
    checkpoint_size_bytes: number | null;
    telemetry_status: Availability;
  };
  aggregation_config: Record<string, unknown> | null;
  population_metrics: PopulationMetricsV1 | null;
  runtime_version: string | null;
  evidence_contract_version: string | null;
  provenance: {
    schema_version: "dashboard-provenance-v1";
    git_revision: string | null;
    device: string | null;
    artifact: ArtifactRefV1;
    adapter_id: string;
    raw_schema_version?: string | null;
    checkpoint_ref?: string | null;
    indexed_at?: string;
    duplicate_artifact_count: number;
    duplicate_artifact_refs: string[];
  };
  status: "COMPLETE" | "PARTIAL" | "INVALID";
  warnings: string[];
}

export interface QualityMetricsV1 {
  schema_version: "dashboard-quality-metrics-v1";
  split: "validation" | "test" | "unknown";
  count: number | null;
  loss?: number | null;
  accuracy: number | null;
  macro_task_accuracy: number | null;
  invalid_output_rate: number | null;
  by_task: Record<string, number> | null;
  by_difficulty?: Record<string, number> | null;
  by_task_difficulty?: Record<string, number> | null;
  uncertainty_precision?: number | null;
  uncertainty_recall?: number | null;
}

export interface PopulationMetricsV1 {
  schema_version: "dashboard-population-metrics-v1";
  population_width: number | null;
  execution_backend: string | null;
  evidence_reducer_accuracy: number | null;
  majority_vote_accuracy: number | null;
  mean_logit_accuracy: number | null;
  mean_probability_accuracy: number | null;
  oracle_any_correct_coverage: number | null;
  evidence_utilization_gap: number | null;
  individual_worker_accuracies: number[] | null;
  all_wrong_rate?: number | null;
  minority_rescue_opportunity_rate?: number | null;
  minority_rescue_rate?: number | null;
  minority_suppression_rate?: number | null;
  majority_harm_rate?: number | null;
  mean_disagreement_entropy?: number | null;
  mean_population_uncertainty?: number | null;
  mean_invalid_label_mass?: number | null;
  by_task?: Record<string, unknown> | null;
}

export interface IndexErrorV1 {
  schema_version: "dashboard-index-error-v1";
  error_id: string;
  artifact: ArtifactRefV1;
  phase: string;
  severity: "WARNING" | "ERROR";
  message: string;
  adapter_id: string | null;
  recovered_identity?: Record<string, unknown> | null;
}

export interface ExperimentListResponseV1 {
  schema_version: "dashboard-experiment-list-response-v1";
  items: ExperimentRunV1[];
  total: number;
  limit: number;
  offset: number;
}
