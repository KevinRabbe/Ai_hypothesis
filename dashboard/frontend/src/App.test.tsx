import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { DashboardStatusV1, ExperimentRunV1, IndexErrorV1 } from "./api/types";

const state = vi.hoisted(() => ({
  status: null as DashboardStatusV1 | null,
  experiments: [] as ExperimentRunV1[],
  indexErrors: [] as IndexErrorV1[],
  failStatus: false,
  failExperiments: false,
  failIndexErrors: false,
  failExperimentLookup: false
}));

vi.mock("./api/client", () => {
  class ApiError extends Error {
    constructor(message: string) {
      super(message);
    }
  }
  return {
    ApiError,
    apiClient: {
      status: async () => {
        if (state.failStatus) throw new ApiError("Dashboard API is unreachable.");
        return state.status;
      },
      experiments: async () => {
        if (state.failExperiments) throw new ApiError("Dashboard API is unreachable.");
        return {
          schema_version: "dashboard-experiment-list-response-v1",
          items: state.experiments,
          total: state.experiments.length,
          limit: 100,
          offset: 0
        };
      },
      experiment: async (id: string) => {
        if (state.failExperimentLookup) throw new ApiError("Experiment was not found.");
        const found = state.experiments.find((item) => item.identity.experiment_id === id);
        if (!found) throw new ApiError("Experiment was not found.");
        return found;
      },
      indexErrors: async () => {
        if (state.failIndexErrors) throw new ApiError("Dashboard API is unreachable.");
        return { items: state.indexErrors, total: state.indexErrors.length };
      },
      reindex: async () => state.status
    }
  };
});

function baseStatus(overrides: Partial<DashboardStatusV1> = {}): DashboardStatusV1 {
  return {
    schema_version: "dashboard-status-v1",
    service_status: "HEALTHY",
    index_status: "READY",
    result_directory_status: "EMPTY",
    indexed_experiment_count: 0,
    indexing_error_count: 0,
    group_count: 0,
    complete_group_count: 0,
    population_experiment_count: 0,
    last_indexed_at: null,
    ...overrides
  };
}

function experiment(overrides: Partial<ExperimentRunV1> = {}): ExperimentRunV1 {
  return {
    schema_version: "dashboard-experiment-run-v1",
    identity: {
      schema_version: "dashboard-experiment-identity-v1",
      experiment_id: "run_step1",
      experiment_name: "step01_checkpoint_50k_extended_15k",
      research_step: "Step 1",
      experiment_type: "single_worker",
      benchmark_version: "step01-v0",
      architecture_version: "step01-unit-v0",
      run_variant: null
    },
    architecture: {
      schema_version: "dashboard-architecture-identity-v1",
      architecture_id: "arch_step1",
      architecture_version: "step01-unit-v0",
      model_type: "single_worker",
      parameter_count: 50268,
      parameter_count_status: "AVAILABLE",
      worker_parameter_count: null,
      worker_parameter_count_status: "NOT_APPLICABLE",
      population_width: null,
      population_width_status: "NOT_APPLICABLE",
      total_learned_parameters: null,
      total_learned_parameters_status: "NOT_APPLICABLE",
      config_fingerprint: "abc",
      display_label: "50,268 params"
    },
    training: {
      schema_version: "dashboard-training-summary-v1",
      seed: 1,
      best_step: 100,
      best_validation_score: 0.94,
      best_validation: null,
      best_validation_status: "AVAILABLE",
      latest_validation: {
        schema_version: "dashboard-quality-metrics-v1",
        split: "validation",
        count: 10,
        loss: null,
        accuracy: 0.93,
        macro_task_accuracy: 0.93,
        uncertainty_precision: null,
        uncertainty_recall: null,
        invalid_output_rate: 0,
        by_task: null,
        by_difficulty: null,
        by_task_difficulty: null
      },
      latest_validation_status: "AVAILABLE",
      training_duration_seconds: 120,
      threshold_steps: { "0.9000": 50 }
    },
    quality: {
      validation: null,
      test: {
        schema_version: "dashboard-quality-metrics-v1",
        split: "test",
        count: 10,
        loss: null,
        accuracy: 0.95,
        macro_task_accuracy: 0.95,
        uncertainty_precision: null,
        uncertainty_recall: null,
        invalid_output_rate: 0,
        by_task: null,
        by_difficulty: null,
        by_task_difficulty: null
      }
    },
    performance: {
      schema_version: "dashboard-performance-metrics-v1",
      training_duration_seconds: 120,
      checkpoint_size_bytes: null,
      inference: [],
      telemetry_status: "NOT_AVAILABLE"
    },
    population_metrics: null,
    runtime_version: null,
    evidence_contract_version: null,
    aggregation_config: null,
    provenance: {
      schema_version: "dashboard-provenance-v1",
      git_revision: "abcdef123456",
      device: "cuda",
      artifact: {
        schema_version: "dashboard-artifact-ref-v1",
        artifact_id: "artifact_1",
        artifact_ref: "step01/run/result.json"
      },
      checkpoint_ref: null,
      indexed_at: "2026-07-18T00:00:00Z",
      adapter_id: "step01-result-v1",
      raw_schema_version: "step01-v0",
      duplicate_artifact_count: 0,
      duplicate_artifact_refs: []
    },
    status: "COMPLETE",
    warnings: [],
    ...overrides
  } as ExperimentRunV1;
}

function populationExperiment(): ExperimentRunV1 {
  return experiment({
    identity: {
      schema_version: "dashboard-experiment-identity-v1",
      experiment_id: "run_step2",
      experiment_name: "step02_population_2w",
      research_step: "Step 2",
      experiment_type: "population",
      benchmark_version: "step01-v0",
      architecture_version: null,
      run_variant: "validation"
    },
    architecture: {
      schema_version: "dashboard-architecture-identity-v1",
      architecture_id: "arch_step2",
      architecture_version: null,
      model_type: "population",
      parameter_count: null,
      parameter_count_status: "NOT_APPLICABLE",
      worker_parameter_count: null,
      worker_parameter_count_status: "NOT_AVAILABLE",
      population_width: 2,
      population_width_status: "AVAILABLE",
      total_learned_parameters: null,
      total_learned_parameters_status: "NOT_AVAILABLE",
      config_fingerprint: "def",
      display_label: "Population 2 workers"
    },
    population_metrics: {
      schema_version: "dashboard-population-metrics-v1",
      population_width: 2,
      execution_backend: "vmap",
      evidence_reducer_accuracy: 0.8,
      majority_vote_accuracy: 0.7,
      mean_logit_accuracy: 0.75,
      mean_probability_accuracy: 0.76,
      oracle_any_correct_coverage: 0.9,
      all_wrong_rate: 0.1,
      minority_rescue_opportunity_rate: 0.2,
      minority_rescue_rate: 0.5,
      minority_suppression_rate: 0.25,
      majority_harm_rate: 0.05,
      evidence_utilization_gap: 0.1,
      mean_disagreement_entropy: 0.3,
      mean_population_uncertainty: 0.4,
      mean_invalid_label_mass: 0.05,
      individual_worker_accuracies: [0.7, 0.8],
      by_task: null
    },
    runtime_version: "step02-population-runtime-v0",
    evidence_contract_version: "step02-evidence-v0"
  });
}

function renderRoute(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  state.status = baseStatus();
  state.experiments = [];
  state.indexErrors = [];
  state.failStatus = false;
  state.failExperiments = false;
  state.failIndexErrors = false;
  state.failExperimentLookup = false;
});

afterEach(() => {
  vi.clearAllMocks();
  cleanup();
});

describe("App shell and zero data", () => {
  it("renders all six navigation sections with zero data", async () => {
    renderRoute("/");
    for (const label of ["Overview", "Experiments", "Compare", "Scaling", "Population", "Evidence"]) {
      expect(screen.getByRole("link", { name: label })).toBeInTheDocument();
    }
    expect(await screen.findByText("No Step 1 results indexed")).toBeInTheDocument();
  });

  it("keeps every top-level section reachable with neither Step 1 nor Step 2 data", async () => {
    for (const [route, text] of [
      ["/experiments", "No experiments found."],
      ["/compare", "Comparison selection required"],
      ["/scaling", "Population Organization Matrix"],
      ["/population", "No population experiments"],
      ["/evidence", "Evidence reconstruction unavailable"]
    ]) {
      const view = renderRoute(route);
      expect(await screen.findByText(text)).toBeInTheDocument();
      view.unmount();
    }
  });
});

describe("Experiment states", () => {
  it("renders Overview with Step 1-only data", async () => {
    state.status = baseStatus({ indexed_experiment_count: 1 });
    state.experiments = [experiment()];
    renderRoute("/");
    expect(await screen.findByText("step01_checkpoint_50k_extended_15k")).toBeInTheDocument();
    expect(screen.getByText("50,268 params")).toBeInTheDocument();
  });

  it("renders Experiments empty and populated states", async () => {
    const empty = renderRoute("/experiments");
    expect(await screen.findByText("No experiments found.")).toBeInTheDocument();
    empty.unmount();

    state.experiments = [experiment()];
    renderRoute("/experiments");
    expect(await screen.findByRole("link", { name: "step01_checkpoint_50k_extended_15k" })).toBeInTheDocument();
    expect(screen.getAllByText("95.00%").length).toBeGreaterThanOrEqual(1);
  });

  it("renders indexing errors separately from experiment rows", async () => {
    state.indexErrors = [
      {
        schema_version: "dashboard-index-error-v1",
        error_id: "err_1",
        artifact: {
          schema_version: "dashboard-artifact-ref-v1",
          artifact_id: "artifact_bad",
          artifact_ref: "bad/result.json"
        },
        phase: "VALIDATION",
        severity: "ERROR",
        message: "missing metrics",
        adapter_id: "step02-population-v1",
        recovered_identity: null
      }
    ];
    renderRoute("/experiments");
    expect(await screen.findByText("Index Errors")).toBeInTheDocument();
    expect(screen.getByText("bad/result.json")).toBeInTheDocument();
  });

  it("renders missing numeric values as unavailable rather than zero", async () => {
    state.experiments = [
      experiment({
        identity: { ...experiment().identity, experiment_id: "run_missing", experiment_name: "missing_numeric_run" },
        architecture: { ...experiment().architecture, parameter_count: null },
        training: { ...experiment().training, best_validation_score: null, training_duration_seconds: null },
        quality: { validation: null, test: null }
      })
    ];
    renderRoute("/experiments");
    expect(await screen.findByText("missing_numeric_run")).toBeInTheDocument();
    const row = screen.getByText("missing_numeric_run").closest("tr");
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getAllByText("—").length).toBeGreaterThanOrEqual(3);
  });

  it("renders invalid experiment id as not found", async () => {
    renderRoute("/experiments/not-real");
    expect(await screen.findByText("Experiment not found")).toBeInTheDocument();
  });

  it("renders Step 2-only population data without requiring Step 1", async () => {
    state.status = baseStatus({ indexed_experiment_count: 1, population_experiment_count: 1 });
    state.experiments = [populationExperiment()];
    renderRoute("/population");
    expect(await screen.findByText("Evidence Reducer")).toBeInTheDocument();
    expect(screen.getByText("80.00%")).toBeInTheDocument();
  });
});

describe("API errors", () => {
  it("renders backend/API error state", async () => {
    state.failExperiments = true;
    renderRoute("/experiments");
    expect(await screen.findByText("API ERROR")).toBeInTheDocument();
  });
});
