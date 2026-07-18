import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { apiClient, ApiError } from "../api/client";
import type { ExperimentRunV1 } from "../api/types";
import { EmptyState, ErrorBanner, formatNumber, formatPercent, MetricTile } from "../components/Display";

export function RunDetail() {
  const { experimentId } = useParams();
  const [run, setRun] = useState<ExperimentRunV1 | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      if (!experimentId) return;
      try {
        setRun(await apiClient.experiment(experimentId));
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : "Experiment lookup failed.");
      }
    }
    void load();
  }, [experimentId]);

  if (error) return <ErrorBanner title="Experiment not found" message={error} />;
  if (!run) return <EmptyState title="Loading experiment" message="Reading normalized run detail." />;

  return (
    <div className="page-stack">
      <section className="panel">
        <h2>{run.identity.experiment_name}</h2>
        <p className="mono">{run.identity.experiment_id}</p>
        <div className="metric-row">
          <MetricTile label="Best Validation" value={formatPercent(run.training.best_validation_score)} />
          <MetricTile label="Latest Validation" value={formatPercent(run.training.latest_validation?.accuracy)} />
          <MetricTile label="Test Accuracy" value={formatPercent(run.quality.test?.accuracy)} />
          <MetricTile label="Parameters" value={formatNumber(run.architecture.parameter_count ?? run.architecture.total_learned_parameters)} />
        </div>
      </section>

      <section className="panel">
        <h2>Summary</h2>
        <dl className="detail-grid">
          <dt>Research Step</dt><dd>{run.identity.research_step}</dd>
          <dt>Type</dt><dd>{run.identity.experiment_type}</dd>
          <dt>Artifact</dt><dd className="mono">{run.provenance.artifact.artifact_ref}</dd>
          <dt>Adapter</dt><dd>{run.provenance.adapter_id}</dd>
          <dt>Git Revision</dt><dd className="mono">{run.provenance.git_revision ?? "—"}</dd>
          <dt>Evidence Contract</dt><dd>{run.evidence_contract_version ?? "N/A"}</dd>
        </dl>
      </section>

      <section className="panel">
        <h2>Training</h2>
        <table>
          <thead><tr><th>Threshold</th><th>First Step Reached</th></tr></thead>
          <tbody>
            {Object.entries(run.training.threshold_steps).map(([threshold, step]) => (
              <tr key={threshold}><td>{formatPercent(Number(threshold))}</td><td>{step ?? "Not reached"}</td></tr>
            ))}
            {Object.keys(run.training.threshold_steps).length === 0 ? (
              <tr><td colSpan={2}>No threshold history available.</td></tr>
            ) : null}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>Quality</h2>
        <dl className="detail-grid">
          <dt>Overall Accuracy</dt><dd>{formatPercent(run.quality.test?.accuracy ?? run.quality.validation?.accuracy)}</dd>
          <dt>Macro Task Accuracy</dt><dd>{formatPercent(run.quality.test?.macro_task_accuracy ?? run.quality.validation?.macro_task_accuracy)}</dd>
          <dt>Invalid Output Rate</dt><dd>{formatPercent(run.quality.test?.invalid_output_rate ?? run.quality.validation?.invalid_output_rate)}</dd>
        </dl>
      </section>

      <section className="panel">
        <h2>Performance</h2>
        <dl className="detail-grid">
          <dt>Training Duration</dt><dd>{run.training.training_duration_seconds ? `${run.training.training_duration_seconds.toFixed(1)}s` : "—"}</dd>
          <dt>Checkpoint Size</dt><dd>{formatNumber(run.performance?.checkpoint_size_bytes)}</dd>
          <dt>Telemetry</dt><dd>{run.performance?.telemetry_status ?? "NOT_AVAILABLE"}</dd>
        </dl>
      </section>

      <section className="panel">
        <h2>Configuration</h2>
        <pre>{JSON.stringify({ runtime_version: run.runtime_version, aggregation_config: run.aggregation_config }, null, 2)}</pre>
      </section>
    </div>
  );
}
