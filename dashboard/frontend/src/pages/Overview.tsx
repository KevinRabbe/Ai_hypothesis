import { useEffect, useState } from "react";
import { apiClient, ApiError } from "../api/client";
import type { DashboardStatusV1, ExperimentRunV1, IndexErrorV1 } from "../api/types";
import { EmptyState, ErrorBanner, formatNumber, formatPercent, MetricTile } from "../components/Display";

export function Overview() {
  const [status, setStatus] = useState<DashboardStatusV1 | null>(null);
  const [experiments, setExperiments] = useState<ExperimentRunV1[]>([]);
  const [errors, setErrors] = useState<IndexErrorV1[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [nextStatus, nextExperiments, nextErrors] = await Promise.all([
          apiClient.status(),
          apiClient.experiments(),
          apiClient.indexErrors()
        ]);
        setStatus(nextStatus);
        setExperiments(nextExperiments.items);
        setErrors(nextErrors.items);
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : "Overview failed.");
      }
    }
    void load();
  }, []);

  if (error) return <ErrorBanner title="API ERROR" message={error} />;

  const step1 = experiments.filter((item) => item.identity.research_step === "Step 1");
  const architectures = Array.from(
    new Map(step1.map((item) => [item.architecture.parameter_count, item])).values()
  ).sort((a, b) => (a.architecture.parameter_count ?? 0) - (b.architecture.parameter_count ?? 0));

  return (
    <div className="page-stack">
      <section className="metric-row">
        <MetricTile label="Indexed Experiments" value={status?.indexed_experiment_count ?? "—"} />
        <MetricTile label="Indexing Errors" value={status?.indexing_error_count ?? "—"} />
        <MetricTile label="Complete Multi-seed Groups" value={status?.complete_group_count ?? "—"} />
        <MetricTile label="Worker Architectures" value={architectures.length} />
        <MetricTile label="Population Runs" value={status?.population_experiment_count ?? "—"} />
      </section>

      <section className="panel">
        <h2>Step 1 Architecture-Size Summary</h2>
        {architectures.length === 0 ? (
          <EmptyState title="No Step 1 results indexed" message="No experiment results have been indexed yet." />
        ) : (
          <table>
            <thead>
              <tr>
                <th>Architecture</th>
                <th>Parameters</th>
                <th>Seed</th>
                <th>Best Validation</th>
                <th>Test Accuracy</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {architectures.map((item) => (
                <tr key={item.identity.experiment_id}>
                  <td>{item.architecture.display_label}</td>
                  <td>{formatNumber(item.architecture.parameter_count)}</td>
                  <td>{item.training.seed ?? "—"}</td>
                  <td>{formatPercent(item.training.best_validation_score)}</td>
                  <td>{formatPercent(item.quality.test?.accuracy)}</td>
                  <td>{item.training.training_duration_seconds ? `${(item.training.training_duration_seconds / 60).toFixed(1)}m` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2>Latest Indexed Artifacts</h2>
        {experiments.length === 0 ? (
          <EmptyState title="No artifacts indexed" message="The dashboard is ready and waiting for result artifacts." />
        ) : (
          <table>
            <thead>
              <tr><th>Experiment</th><th>Artifact</th><th>Status</th></tr>
            </thead>
            <tbody>
              {experiments.slice(0, 8).map((item) => (
                <tr key={item.identity.experiment_id}>
                  <td>{item.identity.experiment_name}</td>
                  <td className="mono">{item.provenance.artifact.artifact_ref}</td>
                  <td>{item.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {errors.length > 0 ? (
        <section className="panel warning">
          <h2>Recent Indexing Problems</h2>
          <table>
            <thead><tr><th>Artifact</th><th>Phase</th><th>Message</th></tr></thead>
            <tbody>
              {errors.slice(0, 6).map((item) => (
                <tr key={item.error_id}>
                  <td className="mono">{item.artifact.artifact_ref}</td>
                  <td>{item.phase}</td>
                  <td>{item.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
    </div>
  );
}
