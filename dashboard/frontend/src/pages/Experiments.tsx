import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { apiClient, ApiError } from "../api/client";
import type { ExperimentRunV1, IndexErrorV1 } from "../api/types";
import { EmptyState, ErrorBanner, formatDuration, formatNumber, formatPercent } from "../components/Display";

export function Experiments() {
  const [items, setItems] = useState<ExperimentRunV1[]>([]);
  const [errors, setErrors] = useState<IndexErrorV1[]>([]);
  const [search, setSearch] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [experiments, indexErrors] = await Promise.all([
          apiClient.experiments(),
          apiClient.indexErrors()
        ]);
        setItems(experiments.items);
        setErrors(indexErrors.items);
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : "Experiments failed.");
      }
    }
    void load();
  }, []);

  const filtered = useMemo(() => {
    const needle = search.toLowerCase();
    return items.filter((item) =>
      [item.identity.experiment_name, item.identity.experiment_id, item.provenance.git_revision ?? ""]
        .join(" ")
        .toLowerCase()
        .includes(needle)
    );
  }, [items, search]);

  if (error) return <ErrorBanner title="API ERROR" message={error} />;

  return (
    <div className="page-stack">
      <section className="panel">
        <h2>Experiments</h2>
        <div className="filters">
          <label>
            Search
            <input value={search} onChange={(event) => setSearch(event.target.value)} />
          </label>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Research Step</th>
                <th>Architecture</th>
                <th>Parameter Count</th>
                <th>Total Parameter Budget</th>
                <th>Population Width</th>
                <th>Seed</th>
                <th>Split</th>
                <th>Best Validation</th>
                <th>Test Accuracy</th>
                <th>Training Duration</th>
                <th>Git Revision</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => (
                <tr key={item.identity.experiment_id}>
                  <td><Link to={`/experiments/${item.identity.experiment_id}`}>{item.identity.experiment_name}</Link></td>
                  <td>{item.identity.research_step}</td>
                  <td>{item.architecture.display_label}</td>
                  <td>{formatNumber(item.architecture.parameter_count)}</td>
                  <td>{formatNumber(item.architecture.total_learned_parameters)}</td>
                  <td>{formatNumber(item.architecture.population_width)}</td>
                  <td>{item.training.seed ?? "—"}</td>
                  <td>{item.quality.test ? "test" : item.quality.validation ? "validation" : "—"}</td>
                  <td>{formatPercent(item.training.best_validation_score)}</td>
                  <td>{formatPercent(item.quality.test?.accuracy)}</td>
                  <td>{formatDuration(item.training.training_duration_seconds)}</td>
                  <td className="mono">{item.provenance.git_revision?.slice(0, 8) ?? "—"}</td>
                  <td>{item.status}</td>
                </tr>
              ))}
              {filtered.length === 0 ? (
                <tr><td colSpan={13}>No experiments found.</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      {errors.length ? (
        <section className="panel warning">
          <h2>Index Errors</h2>
          <table>
            <thead><tr><th>Artifact</th><th>Phase</th><th>Message</th></tr></thead>
            <tbody>
              {errors.map((item) => (
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
