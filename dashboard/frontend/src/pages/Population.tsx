import { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { ExperimentRunV1 } from "../api/types";
import { EmptyState, formatPercent, MetricTile } from "../components/Display";

export function Population() {
  const [runs, setRuns] = useState<ExperimentRunV1[]>([]);
  useEffect(() => {
    apiClient.experiments().then((response) => {
      setRuns(response.items.filter((item) => item.identity.experiment_type === "population"));
    }).catch(() => setRuns([]));
  }, []);
  if (!runs.length) {
    return <EmptyState title="No population experiments" message="No population experiments are currently available." />;
  }
  const run = runs[0];
  const metrics = run.population_metrics;
  return (
    <div className="page-stack">
      <section className="metric-row">
        <MetricTile label="Population Width" value={metrics?.population_width ?? "—"} />
        <MetricTile label="Evidence Reducer" value={formatPercent(metrics?.evidence_reducer_accuracy)} />
        <MetricTile label="Majority Vote" value={formatPercent(metrics?.majority_vote_accuracy)} />
        <MetricTile label="Oracle Coverage" value={formatPercent(metrics?.oracle_any_correct_coverage)} />
      </section>
      <section className="panel">
        <h2>Population Diagnostics</h2>
        <p>Full population analytics UI is reserved for a later slice. Only normalized Step 2 metrics that exist in artifacts are shown here.</p>
      </section>
    </div>
  );
}
