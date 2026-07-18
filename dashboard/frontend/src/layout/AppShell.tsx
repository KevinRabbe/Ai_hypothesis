import { useEffect, useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { apiClient, ApiError } from "../api/client";
import type { DashboardStatusV1 } from "../api/types";
import { StatusBadge } from "../components/Display";

const navItems = [
  ["Overview", "/"],
  ["Experiments", "/experiments"],
  ["Compare", "/compare"],
  ["Scaling", "/scaling"],
  ["Population", "/population"],
  ["Evidence", "/evidence"]
];

export function AppShell({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<DashboardStatusV1 | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  async function loadStatus() {
    try {
      setError(null);
      setStatus(await apiClient.status());
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Dashboard status failed.");
    }
  }

  async function reindex() {
    setRefreshing(true);
    try {
      setError(null);
      setStatus(await apiClient.reindex());
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Reindex failed.");
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    void loadStatus();
  }, []);

  return (
    <div className="app-shell">
      <aside className="side-nav">
        <div className="brand">AI Hypothesis</div>
        <nav>
          {navItems.map(([label, to]) => (
            <NavLink key={to} to={to} end={to === "/"}>
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="main-frame">
        <header className="top-bar">
          <div>
            <h1>Research Dashboard</h1>
            <p>Local experiment evidence console</p>
          </div>
          <div className="top-actions">
            {status ? (
              <StatusBadge tone={status.service_status === "HEALTHY" ? "ok" : "warn"}>
                {status.index_status}
              </StatusBadge>
            ) : (
              <StatusBadge>LOADING</StatusBadge>
            )}
            <button onClick={reindex} disabled={refreshing}>
              {refreshing ? "Indexing" : "Refresh"}
            </button>
          </div>
        </header>
        {error ? <div className="api-error">API ERROR: {error}</div> : null}
        <main>{children}</main>
      </div>
    </div>
  );
}
