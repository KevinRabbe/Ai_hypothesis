import type { ReactNode } from "react";

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat().format(value);
}

export function formatDuration(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value < 120) return `${value.toFixed(1)}s`;
  return `${(value / 60).toFixed(1)}m`;
}

export function MetricTile({
  label,
  value,
  detail
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
}) {
  return (
    <section className="metric-tile">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {detail ? <div className="metric-detail">{detail}</div> : null}
    </section>
  );
}

export function StatusBadge({ children, tone = "neutral" }: { children: ReactNode; tone?: string }) {
  return <span className={`status-badge ${tone}`}>{children}</span>;
}

export function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <section className="empty-state">
      <h2>{title}</h2>
      <p>{message}</p>
    </section>
  );
}

export function ErrorBanner({ title, message }: { title: string; message: string }) {
  return (
    <section className="error-banner">
      <strong>{title}</strong>
      <span>{message}</span>
    </section>
  );
}
