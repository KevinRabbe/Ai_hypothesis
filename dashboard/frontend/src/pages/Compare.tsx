import { EmptyState } from "../components/Display";

export function Compare() {
  return (
    <EmptyState
      title="Comparison selection required"
      message="At least two compatible experiment runs are required for comparison."
    />
  );
}
