import { EmptyState } from "../components/Display";

export function Evidence() {
  return (
    <EmptyState
      title="Evidence reconstruction unavailable"
      message="Select a compatible population experiment and validation sample after the reconstruction service is implemented. Test-set detail remains disabled by default."
    />
  );
}
