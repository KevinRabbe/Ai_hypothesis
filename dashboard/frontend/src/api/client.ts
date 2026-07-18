import type {
  DashboardStatusV1,
  ExperimentListResponseV1,
  ExperimentRunV1,
  IndexErrorV1
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly kind: "HTTP" | "NETWORK" | "SCHEMA",
    public readonly status?: number
  ) {
    super(message);
  }
}

async function request<T>(
  path: string,
  expectedSchema: string,
  init?: RequestInit
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, init);
  } catch (error) {
    throw new ApiError("Dashboard API is unreachable.", "NETWORK");
  }
  if (!response.ok) {
    throw new ApiError(`Dashboard API returned ${response.status}.`, "HTTP", response.status);
  }
  const payload = await response.json();
  if (payload.schema_version !== expectedSchema) {
    throw new ApiError("Dashboard API contract mismatch.", "SCHEMA");
  }
  return payload as T;
}

export const apiClient = {
  async status(): Promise<DashboardStatusV1> {
    const payload = await request<{ status: DashboardStatusV1 }>(
      "/status",
      "dashboard-status-response-v1"
    );
    return payload.status;
  },
  async experiments(): Promise<ExperimentListResponseV1> {
    return request<ExperimentListResponseV1>(
      "/experiments",
      "dashboard-experiment-list-response-v1"
    );
  },
  async experiment(id: string): Promise<ExperimentRunV1> {
    const payload = await request<{ experiment: ExperimentRunV1 }>(
      `/experiments/${encodeURIComponent(id)}`,
      "dashboard-experiment-detail-response-v1"
    );
    return payload.experiment;
  },
  async indexErrors(): Promise<{ items: IndexErrorV1[]; total: number }> {
    return request<{ items: IndexErrorV1[]; total: number }>(
      "/index-errors",
      "dashboard-index-error-list-response-v1"
    );
  },
  async reindex(): Promise<DashboardStatusV1> {
    const payload = await request<{ status: DashboardStatusV1 }>(
      "/reindex",
      "dashboard-reindex-response-v1",
      { method: "POST" }
    );
    return payload.status;
  }
};
