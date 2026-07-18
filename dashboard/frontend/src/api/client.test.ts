import { afterEach, describe, expect, it, vi } from "vitest";
import { apiClient, ApiError } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("apiClient", () => {
  it("reports unknown API schema versions as contract mismatches", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ schema_version: "unexpected-version" })
      }))
    );

    await expect(apiClient.experiments()).rejects.toMatchObject({
      kind: "SCHEMA",
      message: "Dashboard API contract mismatch."
    } satisfies Partial<ApiError>);
  });

  it("reports backend unavailability", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new Error("network down");
    }));

    await expect(apiClient.status()).rejects.toMatchObject({
      kind: "NETWORK",
      message: "Dashboard API is unreachable."
    } satisfies Partial<ApiError>);
  });
});
