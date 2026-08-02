import type { VerdictResponse } from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 60_000;

export type ApiErrorKind = "timeout" | "network" | "http" | "parse";

/** Typed error for all API client failures — check `.kind` to branch UI behavior. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;

  constructor(message: string, kind: ApiErrorKind, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(
        "The request took too long and was cancelled. Verification can be slow — try again.",
        "timeout"
      );
    }
    throw new ApiError(
      `Could not reach the API at ${API_BASE_URL}. Is the backend running?`,
      "network"
    );
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body?.detail) detail = JSON.stringify(body.detail);
    } catch {
      // Response body wasn't JSON — fall back to statusText.
    }
    throw new ApiError(`Request failed (${response.status}): ${detail}`, "http", response.status);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("The server returned a response that wasn't valid JSON.", "parse");
  }
}

/** POST /verify — retrieve, classify, and aggregate evidence for a claim. */
export function verifyClaim(claim: string): Promise<VerdictResponse> {
  return request<VerdictResponse>("/verify", {
    method: "POST",
    body: JSON.stringify({ claim }),
  });
}

/** GET /verify/{id} — fetch a previously computed verdict. */
export function getVerification(id: string): Promise<VerdictResponse> {
  return request<VerdictResponse>(`/verify/${encodeURIComponent(id)}`);
}

/** GET /verify/{id}/report — trigger a browser download of the .docx report. */
export function downloadReport(id: string): void {
  const url = `${API_BASE_URL}/verify/${encodeURIComponent(id)}/report`;
  const link = document.createElement("a");
  link.href = url;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
}
