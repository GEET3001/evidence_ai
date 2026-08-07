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

/** Filename from Content-Disposition, preferring the RFC 5987 encoded form. */
function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null;

  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(header);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1]);
    } catch {
      // Malformed percent-encoding — fall through to the plain form.
    }
  }

  const plain = /filename="([^"]+)"/i.exec(header);
  return plain ? plain[1] : null;
}

/**
 * GET /verify/{id}/report — download the .docx report.
 *
 * Fetched as a blob rather than pointed at with an anchor. A bare navigation
 * hands the URL to the browser, so a 404 or a 500 renders as a JSON error page
 * in a new tab with no way for the app to notice; going through fetch keeps
 * failures as ApiError and lets the UI report them where the user is.
 */
export async function downloadReport(id: string): Promise<void> {
  const path = `/verify/${encodeURIComponent(id)}/report`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { signal: controller.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("Generating the report took too long.", "timeout");
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
      if (body?.detail) detail = String(body.detail);
    } catch {
      // Error body wasn't JSON — fall back to statusText.
    }
    throw new ApiError(
      `Report download failed (${response.status}): ${detail}`,
      "http",
      response.status
    );
  }

  const blob = await response.blob();
  const filename =
    filenameFromDisposition(response.headers.get("Content-Disposition")) ??
    `evidenceai-report-${id}.docx`;

  const objectUrl = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    link.rel = "noopener";
    document.body.appendChild(link);
    link.click();
    link.remove();
  } finally {
    // Revoking immediately can cancel the download in some browsers; one turn
    // of the event loop is enough for the click to have been handled.
    setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  }
}
