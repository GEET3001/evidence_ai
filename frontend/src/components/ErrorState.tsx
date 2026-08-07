import type { ApiErrorKind } from "@/lib/api";

/**
 * A failed verification, told as what happened and what to do about it.
 *
 * Each kind gets its own recovery line, because "something went wrong" leaves
 * the reader with nothing to act on. The client's own message is kept as the
 * detail line since it carries the specifics — the API URL, the status code.
 */

export type ErrorKind = ApiErrorKind | "unknown";

const COPY: Record<ErrorKind, { headline: string; recovery: string }> = {
  network: {
    headline: "The API is not responding",
    recovery:
      "The backend may not be running yet. Start it, wait for the models to load, then verify again.",
  },
  timeout: {
    headline: "The request timed out",
    recovery:
      "Verification ran past 60 seconds. The first request after startup loads the models and is much slower than the rest — try again.",
  },
  http: {
    headline: "The API rejected the request",
    recovery:
      "The claim reached the backend but it could not complete the verification. Check the backend logs for the matching request.",
  },
  parse: {
    headline: "The API returned something unreadable",
    recovery:
      "The response was not valid JSON, which usually means the frontend is pointed at the wrong service.",
  },
  unknown: {
    headline: "The verification failed",
    recovery: "Try again. If it keeps failing, check the backend logs.",
  },
};

export function ErrorState({
  kind,
  detail,
  onRetry,
}: {
  kind: ErrorKind;
  detail: string;
  onRetry: () => void;
}) {
  const copy = COPY[kind];

  return (
    <div className="rounded-lg border border-warn bg-warn-soft p-5" role="alert">
      <h2 className="text-base font-semibold text-warn">{copy.headline}</h2>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-ink">
        {copy.recovery}
      </p>
      <p className="mt-3 font-mono text-[11px] leading-relaxed break-words text-muted">
        {detail}
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-4 rounded border border-warn px-3 py-1.5 font-mono text-[11px] tracking-wide text-warn transition-colors hover:bg-warn hover:text-surface"
      >
        Try again
      </button>
    </div>
  );
}
