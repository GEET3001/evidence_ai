"use client";

import { useState } from "react";
import { ApiError, downloadReport } from "@/lib/api";

/**
 * Export the verdict as a .docx report.
 *
 * The report is built server-side from the stored verdict, so this button is
 * only meaningful once that verdict has been persisted — which happens as part
 * of the verification that produced the id. Failures are shown here rather
 * than thrown away, since a download that silently does nothing is the worst
 * of the available outcomes.
 */
export function DownloadReport({ verificationId }: { verificationId: string }) {
  const [status, setStatus] = useState<"idle" | "working">("idle");
  const [error, setError] = useState<string | null>(null);

  async function handleClick() {
    setStatus("working");
    setError(null);
    try {
      await downloadReport(verificationId);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "The report could not be generated."
      );
    } finally {
      setStatus("idle");
    }
  }

  return (
    <div className="flex flex-col items-start gap-2">
      <button
        type="button"
        onClick={() => void handleClick()}
        disabled={status === "working"}
        className="rounded border border-rule-strong bg-surface px-4 py-2 text-sm font-medium text-ink transition-colors hover:border-ink disabled:cursor-not-allowed disabled:opacity-50"
      >
        {status === "working" ? "Preparing report…" : "Download report (.docx)"}
      </button>
      <p className="max-w-md font-mono text-[10px] leading-relaxed text-faint">
        Full analysis: verdict, evidence table, conflicting findings,
        limitations, and the methodology appendix.
      </p>
      {error ? (
        <p className="max-w-md text-xs leading-relaxed text-warn" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
