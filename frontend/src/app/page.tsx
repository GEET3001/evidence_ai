"use client";

import { useCallback, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ErrorState, type ErrorKind } from "@/components/ErrorState";
import { VerifyingState } from "@/components/VerifyingState";
import { ResultsView } from "@/components/results/ResultsView";
import { ApiError, verifyClaim } from "@/lib/api";
import type { VerdictResponse } from "@/types";

const EXAMPLE_CLAIMS = [
  "Mindfulness-based interventions reduce ADHD symptoms in adults",
  "Exercise is as effective as antidepressants for major depression",
  "Eating chocolate cake improves stock market forecasting",
];

interface FailedRequest {
  kind: ErrorKind;
  detail: string;
}

export default function Home() {
  const [claim, setClaim] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<FailedRequest | null>(null);
  const [result, setResult] = useState<VerdictResponse | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const runVerification = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      setResult(await verifyClaim(trimmed));
    } catch (err) {
      setError(
        err instanceof ApiError
          ? { kind: err.kind, detail: err.message }
          : { kind: "unknown", detail: String(err) }
      );
    } finally {
      setLoading(false);
    }
  }, []);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!loading) void runVerification(claim);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      if (!loading) void runVerification(claim);
    }
  }

  function fillFromExample(example: string) {
    setClaim(example);
    inputRef.current?.focus();
  }

  const canSubmit = claim.trim().length > 0 && !loading;
  const showEmptyState = !loading && !error && !result;

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-10 px-6 py-10">
      <header>
        <p className="font-mono text-[11px] font-semibold tracking-[0.18em] text-faint uppercase">
          EvidenceAI
        </p>
        <h1 className="mt-2 max-w-2xl text-2xl leading-tight font-semibold tracking-tight text-balance text-ink sm:text-3xl">
          Check a research claim against the mental health literature.
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
          Every verdict below is traceable to the passages it came from and the
          counts it was computed from.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <label
          htmlFor="claim"
          className="font-mono text-[10px] font-semibold tracking-[0.14em] text-faint uppercase"
        >
          Claim
        </label>
        <textarea
          id="claim"
          ref={inputRef}
          value={claim}
          onChange={(event) => setClaim(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Mindfulness meditation reduces anxiety in college students"
          rows={3}
          className="w-full resize-y rounded-lg border border-rule bg-surface p-4 font-body text-[15px] leading-relaxed text-ink placeholder:text-faint focus:border-ink focus:outline-none"
        />
        <div className="flex flex-wrap items-center gap-4">
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-opacity disabled:cursor-not-allowed disabled:opacity-35"
          >
            {loading ? "Verifying…" : "Verify claim"}
          </button>
          <span className="font-mono text-[11px] text-faint">⌘/Ctrl + Enter</span>
        </div>
      </form>

      {showEmptyState ? (
        <div className="border-t border-rule pt-6">
          <p className="font-mono text-[10px] font-semibold tracking-[0.14em] text-faint uppercase">
            Or start from one of these
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            {EXAMPLE_CLAIMS.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => fillFromExample(example)}
                className="rounded border border-rule bg-surface px-3 py-2 text-left font-body text-sm text-muted transition-colors hover:border-ink hover:text-ink"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {loading ? <VerifyingState /> : null}

      {error ? (
        <ErrorState
          kind={error.kind}
          detail={error.detail}
          onRetry={() => void runVerification(claim)}
        />
      ) : null}

      {result ? <ResultsView result={result} /> : null}
    </main>
  );
}
