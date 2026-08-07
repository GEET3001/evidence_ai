"use client";

import { useEffect, useState } from "react";

/**
 * What the pipeline is doing while it does it.
 *
 * The elapsed counter is real; the bar is deliberately indeterminate. `/verify`
 * is one request that returns once, so the client cannot know which stage is
 * running — a bar that crept to 90% would be inventing progress it does not
 * have. Naming the three stages tells the reader what the wait is for, which is
 * the honest version of the same reassurance.
 */

const STAGES = [
  {
    name: "Retrieve",
    detail: "BM25 and dense similarity scored across every passage in the corpus",
  },
  {
    name: "Classify",
    detail:
      "An NLI pass per passage, then one per sentence to find the rationale",
  },
  {
    name: "Aggregate",
    detail: "Counts, thresholds, and the certainty grade",
  },
];

const SLOW_AFTER_MS = 10_000;

export function VerifyingState() {
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    const startedAt = performance.now();
    const id = setInterval(() => setElapsedMs(performance.now() - startedAt), 100);
    return () => clearInterval(id);
  }, []);

  return (
    <div
      className="rounded-lg border border-rule bg-surface"
      role="status"
      aria-live="polite"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 px-5 pt-5">
        <h2 className="font-mono text-[11px] font-semibold tracking-[0.14em] text-ink uppercase">
          Verifying
        </h2>
        <p className="font-mono text-sm tabular-nums text-muted">
          {(elapsedMs / 1000).toFixed(1)}s
          <span className="ml-2 text-faint">usually 2–4s</span>
        </p>
      </div>

      <div className="mt-4 h-0.5 w-full overflow-hidden bg-sunk">
        <div className="sweep relative h-full w-full" />
      </div>

      <ol className="flex flex-col">
        {STAGES.map((stage, index) => (
          <li
            key={stage.name}
            className={`flex gap-4 px-5 py-3.5 ${index > 0 ? "border-t border-rule" : ""}`}
          >
            <span className="w-20 shrink-0 font-mono text-[11px] tracking-[0.1em] text-ink uppercase">
              {stage.name}
            </span>
            <span className="text-xs leading-relaxed text-muted">
              {stage.detail}
            </span>
          </li>
        ))}
      </ol>

      {elapsedMs > SLOW_AFTER_MS ? (
        <p className="border-t border-rule bg-sunk px-5 py-3 text-xs leading-relaxed text-muted">
          Longer than usual. The first verification after the backend starts
          loads the models, which adds several seconds.
        </p>
      ) : null}
    </div>
  );
}
