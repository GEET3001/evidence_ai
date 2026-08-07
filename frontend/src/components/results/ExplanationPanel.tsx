import type { VerdictResponse } from "@/types";

/**
 * Why the pipeline said what it said, plus what would undercut it.
 *
 * The explanation arrives as a string the backend assembled from counts and
 * thresholds. Numbers inside it are set in monospace here, which is the same
 * rule the rest of the page follows: monospace means a value something
 * computed, serif means text quoted out of a paper. That distinction is worth
 * showing a reader, not just recording in the report.
 */

const NUMBER = /(\d+(?:\.\d+)?)/g;

export function ExplanationPanel({ result }: { result: VerdictResponse }) {
  // A claim the corpus does not cover has no passages behind it, so the note
  // must not claim quotations it did not make.
  const hasEvidence = result.evidence.length > 0;

  return (
    <div className="flex flex-col gap-5">
      <div className="rounded-lg border border-rule bg-surface p-5">
        <p className="text-[15px] leading-relaxed text-ink">
          {result.explanation.split(NUMBER).map((chunk, index) =>
            index % 2 === 1 ? (
              <span key={index} className="font-mono text-sm tabular-nums">
                {chunk}
              </span>
            ) : (
              <span key={index}>{chunk}</span>
            )
          )}
        </p>
      </div>

      <div>
        <h3 className="font-mono text-[11px] font-semibold tracking-[0.14em] text-ink uppercase">
          Limitations
        </h3>
        <ul className="mt-3 flex flex-col gap-2.5">
          {result.limitations.map((limitation, index) => (
            <li
              key={index}
              className="flex gap-3 text-sm leading-relaxed text-muted"
            >
              <span aria-hidden className="mt-2 size-1.5 shrink-0 rounded-full bg-rule-strong" />
              <span>{limitation}</span>
            </li>
          ))}
        </ul>
      </div>

      <p className="rounded-lg border border-dashed border-rule-strong bg-sunk p-4 text-xs leading-relaxed text-muted">
        <span className="font-semibold text-ink">
          How this explanation was produced.
        </span>{" "}
        It is assembled from the counts and thresholds above
        {hasEvidence
          ? " and from sentences quoted verbatim out of the source passages"
          : ""}
        . No language model writes any of it, so every part of it can be traced
        back to a number{hasEvidence ? " or a quotation" : ""} you can check on
        this page. That is a deliberate constraint: an explanation that reads
        well but was generated after the fact cannot be verified against the
        decision it claims to describe.
      </p>
    </div>
  );
}
