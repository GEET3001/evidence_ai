import {
  CONFIDENT_VERDICT_COPY,
  INSUFFICIENT_COPY,
  INSUFFICIENT_FALLBACK,
  formatScore,
  verdictAccentSegments,
} from "@/lib/verdict-display";
import type { VerdictResponse } from "@/types";
import { GradeLadder } from "./GradeLadder";
import { StanceBar } from "./StanceBar";

/**
 * The verdict, and the page's whole visual thesis.
 *
 * A verdict the pipeline issued sits on a solid dark field. The absence of one
 * never does: it renders on paper, hatched like an unfilled cell in a data
 * table, with the hierarchy inverted so the *reason* is the headline and the
 * word "verdict" shrinks to a caption. The two states are not a colour apart —
 * they are different objects, which is the point. An insufficient result that
 * merely looked like a grey answer would still read as an answer.
 */
export function VerdictBanner({ result }: { result: VerdictResponse }) {
  const insufficient = result.verdict === "INSUFFICIENT_EVIDENCE";

  return insufficient ? (
    <NoVerdictBanner result={result} />
  ) : (
    <IssuedVerdictBanner result={result} />
  );
}

function ClaimLine({ claim, tone }: { claim: string; tone: "light" | "dark" }) {
  const onDark = tone === "dark";
  return (
    <div>
      <p
        className={`font-mono text-[10px] font-semibold tracking-[0.14em] uppercase ${
          onDark ? "text-banner-muted" : "text-faint"
        }`}
      >
        Claim
      </p>
      <p
        className={`mt-2 font-body text-xl leading-snug text-balance sm:text-2xl ${
          onDark ? "text-banner-ink" : "text-ink"
        }`}
      >
        {claim}
      </p>
    </div>
  );
}

function IssuedVerdictBanner({ result }: { result: VerdictResponse }) {
  const copy =
    CONFIDENT_VERDICT_COPY[
      result.verdict as keyof typeof CONFIDENT_VERDICT_COPY
    ];
  const segments = verdictAccentSegments(result.verdict);
  const total =
    result.support_count + result.contradict_count + result.neutral_count;

  return (
    <div className="flex overflow-hidden rounded-lg border border-banner-rule bg-banner">
      <div
        className="flex w-1.5 shrink-0 flex-col"
        aria-hidden
      >
        {segments.map((className) => (
          <span key={className} className={`flex-1 ${className}`} />
        ))}
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-7 p-6 sm:p-8">
        <ClaimLine claim={result.claim} tone="dark" />

        <div className="flex flex-col gap-7 border-t border-banner-rule pt-7 sm:flex-row sm:items-start sm:justify-between sm:gap-10">
          <div className="min-w-0">
            <h1 className="text-4xl leading-none font-semibold tracking-tight text-banner-ink sm:text-5xl">
              {copy.label}
            </h1>
            <p className="mt-3 max-w-md text-sm leading-relaxed text-banner-muted">
              {copy.sentence}
            </p>
          </div>
          <div className="shrink-0">
            <GradeLadder certainty={result.grade_certainty} tone="dark" />
          </div>
        </div>

        {total > 0 ? (
          <div className="flex flex-col gap-2">
            <StanceBar
              support={result.support_count}
              contradict={result.contradict_count}
              neutral={result.neutral_count}
              tone="dark"
            />
            <p className="font-mono text-[11px] text-banner-muted">
              {result.support_count} supporting · {result.contradict_count}{" "}
              contradicting · {result.neutral_count} neutral
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function NoVerdictBanner({ result }: { result: VerdictResponse }) {
  const copy = result.insufficient_reason
    ? INSUFFICIENT_COPY[result.insufficient_reason]
    : INSUFFICIENT_FALLBACK;
  const notCovered = result.insufficient_reason === "NOT_COVERED_BY_CORPUS";

  return (
    <div className="hatch rounded-lg border-2 border-dashed border-rule-strong bg-surface">
      <div className="flex flex-col gap-7 p-6 sm:p-8">
        <ClaimLine claim={result.claim} tone="light" />

        <div className="border-t border-dashed border-rule-strong pt-7">
          <p className="font-mono text-[10px] font-semibold tracking-[0.14em] text-faint uppercase">
            No verdict issued
          </p>
          <h1 className="mt-2 max-w-xl text-2xl leading-tight font-semibold tracking-tight text-balance text-ink sm:text-3xl">
            {copy.headline}
          </h1>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-muted">
            {copy.sentence}
          </p>
        </div>

        {notCovered && result.coverage ? (
          <dl className="flex flex-wrap gap-x-10 gap-y-4 border-t border-dashed border-rule-strong pt-6">
            <CoverageReadout
              term="Closest passage"
              value={formatScore(result.coverage.max_cosine)}
            />
            <CoverageReadout
              term={`Top ${result.coverage.top_k} mean`}
              value={formatScore(result.coverage.mean_topk_cosine)}
            />
            <div className="max-w-sm text-xs leading-relaxed text-muted">
              Absolute cosine similarity against the whole corpus, not a rank —
              it measures whether relevant material exists at all. Both floors
              are in the explanation below.
            </div>
          </dl>
        ) : null}
      </div>
    </div>
  );
}

function CoverageReadout({ term, value }: { term: string; value: string }) {
  return (
    <div>
      <dt className="font-mono text-[10px] font-semibold tracking-[0.14em] text-faint uppercase">
        {term}
      </dt>
      <dd className="mt-1 font-mono text-2xl text-ink tabular-nums">{value}</dd>
    </div>
  );
}
