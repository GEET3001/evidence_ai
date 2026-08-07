import { formatSeconds } from "@/lib/verdict-display";
import type { VerdictResponse } from "@/types";
import { ConflictingFindings } from "./ConflictingFindings";
import { DownloadReport } from "./DownloadReport";
import { EvidenceCounts } from "./EvidenceCounts";
import { EvidenceList } from "./EvidenceList";
import { ExplanationPanel } from "./ExplanationPanel";
import { PICOPanel } from "./PICOPanel";
import { Section } from "./Section";
import { VerdictBanner } from "./VerdictBanner";

/**
 * The results screen.
 *
 * Sections that depend on evidence are dropped rather than shown empty. When
 * the corpus does not cover a claim the backend deliberately returns no
 * evidence at all, because listing the best of an irrelevant set beside a
 * non-verdict is the confusion the coverage gate exists to prevent.
 */
export function ResultsView({ result }: { result: VerdictResponse }) {
  const hasEvidence = result.evidence.length > 0;
  const hasConflicts = result.conflicting_pairs.length > 0;
  const notCovered = result.insufficient_reason === "NOT_COVERED_BY_CORPUS";

  return (
    <div className="flex flex-col gap-10">
      <VerdictBanner result={result} />

      {notCovered ? (
        <aside className="rounded-lg border border-rule bg-surface p-5">
          <h2 className="font-mono text-[11px] font-semibold tracking-[0.14em] text-ink uppercase">
            Where to go from here
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
            The corpus is mental health research literature. Claims from other
            fields, or about topics it has not indexed, will land here no matter
            how they are worded. If the claim is in scope, narrowing it to a
            named intervention and outcome usually finds the passages.
          </p>
        </aside>
      ) : null}

      <Section label="Claim" meta="Parsed into PICO elements">
        <PICOPanel pico={result.pico} evidence={result.evidence} />
      </Section>

      {hasEvidence ? (
        <Section label="Counts" meta="What the verdict was computed from">
          <EvidenceCounts result={result} />
        </Section>
      ) : null}

      <Section label="Reasoning" meta="Derived, not generated">
        <ExplanationPanel result={result} />
      </Section>

      {hasConflicts ? (
        <Section label="Conflicts" meta="Passages that disagree directly">
          <ConflictingFindings result={result} />
        </Section>
      ) : null}

      {hasEvidence ? (
        <Section
          label="Evidence"
          meta={`${result.evidence.length} passage${
            result.evidence.length === 1 ? "" : "s"
          } above the relevance threshold`}
        >
          <EvidenceList evidence={result.evidence} />
        </Section>
      ) : null}

      <Section label="Export" meta="The full analysis as a document">
        <DownloadReport verificationId={result.verification_id} />
      </Section>

      <Section label="Run">
        <dl className="flex flex-wrap gap-x-10 gap-y-3 font-mono text-[11px] text-muted">
          <div>
            <dt className="text-faint">Verification ID</dt>
            <dd className="mt-1 break-all text-ink">{result.verification_id}</dd>
          </div>
          <div>
            <dt className="text-faint">Response time</dt>
            <dd className="mt-1 tabular-nums text-ink">
              {formatSeconds(result.response_time_ms)}
            </dd>
          </div>
        </dl>
      </Section>
    </div>
  );
}
