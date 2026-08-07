import { STANCE_STYLE } from "@/lib/verdict-display";
import type { VerdictResponse } from "@/types";
import { StanceBar } from "./StanceBar";

/**
 * The counts the verdict was computed from, as a readout.
 *
 * "Assessed" rather than "retrieved": the response carries the passages that
 * cleared the relevance threshold, not everything the retriever touched, and
 * labelling the smaller number as the larger one would overstate the search.
 */
export function EvidenceCounts({ result }: { result: VerdictResponse }) {
  const assessed = result.evidence.length;
  const papers = new Set(result.evidence.map((item) => item.paper.paper_id)).size;

  return (
    <div className="rounded-lg border border-rule bg-surface p-5">
      <dl className="flex flex-wrap gap-x-10 gap-y-5">
        <Readout
          term="Supporting"
          value={result.support_count}
          className={STANCE_STYLE.SUPPORT.text}
        />
        <Readout
          term="Contradicting"
          value={result.contradict_count}
          className={STANCE_STYLE.CONTRADICT.text}
        />
        <Readout
          term="Neutral"
          value={result.neutral_count}
          className={STANCE_STYLE.NEUTRAL.text}
        />
        <span aria-hidden className="hidden w-px self-stretch bg-rule sm:block" />
        <Readout term="Passages assessed" value={assessed} className="text-ink" />
        <Readout term="Source papers" value={papers} className="text-ink" />
      </dl>

      <div className="mt-6">
        <StanceBar
          support={result.support_count}
          contradict={result.contradict_count}
          neutral={result.neutral_count}
        />
      </div>

      <p className="mt-4 text-xs leading-relaxed text-muted">
        Counts cover the passages that cleared the relevance threshold. The
        verdict is decided on the directional ones only — neutral passages are
        reported but never break a tie.
      </p>
    </div>
  );
}

function Readout({
  term,
  value,
  className,
}: {
  term: string;
  value: number;
  className: string;
}) {
  return (
    <div>
      <dt className="font-mono text-[10px] font-semibold tracking-[0.14em] text-faint uppercase">
        {term}
      </dt>
      <dd className={`mt-1 font-mono text-3xl leading-none tabular-nums ${className}`}>
        {value}
      </dd>
    </div>
  );
}
