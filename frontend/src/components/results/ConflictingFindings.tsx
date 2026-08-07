import type { EvidenceItem, VerdictResponse } from "@/types";
import { EvidenceCard } from "./EvidenceCard";

/**
 * Passages that directly disagree, read against each other.
 *
 * Kept out of the main list on purpose. In a single relevance-ordered list a
 * contradiction is something the reader has to notice; here the two sides are
 * put next to each other so it cannot be missed. Shown whenever any pair
 * exists, including under a SUPPORTED verdict — a majority is not a consensus.
 */
export function ConflictingFindings({ result }: { result: VerdictResponse }) {
  if (result.conflicting_pairs.length === 0) return null;

  const byId = new Map(result.evidence.map((item) => [item.passage.passage_id, item]));
  const supporting: EvidenceItem[] = [];
  const contradicting: EvidenceItem[] = [];
  const seen = new Set<string>();

  for (const [supportId, contradictId] of result.conflicting_pairs) {
    const support = byId.get(supportId);
    if (support && !seen.has(supportId)) {
      seen.add(supportId);
      supporting.push(support);
    }
    const contradict = byId.get(contradictId);
    if (contradict && !seen.has(contradictId)) {
      seen.add(contradictId);
      contradicting.push(contradict);
    }
  }

  if (supporting.length === 0 || contradicting.length === 0) return null;

  return (
    <div className="flex flex-col gap-5">
      <p className="text-sm leading-relaxed text-muted">
        {supporting.length} supporting and {contradicting.length} contradicting
        passage{contradicting.length === 1 ? "" : "s"} were retrieved for this
        claim, giving {result.conflicting_pairs.length} direct
        {result.conflicting_pairs.length === 1 ? " conflict" : " conflicts"}{" "}
        between them. Each side is shown by the sentence the classifier keyed
        on.
      </p>

      <div className="grid gap-5 lg:grid-cols-2">
        <ConflictColumn
          heading="Supports the claim"
          items={supporting}
          accent="bg-support"
        />
        <ConflictColumn
          heading="Contradicts the claim"
          items={contradicting}
          accent="bg-contradict"
        />
      </div>
    </div>
  );
}

function ConflictColumn({
  heading,
  items,
  accent,
}: {
  heading: string;
  items: EvidenceItem[];
  accent: string;
}) {
  return (
    <div className="flex flex-col gap-3">
      <h3 className="flex items-center gap-2 font-mono text-[11px] font-semibold tracking-[0.14em] text-ink uppercase">
        <span aria-hidden className={`size-2 rounded-full ${accent}`} />
        {heading}
        <span className="tabular-nums font-normal text-faint">{items.length}</span>
      </h3>
      <ol className="flex flex-col gap-3">
        {items.map((item) => (
          <li key={item.passage.passage_id}>
            <EvidenceCard item={item} variant="compact" />
          </li>
        ))}
      </ol>
    </div>
  );
}
