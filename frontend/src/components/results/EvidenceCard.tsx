import {
  STANCE_COPY,
  STANCE_STYLE,
  TIER_LABEL,
  formatAuthors,
  formatScore,
  isKnownRetracted,
  segmentPassage,
} from "@/lib/verdict-display";
import type { EvidenceItem } from "@/types";

/**
 * One retrieved passage and the paper it came from.
 *
 * The passage is set in serif because it is quoted source text, not our words,
 * and the sentences the classifier keyed on are marked in place inside it. The
 * rationale is the model's own readout — marking it here is what makes the
 * explanation checkable rather than merely plausible.
 *
 * `compact` drops the passage body and shows only the rationale, for the
 * conflicting-findings view where two passages are read against each other.
 */
export function EvidenceCard({
  item,
  variant = "full",
}: {
  item: EvidenceItem;
  variant?: "full" | "compact";
}) {
  const style = STANCE_STYLE[item.stance];
  const retracted = isKnownRetracted(item);
  const { segments, unmatched } = segmentPassage(
    item.passage.text,
    item.rationale_sentences
  );

  // A caption only earns its place when there is something for it to point at.
  // A neutral passage has no rationale at all, and one whose rationale did not
  // match is already labelled by the block below — captioning either would be
  // telling the reader to look at a mark that is not there.
  const hasMark = segments.some((segment) => segment.isRationale);
  const caption =
    variant === "full"
      ? hasMark
        ? "Marked text is the sentence the classifier keyed on"
        : null
      : item.rationale_sentences.length > 0
        ? "Sentence the classifier keyed on"
        : "No rationale sentence reported — passage shown in full";

  return (
    <article className="flex overflow-hidden rounded-lg border border-rule bg-surface">
      <span aria-hidden className={`w-1 shrink-0 ${style.fill}`} />

      <div className="min-w-0 flex-1">
        {retracted ? (
          <p className="flex items-start gap-2 border-b border-warn bg-warn-soft px-5 py-3 text-sm leading-relaxed font-medium text-warn">
            <span aria-hidden>⚠</span>
            <span>
              This paper is flagged as retracted in OpenAlex. Do not treat it as
              evidence for or against the claim.
            </span>
          </p>
        ) : null}

        <div className="flex flex-wrap items-center justify-between gap-x-5 gap-y-2 px-5 py-3">
          <div className="flex items-center gap-3">
            <span
              className={`rounded border px-2 py-0.5 font-mono text-[11px] font-semibold tracking-wide ${style.text} ${style.bg} ${style.border}`}
            >
              {STANCE_COPY[item.stance].label}
            </span>
            {item.passage.section ? (
              <span className="font-mono text-[11px] text-faint">
                {item.passage.section}
              </span>
            ) : null}
          </div>

          <dl className="flex items-center gap-5 font-mono text-[11px] text-muted">
            <div className="flex items-center gap-1.5">
              <dt className="text-faint">Confidence</dt>
              <dd className="tabular-nums">
                {formatScore(item.stance_confidence)}
              </dd>
            </div>
            <div
              className="flex items-center gap-1.5"
              title="Normalised within this result set — comparable between these passages, not between claims."
            >
              <dt className="text-faint">Rank</dt>
              <dd className="tabular-nums">
                {formatScore(item.relevance_score)}
              </dd>
            </div>
          </dl>
        </div>

        {item.population_match === false ? (
          <p className="mx-5 mb-3 rounded border border-contradict bg-contradict-soft px-3 py-2 font-mono text-[11px] leading-relaxed text-contradict">
            Indirect evidence — this passage studies a different population from
            the one in the claim.
          </p>
        ) : null}

        <div className="border-t border-rule px-5 py-4">
          {variant === "full" ? (
            <p className="font-body text-[15px] leading-[1.7] text-ink">
              {segments.map((segment, index) =>
                segment.isRationale ? (
                  <mark
                    key={index}
                    className={`rounded-sm px-0.5 decoration-2 underline-offset-4 ${style.bg} ${style.text} underline`}
                  >
                    {segment.text}
                  </mark>
                ) : (
                  <span key={index}>{segment.text}</span>
                )
              )}
            </p>
          ) : (
            <p className="font-body text-[15px] leading-[1.7] text-ink">
              {item.rationale_sentences.length > 0
                ? item.rationale_sentences.join(" ")
                : item.passage.text}
            </p>
          )}

          {variant === "full" && unmatched.length > 0 ? (
            <div className="mt-3 border-l-2 border-rule-strong pl-3">
              <p className="font-mono text-[10px] tracking-[0.1em] text-faint uppercase">
                Rationale, not locatable in the passage above
              </p>
              {unmatched.map((sentence, index) => (
                <p key={index} className="mt-1 font-body text-sm text-muted">
                  {sentence}
                </p>
              ))}
            </div>
          ) : null}

          {caption ? (
            <p className="mt-3 font-mono text-[10px] tracking-[0.1em] text-faint uppercase">
              {caption}
            </p>
          ) : null}
        </div>

        <SourceDetails item={item} />
      </div>
    </article>
  );
}

function SourceDetails({ item }: { item: EvidenceItem }) {
  const { paper } = item;
  const tier = paper.publication_tier ? TIER_LABEL[paper.publication_tier] : null;

  return (
    <footer className="border-t border-rule bg-sunk px-5 py-4">
      <a
        href={paper.source_url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm leading-snug font-medium text-ink underline decoration-rule-strong underline-offset-2 hover:decoration-ink"
      >
        {paper.title}
        <span aria-hidden className="ml-1 text-faint">
          ↗
        </span>
      </a>

      <p className="mt-1.5 text-xs leading-relaxed text-muted">
        {formatAuthors(paper.authors)} · {paper.year}
        {paper.journal ? ` · ${paper.journal}` : ""}
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {tier ? (
          <Tag>{tier}</Tag>
        ) : (
          <Tag muted>Study design not classified</Tag>
        )}
        {paper.is_preprint ? <Tag tone="warn">Not peer reviewed</Tag> : null}
        {paper.is_open_access && paper.open_access_url ? (
          <a
            href={paper.open_access_url}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded border border-rule-strong px-2 py-0.5 font-mono text-[11px] text-muted hover:border-ink hover:text-ink"
          >
            Open access ↗
          </a>
        ) : null}
        {paper.doi ? (
          <a
            href={`https://doi.org/${paper.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded border border-rule-strong px-2 py-0.5 font-mono text-[11px] text-muted hover:border-ink hover:text-ink"
          >
            doi:{paper.doi}
          </a>
        ) : null}
      </div>
    </footer>
  );
}

function Tag({
  children,
  tone,
  muted,
}: {
  children: React.ReactNode;
  tone?: "warn";
  muted?: boolean;
}) {
  const classes =
    tone === "warn"
      ? "border-warn bg-warn-soft text-warn font-semibold"
      : muted
        ? "border-rule text-faint"
        : "border-rule-strong text-muted";
  return (
    <span
      className={`rounded border px-2 py-0.5 font-mono text-[11px] tracking-wide ${classes}`}
    >
      {children}
    </span>
  );
}
