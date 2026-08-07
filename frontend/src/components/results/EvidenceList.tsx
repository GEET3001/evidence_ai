"use client";

import { useMemo, useState } from "react";
import { STANCE_STYLE } from "@/lib/verdict-display";
import type { EvidenceItem, Stance } from "@/types";
import { EvidenceCard } from "./EvidenceCard";

type StanceFilter = Stance | "ALL";
type SortKey = "relevance" | "year";

const FILTERS: Array<{ key: StanceFilter; label: string }> = [
  { key: "ALL", label: "All" },
  { key: "SUPPORT", label: "Supporting" },
  { key: "CONTRADICT", label: "Contradicting" },
  { key: "NEUTRAL", label: "Neutral" },
];

const SORTS: Array<{ key: SortKey; label: string }> = [
  { key: "relevance", label: "Relevance" },
  { key: "year", label: "Newest" },
];

export function EvidenceList({ evidence }: { evidence: EvidenceItem[] }) {
  const [stance, setStance] = useState<StanceFilter>("ALL");
  const [sort, setSort] = useState<SortKey>("relevance");

  const counts = useMemo(() => {
    const byStance = { SUPPORT: 0, CONTRADICT: 0, NEUTRAL: 0 };
    for (const item of evidence) byStance[item.stance] += 1;
    return { ALL: evidence.length, ...byStance };
  }, [evidence]);

  const visible = useMemo(() => {
    const filtered =
      stance === "ALL"
        ? evidence
        : evidence.filter((item) => item.stance === stance);
    return [...filtered].sort((a, b) =>
      sort === "relevance"
        ? b.relevance_score - a.relevance_score
        : b.paper.year - a.paper.year
    );
  }, [evidence, stance, sort]);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
        <div className="flex flex-wrap items-center gap-1.5">
          {FILTERS.map((filter) => {
            const count = counts[filter.key];
            const active = stance === filter.key;
            const accent =
              filter.key !== "ALL" ? STANCE_STYLE[filter.key].text : "";
            return (
              <button
                key={filter.key}
                type="button"
                onClick={() => setStance(filter.key)}
                disabled={count === 0}
                aria-pressed={active}
                className={`rounded border px-3 py-1.5 font-mono text-[11px] tracking-wide transition-colors disabled:cursor-not-allowed disabled:opacity-35 ${
                  active
                    ? "border-ink bg-ink text-paper"
                    : `border-rule bg-surface hover:border-rule-strong ${accent || "text-muted"}`
                }`}
              >
                {filter.label}{" "}
                <span className="tabular-nums opacity-70">{count}</span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <span
            id="evidence-sort-label"
            className="font-mono text-[10px] tracking-[0.14em] text-faint uppercase"
          >
            Sort
          </span>
          <div
            className="flex items-center gap-1.5"
            role="group"
            aria-labelledby="evidence-sort-label"
          >
            {SORTS.map((option) => (
              <button
                key={option.key}
                type="button"
                onClick={() => setSort(option.key)}
                aria-pressed={sort === option.key}
                className={`rounded border px-3 py-1.5 font-mono text-[11px] tracking-wide transition-colors ${
                  sort === option.key
                    ? "border-ink bg-ink text-paper"
                    : "border-rule bg-surface text-muted hover:border-rule-strong"
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {visible.length === 0 ? (
        <p className="rounded-lg border border-dashed border-rule-strong bg-surface px-5 py-8 text-center text-sm text-muted">
          No passages carry that stance. Choose another filter to keep reading.
        </p>
      ) : (
        <ol className="flex flex-col gap-4">
          {visible.map((item) => (
            <li key={item.passage.passage_id}>
              <EvidenceCard item={item} />
            </li>
          ))}
        </ol>
      )}

      <p className="text-xs leading-relaxed text-muted">
        Rank is normalised within this result set, so it orders these passages
        against each other and means nothing across different claims. Whether
        the corpus covers a claim at all is decided separately, on absolute
        similarity.
      </p>
    </div>
  );
}
