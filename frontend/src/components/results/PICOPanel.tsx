import type { EvidenceItem, PICOClaim } from "@/types";

/**
 * The claim decomposed into Population / Intervention / Comparison / Outcome.
 *
 * The single-letter markers are the acronym itself, so they carry meaning a
 * reader in this field already knows. Unextracted elements are drawn as ruled
 * blanks rather than hidden: an element the parser could not find is a fact
 * about how much of the claim was understood, and hiding the row would make a
 * partial parse look like a complete one.
 */

const ELEMENTS = [
  { key: "population", letter: "P", label: "Population" },
  { key: "intervention", letter: "I", label: "Intervention" },
  { key: "comparison", letter: "C", label: "Comparison" },
  { key: "outcome", letter: "O", label: "Outcome" },
] as const;

export function PICOPanel({
  pico,
  evidence,
}: {
  pico: PICOClaim;
  evidence: EvidenceItem[];
}) {
  const indirectCount = evidence.filter(
    (item) => item.population_match === false
  ).length;
  const extracted = ELEMENTS.filter(({ key }) => pico[key]).length;

  return (
    <div className="overflow-hidden rounded-lg border border-rule bg-surface">
      <dl>
        {ELEMENTS.map(({ key, letter, label }, index) => {
          const value = pico[key];
          return (
            <div
              key={key}
              className={`flex gap-4 px-5 py-4 ${index > 0 ? "border-t border-rule" : ""}`}
            >
              <dt className="flex shrink-0 items-baseline gap-3">
                <span className="font-mono text-sm font-semibold text-faint">
                  {letter}
                </span>
                <span className="w-24 font-mono text-[11px] tracking-[0.1em] text-muted uppercase">
                  {label}
                </span>
              </dt>
              <dd className="min-w-0 flex-1">
                {value ? (
                  <span className="font-body text-[15px] leading-relaxed text-ink">
                    {value}
                  </span>
                ) : (
                  <span className="font-mono text-xs text-faint">
                    Not extracted
                  </span>
                )}
                {key === "population" && indirectCount > 0 ? (
                  <p className="mt-2 inline-flex items-center gap-2 rounded border border-contradict bg-contradict-soft px-2 py-1 font-mono text-[11px] text-contradict">
                    Indirect evidence — {indirectCount}{" "}
                    {indirectCount === 1 ? "passage studies" : "passages study"} a
                    different population
                  </p>
                ) : null}
              </dd>
            </div>
          );
        })}
      </dl>

      {extracted === 0 ? (
        <p className="border-t border-rule bg-sunk px-5 py-3 text-xs leading-relaxed text-muted">
          Structured PICO extraction is not implemented, so the claim was
          retrieved against and assessed exactly as written. Nothing here was
          inferred.
        </p>
      ) : null}
    </div>
  );
}
