import { GRADE_COPY } from "@/lib/verdict-display";
import type { GradeCertainty } from "@/types";

/**
 * GRADE certainty in GRADE's own notation: four circles, filled to the level.
 * Drawn rather than spelled out because the audience for this page reads the
 * notation directly, and because a word alone hides that there are four rungs.
 */
export function GradeLadder({
  certainty,
  tone = "light",
}: {
  certainty: GradeCertainty;
  tone?: "light" | "dark";
}) {
  const { label, filled, note } = GRADE_COPY[certainty];
  const onDark = tone === "dark";

  return (
    <div className="flex flex-col gap-1.5">
      <span
        className={`font-mono text-[10px] font-semibold tracking-[0.14em] uppercase ${
          onDark ? "text-banner-muted" : "text-faint"
        }`}
      >
        Certainty
      </span>
      <div className="flex items-center gap-2">
        <span
          className="flex items-center gap-1"
          role="img"
          aria-label={`GRADE certainty ${label.toLowerCase()}, ${filled} of 4`}
        >
          {[0, 1, 2, 3].map((i) => (
            <span
              key={i}
              aria-hidden
              className={`size-2.5 rounded-full border ${
                onDark ? "border-banner-ink" : "border-ink"
              } ${
                i < filled
                  ? onDark
                    ? "bg-banner-ink"
                    : "bg-ink"
                  : "bg-transparent opacity-45"
              }`}
            />
          ))}
        </span>
        <span
          className={`text-sm font-medium ${onDark ? "text-banner-ink" : "text-ink"}`}
        >
          {label}
        </span>
      </div>
      <p
        className={`max-w-[22rem] text-xs leading-relaxed ${
          onDark ? "text-banner-muted" : "text-muted"
        }`}
      >
        {note}
      </p>
    </div>
  );
}
