/**
 * Presentation vocabulary for a verdict: the words, the counts, the ladder.
 *
 * Kept out of the components so every surface says the same thing the same way,
 * and so the one rule that matters most is enforced in one place: a verdict is
 * a statement about the retrieved evidence, never a recommendation to the
 * reader. "The evidence supports X", never "you should do X".
 */

import type {
  EvidenceItem,
  GradeCertainty,
  InsufficientReason,
  PublicationTier,
  Stance,
  Verdict,
  VerdictResponse,
} from "@/types";

/** A verdict the pipeline actually issued. */
export const CONFIDENT_VERDICT_COPY: Record<
  Exclude<Verdict, "INSUFFICIENT_EVIDENCE">,
  { label: string; sentence: string }
> = {
  SUPPORTED: {
    label: "Supported",
    sentence: "The retrieved evidence supports this claim.",
  },
  CONTRADICTED: {
    label: "Contradicted",
    sentence: "The retrieved evidence contradicts this claim.",
  },
  CONFLICTING: {
    label: "Conflicting",
    sentence: "The retrieved literature is split on this claim.",
  },
};

/**
 * The two ways of having no verdict, which are not interchangeable. One says
 * the corpus is the wrong place to ask; the other says it is the right place
 * but the evidence did not carry a direction.
 */
export const INSUFFICIENT_COPY: Record<
  InsufficientReason,
  { headline: string; sentence: string }
> = {
  NOT_COVERED_BY_CORPUS: {
    headline: "Not covered by this corpus",
    sentence:
      "Nothing in the corpus is about this claim, so no assessment was attempted. This is not evidence the claim is false.",
  },
  EVIDENCE_INCONCLUSIVE: {
    headline: "Evidence inconclusive",
    sentence:
      "The corpus covers this topic, but too little of what was retrieved took a position either way to settle the claim.",
  },
};

export const INSUFFICIENT_FALLBACK = {
  headline: "No verdict issued",
  sentence:
    "The retrieved evidence did not meet the thresholds required to issue a verdict.",
};

/**
 * GRADE-inspired certainty, drawn the way GRADE draws it: four circles, filled
 * to the level. Researchers read this notation on sight.
 */
export const GRADE_COPY: Record<
  GradeCertainty,
  { label: string; filled: number; note: string }
> = {
  HIGH: {
    label: "High",
    filled: 4,
    note: "Further evidence is unlikely to change this reading.",
  },
  MODERATE: {
    label: "Moderate",
    filled: 3,
    note: "Further evidence could change this reading.",
  },
  LOW: {
    label: "Low",
    filled: 2,
    note: "Further evidence is likely to change this reading.",
  },
  VERY_LOW: {
    label: "Very low",
    filled: 1,
    note: "This reading is very uncertain.",
  },
};

export const STANCE_COPY: Record<Stance, { label: string; verb: string }> = {
  SUPPORT: { label: "Supports", verb: "supports the claim" },
  CONTRADICT: { label: "Contradicts", verb: "contradicts the claim" },
  NEUTRAL: { label: "Neutral", verb: "takes no position on the claim" },
};

export const TIER_LABEL: Record<PublicationTier, string> = {
  META_ANALYSIS: "Meta-analysis",
  SYSTEMATIC_REVIEW: "Systematic review",
  RCT: "Randomised trial",
  COHORT: "Cohort study",
  CASE_CONTROL: "Case-control study",
  CROSS_SECTIONAL: "Cross-sectional study",
  CASE_REPORT: "Case report",
  OTHER: "Other design",
};

/** Tailwind classes per stance, so a colour is only chosen in one place. */
export const STANCE_STYLE: Record<
  Stance,
  { text: string; bg: string; border: string; fill: string }
> = {
  SUPPORT: {
    text: "text-support",
    bg: "bg-support-soft",
    border: "border-support",
    fill: "bg-support",
  },
  CONTRADICT: {
    text: "text-contradict",
    bg: "bg-contradict-soft",
    border: "border-contradict",
    fill: "bg-contradict",
  },
  NEUTRAL: {
    text: "text-neutral",
    bg: "bg-neutral-soft",
    border: "border-neutral",
    fill: "bg-neutral",
  },
};

/**
 * The accent edge a verdict carries, as stacked segments. CONFLICTING returns
 * two: a hard split reads as two positions held at once, which is the finding.
 */
export function verdictAccentSegments(verdict: Verdict): string[] {
  if (verdict === "SUPPORTED") return ["bg-support"];
  if (verdict === "CONTRADICTED") return ["bg-contradict"];
  if (verdict === "CONFLICTING") return ["bg-support", "bg-contradict"];
  return ["bg-rule-strong"];
}

export function isInsufficient(
  response: Pick<VerdictResponse, "verdict">
): boolean {
  return response.verdict === "INSUFFICIENT_EVIDENCE";
}

export function formatAuthors(authors: string[], max = 3): string {
  if (authors.length === 0) return "Author not recorded";
  if (authors.length <= max) return authors.join(", ");
  return `${authors.slice(0, max).join(", ")} +${authors.length - max} more`;
}

/** Two decimals, always — a score that renders as "0.9" reads as less precise. */
export function formatScore(value: number): string {
  return value.toFixed(2);
}

export function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

/** A paper we have positively verified as retracted, not merely never checked. */
export function isKnownRetracted(item: EvidenceItem): boolean {
  return item.paper.openalex_checked && item.paper.is_retracted;
}

export interface PassageSegment {
  text: string;
  isRationale: boolean;
}

/**
 * Split a passage so the sentences the classifier keyed on can be marked in
 * place. The backend picks rationales by re-splitting this same passage, so
 * they are exact substrings; anything that fails to match is dropped rather
 * than approximated, and the caller reports it separately.
 */
export function segmentPassage(
  passageText: string,
  rationaleSentences: string[]
): { segments: PassageSegment[]; unmatched: string[] } {
  const ranges: Array<[number, number]> = [];
  const unmatched: string[] = [];

  for (const raw of rationaleSentences) {
    const needle = raw.trim();
    if (!needle) continue;
    const start = passageText.indexOf(needle);
    if (start === -1) {
      unmatched.push(needle);
      continue;
    }
    ranges.push([start, start + needle.length]);
  }

  ranges.sort((a, b) => a[0] - b[0]);

  const segments: PassageSegment[] = [];
  let cursor = 0;
  for (const [start, end] of ranges) {
    // Overlaps would double-render the same characters.
    if (start < cursor) continue;
    if (start > cursor) {
      segments.push({ text: passageText.slice(cursor, start), isRationale: false });
    }
    segments.push({ text: passageText.slice(start, end), isRationale: true });
    cursor = end;
  }
  if (cursor < passageText.length) {
    segments.push({ text: passageText.slice(cursor), isRationale: false });
  }

  return { segments, unmatched };
}
