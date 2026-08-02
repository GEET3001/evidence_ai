// Source of truth is backend/app/models.py — keep in sync.
//
// These types mirror the backend's Pydantic schemas field-for-field.
// Enum values are reproduced exactly (including casing) since they are
// serialised as plain strings over the wire.

export type SourceDatabase =
  | "pubmed"
  | "pmc"
  | "psyarxiv"
  | "medrxiv"
  | "cochrane"
  | "manual";

export type Stance = "SUPPORT" | "CONTRADICT" | "NEUTRAL";

export type Verdict =
  | "SUPPORTED"
  | "CONTRADICTED"
  | "CONFLICTING"
  | "INSUFFICIENT_EVIDENCE";

export type GradeCertainty = "HIGH" | "MODERATE" | "LOW" | "VERY_LOW";

export interface Paper {
  paper_id: string;
  title: string;
  authors: string[];
  year: number;
  abstract: string;
  source_url: string;
  source_database: SourceDatabase;
  /** e.g. "RCT", "meta-analysis", "observational" */
  publication_type: string | null;
  /** True if not peer reviewed (e.g. psyarxiv, medrxiv postings). */
  is_preprint: boolean;
  doi: string | null;
  /** ISO 8601 timestamp. */
  retrieved_at: string;
}

export interface Passage {
  passage_id: string;
  paper_id: string;
  text: string;
  char_start: number;
  char_end: number;
  /** e.g. "Results", "Abstract" */
  section: string | null;
}

export interface PICOClaim {
  raw_claim: string;
  population: string | null;
  intervention: string | null;
  comparison: string | null;
  outcome: string | null;
}

export interface EvidenceItem {
  passage: Passage;
  /** Denormalised parent paper, for display. */
  paper: Paper;
  relevance_score: number;
  stance: Stance;
  stance_confidence: number;
  /** The sentence(s) within the passage the classifier keyed on. */
  rationale_sentences: string[];
  /**
   * Whether this passage's population matches the claim's PICO population.
   * null if indeterminate (flags PICO indirectness).
   */
  population_match: boolean | null;
}

export interface VerdictResponse {
  verification_id: string;
  claim: string;
  pico: PICOClaim;
  verdict: Verdict;
  grade_certainty: GradeCertainty;
  support_count: number;
  contradict_count: number;
  neutral_count: number;
  evidence: EvidenceItem[];
  /** Pairs of passage_ids whose stances directly conflict. */
  conflicting_pairs: [string, string][];
  explanation: string;
  limitations: string[];
  response_time_ms: number;
}
