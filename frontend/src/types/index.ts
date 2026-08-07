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

/**
 * Why a verdict came back INSUFFICIENT_EVIDENCE. Both are the same verdict to
 * the caller; they differ in what went wrong, and the UI must not conflate them.
 */
export type InsufficientReason =
  /** Nothing in the corpus is about this claim at all. */
  | "NOT_COVERED_BY_CORPUS"
  /** The corpus covers it, but too few sources qualified or none had a direction. */
  | "EVIDENCE_INCONCLUSIVE";

/**
 * Raw retrieval signals, computed before normalisation and therefore comparable
 * across claims — unlike `EvidenceItem.relevance_score`, which is min-max
 * normalised per query. These are what the corpus-coverage gate thresholds.
 */
export interface CoverageSignals {
  /** Highest raw claim-passage cosine in the corpus. */
  max_cosine: number;
  /** Mean raw cosine over the top-k passages by cosine. */
  mean_topk_cosine: number;
  /** Highest raw BM25 score in the corpus. Diagnostic only; not gated on. */
  top_bm25: number;
  /** k used for mean_topk_cosine. */
  top_k: number;
}

export type GradeCertainty = "HIGH" | "MODERATE" | "LOW" | "VERY_LOW";

export type PublicationTier =
  | "META_ANALYSIS"
  | "SYSTEMATIC_REVIEW"
  | "RCT"
  | "COHORT"
  | "CASE_CONTROL"
  | "CROSS_SECTIONAL"
  | "CASE_REPORT"
  | "OTHER";

export interface Paper {
  paper_id: string;
  title: string;
  authors: string[];
  year: number;
  abstract: string;
  source_url: string;
  source_database: SourceDatabase;
  /** Every source_database this paper was independently found in (post-merge). */
  all_source_databases: SourceDatabase[];
  /** e.g. "RCT", "meta-analysis", "observational" */
  publication_type: string | null;
  /** Normalised study-design tier, derived from PubMed's PublicationTypeList/MeSH headings where available. */
  publication_tier: PublicationTier | null;
  /** MeSH descriptor terms, for retrieval filtering. */
  mesh_terms: string[];
  journal: string | null;
  pmid: string | null;
  /** Slug identifying a genuinely-contested claim this paper was retrieved for. null for ordinary corpus papers. */
  contested_topic: string | null;
  /** True if not peer reviewed (e.g. psyarxiv, medrxiv postings). */
  is_preprint: boolean;
  doi: string | null;
  /** ISO 8601 timestamp. */
  retrieved_at: string;

  /** Whether this paper has actually been looked up in OpenAlex — false means "not checked", not "checked and clean". */
  openalex_checked: boolean;
  /** OpenAlex retraction flag. Always a definite boolean; check openalex_checked before trusting a false here. */
  is_retracted: boolean;
  cited_by_count: number | null;
  is_open_access: boolean | null;
  open_access_url: string | null;
  concepts: OpenAlexConcept[];
  referenced_works_count: number | null;
  author_institutions: string[];
}

export interface OpenAlexConcept {
  display_name: string;
  score: number;
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
  /** Set only when verdict is INSUFFICIENT_EVIDENCE. */
  insufficient_reason: InsufficientReason | null;
  /** Absolute retrieval signals behind the corpus-coverage gate. */
  coverage: CoverageSignals | null;
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
