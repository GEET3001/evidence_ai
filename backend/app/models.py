"""Schemas shared by the API, the pipeline, and the frontend contract."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceDatabase(str, Enum):
    """Where a paper was retrieved from."""

    PUBMED = "pubmed"
    PMC = "pmc"
    PSYARXIV = "psyarxiv"
    MEDRXIV = "medrxiv"
    COCHRANE = "cochrane"
    MANUAL = "manual"


class Stance(str, Enum):
    """Classification label for a passage relative to a claim."""

    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"


class Verdict(str, Enum):
    """Aggregate verdict for a claim."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class GradeCertainty(str, Enum):
    """GRADE-inspired confidence rating for the verdict."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class PublicationTier(str, Enum):
    """Normalised study-design tier, ordered roughly by evidence strength."""

    META_ANALYSIS = "META_ANALYSIS"
    SYSTEMATIC_REVIEW = "SYSTEMATIC_REVIEW"
    RCT = "RCT"
    COHORT = "COHORT"
    CASE_CONTROL = "CASE_CONTROL"
    CROSS_SECTIONAL = "CROSS_SECTIONAL"
    CASE_REPORT = "CASE_REPORT"
    OTHER = "OTHER"


class OpenAlexConcept(BaseModel):
    """A topic OpenAlex assigns to a work, with its confidence score."""

    display_name: str
    score: float


class Paper(BaseModel):
    """A source research paper in the corpus."""

    paper_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    source_url: str
    source_database: SourceDatabase
    all_source_databases: list[SourceDatabase] = Field(
        default_factory=list,
        description="Every source database this paper was independently found in.",
    )
    publication_type: str | None = Field(
        default=None, description='e.g. "RCT", "meta-analysis", "observational"'
    )
    publication_tier: PublicationTier | None = Field(
        default=None,
        description="Study-design tier, from PubMed publication types and MeSH headings.",
    )
    mesh_terms: list[str] = Field(
        default_factory=list, description="MeSH descriptor terms, for retrieval filtering."
    )
    journal: str | None = None
    pmid: str | None = None
    contested_topic: str | None = Field(
        default=None,
        description="Slug of the contested claim this paper was retrieved for, "
        "or None for ordinary corpus papers.",
    )
    is_preprint: bool = Field(
        description="True if not peer reviewed (e.g. psyarxiv, medrxiv postings)."
    )
    doi: str | None = None
    retrieved_at: datetime

    # --- OpenAlex enrichment (see app/ingestion/openalex_enrich.py) ---
    openalex_checked: bool = Field(
        default=False,
        description="Whether this paper was looked up in OpenAlex. The signals "
        "below are only verified when this is True.",
    )
    is_retracted: bool = Field(
        default=False,
        description="OpenAlex retraction flag. Never null, so check "
        "openalex_checked before trusting a False.",
    )
    cited_by_count: int | None = None
    is_open_access: bool | None = None
    open_access_url: str | None = None
    concepts: list[OpenAlexConcept] = Field(default_factory=list)
    referenced_works_count: int | None = None
    author_institutions: list[str] = Field(default_factory=list)


class Passage(BaseModel):
    """A chunk of text extracted from a paper, with its offsets in the source."""

    passage_id: str
    paper_id: str
    text: str
    char_start: int
    char_end: int
    section: str | None = Field(default=None, description='e.g. "Results", "Abstract"')


class PICOClaim(BaseModel):
    """A claim decomposed into Population / Intervention / Comparison / Outcome.

    Only raw_claim is required; extraction may not identify all four elements.
    """

    raw_claim: str
    population: str | None = None
    intervention: str | None = None
    comparison: str | None = None
    outcome: str | None = None


class EvidenceItem(BaseModel):
    """A single retrieved passage, classified against the claim."""

    passage: Passage
    paper: Paper = Field(description="Denormalised parent paper, for display.")
    relevance_score: float
    stance: Stance
    stance_confidence: float
    rationale_sentences: list[str] = Field(
        description="The sentence(s) within the passage the classifier keyed on."
    )
    population_match: bool | None = Field(
        default=None,
        description="Whether this passage's population matches the claim's. "
        "None if indeterminate.",
    )


class ClaimRequest(BaseModel):
    """Incoming request to verify a research claim."""

    claim: str = Field(..., min_length=1, description="The research claim to verify.")


class VerdictResponse(BaseModel):
    """Final response returned to the client."""

    verification_id: str
    claim: str
    pico: PICOClaim
    verdict: Verdict
    grade_certainty: GradeCertainty
    support_count: int
    contradict_count: int
    neutral_count: int
    evidence: list[EvidenceItem]
    conflicting_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Pairs of passage_ids whose stances directly conflict.",
    )
    explanation: str
    limitations: list[str]
    response_time_ms: float
