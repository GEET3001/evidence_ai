"""Pydantic schemas shared across the API, pipeline, and frontend contract.

These models are the interface both the backend pipeline and the frontend
build against — field names and enum values here are load-bearing.
"""

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


class Paper(BaseModel):
    """A source research paper in the corpus."""

    paper_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    source_url: str
    source_database: SourceDatabase
    publication_type: str | None = Field(
        default=None, description='e.g. "RCT", "meta-analysis", "observational"'
    )
    is_preprint: bool = Field(
        description="True if not peer reviewed (e.g. psyarxiv, medrxiv postings)."
    )
    doi: str | None = None
    retrieved_at: datetime


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

    Fields other than raw_claim are optional because PICO extraction from a
    free-text claim does not always identify all four elements.
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
        description="Whether this passage's population matches the claim's PICO "
        "population. None if indeterminate (flags PICO indirectness).",
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
