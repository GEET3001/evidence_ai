"""A description of the corpus a report was produced against.

Measured from the loaded index at request time rather than written down as
constants. The corpus is rebuilt by the ingestion pipeline, so a hardcoded
"183 papers" would quietly become a false statement in a document whose whole
purpose is to be precise about what the evidence rests on.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusProfile:
    paper_count: int
    passage_count: int
    preprint_count: int
    earliest_year: int | None
    latest_year: int | None
    tier_counts: dict[str, int] = field(default_factory=dict)
    unclassified_tier_count: int = 0
    retraction_checked_count: int = 0
    retracted_count: int = 0

    @classmethod
    def from_index(cls, retrieval) -> "CorpusProfile":
        """Build a profile from a loaded RetrievalIndex.

        Typed loosely on purpose: importing RetrievalIndex here would pull faiss
        and torch into any module that only wants to describe a corpus.
        """
        papers = list(retrieval.papers_by_id.values())
        years = [p.year for p in papers if p.year]
        tiers = Counter(
            p.publication_tier.value for p in papers if p.publication_tier is not None
        )

        return cls(
            paper_count=len(papers),
            passage_count=len(retrieval.passages),
            preprint_count=sum(1 for p in papers if p.is_preprint),
            earliest_year=min(years) if years else None,
            latest_year=max(years) if years else None,
            tier_counts=dict(tiers.most_common()),
            unclassified_tier_count=sum(1 for p in papers if p.publication_tier is None),
            retraction_checked_count=sum(1 for p in papers if p.openalex_checked),
            retracted_count=sum(
                1 for p in papers if p.openalex_checked and p.is_retracted
            ),
        )

    def scope_sentence(self) -> str:
        span = (
            f", published {self.earliest_year}–{self.latest_year}"
            if self.earliest_year and self.latest_year
            else ""
        )
        return (
            f"{self.paper_count} mental health research papers{span}, indexed as "
            f"{self.passage_count} passages"
        )
