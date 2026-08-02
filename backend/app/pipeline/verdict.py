"""Aggregate classified evidence into a VerdictResponse.

Thresholds (MIN_RELEVANT_SOURCES, MIN_SIMILARITY, TIE_MARGIN) come from
config.py. grade_certainty is GRADE-*inspired*, not real GRADE (see
GradeCertainty's docstring) — a small, deterministic point system based on
evidence-tier composition, not a model. explanation/limitations are template
strings filled with real computed numbers, never LLM-generated (README:
"explanations are faithful — derived from rationale sentences and computed
values").
"""

from __future__ import annotations

from app.config import settings
from app.models import (
    EvidenceItem,
    GradeCertainty,
    PICOClaim,
    PublicationTier,
    Stance,
    Verdict,
    VerdictResponse,
)

_HIGH_TIERS = {PublicationTier.META_ANALYSIS, PublicationTier.SYSTEMATIC_REVIEW}
_LOW_TIERS = {PublicationTier.CASE_REPORT, PublicationTier.OTHER}

_LEVELS = [GradeCertainty.VERY_LOW, GradeCertainty.LOW, GradeCertainty.MODERATE, GradeCertainty.HIGH]


def _shift(level: GradeCertainty, delta: int) -> GradeCertainty:
    idx = _LEVELS.index(level)
    return _LEVELS[max(0, min(len(_LEVELS) - 1, idx + delta))]


def aggregate(
    verification_id: str,
    claim: str,
    evidence: list[EvidenceItem],
    response_time_ms: float,
) -> VerdictResponse:
    qualifying = [e for e in evidence if e.relevance_score >= settings.MIN_SIMILARITY]
    support = [e for e in qualifying if e.stance == Stance.SUPPORT]
    contradict = [e for e in qualifying if e.stance == Stance.CONTRADICT]
    neutral = [e for e in qualifying if e.stance == Stance.NEUTRAL]
    directional_count = len(support) + len(contradict)

    if len(qualifying) < settings.MIN_RELEVANT_SOURCES or directional_count == 0:
        verdict = Verdict.INSUFFICIENT_EVIDENCE
        grade = GradeCertainty.VERY_LOW
    else:
        support_share = len(support) / directional_count
        contradict_share = len(contradict) / directional_count

        if abs(support_share - contradict_share) <= settings.TIE_MARGIN:
            verdict = Verdict.CONFLICTING
            grade = GradeCertainty.LOW
        elif support_share > contradict_share:
            verdict = Verdict.SUPPORTED
            grade = GradeCertainty.MODERATE
        else:
            verdict = Verdict.CONTRADICTED
            grade = GradeCertainty.MODERATE

        grade = _adjust_grade(grade, qualifying, directional_count)

    conflicting_pairs = [
        (s.passage.passage_id, c.passage.passage_id) for s in support for c in contradict
    ]

    return VerdictResponse(
        verification_id=verification_id,
        claim=claim,
        pico=PICOClaim(raw_claim=claim),
        verdict=verdict,
        grade_certainty=grade,
        support_count=len(support),
        contradict_count=len(contradict),
        neutral_count=len(neutral),
        evidence=qualifying,
        conflicting_pairs=conflicting_pairs,
        explanation=_build_explanation(verdict, qualifying, support, contradict, neutral),
        limitations=_build_limitations(qualifying),
        response_time_ms=response_time_ms,
    )


def _adjust_grade(
    grade: GradeCertainty, qualifying: list[EvidenceItem], directional_count: int
) -> GradeCertainty:
    """Small, deterministic adjustments from a directional-verdict base grade.

    The corpus is currently ~56% unclassified/OTHER tier, so "no strong tier
    signal" is the common case here, not an edge case — the rule is built
    around that, not around an idealized all-tiers-populated corpus.
    """
    n = len(qualifying)
    high_tier_count = sum(1 for e in qualifying if e.paper.publication_tier in _HIGH_TIERS)
    low_tier_count = sum(
        1 for e in qualifying if e.paper.publication_tier in _LOW_TIERS or e.paper.publication_tier is None
    )
    preprint_count = sum(1 for e in qualifying if e.paper.is_preprint)

    if high_tier_count >= 2 and directional_count >= 5:
        grade = _shift(grade, 1)
    if n and low_tier_count / n > 0.5:
        grade = _shift(grade, -1)
    if n and preprint_count / n > 0.3:
        grade = _shift(grade, -1)
    if n == settings.MIN_RELEVANT_SOURCES:
        grade = _shift(grade, -1)
    return grade


def _build_explanation(
    verdict: Verdict,
    qualifying: list[EvidenceItem],
    support: list[EvidenceItem],
    contradict: list[EvidenceItem],
    neutral: list[EvidenceItem],
) -> str:
    n = len(qualifying)
    n_papers = len({e.paper.paper_id for e in qualifying})

    if verdict == Verdict.INSUFFICIENT_EVIDENCE:
        return (
            f"Only {n} passage(s) met the relevance threshold "
            f"(similarity >= {settings.MIN_SIMILARITY}) for this claim, and/or "
            f"none had a directional (supporting or contradicting) stance; at "
            f"least {settings.MIN_RELEVANT_SOURCES} directional sources are "
            f"required to issue a verdict."
        )

    base = (
        f"Retrieved {n} relevant passage(s) from {n_papers} paper(s) "
        f"(similarity >= {settings.MIN_SIMILARITY}): {len(support)} support, "
        f"{len(contradict)} contradict, {len(neutral)} neutral."
    )
    if verdict == Verdict.CONFLICTING:
        return (
            base + " Supporting and contradicting evidence are close enough in "
            "share that the retrieved literature is treated as genuinely mixed "
            "on this claim."
        )
    if verdict == Verdict.SUPPORTED:
        return base + " Supporting evidence outweighs contradicting evidence."
    return base + " Contradicting evidence outweighs supporting evidence."


def _build_limitations(qualifying: list[EvidenceItem]) -> list[str]:
    n = len(qualifying)
    if n == 0:
        return ["No passages met the relevance threshold for this claim."]

    limitations = []
    if n == settings.MIN_RELEVANT_SOURCES:
        limitations.append(
            f"Evidence base is at the minimum threshold ({n} sources); "
            "confidence would improve with a larger evidence base."
        )

    preprint_count = sum(1 for e in qualifying if e.paper.is_preprint)
    if preprint_count:
        limitations.append(
            f"{preprint_count} of {n} sources are preprints and have not been "
            "peer reviewed."
        )

    unclassified = sum(1 for e in qualifying if e.paper.publication_tier is None)
    if unclassified:
        limitations.append(
            f"{unclassified} of {n} sources have no verified study-design tier "
            "classification, which limits how much weight can be placed on "
            "evidence quality."
        )

    if not limitations:
        limitations.append("No major limitations identified for this evidence set.")
    return limitations
