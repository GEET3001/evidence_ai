"""Aggregate classified evidence into a VerdictResponse.

Thresholds come from config.py. `grade_certainty` is GRADE-inspired rather than
real GRADE: a deterministic set of adjustments over evidence-tier composition.
Explanations are built from computed counts, never generated text.
"""

from __future__ import annotations

from app.config import settings
from app.models import (
    CoverageSignals,
    EvidenceItem,
    GradeCertainty,
    InsufficientReason,
    PICOClaim,
    PublicationTier,
    Stance,
    Verdict,
    VerdictResponse,
)

_HIGH_TIERS = {PublicationTier.META_ANALYSIS, PublicationTier.SYSTEMATIC_REVIEW}
_LOW_TIERS = {PublicationTier.CASE_REPORT, PublicationTier.OTHER}

_LEVELS = [
    GradeCertainty.VERY_LOW,
    GradeCertainty.LOW,
    GradeCertainty.MODERATE,
    GradeCertainty.HIGH,
]

# Thresholds for the grade adjustments in _adjust_grade.
_UPGRADE_MIN_HIGH_TIER = 2
_UPGRADE_MIN_DIRECTIONAL = 5
_DOWNGRADE_LOW_TIER_SHARE = 0.5
_DOWNGRADE_PREPRINT_SHARE = 0.3


def _shift(level: GradeCertainty, delta: int) -> GradeCertainty:
    idx = _LEVELS.index(level)
    return _LEVELS[max(0, min(len(_LEVELS) - 1, idx + delta))]


def is_covered_by_corpus(coverage: CoverageSignals) -> bool:
    """Whether the corpus contains anything genuinely about this claim.

    Both signals must clear their floor. max_cosine alone is noisy — one
    incidentally-similar passage can carry an otherwise unrelated claim over
    the line — so the top-k mean is required as well, which off-corpus claims
    fail uniformly.
    """
    return (
        coverage.max_cosine >= settings.MIN_MAX_COSINE
        and coverage.mean_topk_cosine >= settings.MIN_MEAN_TOPK_COSINE
    )


def aggregate(
    verification_id: str,
    claim: str,
    evidence: list[EvidenceItem],
    coverage: CoverageSignals,
    response_time_ms: float,
) -> VerdictResponse:
    # The coverage gate runs before anything else. Ranking a claim the corpus
    # knows nothing about still produces a top hit and a stance for it, so
    # every downstream count is meaningless until this passes.
    if not is_covered_by_corpus(coverage):
        return VerdictResponse(
            verification_id=verification_id,
            claim=claim,
            pico=PICOClaim(raw_claim=claim),
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            insufficient_reason=InsufficientReason.NOT_COVERED_BY_CORPUS,
            coverage=coverage,
            grade_certainty=GradeCertainty.VERY_LOW,
            support_count=0,
            contradict_count=0,
            neutral_count=0,
            # Deliberately empty. The retrieved passages are the best of a bad
            # set, and listing them as "evidence" next to this verdict is the
            # exact confusion that made the old behaviour misleading.
            evidence=[],
            conflicting_pairs=[],
            explanation=_build_not_covered_explanation(coverage),
            limitations=[
                "This claim appears to fall outside the corpus's subject area, "
                "so no assessment was attempted. It is not evidence the claim "
                "is false — only that this corpus cannot speak to it.",
                "The corpus covers mental health research; claims from other "
                "domains will not be verifiable here.",
            ],
            response_time_ms=response_time_ms,
        )

    qualifying = [e for e in evidence if e.relevance_score >= settings.MIN_SIMILARITY]
    support = [e for e in qualifying if e.stance == Stance.SUPPORT]
    contradict = [e for e in qualifying if e.stance == Stance.CONTRADICT]
    neutral = [e for e in qualifying if e.stance == Stance.NEUTRAL]
    directional_count = len(support) + len(contradict)
    insufficient_reason = None

    if len(qualifying) < settings.MIN_RELEVANT_SOURCES or directional_count == 0:
        verdict = Verdict.INSUFFICIENT_EVIDENCE
        insufficient_reason = InsufficientReason.EVIDENCE_INCONCLUSIVE
        grade = GradeCertainty.VERY_LOW
    else:
        # Shares are taken over directional evidence only, so the amount of
        # neutral evidence retrieved can't shift a verdict across TIE_MARGIN.
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
        insufficient_reason=insufficient_reason,
        coverage=coverage,
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
    """Adjust a base grade by the tier and provenance mix of the evidence."""
    n = len(qualifying)
    high_tier_count = sum(1 for e in qualifying if e.paper.publication_tier in _HIGH_TIERS)
    low_tier_count = sum(
        1
        for e in qualifying
        if e.paper.publication_tier in _LOW_TIERS or e.paper.publication_tier is None
    )
    preprint_count = sum(1 for e in qualifying if e.paper.is_preprint)

    if high_tier_count >= _UPGRADE_MIN_HIGH_TIER and directional_count >= _UPGRADE_MIN_DIRECTIONAL:
        grade = _shift(grade, 1)
    if n and low_tier_count / n > _DOWNGRADE_LOW_TIER_SHARE:
        grade = _shift(grade, -1)
    if n and preprint_count / n > _DOWNGRADE_PREPRINT_SHARE:
        grade = _shift(grade, -1)
    if n == settings.MIN_RELEVANT_SOURCES:
        grade = _shift(grade, -1)
    return grade


def _build_not_covered_explanation(coverage: CoverageSignals) -> str:
    return (
        f"No assessment was made: this claim does not appear to be about "
        f"anything in the corpus. The closest passage scored {coverage.max_cosine:.3f} "
        f"cosine similarity against the claim (floor {settings.MIN_MAX_COSINE:.3f}), "
        f"and the top {coverage.top_k} averaged {coverage.mean_topk_cosine:.3f} "
        f"(floor {settings.MIN_MEAN_TOPK_COSINE:.3f}). These are absolute "
        f"similarities, not ranks, so they measure whether relevant material "
        f"exists at all rather than which passage happened to rank first."
    )


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
