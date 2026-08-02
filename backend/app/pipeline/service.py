"""Startup-loaded singleton wiring retrieval + stance + verdict together.

Loaded once at FastAPI startup (see main.py's lifespan handler), not
per-request and not lazily on first call — bart-large-mnli alone is ~1.6GB
and slow to load; a lazy first load would make the first real user request
pathologically slow, and would defer a missing-index error until a live
request instead of surfacing it at boot.
"""

from __future__ import annotations

import time
import uuid

from app.models import EvidenceItem, VerdictResponse
from app.pipeline import verdict as verdict_module
from app.pipeline.retrieval import RetrievalIndex
from app.pipeline.stance import StanceClassifier


class PipelineService:
    def __init__(self) -> None:
        self.retrieval = RetrievalIndex()
        self.stance_classifier = StanceClassifier()

    def verify(self, claim: str) -> VerdictResponse:
        start = time.perf_counter()
        verification_id = str(uuid.uuid4())

        results = self.retrieval.search(claim)
        evidence: list[EvidenceItem] = []
        for passage, paper, relevance_score in results:
            stance_result = self.stance_classifier.classify(passage.text, claim)
            evidence.append(
                EvidenceItem(
                    passage=passage,
                    paper=paper,
                    relevance_score=relevance_score,
                    stance=stance_result.stance,
                    stance_confidence=stance_result.confidence,
                    rationale_sentences=stance_result.rationale_sentences,
                    population_match=None,  # PICO extraction is out of scope for now
                )
            )

        response_time_ms = (time.perf_counter() - start) * 1000
        return verdict_module.aggregate(verification_id, claim, evidence, response_time_ms)


_service: PipelineService | None = None


def get_service() -> PipelineService:
    if _service is None:
        raise RuntimeError(
            "Pipeline service not initialized — this should be loaded at FastAPI "
            "startup (see main.py's lifespan handler)."
        )
    return _service


def init_service() -> PipelineService:
    global _service
    _service = PipelineService()
    return _service
