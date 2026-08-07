"""Hybrid BM25 + dense retrieval over the indexed passage corpus.

Ranking and gating are deliberately separate signals.

**Ranking** fuses BM25 and dense similarity after min-max normalizing each to
[0,1]. Normalization makes the two comparable, which is what a good ranking
needs, and the fused score is what orders results.

**Gating** cannot use that fused score. Min-max normalization is per query, so
the top-ranked passage of *every* query is pushed toward 1.0 — including a
query about nothing in the corpus. Thresholding it therefore answers "is this
the best of what we found?", never "is this actually relevant?". The raw,
pre-normalization cosine and BM25 scores are carried through untouched in
CoverageSignals so the verdict layer can ask the second question.
"""

from __future__ import annotations

import app._thread_limits  # noqa: F401  (must precede the faiss/torch imports below)

import json
import re
from dataclasses import dataclass

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models import CoverageSignals, Paper, Passage


@dataclass(frozen=True)
class RetrievedPassage:
    """One ranked hit, carrying both the fused ranking score and raw signals."""

    passage: Passage
    paper: Paper
    fused_score: float  # normalized; ordering only, never a relevance threshold
    raw_cosine: float  # absolute, comparable across queries
    raw_bm25: float  # absolute, comparable across queries


@dataclass(frozen=True)
class SearchResult:
    passages: list[RetrievedPassage]
    coverage: CoverageSignals


_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class RetrievalIndex:
    """Holds the prebuilt FAISS index, passages, and corpus for reuse across requests."""

    def __init__(self) -> None:
        faiss_path = settings.index_dir / "faiss.index"
        passages_path = settings.index_dir / "passages.json"
        if not faiss_path.exists() or not passages_path.exists():
            raise FileNotFoundError(
                f"No index found at {settings.index_dir}. Run "
                "`python -m app.indexing.build_index` first."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
        self.faiss_index = faiss.read_index(str(faiss_path))

        with open(passages_path, encoding="utf-8") as f:
            passage_dicts = json.load(f)
        self.passages: list[Passage] = [Passage.model_validate(p) for p in passage_dicts]

        if self.faiss_index.ntotal != len(self.passages):
            raise ValueError(
                f"FAISS index has {self.faiss_index.ntotal} vectors but "
                f"passages.json has {len(self.passages)} entries — index and "
                "passage file are out of sync. Rebuild via "
                "`python -m app.indexing.build_index`."
            )

        with open(settings.corpus_path, encoding="utf-8") as f:
            corpus_dicts = json.load(f)
        self.papers_by_id: dict[str, Paper] = {
            p["paper_id"]: Paper.model_validate(p) for p in corpus_dicts
        }

        tokenized_passages = [_tokenize(p.text) for p in self.passages]
        self.bm25 = BM25Okapi(tokenized_passages) if tokenized_passages else None

    def raw_scores(self, claim: str) -> tuple[np.ndarray, np.ndarray]:
        """Raw cosine and BM25 scores for every passage, before normalization.

        Split out so the calibration script can collect these over many claims
        without paying for ranking it does not use.
        """
        n = len(self.passages)
        claim_embedding = self.embedding_model.encode(
            [claim], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        # k=n over an exact IndexFlatIP scores every passage, so a passage that
        # ranks poorly on one signal can still be recovered by the other. Both
        # index and query vectors are L2-normalized, so inner product is cosine.
        distances, indices = self.faiss_index.search(claim_embedding, n)
        cosine_scores = np.zeros(n, dtype="float32")
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                cosine_scores[idx] = score

        bm25_scores = np.array(self.bm25.get_scores(_tokenize(claim)), dtype="float32")
        return cosine_scores, bm25_scores

    def coverage_signals(self, cosine_scores: np.ndarray, bm25_scores: np.ndarray) -> CoverageSignals:
        """Absolute corpus-coverage signals from raw scores.

        Deliberately computed over the top-k by *cosine*, not over the fused
        ranking: the whole point is a gate that does not inherit the fusion
        weights it is supposed to be independent of.
        """
        k = min(settings.COVERAGE_TOP_K, len(cosine_scores))
        top_cosines = np.sort(cosine_scores)[::-1][:k]
        return CoverageSignals(
            max_cosine=float(top_cosines[0]) if k else 0.0,
            mean_topk_cosine=float(top_cosines.mean()) if k else 0.0,
            top_bm25=float(bm25_scores.max()) if len(bm25_scores) else 0.0,
            top_k=k,
        )

    def search(self, claim: str, top_k: int | None = None) -> SearchResult:
        """Rank passages against the claim and report absolute coverage signals."""
        top_k = top_k or settings.RETRIEVAL_TOP_K
        n = len(self.passages)
        if n == 0 or self.bm25 is None:
            empty = CoverageSignals(max_cosine=0.0, mean_topk_cosine=0.0, top_bm25=0.0, top_k=0)
            return SearchResult(passages=[], coverage=empty)

        cosine_scores, bm25_scores = self.raw_scores(claim)
        coverage = self.coverage_signals(cosine_scores, bm25_scores)

        dense_norm = _min_max_normalize(cosine_scores)
        bm25_norm = _min_max_normalize(bm25_scores)
        fused = settings.BM25_WEIGHT * bm25_norm + settings.DENSE_WEIGHT * dense_norm

        order = np.argsort(-fused)[:top_k]
        results = []
        for idx in order:
            passage = self.passages[idx]
            paper = self.papers_by_id.get(passage.paper_id)
            if paper is None:
                continue
            results.append(
                RetrievedPassage(
                    passage=passage,
                    paper=paper,
                    fused_score=float(fused[idx]),
                    raw_cosine=float(cosine_scores[idx]),
                    raw_bm25=float(bm25_scores[idx]),
                )
            )
        return SearchResult(passages=results, coverage=coverage)
