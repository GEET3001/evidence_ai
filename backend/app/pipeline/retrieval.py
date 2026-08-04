"""Hybrid BM25 + dense retrieval over the indexed passage corpus.

Both signals score every passage, are normalized to [0,1], then fused by the
weights in config.py. The fused score is what MIN_SIMILARITY thresholds against.
"""

from __future__ import annotations

import app._thread_limits  # noqa: F401  (must precede the faiss/torch imports below)

import json
import re

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models import Paper, Passage

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

    def search(self, claim: str, top_k: int | None = None) -> list[tuple[Passage, Paper, float]]:
        """Return up to top_k (passage, paper, fused_relevance_score) tuples, ranked."""
        top_k = top_k or settings.RETRIEVAL_TOP_K
        n = len(self.passages)
        if n == 0 or self.bm25 is None:
            return []

        claim_embedding = self.embedding_model.encode(
            [claim], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        # k=n over an exact IndexFlatIP scores every passage, so a passage that
        # ranks poorly on one signal can still be recovered by the other.
        distances, indices = self.faiss_index.search(claim_embedding, n)
        dense_scores = np.zeros(n, dtype="float32")
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                dense_scores[idx] = score

        bm25_scores = np.array(self.bm25.get_scores(_tokenize(claim)), dtype="float32")

        dense_norm = _min_max_normalize(dense_scores)
        bm25_norm = _min_max_normalize(bm25_scores)
        fused = settings.BM25_WEIGHT * bm25_norm + settings.DENSE_WEIGHT * dense_norm

        order = np.argsort(-fused)[:top_k]
        results = []
        for idx in order:
            passage = self.passages[idx]
            paper = self.papers_by_id.get(passage.paper_id)
            if paper is None:
                continue
            results.append((passage, paper, float(fused[idx])))
        return results
