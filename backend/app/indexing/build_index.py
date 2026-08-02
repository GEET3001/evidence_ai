"""CLI to build the FAISS + passage index from data/corpus.json.

Writes to data/index/ (gitignored) — must be re-run locally after cloning or
whenever the corpus changes, before the API can serve real /verify results.

Usage:
    python -m app.indexing.build_index
"""

from __future__ import annotations

import json
import sys

import faiss
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.indexing.chunking import chunk_paper
from app.models import Paper


def load_corpus() -> list[Paper]:
    with open(settings.corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Paper.model_validate(p) for p in data]


def build() -> None:
    papers = load_corpus()
    retracted = [p for p in papers if p.is_retracted]
    eligible = [p for p in papers if not p.is_retracted]

    print(f"Loaded {len(papers)} papers ({len(retracted)} retracted, excluded from index).")
    for p in retracted:
        print(f"  excluded (retracted): {p.paper_id}: {p.title[:70]}")

    print(f"Loading embedding model: {settings.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    print("Chunking abstracts into passages...")
    passages = []
    for paper in eligible:
        passages.extend(chunk_paper(paper, model.tokenizer))
    print(f"Produced {len(passages)} passages from {len(eligible)} papers.")

    if not passages:
        print("No passages to index — aborting.", file=sys.stderr)
        raise SystemExit(1)

    print("Embedding passages...")
    texts = [p.text for p in passages]
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.index_dir / "faiss.index"))

    # Written in FAISS-insertion order — that ordering IS the contract with
    # pipeline.retrieval (row i of the index <-> passages[i]).
    with open(settings.index_dir / "passages.json", "w", encoding="utf-8") as f:
        json.dump(
            [p.model_dump(mode="json") for p in passages], f, indent=2, ensure_ascii=False
        )

    meta = {
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dim": dim,
        "paper_count": len(eligible),
        "excluded_retracted_count": len(retracted),
        "passage_count": len(passages),
    }
    with open(settings.index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("Index build summary")
    print("=" * 70)
    for key, value in meta.items():
        print(f"  {key}: {value}")
    print(f"\nWrote FAISS index + passages.json + meta.json to {settings.index_dir}")


if __name__ == "__main__":
    build()
