"""Application settings: paths, model names, and tunable constants.

Pipeline constants belong here rather than inline at their call sites.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: backend/app/config.py -> backend/app -> backend -> evidenceai
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    """Central configuration, overridable via environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Identity ---
    # Quoted in generated reports and served by the API, so a report can be
    # traced back to the version of the system that produced it.
    APP_VERSION: str = "0.1.0"

    # --- Paths ---
    data_dir: Path = DATA_DIR
    raw_dir: Path = DATA_DIR / "raw"
    corpus_path: Path = DATA_DIR / "corpus.json"
    index_dir: Path = DATA_DIR / "index"
    verifications_db_path: Path = DATA_DIR / "verifications.db"

    # --- NCBI E-utilities identification (see .env.example for how to get a free API key) ---
    NCBI_TOOL: str = "EvidenceAI"
    NCBI_EMAIL: str = "geetnatu73@gmail.com"
    NCBI_API_KEY: str | None = None

    # --- OpenAlex (see app/ingestion/openalex_enrich.py) ---
    # mailto puts requests in OpenAlex's "polite pool" for faster, more reliable responses.
    OPENALEX_MAILTO: str = "geetnatu73@gmail.com"

    # --- Models ---
    EMBEDDING_MODEL: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    NLI_BASELINE_MODEL: str = "facebook/bart-large-mnli"
    STANCE_MODEL_PATH: str = "models/stance-deberta"  # fine-tuned checkpoint, produced in Colab
    # Explicit rather than inferred from whether the path exists, so asking for
    # the fine-tuned model can never silently fall back to zero-shot.
    STANCE_MODEL_MODE: Literal["zeroshot", "finetuned"] = "zeroshot"

    # --- Chunking ---
    CHUNK_SIZE_TOKENS: int = 256
    CHUNK_OVERLAP_TOKENS: int = 32

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = 10
    BM25_WEIGHT: float = 0.4
    DENSE_WEIGHT: float = 0.6

    # --- Corpus-coverage gate ---
    # Answers "is this claim about anything in the corpus at all?", which
    # MIN_SIMILARITY structurally cannot: it thresholds a per-query
    # min-max-normalized score, so the top hit of every query approaches 1.0
    # regardless of true relevance. These operate on raw, pre-normalization
    # scores and are therefore comparable across queries.
    #
    # Values are calibrated empirically, not guessed — see
    # eval/results/similarity_calibration.md and
    # `python -m app.pipeline.calibrate_coverage` to regenerate. Rebuilding the
    # index or changing EMBEDDING_MODEL invalidates them; re-run calibration.
    COVERAGE_TOP_K: int = 10
    # max_cosine does not separate the two claim sets on its own; it sits below
    # the lowest observed in-domain value as a safety net. mean_topk_cosine is
    # the signal that actually separates, by a thin 0.006 margin.
    MIN_MAX_COSINE: float = 0.911
    MIN_MEAN_TOPK_COSINE: float = 0.905

    # --- Verdict / insufficient-evidence thresholds ---
    # Minimum number of distinct sources required before a verdict is issued
    # rather than "insufficient evidence".
    MIN_RELEVANT_SOURCES: int = 3
    # Minimum *relative* rank score for a passage to count as relevant, applied
    # only after the coverage gate above has established the claim is on-topic.
    MIN_SIMILARITY: float = 0.45
    # Maximum allowed gap between support-share and contradict-share for a
    # verdict to be called "mixed" instead of a clear support/contradict.
    TIE_MARGIN: float = 0.1


settings = Settings()
