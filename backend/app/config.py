"""Application settings: paths, model names, and tunable constants.

All magic numbers used elsewhere in the pipeline (chunking, retrieval,
verdict thresholds) must be defined here, not inlined at call sites.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: backend/app/config.py -> backend/app -> backend -> evidenceai
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    """Central configuration for EvidenceAI.

    Values can be overridden via environment variables or a `.env` file
    (see `.env.example`).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Paths ---
    data_dir: Path = DATA_DIR
    raw_dir: Path = DATA_DIR / "raw"
    corpus_path: Path = DATA_DIR / "corpus.json"
    index_dir: Path = DATA_DIR / "index"

    # --- Models ---
    EMBEDDING_MODEL: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    NLI_BASELINE_MODEL: str = "facebook/bart-large-mnli"
    STANCE_MODEL_PATH: str = "models/stance-deberta"  # fine-tuned checkpoint, produced in Colab

    # --- Chunking ---
    CHUNK_SIZE_TOKENS: int = 256
    CHUNK_OVERLAP_TOKENS: int = 32

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = 10
    BM25_WEIGHT: float = 0.4
    DENSE_WEIGHT: float = 0.6

    # --- Verdict / insufficient-evidence thresholds ---
    # Minimum number of distinct sources required before a verdict is issued
    # rather than "insufficient evidence".
    MIN_RELEVANT_SOURCES: int = 3
    # Minimum retrieval similarity score for a passage to count as relevant.
    MIN_SIMILARITY: float = 0.45
    # Maximum allowed gap between support-share and contradict-share for a
    # verdict to be called "mixed" instead of a clear support/contradict.
    TIE_MARGIN: float = 0.1


settings = Settings()
