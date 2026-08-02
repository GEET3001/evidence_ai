"""Shared sentence-splitting used by both chunking (indexing) and rationale
extraction (pipeline.stance), so indexed sentence boundaries and the sentences
shown as a stance's rationale never silently drift apart.

Regex-based rather than NLTK/spaCy, consistent with this codebase's existing
style (validate_corpus.py, runner.py are `re`-only). Known limitation: doesn't
handle abbreviations ("e.g.", "Fig.", "vs.") — accepted as a documented
tradeoff rather than a new dependency for a corpus of abstracts.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Empty/whitespace-only input returns []."""
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
