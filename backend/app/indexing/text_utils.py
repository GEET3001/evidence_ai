"""Sentence splitting shared by chunking and rationale extraction.

Both use this so indexed passage boundaries and the sentences reported as a
stance rationale stay aligned. Does not handle abbreviations such as "e.g.".
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
