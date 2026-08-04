"""Split a paper's abstract into overlapping passage chunks for indexing.

Sentences are packed up to CHUNK_SIZE_TOKENS with CHUNK_OVERLAP_TOKENS of
trailing overlap. Chunks always break on sentence boundaries, since stance
rationale extraction re-splits passages into sentences downstream.

The caller passes in the embedding model's own tokenizer so token counts here
match the model that will embed the result.
"""

from __future__ import annotations

from typing import Protocol

from app.config import settings
from app.indexing.text_utils import split_sentences
from app.models import Paper, Passage


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def _count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _sentence_offsets(abstract: str, sentences: list[str]) -> list[tuple[int, int]]:
    """Locate each sentence's (char_start, char_end) in the original abstract."""
    offsets = []
    cursor = 0
    for sentence in sentences:
        idx = abstract.find(sentence, cursor)
        if idx == -1:
            idx = cursor
        offsets.append((idx, idx + len(sentence)))
        cursor = idx + len(sentence)
    return offsets


def chunk_paper(
    paper: Paper,
    tokenizer: Tokenizer,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Passage]:
    """Chunk one paper's abstract into Passages. Returns [] for an empty abstract."""
    chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
    overlap = overlap or settings.CHUNK_OVERLAP_TOKENS

    sentences = split_sentences(paper.abstract)
    if not sentences:
        return []

    offsets = _sentence_offsets(paper.abstract, sentences)
    token_counts = [_count_tokens(tokenizer, s) for s in sentences]
    n = len(sentences)

    spans: list[tuple[int, int]] = []  # (start_idx, end_idx_inclusive) into sentences
    start = 0
    prev_end = -1
    while start < n:
        end = start
        total = 0
        # Always take at least one sentence, even if it alone exceeds chunk_size.
        while end < n and (total == 0 or total + token_counts[end] <= chunk_size):
            total += token_counts[end]
            end += 1
        end -= 1

        # An oversized sentence can make the overlap rewind re-pack the same
        # trailing sentences forever. If this chunk ended where the last one
        # did, skip past it rather than emit a near-duplicate.
        if end == prev_end:
            start = end + 1
            continue

        spans.append((start, end))
        prev_end = end

        if end == n - 1:
            break

        # Rewind from the chunk's end to find where ~overlap tokens of trailing
        # context begins; that's the next chunk's start.
        back = end
        overlap_tokens = 0
        while back > start and overlap_tokens < overlap:
            overlap_tokens += token_counts[back]
            back -= 1
        next_start = back + 1
        start = next_start if next_start > start else end + 1

    passages = []
    for i, (s_idx, e_idx) in enumerate(spans):
        char_start, _ = offsets[s_idx]
        _, char_end = offsets[e_idx]
        passages.append(
            Passage(
                passage_id=f"{paper.paper_id}_p{i:03d}",
                paper_id=paper.paper_id,
                # Sliced from the source rather than rejoined, so the text
                # matches abstract[char_start:char_end] exactly.
                text=paper.abstract[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                section="Abstract",
            )
        )
    return passages
