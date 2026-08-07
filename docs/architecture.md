# Architecture

## Pipeline

A claim goes through three stages, wired together by `pipeline/service.py` and
exposed as `POST /verify`:

```
claim ──▶ retrieval ──▶ stance classification ──▶ verdict aggregation ──▶ response
          BM25 + dense    NLI per passage          counts, thresholds,
          over FAISS      + rationale sentence     certainty grade
```

The corpus is built offline by a separate pipeline: scrape, merge and dedupe,
enrich from OpenAlex, chunk, embed, index.

## Design decisions

### Retrieval scores every passage

Both BM25 and dense similarity score the whole corpus rather than each
retriever returning its own top-N for fusion. At a few hundred passages this
costs nothing, and it avoids the common hybrid-search failure where a passage
that ranks well on one signal is dropped because it fell outside the other
retriever's cutoff.

Scores are min-max normalized per query before fusion, which bounds the fused
score in [0,1] and lets two incomparable scales be summed. That score is used
for **ordering only**. It cannot also serve as a relevance threshold: because
normalization is relative to the current query, the top-ranked passage of an
unrelated claim is stretched toward 1.0 too. Raw pre-normalization cosine and
BM25 are therefore carried alongside it, and the corpus-coverage gate thresholds
those instead — see "ranking and gating are separate" in the README.

### Rationale sentences come from the classifier

`EvidenceItem.rationale_sentences` is produced by re-running the stance model on
each sentence of the passage and keeping whichever scores highest for the label
the passage received. It is a readout of the model that made the call, not a
separate similarity heuristic, which is what makes the explanation faithful
rather than merely plausible. The mechanism is model-agnostic, so it applies
unchanged to a fine-tuned checkpoint.

### Stance labels are matched by name

The model's own `id2label` is mapped onto SUPPORT / CONTRADICT / NEUTRAL by
label name. A fine-tuned checkpoint is not guaranteed to preserve
bart-large-mnli's label ordering, and matching by index would silently invert
verdicts if it did not.

### Verdict shares exclude neutral evidence

Support and contradict shares are computed over directional evidence only.
Including neutral passages in the denominator would let the amount of
background material retrieved shift a verdict across `TIE_MARGIN`, which is
unrelated to whether the literature actually disagrees.

`grade_certainty` is GRADE-inspired, not GRADE: a deterministic set of
adjustments over publication tier, preprint share, and evidence volume. Roughly
a third of the corpus has no tier classification, so the rules treat a weak tier
signal as the normal case.

### Retracted papers are excluded at index time

`build_index.py` drops any paper flagged `is_retracted` before chunking, so a
retracted paper cannot be retrieved as evidence at all. `is_retracted` is always
a definite boolean and is paired with `openalex_checked`, so "not retracted" and
"never checked" stay distinguishable.

### Chunking is sentence-aligned

Passages are packed to a token budget on sentence boundaries rather than sliced
by a token window, because stance rationale extraction re-splits passages into
sentences downstream and a truncated sentence would corrupt that. Token counts
come from the embedding model's own tokenizer so they match the model that
embeds the result.

### Models load once at startup

`main.py`'s lifespan handler builds the pipeline singleton before the app serves
traffic. Loading lazily would push a multi-second model load onto the first user
request and defer a missing-index error until runtime instead of boot.

### BLAS threading is pinned to one thread

faiss-cpu and torch each bundle an OpenMP runtime and size their thread pools
when their native libraries load. Running both multithreaded in one process
segfaults on Windows. `app/_thread_limits.py` sets the relevant environment
variables and must be imported before either library. Passages are scored one at
a time, so intra-op parallelism was not buying throughput here anyway.

## Not implemented

- **Verdict persistence.** Verdicts are computed per request and never stored,
  so `GET /verify/{id}` and the report export return 501.
- **PICO extraction.** `PICOClaim` carries only `raw_claim`, and
  `population_match` is always `None`.
- **Automated tests.** Verification has been manual: live endpoint calls, the
  corpus validation report, and the classifier comparison script.
