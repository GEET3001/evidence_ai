# Architecture

- [The offline/online split](#the-offlineonline-split)
- [Verdict decision tree](#verdict-decision-tree)
- [Design decisions](#design-decisions)
- [Not implemented](#not-implemented)

## The offline/online split

![EvidenceAI architecture](diagrams/architecture.png)

The division down the middle of that diagram is the most important thing about
this system's shape.

**Offline** — scraping, merging, OpenAlex enrichment, validation, chunking and
embedding — runs once, by hand, and produces two files: `data/corpus.json` and
`data/index/`. It calls the network heavily and takes a long time. None of it
happens while a user waits.

**Online** — retrieval, the coverage gate, stance classification, aggregation
and explanation — runs on every `POST /verify` and makes **no network calls at
all**. Both models and the index are loaded once into a process-level singleton
by `main.py`'s lifespan handler, so a request is pure local computation.

That split is why a verification takes 2–4 seconds rather than minutes, and why
the system works with no internet connection once built.

```
claim ──▶ retrieval ──▶ coverage gate ──▶ stance ──▶ aggregation ──▶ explanation
          BM25+dense      raw cosine       NLI per     counts,        assembled
          over FAISS      floors           passage     thresholds     from values
```

One structural detail worth noting: when the coverage gate rejects a claim, the
pipeline skips classification **entirely** rather than classifying and then
discarding. That is 20–40 NLI forward passes not run, which is why an off-corpus
claim returns in about 10 ms against 2–4 s for a covered one.

## Verdict decision tree

![Verdict decision tree](diagrams/verdict-decision-tree.png)

This is a transcription of `pipeline/verdict.py`, and it exists to make one
claim checkable: **abstention here is principled, not ad hoc.** There are
exactly three routes to `INSUFFICIENT_EVIDENCE`, each with a different cause:

| Route | Reason returned | What it means |
|---|---|---|
| Coverage gate fails | `NOT_COVERED_BY_CORPUS` | Nothing in the corpus is about this claim. Evidence list is returned **empty** and classification never runs. |
| Fewer than 3 qualifying passages | `EVIDENCE_INCONCLUSIVE` | The corpus covers the topic, but too little cleared the relevance threshold to aggregate. |
| Zero directional passages | `EVIDENCE_INCONCLUSIVE` | Material was retrieved and classified, but every passage came back neutral. |

The first is a statement about the corpus. The other two are statements about
the evidence. Collapsing them — which is what the system did before the
coverage gate existed — is what let "eating chocolate cake improves stock
market forecasting" come back as a confident `CONTRADICTED`.

> **Naming caveat.** `MIN_RELEVANT_SOURCES` is compared against
> `len(qualifying)`, which counts **passages, not distinct papers**. Three
> passages chunked out of one paper satisfy it. The constant's name and its
> comment in `config.py` both say "sources", so the guarantee is weaker than
> the name implies. Documented here rather than quietly corrected, because
> tightening it to distinct papers would change verdicts and belongs with a
> re-run of the evaluation.

## Design decisions

### Explanations are derived, never generated

`explanation` is a string assembled from counts, thresholds, and sentences
quoted verbatim out of the retrieved passages. No language model writes any
part of it.

This is the central design commitment of the project, and it is a deliberate
trade of fluency for verifiability. An LLM could write a much better-reading
paragraph. It could also write a well-reading paragraph that does not describe
what the system actually computed, and nothing in the output would show the
difference. A generated explanation is a *plausible* account of a decision; it
is not evidence about that decision.

Everything in an EvidenceAI explanation traces to one of two things a reader
can check on the same page: a number that appears in the counts, or a quotation
that appears in a passage. Rationale sentences are the load-bearing part —
`EvidenceItem.rationale_sentences` is produced by re-running the stance model
on each sentence of the passage and keeping whichever scores highest for the
label that passage already received. It is a readout of the classifier that
made the call, not a separate similarity heuristic run afterwards to justify
it. The mechanism is model-agnostic, so it would apply unchanged to a
fine-tuned checkpoint.

The frontend and the .docx report both surface this to the user rather than
keeping it as an internal note, because "this explanation was derived, not
written" is exactly the property that should inform how much the reader trusts
it.

### Ranking and gating use different signals

Two different questions get asked of retrieval and they cannot share a score.

**Ranking — "which passages are most relevant?"** Hybrid BM25 + dense fusion,
min-max normalised per query, weighted 0.4 / 0.6. Normalisation is correct
here: it puts two incomparable scales on a common footing so they can be summed.

**Gating — "is this claim about anything in the corpus at all?"** The same
normalised score structurally cannot answer this. Because normalisation is
relative to the current query, the top-ranked passage of *every* query is
stretched toward 1.0 — including a query about nothing in the corpus. A
threshold on it asks "is this the best of what we found?" and can never ask "is
any of this actually relevant?".

So the gate uses absolute signals captured **before** normalisation and carried
through `CoverageSignals` untouched:

- `max_cosine` — highest raw claim-passage cosine. Bounded, so comparable across queries.
- `mean_topk_cosine` — mean raw cosine over the top `COVERAGE_TOP_K`. Steadier than the max, where one incidentally-similar passage can carry an unrelated claim over the line.
- `top_bm25` — measured and returned but deliberately **not** gated on. It is unbounded and corpus-frequency-dependent, so there is no stable cross-query scale to put a fixed floor on. Kept as a diagnostic.

Thresholds are measured, not guessed — the useful range of raw cosine is a
property of the embedding model and this corpus. See
[`eval/results/similarity_calibration.md`](../eval/results/similarity_calibration.md)
and the [final report](final-report.md#the-coverage-gate) for the calibration
and its honest failure modes.

### Zero-shot was retained over the fine-tuned model

`STANCE_MODEL_MODE` defaults to `zeroshot` (`facebook/bart-large-mnli`). A
fine-tuned DeBERTa-v3 checkpoint exists as a supported path and measured
**worse** on the metric that mattered.

Two training runs, one on leaked data and one on corrected data, both landed
below the zero-shot baseline on CONTRADICT recall — 52.9% baseline against
29.4% and then 17.6%. The clean run only tied on macro F1 (0.518 vs 0.521)
while buying 7 points of accuracy. Since the entire purpose of fine-tuning was
to improve CONTRADICT detection, and a claim-verification system that cannot
recognise contradicting evidence is failing at its core task, the accuracy gain
does not buy anything worth having.

The full experimental narrative — hypothesis, the label leak, its diagnosis and
fix, the retrain, and the refutation — is in the
[final report](final-report.md#the-fine-tuning-experiment). It is the most
substantial piece of empirical work in the project.

### BLAS threading is pinned to one thread

`faiss-cpu` and `torch` each bundle their own OpenMP runtime — `vcomp140.dll` +
`libopenblas.dll` for faiss, MKL-backed `libiomp5md.dll` for torch — and each
sizes its internal thread pool to the CPU core count **at native-library-load
time**. Running both multithreaded in one process reliably segfaults on Windows
(SIGSEGV, exit 139, no Python traceback) on first concurrent use.

`app/_thread_limits.py` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` to 1 and **must be imported before torch or faiss
anywhere in the process**. Three things were confirmed experimentally rather
than assumed:

- RSS at crash was ~480 MB on a 16 GB machine, so this was a threading bug despite superficially resembling memory pressure.
- Calling `torch.set_num_threads(1)` / `faiss.omp_set_num_threads(1)` *after* import does **not** prevent it. The variables are read at DLL-load time.
- `KMP_DUPLICATE_LIB_OK=TRUE` alone does not fix it — this is a concurrent-thread-pool crash, not a duplicate-symbol abort.

Single-threaded BLAS costs nothing here: passages are scored one at a time over
a few hundred passages, so intra-op parallelism was never buying throughput.

### Models run on GPU

`stance_model_device` is the single largest factor in responsiveness. A
`/verify` call runs 20–40 `bart-large-mnli` forward passes — one per retrieved
passage, plus one per sentence for rationale extraction — which takes roughly
**140 s on CPU and 2–4 s on an RTX 3050**. That is not a tuning difference; it
is the difference between a demo and an unusable system.

`/health` reports the device it actually landed on, because the failure is
silent: install torch from the CPU wheel index and everything still works, just
seventy times slower.

### Retrieval scores every passage

Both BM25 and dense similarity score the whole corpus rather than each
retriever returning its own top-N for fusion. At a few hundred passages this
costs nothing, and it avoids the common hybrid-search failure where a passage
that ranks well on one signal is dropped for falling outside the other
retriever's cutoff.

### Stance labels are matched by name

The model's own `id2label` is mapped onto SUPPORT / CONTRADICT / NEUTRAL by
label *name*. A fine-tuned checkpoint is not guaranteed to preserve
`bart-large-mnli`'s label ordering, and matching by index would silently invert
verdicts if it did not.

### Verdict shares exclude neutral evidence

Support and contradict shares are computed over directional evidence only.
Including neutral passages in the denominator would let the volume of
background material retrieved shift a verdict across `TIE_MARGIN`, which is
unrelated to whether the literature actually disagrees.

`grade_certainty` is GRADE-*inspired*, not GRADE: a deterministic set of
adjustments over publication tier, preprint share, and evidence volume. Roughly
a third of the corpus carries no tier classification, so the rules treat a weak
tier signal as the normal case. No reviewer assesses risk of bias, imprecision
or publication bias, as real GRADE requires.

### Retracted papers are excluded at index time

`build_index.py` drops any paper flagged `is_retracted` before chunking, so a
retracted paper cannot be retrieved as evidence at all. `is_retracted` is always
a definite boolean and is paired with `openalex_checked`, so "not retracted" and
"never checked" stay distinguishable. The UI and report still carry a retraction
warning for defence in depth, since enrichment can run after an index build.

### Chunking is sentence-aligned

Passages are packed to a token budget on sentence boundaries rather than sliced
by a token window, because rationale extraction re-splits passages into
sentences downstream and a truncated sentence would corrupt that. Token counts
come from the embedding model's own tokenizer, so they match the model that
embeds the result.

### Verdicts persist to SQLite

`POST /verify` writes the full `VerdictResponse` as JSON into
`data/verifications.db` so `GET /verify/{id}` and the .docx report work on a
later request. One file next to the corpus, no service to run.

The write is best-effort by design: the verdict has already been computed and
is about to be returned, so a storage failure logs to stderr and is swallowed
rather than turning a successful verification into a 500.

## Not implemented

- **PICO extraction.** `PICOClaim` carries only `raw_claim`; `population_match` is always `None`. Claims are retrieved against and assessed exactly as written, and population indirectness is not detected. The UI renders this as explicit blanks rather than hiding the panel.
- **Automated tests.** Verification has been manual and instrumented: live endpoint calls, the corpus validation report, the classifier comparison script, the coverage calibration script, and the evaluation harness.
- **A labelled evaluation set.** The harness and the 50 candidate claims exist; the labels do not yet. See [`eval/README.md`](../eval/README.md).
