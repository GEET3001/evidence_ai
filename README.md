# EvidenceAI

**EvidenceAI** is an explainable research claim verification system for
mental health literature. A user submits a claim (e.g. "Mindfulness
meditation reduces symptoms of generalized anxiety disorder"); the system
retrieves relevant passages from a corpus of mental health research papers,
classifies each passage as supporting, contradicting, or neutral toward the
claim, aggregates those classifications into a verdict with a certainty
rating, and explains its reasoning with the underlying evidence attached.

## Scope

EvidenceAI is a **literature triage aid** for researchers, students,
clinicians, and journalists. **It is not a diagnostic tool, not treatment
advice, and not intended for patients making care decisions.** Verdicts
reflect what the retrieved literature says, not clinical guidance — always
consult a qualified professional for health decisions.

See [`docs/architecture.md`](docs/architecture.md) for the pipeline design and
the reasoning behind the main technical choices.

## Project Structure

```
evidenceai/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint
│   │   ├── models.py          # Pydantic schemas
│   │   ├── config.py          # settings, paths, constants
│   │   ├── ingestion/         # scrapers + API clients
│   │   ├── indexing/          # chunking, embedding, FAISS
│   │   └── pipeline/          # retrieval, classification, aggregation
│   ├── requirements.txt
│   └── .env.example
├── frontend/                  # Next.js app
├── data/
│   ├── raw/                   # scraped HTML/JSON, untouched (gitignored)
│   ├── corpus.json            # cleaned papers (committed)
│   └── index/                 # FAISS index + metadata lookup (gitignored)
├── eval/
│   ├── claims.csv             # hand-labelled eval claims (committed)
│   └── results/
├── docs/
│   └── architecture.md
└── notebooks/                 # data prep + Colab training
```

## Setup

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env
python -m app.indexing.build_index
uvicorn app.main:app --reload
```

Install torch first, or pip resolves the much larger CUDA-bundled wheel. The
index build is required before `/verify` will return anything — `data/index/`
is gitignored. The API is then available at `http://localhost:8000` (docs at
`/docs`).

To run on a CUDA GPU instead, install the matching torch build rather than the
CPU one, e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu130` — match the tag to the CUDA version
reported by `nvidia-smi`.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000`.

## Running a demo

Use `.\demo.ps1` rather than `.\dev.ps1`; it starts uvicorn without `--reload`,
which removes the reload watcher's second process and its restart cycles.

Check `http://localhost:8000/health` first — it reports the loaded models and
the device they are on:

```json
{"status": "ok", "pipeline_loaded": true, "rss_mb": 924.3,
 "models": {"embedding_model": "...", "stance_model_mode": "zeroshot",
            "stance_model_source": "facebook/bart-large-mnli",
            "stance_model_device": "cuda", "passages_indexed": 446}}
```

`stance_model_device` is the single biggest factor in responsiveness: a
`/verify` call takes roughly 140s on CPU and 2-4s on an RTX 3050, since a
request runs 20-40 bart-large-mnli forward passes. If it reads `cpu`
unexpectedly, torch was installed from the CPU wheel index. Send one throwaway
`/verify` call before recording so the first real request is not paying for
lazy CUDA context init.

## Design: ranking and gating are separate

Two different questions get asked of retrieval, and they need different scores.

**Ranking — "which passages are most relevant?"** Hybrid BM25 + dense fusion,
min-max normalized per query. Normalization is correct here: it puts two
incomparable scales on a common footing so they can be summed.

**Gating — "is this claim about anything in the corpus at all?"** The same
normalized score cannot answer this. Because it is relative to the current
query, the top-ranked passage of *every* query is stretched toward 1.0,
including a query the corpus knows nothing about. A threshold on it asks "is
this the best of what we found?", never "is any of this actually relevant?".
Off-corpus claims sailed through it: "eating chocolate cake improves stock
market forecasting" came back a confident `CONTRADICTED`.

So gating uses absolute signals, captured **before** any normalization and
carried through `RetrievedPassage` / `CoverageSignals` untouched:

- `max_cosine` — highest raw claim-passage cosine. Bounded, so comparable
  across queries.
- `mean_topk_cosine` — mean raw cosine over the top `COVERAGE_TOP_K`.
  Off-corpus claims are uniformly low, which makes the mean a steadier signal
  than the max, where one incidentally-similar passage can carry an unrelated
  claim over the line.
- `top_bm25` — measured and returned, but deliberately **not** part of the
  gate. It is unbounded and corpus-frequency-dependent, so there is no stable
  cross-query scale to put a fixed floor on. It is kept as a diagnostic:
  near-zero lexical overlap is strong independent evidence of an off-corpus
  claim.

A claim failing either cosine floor returns `INSUFFICIENT_EVIDENCE` with
`insufficient_reason = NOT_COVERED_BY_CORPUS`, and no evidence list — the
retrieved passages are the best of a bad set, and showing them as "evidence"
was the exact confusion this fixes. That is a distinct reason from
`EVIDENCE_INCONCLUSIVE`, which means the corpus does cover the claim but the
qualifying evidence carried no direction. Both read as `INSUFFICIENT_EVIDENCE`
to the user; the difference shows up in the explanation and limitations.

**Thresholds are calibrated, not guessed.** The useful range of raw cosine is a
property of the embedding model and this corpus, so it has to be measured.
`python -m app.pipeline.calibrate_coverage` scores 20 in-domain mental health
claims against 20 off-corpus ones (other scientific domains, nonsense claims,
and near-miss biomedical claims), prints both distributions, and picks floors
from the gap. Full data and method:
[`eval/results/similarity_calibration.md`](eval/results/similarity_calibration.md).
Results are `MIN_MAX_COSINE = 0.911` and `MIN_MEAN_TOPK_COSINE = 0.905` in
`config.py`; end to end this turned 9 confident-but-wrong off-corpus verdicts
into `NOT_COVERED_BY_CORPUS` while changing no in-domain verdict at all.

**Residual failure mode, honestly.** This is a working guard, not a solved
problem:

- `mean_topk_cosine` separates the two claim sets by **0.006** — real, but thin.
- `max_cosine` does **not** separate them (highest off-corpus 0.926 exceeds
  lowest in-domain 0.916), so it is set below the lowest in-domain observation
  as a safety net rather than tuned up at the cost of false rejections.
- Recalibrating on half the claims and testing on the held-out half catches
  9/9 off-corpus but wrongly rejects 4/10 in-domain — the honest generalization
  signal at this sample size.

Near-miss claims are where it will break: real biomedical claims that sound in
scope but aren't ("statins reduce cardiovascular mortality", "metformin
improves glycaemic control") sit closest to the boundary and are caught with
the least margin. Rebuilding the index or changing `EMBEDDING_MODEL` invalidates
these thresholds; re-run the calibration.

## Model Provenance

The stance classifier runs in one of two modes, set by `STANCE_MODEL_MODE` in
`backend/.env`:

- **`zeroshot`** (default) — `facebook/bart-large-mnli`, zero-shot NLI, no
  training. Works out of the box.
- **`finetuned`** — requires a checkpoint at `STANCE_MODEL_PATH` (default
  `backend/models/stance-deberta/`). Startup fails if it is missing rather than
  falling back to zero-shot. Kept as a supported path, but it measured worse
  than the baseline on the metric that matters — see the comparison below.

**Base model:** `microsoft/deberta-v3-base`, falling back to `roberta-base` if
the DeBERTa tokenizer fails to load. The 3-way head is labelled SUPPORT /
CONTRADICT / NEUTRAL at training time, which `stance.py` matches by name, so a
checkpoint drops in without code changes.

**Training data:** [SciFact](https://huggingface.co/datasets/allenai/scifact),
prepared by `notebooks/prepare_scifact.py` (which also documents why HealthVer
and PUBHEALTH were considered and declined as extra CONTRADICT sources).
SciFact's official `test` split carries no evidence annotations, so evaluation
uses a stratified 10% split carved out of `train.jsonl` at seed 42. Class
distribution across train+validation: SUPPORT 47.8%, CONTRADICT 26.6%, NEUTRAL
25.5% (1,739 triples).

**Training config** (`notebooks/train_stance.ipynb`, free Colab T4): fp16,
batch size 16 with gradient accumulation 2, 4 epochs, LR 2e-5, warmup ratio
0.1, sentence-pair tokenization truncated at 256 tokens on the passage side.
Class weighting is switchable via `WEIGHTING_MODE` and defaults to off. The
best checkpoint is selected on macro F1 rather than accuracy, which the
majority classes would dominate, or a single class's recall, which can improve
at the other classes' expense.

**Headline metrics** (129-example held-out split, measured by
`app/pipeline/compare_classifiers.py` — full report in
`eval/results/classifier_comparison.md`). The fine-tuned column is the
checkpoint trained on leak-corrected data:

| Metric | Zero-shot baseline | Fine-tuned |
|---|---|---|
| Accuracy | 51.9% | 58.9% |
| Macro F1 | 0.521 | 0.518 |
| CONTRADICT recall | 52.9% | **17.6%** |
| SUPPORT recall | 29.0% | 82.3% |
| NEUTRAL recall | 93.9% | 57.6% |
| Mean inference / passage | 36.4 ms | 26.1 ms |

**The default mode remains `zeroshot`.** Fine-tuning buys 7 points of
accuracy, ties on macro F1, and makes CONTRADICT recall substantially *worse*
(52.9% → 17.6%) — the specific weakness fine-tuning was meant to address. The
confusion matrix shows the failure directly: 25 of 34 CONTRADICT examples are
predicted SUPPORT. The model is not distinguishing the direction of the
evidence; it predicts SUPPORT for 89 of 129 examples (69%) when SUPPORT is
only 48% of the set.

**A label leak was found and fixed, and it was not the cause.** An earlier
checkpoint scored a perfect 1.000 precision *and* recall on NEUTRAL, which is
not a real result. `build_triples` gave SUPPORT/CONTRADICT rows only their
annotated rationale sentences while NEUTRAL rows got the entire abstract — a
median 1158 chars against ~200. A single length threshold predicted NEUTRAL at
99.0% accuracy against a 74.5% majority-class floor, so the model was learning
"long passage = NEUTRAL" instead of learning stance, and macro F1 was inflated
by a free 1.000 component.

`prepare_scifact.py` now samples a contiguous span for NEUTRAL rows, sized by
drawing from the empirical rationale-length distribution. Measured after the
fix: **74.5% separability — exactly the majority-class floor**, i.e. passage
length carries no information about the label. `report_length_leak()` re-checks
this on every run and warns if it regresses. Class balance and triple counts
are unchanged, so the held-out split indices remain valid.

Retraining on the corrected data confirmed the fix worked — NEUTRAL fell from
a fake 1.000/1.000 to a real 0.655/0.576, so the model is no longer riding
passage length — but it did **not** recover CONTRADICT recall, which fell
further (29.4% → 17.6%). The leak was a real measurement artifact inflating
the headline numbers; it was not what broke the CONTRADICT class.

**This line of work is closed.** Zero-shot is what ships. Two fine-tuning runs
on the same data — one leaked, one clean — both landed below the baseline on
CONTRADICT recall, and the clean run only tied it on macro F1. The untested
hypothesis is class imbalance (SUPPORT 47.8% vs CONTRADICT 26.6%) against
`WEIGHTING_MODE = 'none'`; anyone picking this up should try
`WEIGHTING_MODE = 'balanced'` first. But 1,739 SciFact triples is a small
corpus for teaching a 3-way directional distinction, and the honest reading is
that this dataset at this size does not beat an NLI model that was pretrained
on far more entailment data.

The pre-fix run's own notebook metrics are kept at
`eval/results/held_out_test_metrics_leaked_run.json` as the record of what a
leaked split looks like from the inside — note the 1.000/1.000/1.000 NEUTRAL
row.

Baseline numbers shift slightly against the previously recorded run (50.4% →
51.9% accuracy) because the leak fix rewrote the NEUTRAL passages, so the
zero-shot model is being scored on different text for a third of the split.
The split indices themselves are unchanged.

### Reproducing the fine-tuned checkpoint

Not required to run EvidenceAI — `zeroshot` is the default and needs no
checkpoint. These steps exist so the measured comparison above can be
reproduced or extended.

1. Run `notebooks/prepare_scifact.py` to produce
   `data/scifact/{train,validation,test}.jsonl`. Keep `--seed` fixed across a
   train/eval cycle — it seeds NEUTRAL span sampling, so changing it changes
   the passages underneath a checkpoint.
2. Upload all three files to `MyDrive/EvidenceAI/data/scifact/` — including
   the 0-byte `test.jsonl`, which the notebook opens unconditionally — then
   run `notebooks/train_stance.ipynb` end to end on a free Colab T4. The
   project folder must be named exactly `EvidenceAI`; `PROJECT_DIR` is
   hardcoded.
3. Unzip the model from the notebook's final cell into
   `backend/models/stance-deberta/`. That directory is gitignored.
4. Set `STANCE_MODEL_MODE=finetuned` in `backend/.env`.
5. Run `python -m app.pipeline.compare_classifiers` for the real
   fine-tuned-vs-baseline numbers.

Training needs the T4: a 4GB laptop GPU cannot hold this fine-tune. DeBERTa-v3's
128100-token vocabulary makes the embedding matrix alone cost 393MB of gradient
plus 786MB of AdamW state on top of the rest of the model, which OOMs a 4GB card
even with gradient checkpointing and the embeddings frozen (measured).

## Constraints

- Zero budget: no paid APIs. Everything runs on free, local, or free-tier
  resources.
- Model training happens in Google Colab (free T4 GPU); inference runs
  locally.
- Explanations are faithful — derived from rationale sentences and computed
  values, never free-text generated by a language model.
