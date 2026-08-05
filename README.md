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

**Known limitation.** `MIN_SIMILARITY` is compared against a per-query,
min-max-normalized relevance score, so the top-ranked passage for any claim —
including one unrelated to the corpus — is stretched toward 1.0 and can clear
the threshold. An off-corpus claim ("eating chocolate cake improves stock
market forecasting") returned a confident `CONTRADICTED` verdict rather than
`INSUFFICIENT_EVIDENCE`. `INSUFFICIENT_EVIDENCE` still fires correctly when
qualifying evidence carries no directional signal, just not for "this claim is
unrelated to the corpus". Fixing this needs an absolute similarity scale rather
than a per-query one.

## Model Provenance

The stance classifier runs in one of two modes, set by `STANCE_MODEL_MODE` in
`backend/.env`:

- **`zeroshot`** (default) — `facebook/bart-large-mnli`, zero-shot NLI, no
  training. Works out of the box.
- **`finetuned`** — requires a checkpoint at `STANCE_MODEL_PATH` (default
  `backend/models/stance-deberta/`). Startup fails if it is missing rather than
  falling back to zero-shot.

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

**Headline metrics — SUPERSEDED, retrain pending.** The numbers below come
from a checkpoint trained on data that carried a label leak (see the artifact
note after the table). The leak is now fixed in `prepare_scifact.py`, but the
model has not yet been retrained on the corrected data, so these remain the
only measured results. Treat them as the record of a failed attempt, not as
the model's ceiling.

(129-example held-out split, measured by `app/pipeline/compare_classifiers.py`
— full report in `eval/results/classifier_comparison.md`):

| Metric | Zero-shot baseline | Fine-tuned |
|---|---|---|
| Accuracy | 50.4% | 71.3% |
| Macro F1 | 0.507 | 0.692 |
| CONTRADICT recall | 52.9% | **29.4%** |
| SUPPORT recall | 29.0% | 79.0% |
| NEUTRAL recall | 87.9% | 100% |
| Mean inference / passage | 109.1 ms | 80.1 ms |

The notebook's own held-out evaluation
(`eval/results/held_out_test_metrics.json`) agrees within noise: 72.9%
accuracy, 0.688 macro F1, 23.5% CONTRADICT recall.

**The default mode remains `zeroshot`.** Accuracy and macro F1 both improved
by a wide margin, but CONTRADICT recall — the specific weakness fine-tuning
was meant to address — got substantially *worse* (52.9% → 29.4%). The
fine-tuned confusion matrix shows why: 24 of 34 CONTRADICT examples are
predicted SUPPORT, so the model largely stopped distinguishing the direction
of the evidence.

**Root cause — NEUTRAL was separable by passage length (FIXED in the data
prep, not yet retrained).** The fine-tuned model scored a perfect 1.000
precision *and* recall on NEUTRAL, which is not a real result. `build_triples`
gave SUPPORT/CONTRADICT rows only their annotated rationale sentences while
NEUTRAL rows got the entire abstract — a median 1158 chars against ~200. A
single length threshold predicted NEUTRAL at 99.0% accuracy against a 74.5%
majority-class floor, so the model learned "long passage = NEUTRAL" instead of
learning stance. That also inflated macro F1, one of whose three components
was a free 1.000, which is why the headline numbers looked like an improvement
while the metric that mattered regressed.

`prepare_scifact.py` now samples a contiguous span for NEUTRAL rows, sized by
drawing from the empirical rationale-length distribution. Measured after the
fix: **74.5% separability — exactly the majority-class floor**, i.e. passage
length carries no information about the label. `report_length_leak()` re-checks
this on every run and warns if it regresses. Class balance and triple counts
are unchanged, so the held-out split indices remain valid.

Retraining on the corrected data is the next step; `WEIGHTING_MODE` tuning is
only worth attempting after that, since the previous run's class behaviour was
confounded by the leak.

### Obtaining or retraining the checkpoint

1. Run `notebooks/prepare_scifact.py` to produce
   `data/scifact/{train,validation,test}.jsonl`. Keep `--seed` fixed across a
   train/eval cycle — it seeds NEUTRAL span sampling, so changing it changes
   the passages underneath a checkpoint.
2. Upload `train.jsonl` and `validation.jsonl` to Google Drive, then run
   `notebooks/train_stance.ipynb` end to end on a free Colab T4.
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
