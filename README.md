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
│   │   ├── pipeline/          # retrieval, classification, aggregation
│   │   └── reporting/         # docx/pdf export
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
│   ├── architecture.md
│   └── diagrams/
└── notebooks/                 # Colab training notebooks
```

## Setup

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
playwright install
copy .env.example .env
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000` (docs at `/docs`).

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000`.

## Demo Mode

The backend loads two transformer models (embedding + stance) plus a FAISS
index in one process. On Windows, running faiss-cpu and torch multithreaded
in the same process is known to segfault (see `app/_thread_limits.py` for
the full diagnosis) — this is fixed by forcing single-threaded BLAS/OpenMP
via env vars set before either library is imported, which is now baked into
the code (no manual steps needed). What *is* still worth doing manually
before recording:

1. **Close memory-heavy apps** (extra browser windows, etc.) — not required
   for correctness anymore (the segfault was a threading bug, not a memory
   one; RSS at crash time was ~480MB on a 16GB machine), but startup is
   faster with less contention.
2. **Start with `.\demo.ps1`**, not `.\dev.ps1` — it runs uvicorn without
   `--reload`. The reload watcher adds a second process and reload cycles
   that are pure risk for a recording, with no benefit once you're not
   editing code.
3. **Check `http://localhost:8000/health` before recording.** It reports:
   ```json
   {"status": "ok", "pipeline_loaded": true, "rss_mb": 924.3,
    "models": {"embedding_model": "...", "stance_model_mode": "zeroshot",
               "stance_model_source": "facebook/bart-large-mnli",
               "stance_model_device": "cuda", "passages_indexed": 446}}
   ```
   Confirm `status: "ok"` and, if this machine has a CUDA GPU,
   `stance_model_device: "cuda"` — the biggest single factor in demo
   responsiveness. On CPU, a single `/verify` call takes **~140s**
   (bart-large-mnli forward passes are compute-bound at ~3s each,
   single-threaded, and a request runs ~20-40 of them). On this
   machine's RTX 3050, the same call takes **~2-4s**. If `stance_model_device`
   ever reads `"cpu"` unexpectedly, torch was reinstalled as the CPU-only
   build — reinstall with `pip install torch --index-url
   https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps`
   (match the cu-tag to what `nvidia-smi`'s "CUDA Version" supports).
4. **Do a throwaway `/verify` call first** so the first real request in the
   recording isn't paying for lazy CUDA context init.

**Known limitation, not caused by the above:** `MIN_SIMILARITY` is compared
against a *per-query, min-max-normalized* relevance score
(`app/pipeline/retrieval.py`), so the top-ranked passage for any claim —
even one entirely unrelated to the corpus — is stretched toward 1.0 and can
still clear the threshold. A live check with an off-corpus claim ("eating
chocolate cake improves stock market forecasting") returned a confident
`CONTRADICTED` verdict off a mukbang-video paper rather than
`INSUFFICIENT_EVIDENCE`. `INSUFFICIENT_EVIDENCE` still fires correctly when
qualifying evidence has no directional (support/contradict) signal at all,
just not for "this claim has nothing to do with the corpus." Avoid picking
an obviously off-topic claim for the demo until this is addressed.

## Model Provenance

The stance classifier (`backend/app/pipeline/stance.py`) can run in two modes,
switched via `STANCE_MODEL_MODE` in `backend/.env` (default `zeroshot`, no
setup required):

- **`zeroshot`** (default) — `facebook/bart-large-mnli`, zero-shot NLI, no
  training. Works out of the box.
- **`finetuned`** — a model fine-tuned specifically for this task. Requires a
  real checkpoint at `STANCE_MODEL_PATH` (default `models/stance-deberta`,
  i.e. `backend/models/stance-deberta/`); the app fails loudly at startup if
  it isn't there rather than silently using zero-shot instead.

**Base model:** `microsoft/deberta-v3-base` (primary), with an automatic
fallback to `roberta-base` if the DeBERTa tokenizer fails to load (see
`notebooks/train_stance.ipynb`). 3-way head: SUPPORT / CONTRADICT / NEUTRAL —
`id2label` is set explicitly at training time to this exact vocabulary, which
`stance.py` already reads by name, so the checkpoint drops in with no code
changes.

**Training data:** [SciFact](https://huggingface.co/datasets/allenai/scifact)
(via HuggingFace, prepared locally by `notebooks/prepare_scifact.py` — see
that script for the exact claim/passage pairing logic, and for why
HealthVer/PUBHEALTH were investigated and explicitly declined as additional
CONTRADICT sources, rather than force-fit). SciFact's own official `test`
split carries no evidence annotations at all, so it isn't used; `train`/
`validation` come from SciFact's official splits. Class distribution
(train+validation combined): SUPPORT 47.8%, CONTRADICT 26.6%, NEUTRAL 25.5%
(1,739 triples).

**Training config** (`notebooks/train_stance.ipynb`, a free Colab T4 GPU):
fp16, batch size 16 (gradient accumulation 2, effective 32), 4 epochs, LR
2e-5, warmup ratio 0.1, sentence-pair tokenization (claim, passage) truncated
on the passage side at 256 tokens, class-weighted cross-entropy loss (SUPPORT
is ~1.8x either other class), best checkpoint selected on **CONTRADICT
recall** on the validation split (not overall accuracy — accuracy is
dominated by the majority classes and would hide the exact weakness this
fine-tune exists to fix), final evaluation on a held-out split carved from
`train.jsonl` (stratified, seed 42) since SciFact's own `test` split can't be
used for this.

**Headline metrics: not yet available.** `notebooks/train_stance.ipynb` has
not been executed against a live GPU — no fine-tuned checkpoint exists in
this repo. What *is* measured and real: the zero-shot baseline it's meant to
improve on scores **50.4% accuracy, 0.507 macro F1, 52.9% CONTRADICT recall,
29.0% SUPPORT recall** on the held-out SciFact split (see
`eval/results/classifier_comparison.md`, generated by
`backend/app/pipeline/compare_classifiers.py`). That script's fine-tuned
columns and recommendation section are fully built, not stubs — re-run it
once a checkpoint exists to fill in the real comparison.

### Obtaining or retraining the checkpoint

1. Run `notebooks/prepare_scifact.py` locally to produce
   `data/scifact/{train,validation,test}.jsonl`.
2. Upload `train.jsonl`/`validation.jsonl` to Google Drive, then run
   `notebooks/train_stance.ipynb` end to end on a free Colab T4 runtime.
3. Download the zipped model from the notebook's final cell (or grab it from
   Drive) and unzip it into `backend/models/stance-deberta/` — this directory
   is gitignored; model binaries don't belong in git.
4. Set `STANCE_MODEL_MODE=finetuned` in `backend/.env`.
5. Optionally run `python -m app.pipeline.compare_classifiers` for the real
   fine-tuned-vs-baseline numbers.

## Constraints

- Zero budget: no paid APIs (no Claude/OpenAI/Gemini calls). Everything runs
  on free, local, or free-tier resources.
- Model training happens in Google Colab (free T4 GPU); inference runs
  locally.
- Explanations are faithful — derived from rationale sentences and computed
  values, never free-text generated by a language model.
