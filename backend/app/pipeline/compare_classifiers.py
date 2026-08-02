"""Rigorous comparison: zero-shot NLI baseline vs. fine-tuned stance classifier.

This is a graded evaluation artifact, not a sanity check — every number here
is measured, not estimated, and nothing is fabricated for a model that hasn't
actually been run against the data.

STATUS AS OF WRITING: no fine-tuned checkpoint exists in this environment —
notebooks/train_stance.ipynb was authored without live GPU/Colab access, so
it has not actually been executed. This script is fully implemented and ready
for both models; run it again once a checkpoint exists at STANCE_MODEL_PATH
(or pass --fine-tuned-path) to get the real side-by-side comparison. Until
then it reports the baseline alone and says so plainly in the output — it
does not invent fine-tuned numbers to fill the table.

METHODOLOGY NOTES:
- The official SciFact `test` split has zero evidence annotations (verified
  while building prepare_scifact.py — it's the held-out shared-task
  leaderboard set). This script reproduces the EXACT held-out test split
  notebooks/train_stance.ipynb carves out of train.jsonl instead (same
  test_size=0.10, same random_state=42) — this is the only way baseline and
  fine-tuned numbers are comparable to what the training notebook itself
  reports for the fine-tuned side.
- "Mean inference time per passage" times the full StanceClassifier.classify()
  call, including its per-sentence rationale re-scoring — that's what actually
  happens per retrieved passage in a real /verify request (see
  pipeline.service), not just a bare forward pass, so it's the number that
  actually feeds response-time expectations.

Usage:
    python -m app.pipeline.compare_classifiers
    python -m app.pipeline.compare_classifiers --fine-tuned-path /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from app.config import settings
from app.ingestion.contested_topics import CONTESTED_CLAIMS
from app.pipeline.stance import StanceClassifier

LABELS = ["SUPPORT", "CONTRADICT", "NEUTRAL"]

SCIFACT_DIR = settings.data_dir / "scifact"
HELD_OUT_TEST_SIZE = 0.10
HELD_OUT_TEST_SEED = 42  # MUST match notebooks/train_stance.ipynb exactly

DEFAULT_CORPUS_SAMPLES_PER_TOPIC = 6  # ~5 contested topics * 6 = ~30, within the requested 20-30
DEFAULT_CORPUS_SAMPLE_SEED = 7

RESULTS_PATH = settings.data_dir.parent / "eval" / "results" / "classifier_comparison.md"


@dataclass
class EvalResult:
    name: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    mean_inference_ms: float
    n: int
    predictions: list[str] = field(default_factory=list)


def load_scifact_held_out_test() -> list[dict]:
    train_path = SCIFACT_DIR / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — run notebooks/prepare_scifact.py first.")
    with open(train_path, encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f if line.strip()]

    labels = [row["label"] for row in train_raw]
    _, held_out_test = train_test_split(
        train_raw, test_size=HELD_OUT_TEST_SIZE, random_state=HELD_OUT_TEST_SEED, stratify=labels
    )
    return held_out_test


def try_load_fine_tuned(path_str: str) -> StanceClassifier | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return StanceClassifier(model_path=str(path))
    except Exception as exc:
        print(f"WARNING: found {path} but failed to load it as a stance model: {exc}", file=sys.stderr)
        return None


def evaluate_model(classifier: StanceClassifier, name: str, triples: list[dict]) -> EvalResult:
    y_true: list[str] = []
    y_pred: list[str] = []
    timings_ms: list[float] = []

    for i, t in enumerate(triples):
        start = time.perf_counter()
        result = classifier.classify(t["passage"], t["claim"])
        timings_ms.append((time.perf_counter() - start) * 1000)
        y_true.append(t["label"])
        y_pred.append(result.stance.value)
        if (i + 1) % 50 == 0:
            print(f"  [{name}] {i + 1}/{len(triples)}")

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    mean_ms = sum(timings_ms) / len(timings_ms) if timings_ms else 0.0

    return EvalResult(
        name=name,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=cm,
        mean_inference_ms=mean_ms,
        n=len(triples),
        predictions=y_pred,
    )


def load_corpus_samples(n_per_topic: int, seed: int) -> list[dict]:
    """Real (claim, passage) pairs from indexed corpus passages belonging to
    papers tagged with a genuinely-contested topic (see contested_topics.py).
    Claim text is the original contested-claim wording, not fabricated."""
    claim_by_slug = {c.slug: c.claim for c in CONTESTED_CLAIMS}

    with open(settings.corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    paper_ids_by_slug: dict[str, set[str]] = {}
    for paper in corpus:
        slug = paper.get("contested_topic")
        if slug:
            paper_ids_by_slug.setdefault(slug, set()).add(paper["paper_id"])

    passages_path = settings.index_dir / "passages.json"
    if not passages_path.exists():
        raise FileNotFoundError(f"{passages_path} not found — run `python -m app.indexing.build_index` first.")
    with open(passages_path, encoding="utf-8") as f:
        passages = json.load(f)
    passages_by_paper: dict[str, list[dict]] = {}
    for p in passages:
        passages_by_paper.setdefault(p["paper_id"], []).append(p)

    rng = random.Random(seed)
    samples = []
    for slug, paper_ids in sorted(paper_ids_by_slug.items()):
        claim = claim_by_slug.get(slug)
        if claim is None:
            continue
        candidates = [p for pid in paper_ids for p in passages_by_paper.get(pid, [])]
        if not candidates:
            continue
        chosen = rng.sample(candidates, min(n_per_topic, len(candidates)))
        for p in chosen:
            samples.append(
                {
                    "slug": slug,
                    "claim": claim,
                    "passage": p["text"],
                    "paper_id": p["paper_id"],
                    "passage_id": p["passage_id"],
                }
            )
    return samples


def compare_on_corpus_sample(
    baseline: StanceClassifier, fine_tuned: StanceClassifier | None, samples: list[dict]
) -> list[dict]:
    rows = []
    for s in samples:
        b_result = baseline.classify(s["passage"], s["claim"])
        row = {
            **s,
            "baseline_stance": b_result.stance.value,
            "baseline_confidence": b_result.confidence,
        }
        if fine_tuned is not None:
            f_result = fine_tuned.classify(s["passage"], s["claim"])
            row["fine_tuned_stance"] = f_result.stance.value
            row["fine_tuned_confidence"] = f_result.confidence
            row["disagree"] = b_result.stance.value != f_result.stance.value
        else:
            row["fine_tuned_stance"] = None
            row["fine_tuned_confidence"] = None
            row["disagree"] = None
        rows.append(row)
    return rows


# --- markdown rendering ---


def render_confusion_matrix(cm: list[list[int]]) -> str:
    lines = ["| True \\ Predicted | " + " | ".join(LABELS) + " |"]
    lines.append("|" + "---|" * (len(LABELS) + 1))
    for i, label in enumerate(LABELS):
        lines.append(f"| **{label}** | " + " | ".join(str(v) for v in cm[i]) + " |")
    return "\n".join(lines)


def render_side_by_side_summary(baseline: EvalResult, fine_tuned: EvalResult | None) -> str:
    lines = ["| Metric | Baseline (bart-large-mnli) | Fine-tuned |", "|---|---|---|"]
    ft_acc = f"{fine_tuned.accuracy:.1%}" if fine_tuned else "*not yet run*"
    ft_f1 = f"{fine_tuned.macro_f1:.3f}" if fine_tuned else "*not yet run*"
    ft_ms = f"{fine_tuned.mean_inference_ms:.1f} ms" if fine_tuned else "*not yet run*"
    lines.append(f"| Overall accuracy | {baseline.accuracy:.1%} | {ft_acc} |")
    lines.append(f"| Macro F1 | {baseline.macro_f1:.3f} | {ft_f1} |")
    lines.append(f"| Mean inference time / passage | {baseline.mean_inference_ms:.1f} ms | {ft_ms} |")
    lines.append(f"| N (held-out test) | {baseline.n} | {fine_tuned.n if fine_tuned else baseline.n} |")
    return "\n".join(lines)


def render_per_class_table(baseline: EvalResult, fine_tuned: EvalResult | None) -> str:
    lines = [
        "| Class | Baseline P | Baseline R | Baseline F1 | Fine-tuned P | Fine-tuned R | Fine-tuned F1 | Support |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        b = baseline.per_class[label]
        if fine_tuned:
            f = fine_tuned.per_class[label]
            f_cells = f"{f['precision']:.3f} | {f['recall']:.3f} | {f['f1']:.3f}"
        else:
            f_cells = "*n/a* | *n/a* | *n/a*"
        lines.append(
            f"| {label} | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} | "
            f"{f_cells} | {b['support']} |"
        )
    return "\n".join(lines)


def render_contradict_callout(baseline: EvalResult, fine_tuned: EvalResult | None) -> str:
    b_recall = baseline.per_class["CONTRADICT"]["recall"]
    support = baseline.per_class["CONTRADICT"]["support"]
    if fine_tuned is None:
        return (
            f"**Baseline CONTRADICT recall: {b_recall:.1%}** (n={support} CONTRADICT examples "
            f"in the held-out test set). This is the documented weakness motivating fine-tuning. "
            f"No fine-tuned-model comparison is available yet — see the status note at the top."
        )
    f_recall = fine_tuned.per_class["CONTRADICT"]["recall"]
    delta = f_recall - b_recall
    verdict = "IMPROVED" if delta > 1e-9 else ("WORSE" if delta < -1e-9 else "UNCHANGED")
    return (
        f"**CONTRADICT recall: baseline {b_recall:.1%} -> fine-tuned {f_recall:.1%} "
        f"({delta:+.1%}, {verdict})** (n={support} CONTRADICT examples in the held-out test set)."
    )


def render_corpus_disagreements(rows: list[dict], fine_tuned_available: bool) -> str:
    lines = []
    if not fine_tuned_available:
        lines.append(
            "*No fine-tuned model available — showing the baseline's predictions on these "
            "corpus samples only. Re-run once a fine-tuned checkpoint exists to see where "
            "the two models disagree.*\n"
        )
        for r in rows:
            lines.append(f"- **[{r['slug']}]** claim: *{r['claim']}*")
            lines.append(f"  - baseline: **{r['baseline_stance']}** (conf={r['baseline_confidence']:.2f})")
            lines.append(f"  - passage ({r['passage_id']}): {r['passage'][:220]}")
            lines.append("")
        return "\n".join(lines)

    disagreements = [r for r in rows if r["disagree"]]
    lines.append(
        f"{len(disagreements)} of {len(rows)} sampled corpus pairs — baseline and fine-tuned "
        f"disagree. Neither prediction is treated as ground truth below; review manually.\n"
    )
    for r in disagreements:
        lines.append(f"- **[{r['slug']}]** claim: *{r['claim']}*")
        lines.append(
            f"  - baseline: **{r['baseline_stance']}** (conf={r['baseline_confidence']:.2f}) "
            f"vs. fine-tuned: **{r['fine_tuned_stance']}** (conf={r['fine_tuned_confidence']:.2f})"
        )
        lines.append(f"  - passage ({r['passage_id']}): {r['passage'][:220]}")
        lines.append("")
    if not disagreements:
        lines.append("*(no disagreements found in this sample)*")
    return "\n".join(lines)


def render_recommendation(baseline: EvalResult, fine_tuned: EvalResult | None) -> str:
    if fine_tuned is None:
        return (
            "**Pending — no real recommendation yet.** No fine-tuned checkpoint exists in this "
            "environment (see status note at the top). Shipping the baseline right now isn't a "
            "recommendation, it's the only option available. Once `notebooks/train_stance.ipynb` "
            "has actually been run on a GPU and its checkpoint is placed at `STANCE_MODEL_PATH` "
            "(or passed via `--fine-tuned-path`), re-run this script. The real recommendation "
            "should weigh, in this order: (1) CONTRADICT recall — the entire reason for "
            "fine-tuning, so a fine-tuned model that doesn't move this number isn't worth "
            "shipping regardless of other gains; (2) whether SUPPORT/NEUTRAL performance "
            "regresses — a model that fixes CONTRADICT by over-predicting it everywhere is worse, "
            "not better; (3) macro F1 as a single-number sanity check, not the deciding factor; "
            "(4) mean inference time — a real cost for a live `/verify` endpoint, not just an "
            "academic concern, especially if the fine-tuned model is larger than bart-large-mnli."
        )

    b_contradict_recall = baseline.per_class["CONTRADICT"]["recall"]
    f_contradict_recall = fine_tuned.per_class["CONTRADICT"]["recall"]
    contradict_improved = f_contradict_recall > b_contradict_recall

    regressions = []
    for label in ("SUPPORT", "NEUTRAL"):
        b_f1 = baseline.per_class[label]["f1"]
        f_f1 = fine_tuned.per_class[label]["f1"]
        if f_f1 < b_f1 - 0.05:  # more than a 5-point F1 drop
            regressions.append(f"{label} F1 dropped {b_f1:.3f} -> {f_f1:.3f}")

    slower = fine_tuned.mean_inference_ms > baseline.mean_inference_ms * 1.5

    if contradict_improved and not regressions:
        verdict = "**Ship the fine-tuned model.**"
    elif contradict_improved and regressions:
        verdict = (
            "**Tradeoff, not a clean win — decide based on what matters more for your use case.** "
            "The fine-tuned model improves the one thing it was built to fix, at a real cost "
            "elsewhere."
        )
    else:
        verdict = (
            "**Do not ship the fine-tuned model as-is.** It does not improve CONTRADICT recall, "
            "which was the entire point of fine-tuning."
        )

    lines = [verdict, ""]
    lines.append(f"- CONTRADICT recall: {'improved' if contradict_improved else 'did not improve'} "
                 f"({b_contradict_recall:.1%} -> {f_contradict_recall:.1%})")
    if regressions:
        lines.append("- Regressions found: " + "; ".join(regressions))
    else:
        lines.append("- No material SUPPORT/NEUTRAL F1 regression (>5 points) found.")
    if slower:
        lines.append(
            f"- Inference is notably slower: {baseline.mean_inference_ms:.1f}ms -> "
            f"{fine_tuned.mean_inference_ms:.1f}ms per passage — factor this into whether the "
            f"CONTRADICT gain is worth the added /verify latency."
        )
    return "\n".join(lines)


def write_report(
    baseline: EvalResult,
    fine_tuned: EvalResult | None,
    corpus_rows: list[dict],
) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = ["# Stance classifier comparison: zero-shot baseline vs. fine-tuned\n"]

    if fine_tuned is None:
        parts.append(
            "> **Status: baseline-only run.** No fine-tuned checkpoint was found at "
            f"`{settings.STANCE_MODEL_PATH}` (or the path passed via `--fine-tuned-path`) when "
            "this report was generated — `notebooks/train_stance.ipynb` has not yet been "
            "executed against a live GPU. Every number below for the baseline is real and "
            "measured; the fine-tuned columns are explicitly marked *not yet run*, not "
            "estimated or filled in. Re-run this script once a checkpoint exists for the real "
            "comparison.\n"
        )

    parts.append("## SciFact held-out test set\n")
    parts.append(
        f"Reproduces the exact held-out split from `notebooks/train_stance.ipynb` "
        f"(`test_size={HELD_OUT_TEST_SIZE}`, `random_state={HELD_OUT_TEST_SEED}`, stratified, "
        f"carved out of `data/scifact/train.jsonl` — SciFact's official `test` split has zero "
        f"usable annotations, see prepare_scifact.py). N={baseline.n}.\n"
    )
    parts.append(render_side_by_side_summary(baseline, fine_tuned) + "\n")

    parts.append("### Per-class precision / recall / F1\n")
    parts.append(render_per_class_table(baseline, fine_tuned) + "\n")

    parts.append("### Headline number: CONTRADICT recall\n")
    parts.append(render_contradict_callout(baseline, fine_tuned) + "\n")

    parts.append("### Confusion matrix — baseline\n")
    parts.append(render_confusion_matrix(baseline.confusion) + "\n")
    if fine_tuned is not None:
        parts.append("### Confusion matrix — fine-tuned\n")
        parts.append(render_confusion_matrix(fine_tuned.confusion) + "\n")

    parts.append("## Corpus contested-topic sample\n")
    parts.append(
        f"{len(corpus_rows)} real (claim, passage) pairs sampled from indexed passages "
        f"belonging to papers tagged with a genuinely-contested topic (see "
        f"`app/ingestion/contested_topics.py`). Predictions are NOT auto-labeled as ground "
        f"truth — for manual review, and may feed a future eval set.\n"
    )
    parts.append(render_corpus_disagreements(corpus_rows, fine_tuned is not None) + "\n")

    parts.append("## Recommendation\n")
    parts.append(render_recommendation(baseline, fine_tuned) + "\n")

    RESULTS_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nWrote report to {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare zero-shot vs. fine-tuned stance classifiers.")
    parser.add_argument("--fine-tuned-path", default=settings.STANCE_MODEL_PATH)
    parser.add_argument("--corpus-samples-per-topic", type=int, default=DEFAULT_CORPUS_SAMPLES_PER_TOPIC)
    parser.add_argument("--corpus-sample-seed", type=int, default=DEFAULT_CORPUS_SAMPLE_SEED)
    args = parser.parse_args()

    print("Loading SciFact held-out test set (reproducing train_stance.ipynb's split)...")
    held_out_test = load_scifact_held_out_test()
    print(f"  {len(held_out_test)} triples")

    print("\nLoading baseline (zero-shot NLI)...")
    baseline = StanceClassifier(model_path=settings.NLI_BASELINE_MODEL)

    print("\nEvaluating baseline on SciFact held-out test...")
    baseline_result = evaluate_model(baseline, "baseline", held_out_test)

    fine_tuned = try_load_fine_tuned(args.fine_tuned_path)
    fine_tuned_result = None
    if fine_tuned is None:
        print(
            f"\nNo fine-tuned model found at {args.fine_tuned_path} — proceeding "
            "baseline-only (see report's status note)."
        )
    else:
        print("\nEvaluating fine-tuned model on SciFact held-out test...")
        fine_tuned_result = evaluate_model(fine_tuned, "fine-tuned", held_out_test)

    print("\nSampling corpus contested-topic pairs...")
    corpus_samples = load_corpus_samples(args.corpus_samples_per_topic, args.corpus_sample_seed)
    print(f"  {len(corpus_samples)} pairs across {len({s['slug'] for s in corpus_samples})} topics")

    print("\nRunning corpus sample through model(s)...")
    corpus_rows = compare_on_corpus_sample(baseline, fine_tuned, corpus_samples)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(render_side_by_side_summary(baseline_result, fine_tuned_result))
    print()
    print(render_contradict_callout(baseline_result, fine_tuned_result))
    print("=" * 70)

    write_report(baseline_result, fine_tuned_result, corpus_rows)


if __name__ == "__main__":
    main()
