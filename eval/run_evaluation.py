"""Evaluate the verification pipeline against a labelled claim set.

Run from the repository root:

    python eval/run_evaluation.py
    python eval/run_evaluation.py --sample 5
    python eval/run_evaluation.py --claims eval/claims.csv --out eval/results/evaluation_report.md

Labelling is incremental by design. Rows in `claims.csv` with an empty
`expected_verdict` are skipped and counted, so the harness is runnable from the
first labelled claim onward rather than only once all 50 exist.

Three kinds of ground truth are involved and they live at different
granularities, which is why they live in different files:

  * `eval/claims.csv` — one row per claim. Verdict-level truth.
  * `eval/relevance.csv` — one row per (claim, paper). Which papers should have
    been retrieved. Optional.
  * `eval/stance_labels.csv` — one row per (claim, passage). What each retrieved
    passage actually says. Optional.

Retrieval and stance metrics are computed only over the claims that carry the
corresponding judgments, and the report states that coverage next to every such
number. A metric computed from nothing is reported as absent, never as zero:
"precision@10 = 0.00" and "precision@10 was never measured" are different
findings and the report must not conflate them.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app._thread_limits  # noqa: F401,E402  (must precede the faiss/torch imports)

from app.models import (  # noqa: E402
    InsufficientReason,
    Stance,
    Verdict,
    VerdictResponse,
)
from app.pipeline.service import PipelineService  # noqa: E402

REQUIRED_COLUMNS = [
    "claim_id",
    "claim_text",
    "domain_topic",
    "expected_verdict",
    "expected_grade",
    "contested",
    "off_corpus",
    "notes",
    "labeller",
    "labelled_date",
]

VERDICT_ORDER = [
    Verdict.SUPPORTED,
    Verdict.CONTRADICTED,
    Verdict.CONFLICTING,
    Verdict.INSUFFICIENT_EVIDENCE,
]

STANCE_ORDER = [Stance.SUPPORT, Stance.CONTRADICT, Stance.NEUTRAL]

# Anything that is not INSUFFICIENT_EVIDENCE is an answer the reader will act
# on, so all three count as confident for the purposes of the safety metric.
CONFIDENT_VERDICTS = {Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.CONFLICTING}

DEFAULT_K = 10


# --- Inputs -----------------------------------------------------------------


@dataclass
class EvalClaim:
    claim_id: str
    claim_text: str
    domain_topic: str
    expected_verdict: Verdict
    expected_grade: str | None
    contested: bool
    off_corpus: bool
    notes: str
    labeller: str
    labelled_date: str

    @property
    def expected_reason(self) -> InsufficientReason | None:
        """Which flavour of INSUFFICIENT_EVIDENCE this claim should produce.

        Derived from `off_corpus` rather than labelled separately: the two are
        the same judgement asked twice. A claim the corpus does not cover must
        come back NOT_COVERED_BY_CORPUS; one it covers without a clear
        direction must come back EVIDENCE_INCONCLUSIVE.
        """
        if self.expected_verdict != Verdict.INSUFFICIENT_EVIDENCE:
            return None
        return (
            InsufficientReason.NOT_COVERED_BY_CORPUS
            if self.off_corpus
            else InsufficientReason.EVIDENCE_INCONCLUSIVE
        )


def _parse_bool(value: str, field_name: str, claim_id: str) -> bool:
    normalised = (value or "").strip().lower()
    if normalised in {"true", "yes", "y", "1"}:
        return True
    if normalised in {"false", "no", "n", "0", ""}:
        return False
    raise ValueError(
        f"Claim '{claim_id}': {field_name} is '{value}', which is neither true "
        f"nor false. Use true/false."
    )


@dataclass
class LoadedClaims:
    labelled: list[EvalClaim]
    unlabelled_ids: list[str]


def load_claims(path: Path) -> LoadedClaims:
    if not path.exists():
        raise FileNotFoundError(f"No claim set at {path}.")

    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}. "
                f"Expected header: {','.join(REQUIRED_COLUMNS)}"
            )

        labelled: list[EvalClaim] = []
        unlabelled: list[str] = []
        seen_ids: set[str] = set()

        for row in reader:
            claim_id = (row.get("claim_id") or "").strip()
            claim_text = (row.get("claim_text") or "").strip()
            if not claim_id and not claim_text:
                continue  # blank spacer row
            if not claim_id:
                raise ValueError(f"A row has claim_text but no claim_id: {claim_text!r}")
            if claim_id in seen_ids:
                raise ValueError(f"Duplicate claim_id '{claim_id}'.")
            seen_ids.add(claim_id)

            raw_verdict = (row.get("expected_verdict") or "").strip().upper()
            if not raw_verdict:
                unlabelled.append(claim_id)
                continue
            if raw_verdict not in Verdict.__members__:
                raise ValueError(
                    f"Claim '{claim_id}': expected_verdict '{raw_verdict}' is not "
                    f"one of {', '.join(Verdict.__members__)}."
                )
            if not claim_text:
                raise ValueError(f"Claim '{claim_id}' is labelled but has no claim_text.")

            grade = (row.get("expected_grade") or "").strip().upper() or None
            labelled.append(
                EvalClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    domain_topic=(row.get("domain_topic") or "").strip(),
                    expected_verdict=Verdict[raw_verdict],
                    expected_grade=grade,
                    contested=_parse_bool(row.get("contested", ""), "contested", claim_id),
                    off_corpus=_parse_bool(row.get("off_corpus", ""), "off_corpus", claim_id),
                    notes=(row.get("notes") or "").strip(),
                    labeller=(row.get("labeller") or "").strip(),
                    labelled_date=(row.get("labelled_date") or "").strip(),
                )
            )

    return LoadedClaims(labelled=labelled, unlabelled_ids=unlabelled)


def load_relevance(path: Path) -> dict[str, set[str]]:
    """claim_id -> set of paper_ids that should have been retrieved."""
    if not path.exists():
        return {}
    judgments: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            claim_id = (row.get("claim_id") or "").strip()
            paper_id = (row.get("paper_id") or "").strip()
            if not claim_id or not paper_id:
                continue
            if _parse_bool(row.get("relevant", "true"), "relevant", claim_id):
                judgments[claim_id].add(paper_id)
    return dict(judgments)


def load_stance_labels(path: Path) -> dict[tuple[str, str], Stance]:
    """(claim_id, passage_id) -> the stance a human read in that passage."""
    if not path.exists():
        return {}
    labels: dict[tuple[str, str], Stance] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            claim_id = (row.get("claim_id") or "").strip()
            passage_id = (row.get("passage_id") or "").strip()
            gold = (row.get("gold_stance") or "").strip().upper()
            if not claim_id or not passage_id or not gold:
                continue
            if gold not in Stance.__members__:
                raise ValueError(
                    f"Stance label for {claim_id}/{passage_id} is '{gold}', not one "
                    f"of {', '.join(Stance.__members__)}."
                )
            labels[(claim_id, passage_id)] = Stance[gold]
    return labels


# --- Running ----------------------------------------------------------------


@dataclass
class EvalResult:
    claim: EvalClaim
    response: VerdictResponse

    @property
    def predicted(self) -> Verdict:
        return self.response.verdict

    @property
    def correct(self) -> bool:
        return self.predicted == self.claim.expected_verdict

    @property
    def retrieved_papers(self) -> list[str]:
        """Retrieved paper ids in rank order, deduplicated, first mention wins."""
        ordered: list[str] = []
        seen: set[str] = set()
        for item in sorted(
            self.response.evidence, key=lambda e: e.relevance_score, reverse=True
        ):
            if item.paper.paper_id not in seen:
                seen.add(item.paper.paper_id)
                ordered.append(item.paper.paper_id)
        return ordered


def run_claims(claims: list[EvalClaim]) -> list[EvalResult]:
    print("Loading retrieval index and stance classifier...", file=sys.stderr)
    service = PipelineService()
    print(f"Ready. Running {len(claims)} claim(s).\n", file=sys.stderr)

    results: list[EvalResult] = []
    for index, claim in enumerate(claims, start=1):
        response = service.verify(claim.claim_text)
        results.append(EvalResult(claim=claim, response=response))
        flag = "ok " if response.verdict == claim.expected_verdict else "MISS"
        print(
            f"  [{index:>3}/{len(claims)}] {flag} {claim.claim_id:<10} "
            f"expected={claim.expected_verdict.value:<22} "
            f"got={response.verdict.value:<22} ({response.response_time_ms:.0f} ms)",
            file=sys.stderr,
        )
    return results


# --- Metrics ----------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    k: int
    judged_claim_ids: list[str] = field(default_factory=list)
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    per_claim: list[dict] = field(default_factory=list)
    off_corpus_total: int = 0
    off_corpus_suppressed: int = 0


def compute_retrieval(
    results: list[EvalResult], relevance: dict[str, set[str]], k: int
) -> RetrievalMetrics:
    """Precision@k, recall@k and MRR over claims with relevance judgments.

    Off-corpus claims are deliberately excluded from these averages. Their
    relevant set is empty by definition, which makes precision trivially 0 and
    recall undefined; averaging them in would drag the numbers around without
    saying anything about ranking quality. What matters for those claims is
    whether retrieval was suppressed at all, so that is counted separately.
    """
    metrics = RetrievalMetrics(k=k)

    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for result in results:
        if result.claim.off_corpus:
            metrics.off_corpus_total += 1
            if not result.response.evidence:
                metrics.off_corpus_suppressed += 1
            continue

        relevant = relevance.get(result.claim.claim_id)
        if not relevant:
            continue

        retrieved = result.retrieved_papers
        top_k = retrieved[:k]
        hits = [paper_id for paper_id in top_k if paper_id in relevant]

        precision = len(hits) / k
        recall = len(hits) / len(relevant)
        first_rank = next(
            (i for i, paper_id in enumerate(retrieved, start=1) if paper_id in relevant),
            None,
        )
        reciprocal = 1.0 / first_rank if first_rank else 0.0

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal)
        metrics.judged_claim_ids.append(result.claim.claim_id)
        metrics.per_claim.append(
            {
                "claim_id": result.claim.claim_id,
                "relevant": len(relevant),
                "retrieved": len(retrieved),
                "hits": len(hits),
                "precision": precision,
                "recall": recall,
                "reciprocal_rank": reciprocal,
                "first_rank": first_rank,
            }
        )

    if precisions:
        metrics.precision_at_k = statistics.mean(precisions)
        metrics.recall_at_k = statistics.mean(recalls)
        metrics.mrr = statistics.mean(reciprocal_ranks)
    return metrics


@dataclass
class StanceMetrics:
    labelled_count: int = 0
    matched_count: int = 0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    accuracy: float | None = None
    macro_f1: float | None = None
    unmatched_labels: list[tuple[str, str]] = field(default_factory=list)


def compute_stance(
    results: list[EvalResult], labels: dict[tuple[str, str], Stance]
) -> StanceMetrics:
    metrics = StanceMetrics(labelled_count=len(labels))
    if not labels:
        return metrics

    predicted_by_key: dict[tuple[str, str], Stance] = {}
    for result in results:
        for item in result.response.evidence:
            predicted_by_key[(result.claim.claim_id, item.passage.passage_id)] = item.stance

    gold: list[str] = []
    pred: list[str] = []
    for key, gold_stance in labels.items():
        predicted = predicted_by_key.get(key)
        if predicted is None:
            # Labelled a passage the pipeline did not return for this claim.
            # Counting it as an error would conflate a retrieval miss with a
            # classification miss, so it is reported separately instead.
            metrics.unmatched_labels.append(key)
            continue
        gold.append(gold_stance.value)
        pred.append(predicted.value)

    metrics.matched_count = len(gold)
    if not gold:
        return metrics

    from sklearn.metrics import classification_report, confusion_matrix

    label_values = [s.value for s in STANCE_ORDER]
    report = classification_report(
        gold, pred, labels=label_values, output_dict=True, zero_division=0
    )
    for label in label_values:
        entry = report.get(label, {})
        metrics.per_class[label] = {
            "precision": entry.get("precision", 0.0),
            "recall": entry.get("recall", 0.0),
            "f1": entry.get("f1-score", 0.0),
            "support": entry.get("support", 0),
        }
    metrics.accuracy = report.get("accuracy")
    metrics.macro_f1 = report.get("macro avg", {}).get("f1-score")

    matrix = confusion_matrix(gold, pred, labels=label_values)
    for i, true_label in enumerate(label_values):
        for j, pred_label in enumerate(label_values):
            metrics.confusion[(true_label, pred_label)] = int(matrix[i][j])
    return metrics


@dataclass
class HighConfidenceError:
    """A confident verdict that was wrong.

    Severity is ordered by how the failure would mislead somebody reading the
    result, not by how far apart the labels are. Asserting a direction on a
    claim the corpus cannot speak to, and inverting the direction of one it
    can, are the two ways this system does real damage.
    """

    claim: EvalClaim
    response: VerdictResponse
    severity: str
    rank: int
    description: str


_SEVERITY_RANK = {
    "POLARITY_INVERSION": 0,
    "CONFIDENT_ON_OFF_CORPUS": 1,
    "CONFIDENT_ON_INCONCLUSIVE": 2,
    "CONTESTED_RESOLVED": 3,
    "OTHER_CONFIDENT_ERROR": 4,
}

_GRADE_WEIGHT = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "VERY_LOW": 3}


def find_high_confidence_errors(results: list[EvalResult]) -> list[HighConfidenceError]:
    errors: list[HighConfidenceError] = []
    opposites = {Verdict.SUPPORTED: Verdict.CONTRADICTED, Verdict.CONTRADICTED: Verdict.SUPPORTED}

    for result in results:
        predicted = result.predicted
        expected = result.claim.expected_verdict
        if predicted not in CONFIDENT_VERDICTS or predicted == expected:
            continue

        if opposites.get(expected) == predicted:
            severity = "POLARITY_INVERSION"
            description = (
                f"Reported {predicted.value} for a claim labelled {expected.value}. "
                f"The system stated the opposite of the labelled reading of the "
                f"evidence."
            )
        elif expected == Verdict.INSUFFICIENT_EVIDENCE and result.claim.off_corpus:
            severity = "CONFIDENT_ON_OFF_CORPUS"
            description = (
                f"Reported {predicted.value} for a claim the corpus does not cover. "
                f"The coverage gate should have returned NOT_COVERED_BY_CORPUS and "
                f"withheld all evidence."
            )
        elif expected == Verdict.INSUFFICIENT_EVIDENCE:
            severity = "CONFIDENT_ON_INCONCLUSIVE"
            description = (
                f"Reported {predicted.value} where the labelled reading is that the "
                f"retrieved evidence has no clear direction."
            )
        elif expected == Verdict.CONFLICTING:
            severity = "CONTESTED_RESOLVED"
            description = (
                f"Reported {predicted.value} on a claim the field genuinely "
                f"disagrees about, presenting a contested question as settled."
            )
        else:
            severity = "OTHER_CONFIDENT_ERROR"
            description = (
                f"Reported {predicted.value} where {expected.value} was expected."
            )

        errors.append(
            HighConfidenceError(
                claim=result.claim,
                response=result.response,
                severity=severity,
                rank=_SEVERITY_RANK[severity],
                description=description,
            )
        )

    errors.sort(
        key=lambda e: (e.rank, _GRADE_WEIGHT.get(e.response.grade_certainty.value, 9))
    )
    return errors


# --- Report -----------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%}"


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "not measured" if value is None else format(value, spec)


def write_confusion_png(results: list[EvalResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [v.value for v in VERDICT_ORDER]
    short = [label.replace("INSUFFICIENT_EVIDENCE", "INSUFFICIENT") for label in labels]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    index = {label: i for i, label in enumerate(labels)}
    for result in results:
        matrix[index[result.claim.expected_verdict.value]][index[result.predicted.value]] += 1

    figure, axes = plt.subplots(figsize=(6.4, 5.6))
    axes.imshow(matrix, cmap="Blues")
    axes.set_xticks(range(len(labels)), short, rotation=35, ha="right", fontsize=9)
    axes.set_yticks(range(len(labels)), short, fontsize=9)
    axes.set_xlabel("Predicted verdict")
    axes.set_ylabel("Expected verdict")
    axes.set_title(f"Verdict confusion matrix (n={len(results)})")

    limit = matrix.max() if matrix.max() else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes.text(
                j,
                i,
                str(matrix[i][j]),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if matrix[i][j] > limit * 0.6 else "black",
            )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_stance_confusion_png(stance: StanceMetrics, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [s.value for s in STANCE_ORDER]
    matrix = np.array(
        [[stance.confusion.get((t, p), 0) for p in labels] for t in labels], dtype=int
    )

    figure, axes = plt.subplots(figsize=(5.6, 5.0))
    axes.imshow(matrix, cmap="Purples")
    axes.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=9)
    axes.set_yticks(range(len(labels)), labels, fontsize=9)
    axes.set_xlabel("Predicted stance")
    axes.set_ylabel("Gold stance")
    axes.set_title(f"Stance confusion matrix (n={stance.matched_count})")

    limit = matrix.max() if matrix.max() else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes.text(
                j,
                i,
                str(matrix[i][j]),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if matrix[i][j] > limit * 0.6 else "black",
            )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_report(
    results: list[EvalResult],
    loaded: LoadedClaims,
    retrieval: RetrievalMetrics,
    stance: StanceMetrics,
    errors: list[HighConfidenceError],
    sample_size: int | None,
    confusion_png: Path,
    stance_png: Path | None,
    report_path: Path,
) -> str:
    lines: list[str] = []
    add = lines.append
    total = len(results)
    correct = sum(1 for r in results if r.correct)

    add("# Evaluation report")
    add("")
    add(
        f"Generated {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')} by "
        f"`eval/run_evaluation.py`. Regenerate with `python eval/run_evaluation.py`."
    )
    add("")

    add("## Claim set")
    add("")
    add("| | Count |")
    add("|---|---|")
    add(f"| Labelled claims evaluated | {total} |")
    add(f"| Unlabelled rows skipped | {len(loaded.unlabelled_ids)} |")
    add(f"| Contested claims | {sum(1 for r in results if r.claim.contested)} |")
    add(f"| Off-corpus claims | {sum(1 for r in results if r.claim.off_corpus)} |")
    add("")
    if sample_size is not None:
        add(
            f"> **Partial run.** `--sample {sample_size}` was used, so this covers "
            f"the first {total} labelled claim(s) in file order, not the whole set."
        )
        add("")
    if loaded.unlabelled_ids:
        preview = ", ".join(loaded.unlabelled_ids[:10])
        suffix = ", …" if len(loaded.unlabelled_ids) > 10 else ""
        add(
            f"> {len(loaded.unlabelled_ids)} row(s) carry no `expected_verdict` and "
            f"were skipped: {preview}{suffix}"
        )
        add("")

    # --- Verdict accuracy ---
    add("## Verdict accuracy")
    add("")
    add(f"**Overall: {_pct(correct, total)}** ({correct}/{total})")
    add("")
    add("| Expected verdict | n | Correct | Accuracy |")
    add("|---|---|---|---|")
    for verdict in VERDICT_ORDER:
        subset = [r for r in results if r.claim.expected_verdict == verdict]
        if not subset:
            continue
        subset_correct = sum(1 for r in subset if r.correct)
        add(
            f"| {verdict.value} | {len(subset)} | {subset_correct} | "
            f"{_pct(subset_correct, len(subset))} |"
        )
    add("")

    contested = [r for r in results if r.claim.contested]
    if contested:
        contested_correct = sum(1 for r in contested if r.correct)
        add(
            f"Contested claims specifically: {_pct(contested_correct, len(contested))} "
            f"({contested_correct}/{len(contested)})."
        )
        add("")

    add("### Confusion matrix")
    add("")
    add(f"![Verdict confusion matrix]({confusion_png.name})")
    add("")
    header = " | ".join(v.value.replace("INSUFFICIENT_EVIDENCE", "INSUFF.") for v in VERDICT_ORDER)
    add(f"| Expected \\ Predicted | {header} |")
    add("|---" * (len(VERDICT_ORDER) + 1) + "|")
    for expected in VERDICT_ORDER:
        row = [f"**{expected.value.replace('INSUFFICIENT_EVIDENCE', 'INSUFF.')}**"]
        for predicted in VERDICT_ORDER:
            row.append(
                str(
                    sum(
                        1
                        for r in results
                        if r.claim.expected_verdict == expected and r.predicted == predicted
                    )
                )
            )
        add("| " + " | ".join(row) + " |")
    add("")

    # --- Grade agreement ---
    graded = [r for r in results if r.claim.expected_grade]
    if graded:
        grade_correct = sum(
            1 for r in graded if r.response.grade_certainty.value == r.claim.expected_grade
        )
        add("### GRADE certainty agreement")
        add("")
        add(
            f"{_pct(grade_correct, len(graded))} ({grade_correct}/{len(graded)}) exact "
            f"agreement over the {len(graded)} claim(s) carrying an expected grade."
        )
        add("")

    # --- Insufficient evidence ---
    add("## Insufficient-evidence detection")
    add("")
    add(
        "The two reasons are measured separately because they are different "
        "failures. `NOT_COVERED_BY_CORPUS` says the corpus cannot speak to the "
        "claim at all; `EVIDENCE_INCONCLUSIVE` says it can, but what was "
        "retrieved took no clear direction. Collapsing them hides whether the "
        "coverage gate is doing its job."
    )
    add("")

    expected_insufficient = [
        r for r in results if r.claim.expected_verdict == Verdict.INSUFFICIENT_EVIDENCE
    ]
    add("| Expected reason | n | Verdict correct | Reason also correct |")
    add("|---|---|---|---|")
    for reason in (
        InsufficientReason.NOT_COVERED_BY_CORPUS,
        InsufficientReason.EVIDENCE_INCONCLUSIVE,
    ):
        subset = [r for r in expected_insufficient if r.claim.expected_reason == reason]
        if not subset:
            continue
        verdict_hits = sum(1 for r in subset if r.predicted == Verdict.INSUFFICIENT_EVIDENCE)
        reason_hits = sum(
            1
            for r in subset
            if r.predicted == Verdict.INSUFFICIENT_EVIDENCE
            and r.response.insufficient_reason == reason
        )
        add(
            f"| {reason.value} | {len(subset)} | {verdict_hits} "
            f"({_pct(verdict_hits, len(subset))}) | {reason_hits} "
            f"({_pct(reason_hits, len(subset))}) |"
        )
    add("")

    false_insufficient = [
        r
        for r in results
        if r.claim.expected_verdict != Verdict.INSUFFICIENT_EVIDENCE
        and r.predicted == Verdict.INSUFFICIENT_EVIDENCE
    ]
    add(
        f"**False insufficient:** {len(false_insufficient)} claim(s) the labels say "
        f"the corpus can answer came back as insufficient. This is the cost side of "
        f"the coverage gate — the error it trades against wrong confident verdicts."
    )
    if false_insufficient:
        add("")
        for result in false_insufficient:
            reason = result.response.insufficient_reason
            add(
                f"- `{result.claim.claim_id}` ({result.claim.expected_verdict.value} "
                f"expected) — reason `{reason.value if reason else 'none'}`, "
                f"{len(result.response.evidence)} passage(s) retrieved: "
                f"*{result.claim.claim_text}*"
            )
    add("")

    # --- High-confidence errors ---
    add("## High-confidence errors")
    add("")
    add(
        "Every case where the system issued a confident verdict "
        "(SUPPORTED, CONTRADICTED or CONFLICTING) that did not match the label. "
        "These are listed individually and never summarised into a rate: an "
        "aggregate cannot tell you whether the system confidently inverted a "
        "finding or merely called a contested question early, and those two "
        "failures do not carry the same cost."
    )
    add("")
    add(f"**Count: {len(errors)}** of {total} evaluated claim(s).")
    add("")

    if not errors:
        add("No confident verdict in this run contradicted its label.")
        add("")
    else:
        counts = Counter(error.severity for error in errors)
        add("| Severity | Count |")
        add("|---|---|")
        for severity, _ in sorted(_SEVERITY_RANK.items(), key=lambda kv: kv[1]):
            if counts.get(severity):
                add(f"| {severity} | {counts[severity]} |")
        add("")

        for position, error in enumerate(errors, start=1):
            response = error.response
            add(f"### {position}. `{error.claim.claim_id}` — {error.severity}")
            add("")
            add(f"> {error.claim.claim_text}")
            add("")
            add(f"| | |")
            add("|---|---|")
            add(f"| Expected | {error.claim.expected_verdict.value} |")
            add(f"| Predicted | {response.verdict.value} |")
            add(f"| Certainty reported | {response.grade_certainty.value} |")
            add(
                f"| Counts | {response.support_count} support / "
                f"{response.contradict_count} contradict / "
                f"{response.neutral_count} neutral |"
            )
            add(f"| Domain topic | {error.claim.domain_topic or '—'} |")
            add(f"| Contested | {'yes' if error.claim.contested else 'no'} |")
            add(f"| Off-corpus | {'yes' if error.claim.off_corpus else 'no'} |")
            if response.coverage:
                add(
                    f"| Coverage | max_cosine {response.coverage.max_cosine:.3f}, "
                    f"mean_topk {response.coverage.mean_topk_cosine:.3f} |"
                )
            add("")
            add(error.description)
            add("")
            if error.claim.notes:
                add(f"Labeller's note: {error.claim.notes}")
                add("")
            top = sorted(
                response.evidence, key=lambda e: e.relevance_score, reverse=True
            )[:3]
            if top:
                add("Passages that drove it:")
                add("")
                for item in top:
                    rationale = " ".join(item.rationale_sentences) or item.passage.text
                    add(
                        f"- **{item.stance.value}** ({item.relevance_score:.2f}) "
                        f"*{item.paper.title}* ({item.paper.year}) — "
                        f"{rationale[:240]}{'…' if len(rationale) > 240 else ''}"
                    )
                add("")

    # --- Retrieval ---
    add("## Retrieval")
    add("")
    judged = len(retrieval.judged_claim_ids)
    if judged:
        add(f"| Metric | Value |")
        add("|---|---|")
        add(f"| Precision@{retrieval.k} | {_fmt(retrieval.precision_at_k)} |")
        add(f"| Recall@{retrieval.k} | {_fmt(retrieval.recall_at_k)} |")
        add(f"| MRR | {_fmt(retrieval.mrr)} |")
        add(f"| Claims with relevance judgments | {judged} |")
        add("")
        add("| Claim | Relevant | Retrieved | Hits | P@k | R@k | First relevant rank |")
        add("|---|---|---|---|---|---|---|")
        for entry in retrieval.per_claim:
            add(
                f"| `{entry['claim_id']}` | {entry['relevant']} | {entry['retrieved']} | "
                f"{entry['hits']} | {entry['precision']:.2f} | {entry['recall']:.2f} | "
                f"{entry['first_rank'] if entry['first_rank'] else '—'} |"
            )
        add("")
    else:
        add(
            "**Not measured.** Precision@k, recall@k and MRR need to know which "
            "papers *should* have been retrieved for each claim, which is a "
            "per-(claim, paper) judgement that `claims.csv` does not carry. Add "
            "rows to `eval/relevance.csv` (`claim_id,paper_id,relevant`) and "
            "re-run. No number is reported here rather than a zero, because "
            "'retrieval scored 0.00' and 'retrieval was never scored' are "
            "different findings."
        )
        add("")

    if retrieval.off_corpus_total:
        add(
            f"**Off-corpus retrieval suppression:** "
            f"{retrieval.off_corpus_suppressed}/{retrieval.off_corpus_total} "
            f"({_pct(retrieval.off_corpus_suppressed, retrieval.off_corpus_total)}) "
            f"of off-corpus claims returned no evidence at all. Off-corpus claims "
            f"are excluded from the ranking metrics above: their relevant set is "
            f"empty, so precision is trivially zero and recall is undefined. "
            f"Whether retrieval was withheld entirely is the measurement that "
            f"applies to them."
        )
        add("")

    # --- Stance classification ---
    add("## Stance classification")
    add("")
    if stance.matched_count:
        add(f"| Class | Precision | Recall | F1 | Support |")
        add("|---|---|---|---|---|")
        for label, values in stance.per_class.items():
            add(
                f"| {label} | {values['precision']:.3f} | {values['recall']:.3f} | "
                f"{values['f1']:.3f} | {int(values['support'])} |"
            )
        add("")
        add(f"Accuracy {_fmt(stance.accuracy)}, macro F1 {_fmt(stance.macro_f1)}, "
            f"over {stance.matched_count} labelled passage(s).")
        add("")
        if stance_png:
            add(f"![Stance confusion matrix]({stance_png.name})")
            add("")
        add("| Gold \\ Predicted | " + " | ".join(s.value for s in STANCE_ORDER) + " |")
        add("|---" * (len(STANCE_ORDER) + 1) + "|")
        for true_label in STANCE_ORDER:
            row = [f"**{true_label.value}**"]
            for predicted_label in STANCE_ORDER:
                row.append(
                    str(stance.confusion.get((true_label.value, predicted_label.value), 0))
                )
            add("| " + " | ".join(row) + " |")
        add("")
        if stance.unmatched_labels:
            add(
                f"{len(stance.unmatched_labels)} labelled passage(s) were not "
                f"returned by the pipeline for their claim and are excluded. They "
                f"are a retrieval miss, not a classification error, and scoring "
                f"them as the latter would misattribute the failure."
            )
            add("")
    else:
        add(
            "**Not measured.** Per-class precision, recall and F1 need a gold "
            "stance for individual passages, which is a per-(claim, passage) "
            "judgement `claims.csv` does not carry. Add rows to "
            "`eval/stance_labels.csv` (`claim_id,passage_id,gold_stance`) and "
            "re-run. The shipped classifier's measured performance on SciFact is "
            "in `eval/results/classifier_comparison.md`; this section measures it "
            "on this corpus instead, which is the harder and more relevant test."
        )
        add("")

    # --- Response time ---
    add("## Response time")
    add("")
    times = sorted(r.response.response_time_ms for r in results)
    if times:
        p95_index = min(len(times) - 1, int(round(0.95 * (len(times) - 1))))
        add("| Metric | Milliseconds |")
        add("|---|---|")
        add(f"| Mean | {statistics.mean(times):.0f} |")
        add(f"| Median | {statistics.median(times):.0f} |")
        add(f"| p95 | {times[p95_index]:.0f} |")
        add(f"| Min | {times[0]:.0f} |")
        add(f"| Max | {times[-1]:.0f} |")
        add("")
        add(
            "Measured server-side across the retrieval, classification and "
            "aggregation stages. Off-corpus claims skip stance classification "
            "entirely once the coverage gate rejects them, so they sit well below "
            "the mean and pull it down."
        )
        add("")

    add("## Per-claim results")
    add("")
    add("| Claim | Topic | Expected | Predicted | Grade | Counts | ms | Result |")
    add("|---|---|---|---|---|---|---|---|")
    for result in results:
        response = result.response
        add(
            f"| `{result.claim.claim_id}` | {result.claim.domain_topic or '—'} | "
            f"{result.claim.expected_verdict.value} | {response.verdict.value} | "
            f"{response.grade_certainty.value} | "
            f"{response.support_count}/{response.contradict_count}/"
            f"{response.neutral_count} | {response.response_time_ms:.0f} | "
            f"{'pass' if result.correct else '**MISS**'} |"
        )
    add("")

    report = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report


# --- CLI --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the verification pipeline against a labelled claim set."
    )
    parser.add_argument("--claims", type=Path, default=REPO_ROOT / "eval" / "claims.csv")
    parser.add_argument("--relevance", type=Path, default=REPO_ROOT / "eval" / "relevance.csv")
    parser.add_argument(
        "--stance-labels", type=Path, default=REPO_ROOT / "eval" / "stance_labels.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "eval" / "results" / "evaluation_report.md"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N labelled claims, in file order. For "
        "smoke-testing the harness before the full set is labelled.",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help=f"Cutoff for retrieval metrics (default {DEFAULT_K})."
    )
    args = parser.parse_args()

    try:
        loaded = load_claims(args.claims)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not loaded.labelled:
        print(
            f"No labelled claims in {args.claims}. Fill in `expected_verdict` for at "
            f"least one row and re-run — the harness works from the first label "
            f"onward, so there is no need to wait for all 50.\n"
            f"({len(loaded.unlabelled_ids)} unlabelled row(s) found.)",
            file=sys.stderr,
        )
        return 1

    claims = loaded.labelled
    if args.sample is not None:
        if args.sample < 1:
            print("ERROR: --sample must be at least 1.", file=sys.stderr)
            return 2
        claims = claims[: args.sample]

    results = run_claims(claims)

    relevance = load_relevance(args.relevance)
    stance_labels = load_stance_labels(args.stance_labels)

    retrieval = compute_retrieval(results, relevance, args.k)
    stance = compute_stance(results, stance_labels)
    errors = find_high_confidence_errors(results)

    confusion_png = args.out.parent / "confusion_matrix.png"
    write_confusion_png(results, confusion_png)

    stance_png = None
    if stance.matched_count:
        stance_png = args.out.parent / "confusion_matrix_stance.png"
        write_stance_confusion_png(stance, stance_png)

    build_report(
        results=results,
        loaded=loaded,
        retrieval=retrieval,
        stance=stance,
        errors=errors,
        sample_size=args.sample,
        confusion_png=confusion_png,
        stance_png=stance_png,
        report_path=args.out,
    )

    correct = sum(1 for r in results if r.correct)
    print("", file=sys.stderr)
    print(f"Verdict accuracy : {correct}/{len(results)} ({_pct(correct, len(results))})", file=sys.stderr)
    print(f"High-confidence errors: {len(errors)}", file=sys.stderr)
    print(f"Report  : {args.out}", file=sys.stderr)
    print(f"Matrix  : {confusion_png}", file=sys.stderr)
    if stance_png:
        print(f"Stance  : {stance_png}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
