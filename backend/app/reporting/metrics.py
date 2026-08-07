"""Measured evaluation results quoted in generated reports.

These are findings, not settings. They come from evaluation runs whose written
output lives in `eval/results/`, and they are reproduced here so a report can
state what this system's accuracy actually is instead of leaving the reader to
assume it was never measured.

They are deliberately not recomputed per request. A report has to quote the
same figures as the written evaluation it cites; deriving them live would let
the two drift apart silently. The cost of that choice is that re-running an
evaluation means editing this module, so each block records the file it came
from and the run that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Stance classifier ------------------------------------------------------
# Source: eval/results/classifier_comparison.md, "SciFact held-out test set".
# The shipped classifier is the zero-shot baseline. Fine-tuning was tried and
# rejected: it moved CONTRADICT recall from 52.9% to 17.6%, which was the one
# thing it existed to improve. Reports quote the baseline because that is what
# actually runs.

STANCE_EVAL_MODEL = "facebook/bart-large-mnli (zero-shot NLI)"
STANCE_EVAL_DATASET = (
    "SciFact held-out split (10%, stratified, carved out of the training set; "
    "SciFact's official test split carries no usable annotations)"
)
STANCE_EVAL_N = 129
STANCE_EVAL_ACCURACY = 0.519
STANCE_EVAL_MACRO_F1 = 0.521


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


STANCE_PER_CLASS: tuple[ClassMetrics, ...] = (
    ClassMetrics("SUPPORT", 0.818, 0.290, 0.429, 62),
    ClassMetrics("CONTRADICT", 0.643, 0.529, 0.581, 34),
    ClassMetrics("NEUTRAL", 0.392, 0.939, 0.554, 33),
)

# The single most consequential number for a reader deciding how much to trust
# a verdict, so it is called out rather than left to be read off the table.
STANCE_HEADLINE = (
    "Recall on SUPPORT is 29.0%: roughly seven in ten genuinely supporting "
    "passages are missed, most of them labelled NEUTRAL. A verdict is therefore "
    "built on a conservative subset of the supporting evidence that exists, so "
    "the source counts in this report are a floor rather than a census."
)

# --- Corpus-coverage gate ---------------------------------------------------
# Source: eval/results/similarity_calibration.md.

COVERAGE_SEPARATING_MARGIN = 0.006
COVERAGE_NEAR_MISS_CAUGHT = 7
COVERAGE_NEAR_MISS_TOTAL = 7
COVERAGE_HOLDOUT_OFF_CORPUS_CAUGHT = 9
COVERAGE_HOLDOUT_OFF_CORPUS_TOTAL = 9
COVERAGE_HOLDOUT_IN_DOMAIN_REJECTED = 4
COVERAGE_HOLDOUT_IN_DOMAIN_TOTAL = 10

COVERAGE_RESIDUAL_GAP = (
    f"The gate that decides whether the corpus covers a claim at all separates "
    f"on-topic from off-topic claims by a margin of only "
    f"{COVERAGE_SEPARATING_MARGIN:.3f} in mean top-k cosine. The hardest "
    f"category tested — real biomedical claims that a mental health corpus "
    f"still cannot answer, such as statin trials or ICU mobilisation — was "
    f"caught {COVERAGE_NEAR_MISS_CAUGHT}/{COVERAGE_NEAR_MISS_TOTAL} of the "
    f"time, but with the two distributions nearly touching. Recalibrating on "
    f"half the claims and scoring the held-out half caught "
    f"{COVERAGE_HOLDOUT_OFF_CORPUS_CAUGHT}/{COVERAGE_HOLDOUT_OFF_CORPUS_TOTAL} "
    f"off-corpus claims while wrongly rejecting "
    f"{COVERAGE_HOLDOUT_IN_DOMAIN_REJECTED}/{COVERAGE_HOLDOUT_IN_DOMAIN_TOTAL} "
    f"valid ones. Claims sitting between those sampled sets should be expected "
    f"to be misclassified in either direction. This is a working guard, not a "
    f"solved problem."
)
