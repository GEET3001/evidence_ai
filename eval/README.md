# Evaluation

## The short version

```bash
python eval/propose_candidates.py --write-claims   # draft candidates + surface their papers
# read eval/candidate_worksheet.md, fill in expected_verdict in eval/claims.csv
python eval/run_evaluation.py --sample 5           # smoke-test on the first 5 labelled claims
python eval/run_evaluation.py                      # full run once you have labels you trust
```

Unlabelled rows are skipped and counted, so the harness runs from the first
label onward. There is no need to finish all 50 before measuring anything.

## Files

| File | What it holds | Who writes it |
|---|---|---|
| `claims.csv` | One row per claim. Verdict-level ground truth. | You |
| `relevance.csv` | One row per (claim, paper). Which papers *should* have been retrieved. Optional. | You |
| `stance_labels.csv` | One row per (claim, passage). What each passage actually says. Optional. | You |
| `candidate_worksheet.md` | 50 drafted candidates with their retrieved papers, for reading before labelling. | `propose_candidates.py` |
| `results/evaluation_report.md` | The measured results. | `run_evaluation.py` |

## Why ground truth lives in three files

The three metrics families need judgements at three different granularities, and
one row per claim cannot carry all of them.

Verdict accuracy needs one label per claim — that is `claims.csv`. Retrieval
precision, recall and MRR need to know which *papers* were relevant to each
claim. Per-class stance precision, recall and F1 need a gold stance for each
retrieved *passage*. Neither fits in a claim row, so each gets its own file, and
both are optional.

If a sidecar is empty, the report says the corresponding metric was **not
measured** rather than printing a zero. "Retrieval scored 0.00" and "retrieval
was never scored" are different findings and the report must not blur them.

## claims.csv

| Column | Notes |
|---|---|
| `claim_id` | Unique. `EV-01`…`EV-50` as drafted. |
| `claim_text` | The claim as submitted to the pipeline, verbatim. |
| `domain_topic` | Free text, used for grouping in the report. |
| `expected_verdict` | `SUPPORTED`, `CONTRADICTED`, `CONFLICTING`, `INSUFFICIENT_EVIDENCE`. **Blank means unlabelled and the row is skipped.** |
| `expected_grade` | `HIGH`, `MODERATE`, `LOW`, `VERY_LOW`. Optional; scored only where present. |
| `contested` | `true`/`false`. Pre-filled for the five seeded topics. |
| `off_corpus` | `true`/`false`. **Decides which insufficient-evidence reason the claim is scored against**, so worth checking rather than trusting. |
| `notes` | Free text. Shown in the report next to any high-confidence error on that claim. |
| `labeller` | Who decided. |
| `labelled_date` | When. |

`expected_verdict` and `expected_grade` are left blank by `propose_candidates.py`
on purpose. The labels are the ground truth and nothing generates them for you.

### How `off_corpus` is used

It is not scored directly. It selects which failure a claim expecting
`INSUFFICIENT_EVIDENCE` is measured against:

- `off_corpus = true` → the system should return `NOT_COVERED_BY_CORPUS`
- `off_corpus = false` → the system should return `EVIDENCE_INCONCLUSIVE`

Both are `INSUFFICIENT_EVIDENCE` to a caller. They are different failures
underneath, and the report measures them separately because collapsing them
hides whether the coverage gate is working.

## relevance.csv

```csv
claim_id,paper_id,relevant
EV-11,pmid_40958241,true
EV-11,pmid_41138947,true
```

`paper_id` values are in `data/corpus.json` and are shown for every paper in the
candidate worksheet. A claim with no rows is skipped for retrieval metrics
rather than scored as zero.

Off-corpus claims are excluded from precision/recall/MRR automatically: their
relevant set is empty, which makes precision trivially zero and recall
undefined. They are measured instead by whether retrieval was suppressed
entirely, which is the thing that matters for them.

## stance_labels.csv

```csv
claim_id,passage_id,gold_stance
EV-11,pmid_40958241_p002,SUPPORT
EV-11,pmid_40958241_p001,NEUTRAL
```

`passage_id` values appear in the API response for a claim. Label only passages
the pipeline actually returned for that claim — a label for a passage that was
never retrieved is reported separately as a retrieval miss, since scoring it as
a classification error would misattribute the failure.

## What the report measures

- **Verdict accuracy**, overall and per expected verdict, plus a confusion matrix (PNG and table).
- **Insufficient-evidence detection**, split into `NOT_COVERED_BY_CORPUS` and `EVIDENCE_INCONCLUSIVE`, plus false-insufficient cases — the cost side of the coverage gate.
- **High-confidence errors**: every case where the system issued a confident verdict that did not match the label, listed individually with the passages that drove it, never summarised into a rate.
- **Retrieval**: precision@k, recall@k, MRR, and off-corpus suppression.
- **Stance classification**: per-class precision/recall/F1 and a confusion matrix.
- **Response time**: mean, median, p95, min, max.

### High-confidence error severities

Ordered by how the failure misleads a reader, not by label distance:

| Severity | Meaning |
|---|---|
| `POLARITY_INVERSION` | Said the opposite of the label. |
| `CONFIDENT_ON_OFF_CORPUS` | Issued a directional verdict on a claim the corpus cannot speak to. The failure the coverage gate exists to prevent. |
| `CONFIDENT_ON_INCONCLUSIVE` | Issued a directional verdict where the evidence has no direction. |
| `CONTESTED_RESOLVED` | Presented a genuinely disputed question as settled. |
| `OTHER_CONFIDENT_ERROR` | Any other confident mismatch. |

Ties break on reported certainty, so a wrong verdict at `HIGH` sorts above the
same error at `VERY_LOW`.

## A caveat on the off-corpus candidates

None of the ten off-corpus candidates reuses a claim from
`results/similarity_calibration.md`. The coverage thresholds were fitted on that
set, so scoring against it would measure fit rather than skill. These are new
claims in the same three difficulty classes — near-miss biomedical, other
domain, and nonsense — with the near-miss group being the one that matters.
