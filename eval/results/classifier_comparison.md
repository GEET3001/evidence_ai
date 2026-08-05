# Stance classifier comparison: zero-shot baseline vs. fine-tuned

## SciFact held-out test set

Reproduces the exact held-out split from `notebooks/train_stance.ipynb` (`test_size=0.1`, `random_state=42`, stratified, carved out of `data/scifact/train.jsonl` — SciFact's official `test` split has zero usable annotations, see prepare_scifact.py). N=129.

| Metric | Baseline (bart-large-mnli) | Fine-tuned |
|---|---|---|
| Overall accuracy | 50.4% | 71.3% |
| Macro F1 | 0.507 | 0.692 |
| Mean inference time / passage | 109.1 ms | 80.1 ms |
| N (held-out test) | 129 | 129 |

### Per-class precision / recall / F1

| Class | Baseline P | Baseline R | Baseline F1 | Fine-tuned P | Fine-tuned R | Fine-tuned F1 | Support |
|---|---|---|---|---|---|---|---|
| SUPPORT | 0.783 | 0.290 | 0.424 | 0.671 | 0.790 | 0.726 | 62 |
| CONTRADICT | 0.621 | 0.529 | 0.571 | 0.435 | 0.294 | 0.351 | 34 |
| NEUTRAL | 0.377 | 0.879 | 0.527 | 1.000 | 1.000 | 1.000 | 33 |

### Headline number: CONTRADICT recall

**CONTRADICT recall: baseline 52.9% -> fine-tuned 29.4% (-23.5%, WORSE)** (n=34 CONTRADICT examples in the held-out test set).

### Confusion matrix — baseline

| True \ Predicted | SUPPORT | CONTRADICT | NEUTRAL |
|---|---|---|---|
| **SUPPORT** | 18 | 9 | 35 |
| **CONTRADICT** | 3 | 18 | 13 |
| **NEUTRAL** | 2 | 2 | 29 |

### Confusion matrix — fine-tuned

| True \ Predicted | SUPPORT | CONTRADICT | NEUTRAL |
|---|---|---|---|
| **SUPPORT** | 49 | 13 | 0 |
| **CONTRADICT** | 24 | 10 | 0 |
| **NEUTRAL** | 0 | 0 | 33 |

## Corpus contested-topic sample

30 real (claim, passage) pairs sampled from indexed passages belonging to papers tagged with a genuinely-contested topic (see `app/ingestion/contested_topics.py`). Predictions are NOT auto-labeled as ground truth — for manual review, and may feed a future eval set.

8 of 30 sampled corpus pairs — baseline and fine-tuned disagree. Neither prediction is treated as ground truth below; review manually.

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.98) vs. fine-tuned: **SUPPORT** (conf=0.52)
  - passage (pmid_35398452_p000): This commentary suggests that neuroscience research on young healthy heavy cannabis users and patients with cannabis-induced psychosis using multimodal assessment of sensorimotor dysfunction (e.g. neuroimaging, clinical 

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **SUPPORT** (conf=0.80) vs. fine-tuned: **NEUTRAL** (conf=0.99)
  - passage (pmid_41057642_p000): Inflammatory changes have been widely reported in psychosis. Cannabis use has been consistently related to increased risk of psychosis, earlier onset, higher rates of relapse and poorer treatment response. However, it is

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **SUPPORT** (conf=0.92) vs. fine-tuned: **NEUTRAL** (conf=0.97)
  - passage (pmid_40149904_p001): The Hill criteria indicated a high likelihood for the contribution of cannabis to schizophrenia development. Cannabinoids likely contribute to chronic psychotic events and schizophrenia, especially if taken during adoles

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.89) vs. fine-tuned: **SUPPORT** (conf=0.56)
  - passage (pmid_40931726_p000): Cannabis use increases the risk of psychosis, but cannabis-based medicinal products may provide additional therapeutic opportunities. Decriminalisation of cannabis has led to wider availability in certain jurisdictions, 

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.99) vs. fine-tuned: **SUPPORT** (conf=0.61)
  - passage (pmid_40474301_p002): By examining expressed emotions within the family context, the findings might shed light on the mechanism of the online MBI in affecting the mental health outcomes of parents and the ADHD symptoms and executive functioni

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **CONTRADICT** (conf=0.66) vs. fine-tuned: **NEUTRAL** (conf=0.99)
  - passage (pmid_35608666_p001): However, time-by-group interaction did not achieve statistical significance for commission errors and hit RT, indicating that the changes over time in these outcomes were not significantly different between the MBI and C

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.95) vs. fine-tuned: **SUPPORT** (conf=0.63)
  - passage (pmid_38032045_p006): Los hallazgos de esta revisión no demostraron ningún daño causado por CISD, CISM, PFA, TRiM, EMDR, asesoramiento grupal o intervenciones de TCC cuando se realizaron en un entorno laboral. Sin embargo, no demuestran de ma

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.98) vs. fine-tuned: **SUPPORT** (conf=0.57)
  - passage (pmid_42010727_p002): CONCLUSION: Responders deployed through a large humanitarian organization in KPK and Punjab demonstrated a high burden of post-traumatic stress symptoms, appears higher than some previously reported estimates among disas


## Recommendation

**Do not ship the fine-tuned model as-is.** It does not improve CONTRADICT recall, which was the entire point of fine-tuning.

- CONTRADICT recall: did not improve (52.9% -> 29.4%)
- No material SUPPORT/NEUTRAL F1 regression (>5 points) found.
