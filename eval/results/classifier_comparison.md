# Stance classifier comparison: zero-shot baseline vs. fine-tuned

> **Status: baseline-only run.** No fine-tuned checkpoint was found at `models/stance-deberta` (or the path passed via `--fine-tuned-path`) when this report was generated — `notebooks/train_stance.ipynb` has not yet been executed against a live GPU. Every number below for the baseline is real and measured; the fine-tuned columns are explicitly marked *not yet run*, not estimated or filled in. Re-run this script once a checkpoint exists for the real comparison.

## SciFact held-out test set

Reproduces the exact held-out split from `notebooks/train_stance.ipynb` (`test_size=0.1`, `random_state=42`, stratified, carved out of `data/scifact/train.jsonl` — SciFact's official `test` split has zero usable annotations, see prepare_scifact.py). N=129.

| Metric | Baseline (bart-large-mnli) | Fine-tuned |
|---|---|---|
| Overall accuracy | 50.4% | *not yet run* |
| Macro F1 | 0.507 | *not yet run* |
| Mean inference time / passage | 1011.3 ms | *not yet run* |
| N (held-out test) | 129 | 129 |

### Per-class precision / recall / F1

| Class | Baseline P | Baseline R | Baseline F1 | Fine-tuned P | Fine-tuned R | Fine-tuned F1 | Support |
|---|---|---|---|---|---|---|---|
| SUPPORT | 0.783 | 0.290 | 0.424 | *n/a* | *n/a* | *n/a* | 62 |
| CONTRADICT | 0.621 | 0.529 | 0.571 | *n/a* | *n/a* | *n/a* | 34 |
| NEUTRAL | 0.377 | 0.879 | 0.527 | *n/a* | *n/a* | *n/a* | 33 |

### Headline number: CONTRADICT recall

**Baseline CONTRADICT recall: 52.9%** (n=34 CONTRADICT examples in the held-out test set). This is the documented weakness motivating fine-tuning. No fine-tuned-model comparison is available yet — see the status note at the top.

### Confusion matrix — baseline

| True \ Predicted | SUPPORT | CONTRADICT | NEUTRAL |
|---|---|---|---|
| **SUPPORT** | 18 | 9 | 35 |
| **CONTRADICT** | 3 | 18 | 13 |
| **NEUTRAL** | 2 | 2 | 29 |

## Corpus contested-topic sample

30 real (claim, passage) pairs sampled from indexed passages belonging to papers tagged with a genuinely-contested topic (see `app/ingestion/contested_topics.py`). Predictions are NOT auto-labeled as ground truth — for manual review, and may feed a future eval set.

*No fine-tuned model available — showing the baseline's predictions on these corpus samples only. Re-run once a fine-tuned checkpoint exists to see where the two models disagree.*

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **SUPPORT** (conf=0.92)
  - passage (pmid_40149904_p001): The Hill criteria indicated a high likelihood for the contribution of cannabis to schizophrenia development. Cannabinoids likely contribute to chronic psychotic events and schizophrenia, especially if taken during adoles

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.97)
  - passage (pmid_38191320_p000): BACKGROUND: Studies to date examining cortical thickness and surface area in young individuals At Risk Mental State (ARMS) of developing psychosis have revealed inconsistent findings, either reporting increased, decrease

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.98)
  - passage (pmid_41057642_p001): Our results demonstrate a differential effect of cannabis use on FW, a surrogate marker of neuroinflammatory processes and suggest that past cannabis use may influence the effects of antipsychotic medication on the brain

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.96)
  - passage (pmid_40156962_p000): BACKGROUND: Distressing psychotic-like experiences (PLEs) in children are associated with an increased risk for psychiatric disorders. Recent studies suggest that different domains of psychotic symptoms could be associat

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.82)
  - passage (pmid_35753121_p001): Parental psychiatric disorders, family structure, sex, frequent alcohol intoxications, daily smoking and illicit substance use other than cannabis were adjusted for. RESULTS: In all, 6552 subjects (49.2 % males) were inc

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.98)
  - passage (pmid_38418416_p000): BACKGROUND: In patients with a psychotic disorder, rates of substance use (tobacco, cannabis, and alcohol) are higher compared to the general population. However, little is known about associations between substance use 

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.91)
  - passage (pmid_42350829_p000): Research suggests that ADHD is characterized by dominant Default-Mode Network (DMN) and diminished Task Positive Network (TPN) activity, which has been linked to detrimentally affecting attention and executive task-depen

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.74)
  - passage (pmid_41138947_p001): These effects were consistently maintained at 1- and 4-months follow-ups. A higher proportion of children showed reliable post-treatment improvements in the intervention groups versus the control group (63.6 % vs. 52.2 %

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.89)
  - passage (pmid_32338110_p000): Objective: This study was the first attempt to explore the efficacy of a mindfulness protocol for children with attention-deficit hyperactivity disorder (ADHD) and oppositional defiant disorder (ODD), and their parents. 

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.93)
  - passage (pmid_41810578_p001): Intent-to-treat analyses showed significantly lower ADHD symptom severity in the intervention group at T1 (baseline-adjusted mean difference = -5.0 points; d = 0.85, p < .001). Significant improvements were also observed

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **SUPPORT** (conf=0.76)
  - passage (pmid_42276684_p000): In the dual pathway model, executive dysfunction (EDF) and delay aversion (DEL) are key mechanisms underlying ADHD. This study aimed to develop and evaluate a pathway model to elucidate the mediating roles of trait mindf

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.96)
  - passage (pmid_35608666_p000): This study examined the effectiveness of a mindfulness-based intervention (MBI) on Conners' continuous performance test scores (CPTs), cardiac vagal control (CVC) assessed by vagally mediated heart rate variability (HRV)

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.75)
  - passage (pmid_25858181_p004): The quality of evidence for the remaining outcomes (that is prevalence of anxiety, prevalence of fear of childbirth, prevalence of general psychological morbidity, health service utilization and attrition from treatment)

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.84)
  - passage (pmid_42189904_p000): Background: A conversation between a victim and a perpetrator of sexual abuse has the potential to reduce posttraumatic stress disorder (PTSD) symptoms in the victim. However, an actual conversation may not always be pos

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=1.00)
  - passage (pmid_41474777_p001): During the clinical debriefs with mothers, a major influence on the mother's perception of her birth was the effectiveness of communication between her and her health care team. CLINICAL IMPLICATIONS: When labor and birt

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.91)
  - passage (pmid_38032045_p000): Background: After a traumatic incident in the workplace organisations want to provide support for their employees to prevent PTSD. However, what is safe and effective to offer has not yet been established, despite many o

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.84)
  - passage (pmid_40283813_p001): The PEDro scale showed that one study was of high methodological quality, four were of acceptable quality, and two had deficiencies. The findings revealed mixed outcomes: while some studies reported a reduction in PTSD s

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.98)
  - passage (pmid_40283813_p000): Background : Rescue teams and emergency services face high levels of mental health problems due to their frequent exposure to traumatic situations. Critical incident stress debriefing (CISD) is widely used as a psycholog

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.85)
  - passage (pmid_42298888_p000): In this study, we first explored the independent associations of problematic social media use and moderate-to-vigorous physical activity (MVPA) with mental health outcomes measured by symptoms of depression and anxiety. 

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.96)
  - passage (pmid_42121104_p001): We weighted the sample to be representative of the city-wide population of 12-15-year-olds and report the median daily screen time spent on social media apps by age, sex, and ethnicity. We used a log-linear model to esti

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **SUPPORT** (conf=0.94)
  - passage (pmid_42298888_p001): We found that problematic social media use was associated with an increased risk of having symptoms of depression ( β = 0.337, p < 0.001) and anxiety ( β = 0.451, p < 0.001), especially in girls. MVPA also buffered the n

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.98)
  - passage (pmid_41015867_p000): BACKGROUND AND OBJECTIVE : Social anxiety arising from intensive social media usage (SMU) among adolescents and youth has gained extensive attention in recent years due to its negative influence on mental health and acad

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.99)
  - passage (pmid_42192480_p003): Sensitivity analyses using data-driven ROC-optimal cut-offs preserved the core graded pattern. CONCLUSION: Co-occurring digital addictions are common among vocational students and exhibit graded associations with adverse

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.98)
  - passage (pmid_42448051_p001): RESULTS: A three-profile solution was selected as optimal based on model fit indices (AIC, BIC, SABIC), entropy, and bootstrapped likelihood ratio test (BLRT). The three profiles represented a gradient of PSMU severity: 

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.96)
  - passage (pmid_41879760_p001): INTERVENTIONS: Patients were randomly assigned 1:1 to receive an individualized dosing regimen of up to 3 escalating doses of GH001 (6, 12, and 18 mg) or a placebo individualized dosing regimen on a single day (day 1). M

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.96)
  - passage (pmid_42061517_p000): OBJECTIVE (S): Perimenopause is associated with an increased risk of depressive symptoms, potentially related to hormonal fluctuations during the menopausal transition. Although hormone replacement therapy (HRT) is widel

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.70)
  - passage (pmid_42061517_p001): In placebo-controlled comparisons, HRT was associated with a small reduction in depressive symptom severity (SMD = -0.23, 95% CI -0.43 to -0.03). Subgroup analyses suggested larger effect estimates for tibolone or select

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.97)
  - passage (pmid_42138922_p001): MAIN OUTCOMES AND MEASURES: The primary end point was between-group difference in change in Montgomery-Åsberg Depression Rating Scale (MADRS) score from baseline to day 8. Secondary end points included MADRS scores on da

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.94)
  - passage (pmid_41925612_p000): Long-term antidepressant pharmacotherapy is frequently continued beyond guideline-recommended durations. This occurs particularly among women, despite limited evidence of ongoing benefit for many people treated for mild-

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.92)
  - passage (pmid_41310599_p002): A monotherapy trial (aggregate data only) showed a larger effect (MD = - 6.32 [- 8.62 to - 4.03]), but there were concerns over selection bias and unblinding. Esketamine increased sedation (RR = 3.70 [2.02-6.78]), dissoc


## Recommendation

**Pending — no real recommendation yet.** No fine-tuned checkpoint exists in this environment (see status note at the top). Shipping the baseline right now isn't a recommendation, it's the only option available. Once `notebooks/train_stance.ipynb` has actually been run on a GPU and its checkpoint is placed at `STANCE_MODEL_PATH` (or passed via `--fine-tuned-path`), re-run this script. The real recommendation should weigh, in this order: (1) CONTRADICT recall — the entire reason for fine-tuning, so a fine-tuned model that doesn't move this number isn't worth shipping regardless of other gains; (2) whether SUPPORT/NEUTRAL performance regresses — a model that fixes CONTRADICT by over-predicting it everywhere is worse, not better; (3) macro F1 as a single-number sanity check, not the deciding factor; (4) mean inference time — a real cost for a live `/verify` endpoint, not just an academic concern, especially if the fine-tuned model is larger than bart-large-mnli.
