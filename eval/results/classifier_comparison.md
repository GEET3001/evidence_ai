# Stance classifier comparison: zero-shot baseline vs. fine-tuned

## SciFact held-out test set

Reproduces the exact held-out split from `notebooks/train_stance.ipynb` (`test_size=0.1`, `random_state=42`, stratified, carved out of `data/scifact/train.jsonl` — SciFact's official `test` split has zero usable annotations, see prepare_scifact.py). N=129.

| Metric | Baseline (bart-large-mnli) | Fine-tuned |
|---|---|---|
| Overall accuracy | 51.9% | 58.9% |
| Macro F1 | 0.521 | 0.518 |
| Mean inference time / passage | 36.4 ms | 26.1 ms |
| N (held-out test) | 129 | 129 |

### Per-class precision / recall / F1

| Class | Baseline P | Baseline R | Baseline F1 | Fine-tuned P | Fine-tuned R | Fine-tuned F1 | Support |
|---|---|---|---|---|---|---|---|
| SUPPORT | 0.818 | 0.290 | 0.429 | 0.573 | 0.823 | 0.675 | 62 |
| CONTRADICT | 0.643 | 0.529 | 0.581 | 0.545 | 0.176 | 0.267 | 34 |
| NEUTRAL | 0.392 | 0.939 | 0.554 | 0.655 | 0.576 | 0.613 | 33 |

### Headline number: CONTRADICT recall

**CONTRADICT recall: baseline 52.9% -> fine-tuned 17.6% (-35.3%, WORSE)** (n=34 CONTRADICT examples in the held-out test set).

### Confusion matrix — baseline

| True \ Predicted | SUPPORT | CONTRADICT | NEUTRAL |
|---|---|---|---|
| **SUPPORT** | 18 | 9 | 35 |
| **CONTRADICT** | 3 | 18 | 13 |
| **NEUTRAL** | 1 | 1 | 31 |

### Confusion matrix — fine-tuned

| True \ Predicted | SUPPORT | CONTRADICT | NEUTRAL |
|---|---|---|---|
| **SUPPORT** | 51 | 4 | 7 |
| **CONTRADICT** | 25 | 6 | 3 |
| **NEUTRAL** | 13 | 1 | 19 |

## Corpus contested-topic sample

30 real (claim, passage) pairs sampled from indexed passages belonging to papers tagged with a genuinely-contested topic (see `app/ingestion/contested_topics.py`). Predictions are NOT auto-labeled as ground truth — for manual review, and may feed a future eval set.

26 of 30 sampled corpus pairs — baseline and fine-tuned disagree. Neither prediction is treated as ground truth below; review manually.

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.64) vs. fine-tuned: **SUPPORT** (conf=0.50)
  - passage (pmid_39903464_p003): The PARF for CUD associated with schizophrenia almost tripled from 3.7% (95% CI, 2.7%-4.7%) during the prelegalization period to 10.3% (95% CI, 8.9%-11.7%) during the legalization period. The PARF in the postlegalization

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **CONTRADICT** (conf=0.81) vs. fine-tuned: **SUPPORT** (conf=0.59)
  - passage (pmid_38908654_p001): Odds ratios (OR) and 95% confidence intervals (CI) were pooled for each neuropsychiatric outcome in the offspring of women exposed to cannabis during pregnancy compared with nonexposed. Data were pooled using random-effe

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.98) vs. fine-tuned: **SUPPORT** (conf=0.61)
  - passage (pmid_38191320_p001): However, reduced left-frontal grey matter thickness was correlated with greater symptom severity and lower function levels; the latter being also correlated with smaller left-frontal surface areas. ARMS individuals with 

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.98) vs. fine-tuned: **SUPPORT** (conf=0.55)
  - passage (pmid_38418416_p000): BACKGROUND: In patients with a psychotic disorder, rates of substance use (tobacco, cannabis, and alcohol) are higher compared to the general population. However, little is known about associations between substance use 

- **[cannabis_psychosis_risk]** claim: *Cannabis use increases the risk of psychosis/schizophrenia.*
  - baseline: **NEUTRAL** (conf=0.95) vs. fine-tuned: **SUPPORT** (conf=0.52)
  - passage (pmid_38418416_p001): Daily cannabis users reported less social participation deficits than non-cannabis users (E = -0.348, SE = 0.145, p = 0.017). Problematic alcohol use was associated with more perceived social support compared to non-alco

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.99) vs. fine-tuned: **SUPPORT** (conf=0.57)
  - passage (pmid_40958241_p002): CONCLUSION: MBIs may be effective in improving core ADHD symptoms and overall functioning in adults with ADHD. However, their effects on emotional well-being and mindfulness skills remain inconclusive. These findings sup

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=1.00) vs. fine-tuned: **SUPPORT** (conf=0.69)
  - passage (pmid_40474301_p000): BACKGROUND: Childhood attention-deficit/hyperactivity disorder (ADHD) has been associated with poor family functioning and higher risks of conflicts in parent-child relationships. Mindfulness-based intervention (MBI) has

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.74) vs. fine-tuned: **SUPPORT** (conf=0.67)
  - passage (pmid_41138947_p001): These effects were consistently maintained at 1- and 4-months follow-ups. A higher proportion of children showed reliable post-treatment improvements in the intervention groups versus the control group (63.6 % vs. 52.2 %

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.97) vs. fine-tuned: **SUPPORT** (conf=0.64)
  - passage (pmid_40275428_p001): 66% of the participants in the intervention group reported satisfaction with the intervention, which was helpful in reducing stress. They were willing to stay on this mindfulness-based programme in the future. The result

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **NEUTRAL** (conf=0.99) vs. fine-tuned: **SUPPORT** (conf=0.70)
  - passage (pmid_40474301_p001): METHODS: This study is a two-arm randomized controlled trial (RCT) study, comparing online family MBI (arm 1) and an online psychoeducation program (arm 2) designed for parents and their ADHD children. The outcome measur

- **[mindfulness_adhd_symptoms]** claim: *Mindfulness-based interventions significantly reduce ADHD symptoms.*
  - baseline: **CONTRADICT** (conf=0.66) vs. fine-tuned: **SUPPORT** (conf=0.62)
  - passage (pmid_35608666_p001): However, time-by-group interaction did not achieve statistical significance for commission errors and hit RT, indicating that the changes over time in these outcomes were not significantly different between the MBI and C

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.66) vs. fine-tuned: **SUPPORT** (conf=0.60)
  - passage (pmid_34318331_p003): A meta-analysis of studies comparing a specific stress control intervention to an active comparator (usually standard stress management education) found no significant effect on PTSD symptom scores (moderate QoE). CONCLU

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.96) vs. fine-tuned: **SUPPORT** (conf=0.69)
  - passage (pmid_35100248_p002): El debriefing psicológico es una intervención para el trauma agudo, que consiste en la verbalización de percepciones, pensamientos y emociones experimentados durante un evento traumático reciente. La evidencia en torno a

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **CONTRADICT** (conf=0.74) vs. fine-tuned: **SUPPORT** (conf=0.72)
  - passage (pmid_42189904_p002): We developed an intervention protocol for a deepfake perpetrator conversation for patients with sexual abuse-related PTSD.The intervention protocol appeared ethically, legally, and clinically feasible, as well as accepta

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.98) vs. fine-tuned: **SUPPORT** (conf=0.69)
  - passage (pmid_40283813_p000): Background : Rescue teams and emergency services face high levels of mental health problems due to their frequent exposure to traumatic situations. Critical incident stress debriefing (CISD) is widely used as a psycholog

- **[psychological_debriefing_ptsd_prevention]** claim: *Single-session psychological debriefing after trauma prevents PTSD.*
  - baseline: **NEUTRAL** (conf=0.95) vs. fine-tuned: **SUPPORT** (conf=0.61)
  - passage (pmid_25858181_p006): AUTHORS' CONCLUSIONS: We did not find any high quality evidence to inform practice, with substantial heterogeneity being found between the studies conducted to date. There is little or no evidence to support either a pos

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **SUPPORT** (conf=0.94) vs. fine-tuned: **CONTRADICT** (conf=0.51)
  - passage (pmid_42298888_p001): We found that problematic social media use was associated with an increased risk of having symptoms of depression ( β = 0.337, p < 0.001) and anxiety ( β = 0.451, p < 0.001), especially in girls. MVPA also buffered the n

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.94) vs. fine-tuned: **SUPPORT** (conf=0.52)
  - passage (pmid_42274377_p002): Estimated risks for all mental health problems were greatest in early adolescence (12-13 years), with the largest effects observed for high depressive symptoms in female participants (> 2 h vs. < 1 h: RD, 10.8 [95% CI, 2

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.95) vs. fine-tuned: **SUPPORT** (conf=0.69)
  - passage (pmid_42475423_p000): The rapid expansion of algorithm-curated short-form video (ACSFV) platforms has raised concerns about their potential association with youth mental health, particularly through personalized content streams that may ampli

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.97) vs. fine-tuned: **SUPPORT** (conf=0.70)
  - passage (pmid_41171508_p000): Digital stress among adolescents is prevalent and likely fluctuates over time. However, existing research has paid limited attention to the developmental course of digital stress and its heterogeneity across individuals.

- **[social_media_adolescent_mental_health]** claim: *Social media use causes increased depression/anxiety in adolescents.*
  - baseline: **NEUTRAL** (conf=0.85) vs. fine-tuned: **SUPPORT** (conf=0.62)
  - passage (pmid_42298888_p000): In this study, we first explored the independent associations of problematic social media use and moderate-to-vigorous physical activity (MVPA) with mental health outcomes measured by symptoms of depression and anxiety. 

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.97) vs. fine-tuned: **SUPPORT** (conf=0.67)
  - passage (pmid_41740754_p000): INTRODUCTION: Current treatments for major depressive episodes may not resolve all symptoms, with residual symptoms predicting disease relapse, recurrence, and functional impairment. This post hoc analysis of Study 403 (

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.98) vs. fine-tuned: **SUPPORT** (conf=0.66)
  - passage (pmid_42331044_p002): Several additional drugs might be efficacious, although they emerged as outliers for either mean age of participants/proportion of females/BD-II participants/psychotic features/rapid cycling/baseline severity of depressi

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.92) vs. fine-tuned: **SUPPORT** (conf=0.64)
  - passage (pmid_41310599_p002): A monotherapy trial (aggregate data only) showed a larger effect (MD = - 6.32 [- 8.62 to - 4.03]), but there were concerns over selection bias and unblinding. Esketamine increased sedation (RR = 3.70 [2.02-6.78]), dissoc

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.91) vs. fine-tuned: **SUPPORT** (conf=0.60)
  - passage (pmid_41412339_p001): Secondary outcomes, including treatment response, remission rates, and global functioning, showed no differences between groups. Scopolamine was associated with milder adverse events, including dizziness and dry mouth, b

- **[ssri_mild_depression_efficacy]** claim: *SSRIs are more effective than placebo for mild-to-moderate depression.*
  - baseline: **NEUTRAL** (conf=0.98) vs. fine-tuned: **SUPPORT** (conf=0.70)
  - passage (pmid_41310599_p000): BACKGROUND: In 2019, the FDA and EMA approved intranasal esketamine for treatment-resistant depression (TRD). The current study re-evaluated its efficacy and safety. METHODS: This registered report presents a systematic 


## Recommendation

**Do not ship the fine-tuned model as-is.** It does not improve CONTRADICT recall, which was the entire point of fine-tuning.

- CONTRADICT recall: did not improve (52.9% -> 17.6%)
- No material SUPPORT/NEUTRAL F1 regression (>5 points) found.
