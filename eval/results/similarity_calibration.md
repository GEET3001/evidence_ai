# Corpus-coverage threshold calibration

Regenerate with `python -m app.pipeline.calibrate_coverage` from `backend/`. These thresholds are specific to this corpus and to `pritamdeka/S-PubMedBert-MS-MARCO` — rebuilding the index or changing the embedding model invalidates them.

## Why this exists

`MIN_SIMILARITY` is compared against a per-query min-max-normalized score. Normalization sets the top-ranked passage of *every* query to 1.0, so the threshold answers "is this the best of what we found?" and never "is any of this actually relevant?". An off-corpus claim therefore sails through it and receives a confident verdict. The gate calibrated here uses raw pre-normalization cosine instead, which is comparable across queries — but its useful range is a property of the embedding model and corpus, so it has to be measured rather than guessed.

## Claim sets

- **in_domain** — 20 claims on topics the corpus covers.
- **other_domain** — 7 off-corpus claims.
- **nonsense** — 6 off-corpus claims.
- **near_miss** — 7 off-corpus claims.

`near_miss` is the category that matters. Those are real biomedical claims that a mental-health corpus still cannot answer, and they sit closest to the decision boundary.

## Distributions

### max_cosine

| Set | n | min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|---|---|
| in_domain | 20 | 0.916 | 0.927 | 0.932 | 0.950 | 0.965 | 0.938 |
| other_domain | 7 | 0.868 | 0.877 | 0.881 | 0.893 | 0.904 | 0.884 |
| nonsense | 6 | 0.861 | 0.865 | 0.869 | 0.873 | 0.880 | 0.870 |
| near_miss | 7 | 0.893 | 0.898 | 0.910 | 0.913 | 0.926 | 0.907 |
| ALL off-corpus | 20 | 0.861 | 0.873 | 0.884 | 0.904 | 0.926 | 0.888 |

```
  in-domain = #    off-corpus = o    overlap = X
   0.861- 0.869 | ooooooooooooooooooooooo                         0#  4o
   0.869- 0.876 | oooooooooooo                                    0#  2o
   0.876- 0.884 | ooooooooooooooooooooooo                         0#  4o
   0.884- 0.891 | oooooo                                          0#  1o
   0.891- 0.898 | ooooooooooooooooo                               0#  3o
   0.898- 0.906 | oooooooooooo                                    0#  2o
   0.906- 0.913 | ooooooooooooooooo                               0#  3o
   0.913- 0.921 | ######                                          1#  0o
   0.921- 0.928 | XXXXXX########################################  8#  1o
   0.928- 0.935 | #################                               3#  0o
   0.935- 0.943 | ######                                          1#  0o
   0.943- 0.950 | #################                               3#  0o
   0.950- 0.958 |                                                 0#  0o
   0.958- 0.965 | #######################                         4#  0o
```

### mean_topk_cosine

| Set | n | min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|---|---|
| in_domain | 20 | 0.909 | 0.917 | 0.922 | 0.938 | 0.955 | 0.926 |
| other_domain | 7 | 0.863 | 0.868 | 0.875 | 0.878 | 0.880 | 0.873 |
| nonsense | 6 | 0.856 | 0.856 | 0.862 | 0.871 | 0.872 | 0.863 |
| near_miss | 7 | 0.884 | 0.890 | 0.896 | 0.903 | 0.903 | 0.895 |
| ALL off-corpus | 20 | 0.856 | 0.868 | 0.876 | 0.890 | 0.903 | 0.878 |

```
  in-domain = #    off-corpus = o    overlap = X
   0.856- 0.863 | ooooooooooooooooooooooooooooooo                 0#  4o
   0.863- 0.871 | ooooooooooooooooooooooo                         0#  3o
   0.871- 0.878 | ooooooooooooooooooooooooooooooo                 0#  4o
   0.878- 0.885 | ooooooooooooooooooooooo                         0#  3o
   0.885- 0.892 | ooooooooooooooo                                 0#  2o
   0.892- 0.899 | ooooooooooooooo                                 0#  2o
   0.899- 0.906 | ooooooooooooooo                                 0#  2o
   0.906- 0.913 | #######################                         3#  0o
   0.913- 0.920 | ###############################                 4#  0o
   0.920- 0.927 | ##############################################  6#  0o
   0.927- 0.934 | ########                                        1#  0o
   0.934- 0.941 | #######################                         3#  0o
   0.941- 0.948 | ########                                        1#  0o
   0.948- 0.955 | ###############                                 2#  0o
```

### top_bm25

| Set | n | min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|---|---|
| in_domain | 20 | 9.385 | 14.236 | 15.585 | 21.382 | 26.968 | 17.067 |
| other_domain | 7 | 6.721 | 7.078 | 9.234 | 11.149 | 12.276 | 9.187 |
| nonsense | 6 | 6.566 | 7.247 | 8.026 | 9.973 | 10.947 | 8.464 |
| near_miss | 7 | 7.942 | 10.283 | 11.505 | 15.989 | 17.900 | 12.633 |
| ALL off-corpus | 20 | 6.566 | 7.880 | 9.973 | 11.505 | 17.900 | 10.176 |

```
  in-domain = #    off-corpus = o    overlap = X
   6.566- 8.024 | oooooooooooooooooooooooooooooooooooooooooooooo  0#  7o
   8.024- 9.481 | XXXXXXXoooooo                                   1#  2o
   9.481-10.938 | oooooooooooooooooooo                            0#  3o
  10.938-12.395 | ooooooooooooooooooooooooooooooooo               0#  5o
  12.395-13.852 | XXXXXXX#############                            3#  1o
  13.852-15.310 | #######################################         6#  0o
  15.310-16.767 | XXXXXXX                                         1#  1o
  16.767-18.224 | XXXXXXX#############                            3#  1o
  18.224-19.681 | #######                                         1#  0o
  19.681-21.139 |                                                 0#  0o
  21.139-22.596 | #############                                   2#  0o
  22.596-24.053 | #######                                         1#  0o
  24.053-25.510 | #######                                         1#  0o
  25.510-26.968 | #######                                         1#  0o
```

## Chosen thresholds

| Signal | Threshold | Distributions | Lowest in-domain | Highest off-corpus |
|---|---|---|---|---|
| `max_cosine` | **0.911** | **OVERLAP** | 0.916 | 0.926 |
| `mean_topk_cosine` | **0.905** | cleanly separated | 0.909 | 0.903 |

- `max_cosine` **does not separate the two sets**: the highest off-corpus value (0.926) is above the lowest in-domain one (0.916). No threshold can catch every off-corpus claim without also rejecting valid ones. It is therefore set below the lowest in-domain observation as a safety net only, catching 17/20 off-corpus claims on its own rather than being tuned up at the cost of false rejections.
- `mean_topk_cosine` separates cleanly (0.903 off-corpus max vs 0.909 in-domain min), so the threshold sits 25% of the way up the gap — deliberately below the midpoint, see the holdout note below.

The two error types are not symmetric, which is what drives that policy. Letting an off-corpus claim through produces a wrong verdict — the bug being fixed. Rejecting an in-domain claim breaks a question the corpus can actually answer. The second is worse, so thresholds are chosen to never reject an observed in-domain claim.

`top_bm25` is measured and carried through the pipeline but is **not** part of the gate. It is an unbounded, corpus-frequency-dependent score rather than a bounded one, so it has no stable cross-query scale to put a fixed floor on. Its distribution is reported above because near-zero lexical overlap is a useful independent diagnostic when a gate decision looks wrong.

## Separation quality (combined gate)

The gate rejects a claim when `max_cosine` is below its floor **or** `mean_topk_cosine` is below its floor.

- Off-corpus claims correctly caught: **20/20**
- In-domain claims wrongly rejected: **0/20**

**Those two numbers are training-set fit** — the thresholds were chosen on these same claims, so they are an upper bound, not an estimate of real-world behaviour. Recalibrating on half the claims and scoring the held-out half gives: **9/9** off-corpus caught, **4/10** in-domain wrongly rejected. With 40 claims total this is indicative only.

**The separating margin is thin: 0.006.** The two distributions do not overlap, but they very nearly touch, so this gate should be expected to misclassify claims that sit between the sets sampled here. It is a working guard, not a solved problem.

| Off-corpus category | Caught | Total |
|---|---|---|
| other_domain | 7 | 7 |
| nonsense | 6 | 6 |
| near_miss | 7 | 7 |

## End-to-end verdict change

The full pipeline was run over all 40 claims twice — once with both floors forced to `0.0` (the pre-fix behaviour) and once with the calibrated values — and the verdicts diffed.

**20 of 40 verdicts changed. All 20 were off-corpus claims. No in-domain verdict changed.**

| | before | after |
|---|---|---|
| Off-corpus given a confident directional verdict | 9 | 0 |
| Off-corpus `INSUFFICIENT_EVIDENCE` | 11 | 20 |
| In-domain verdicts altered | — | 0 |

The 9 off-corpus claims that previously received a confident `CONTRADICTED` are the bug in its plainest form. Examples:

| Claim | Before | After |
|---|---|---|
| Eating chocolate cake improves stock market forecasting. | `CONTRADICTED` (0 support / 1 contradict / 9 neutral) | `INSUFFICIENT_EVIDENCE` — `NOT_COVERED_BY_CORPUS` |
| Left-handed pigeons navigate using prime numbers. | `CONTRADICTED` (0/9/1) | `INSUFFICIENT_EVIDENCE` — `NOT_COVERED_BY_CORPUS` |
| Roman concrete achieves self-healing durability through lime clasts. | `CONTRADICTED` (0/2/8) | `INSUFFICIENT_EVIDENCE` — `NOT_COVERED_BY_CORPUS` |
| Early mobilisation shortens ICU length of stay after cardiac surgery. | `CONTRADICTED` (0/1/9) | `INSUFFICIENT_EVIDENCE` — `NOT_COVERED_BY_CORPUS` |

The remaining 11 off-corpus claims were already `INSUFFICIENT_EVIDENCE`, but for the wrong reason — `EVIDENCE_INCONCLUSIVE`, i.e. "the corpus discusses this but has no clear direction". They now carry `NOT_COVERED_BY_CORPUS` instead. Same verdict to the caller, honest reason underneath.

On the in-domain side nothing moved: 8 `SUPPORTED`, 1 `CONFLICTING`, and 11 `INSUFFICIENT_EVIDENCE` before and after, claim for claim. "Mindfulness-based interventions significantly reduce ADHD symptoms" stays `SUPPORTED`; all 21 claims that were `INSUFFICIENT_EVIDENCE` before are still `INSUFFICIENT_EVIDENCE`.

Note this is the same 40 claims the thresholds were fitted on, so it confirms the gate is wired correctly end to end — it is not independent evidence about generalization. The holdout number above is the one to trust for that.

## Per-claim measurements

| Category | Claim | max_cosine | mean_topk_cosine | top_bm25 | Gated out |
|---|---|---|---|---|---|
| in_domain | Mindfulness-based interventions significantly reduce ADHD symptoms. | 0.965 | 0.955 | 15.98 | no |
| in_domain | Cannabis use increases the risk of psychosis and schizophrenia. | 0.963 | 0.953 | 26.97 | no |
| in_domain | SSRIs are more effective than placebo for mild-to-moderate depression. | 0.943 | 0.939 | 14.24 | no |
| in_domain | Social media use causes increased depression and anxiety in adolescents. | 0.959 | 0.946 | 23.34 | no |
| in_domain | Single-session psychological debriefing after trauma prevents PTSD. | 0.942 | 0.928 | 13.93 | no |
| in_domain | Cognitive behavioural therapy reduces symptoms of major depressive disorder. | 0.926 | 0.922 | 14.44 | no |
| in_domain | Smartphone-delivered mental health interventions reduce anxiety symptoms. | 0.961 | 0.934 | 12.76 | no |
| in_domain | Physical exercise improves depressive symptoms in adults. | 0.916 | 0.909 | 12.69 | no |
| in_domain | Telemedicine is as effective as in-person therapy for common mental disorders. | 0.935 | 0.922 | 15.16 | no |
| in_domain | Mindfulness meditation reduces symptoms of generalized anxiety disorder. | 0.927 | 0.921 | 15.19 | no |
| in_domain | Antidepressants increase the risk of suicidal ideation in young people. | 0.928 | 0.921 | 25.02 | no |
| in_domain | Screen time is associated with poor sleep quality in adolescents. | 0.923 | 0.910 | 19.03 | no |
| in_domain | Internet-delivered cognitive behavioural therapy is effective for insomnia. | 0.930 | 0.914 | 15.18 | no |
| in_domain | Prenatal cannabis exposure affects neurodevelopmental outcomes in offspring. | 0.950 | 0.924 | 21.38 | no |
| in_domain | Antipsychotic medication reduces relapse rates in schizophrenia. | 0.926 | 0.919 | 16.86 | no |
| in_domain | Parental involvement improves outcomes of ADHD interventions in children. | 0.935 | 0.926 | 16.86 | no |
| in_domain | Problematic social media use is associated with body image concerns. | 0.927 | 0.911 | 22.07 | no |
| in_domain | Psychological therapy is effective for post-traumatic stress disorder. | 0.949 | 0.938 | 18.05 | no |
| in_domain | Depression screening in primary care improves patient outcomes. | 0.927 | 0.917 | 9.39 | no |
| in_domain | Mindfulness training reduces occupational stress in healthcare workers. | 0.921 | 0.917 | 12.83 | no |
| other_domain | Quantum entanglement enables faster-than-light communication. | 0.879 | 0.868 | 7.08 | yes |
| other_domain | The CRISPR-Cas9 system can be used to edit plant genomes. | 0.877 | 0.875 | 12.28 | yes |
| other_domain | Reinforcement learning improves robotic grasping accuracy. | 0.887 | 0.870 | 11.15 | yes |
| other_domain | Atlantic hurricane frequency has increased due to ocean warming. | 0.893 | 0.878 | 9.23 | yes |
| other_domain | Sourdough fermentation reduces the glycaemic index of bread. | 0.904 | 0.880 | 9.97 | yes |
| other_domain | Lithium-ion battery degradation accelerates at high charge rates. | 0.881 | 0.877 | 7.88 | yes |
| other_domain | Roman concrete achieves self-healing durability through lime clasts. | 0.868 | 0.863 | 6.72 | yes |
| nonsense | Eating chocolate cake improves stock market forecasting. | 0.861 | 0.856 | 8.37 | yes |
| nonsense | Wearing purple socks increases the boiling point of water. | 0.870 | 0.865 | 10.95 | yes |
| nonsense | Listening to jazz makes bicycles more aerodynamic. | 0.880 | 0.871 | 7.25 | yes |
| nonsense | Counting clouds reduces the tensile strength of steel. | 0.873 | 0.872 | 9.97 | yes |
| nonsense | Alphabetising a bookshelf raises local property taxes. | 0.865 | 0.859 | 6.57 | yes |
| nonsense | Left-handed pigeons navigate using prime numbers. | 0.869 | 0.856 | 7.68 | yes |
| near_miss | Statins reduce the risk of cardiovascular mortality in older adults. | 0.910 | 0.903 | 17.90 | yes |
| near_miss | Metformin improves glycaemic control in type 2 diabetes. | 0.913 | 0.903 | 11.10 | yes |
| near_miss | Early mobilisation shortens ICU length of stay after cardiac surgery. | 0.926 | 0.896 | 15.99 | yes |
| near_miss | Vitamin D supplementation reduces the incidence of bone fractures. | 0.893 | 0.890 | 10.28 | yes |
| near_miss | Proton pump inhibitors increase the risk of chronic kidney disease. | 0.912 | 0.897 | 13.71 | yes |
| near_miss | Physiotherapy improves recovery after anterior cruciate ligament reconstruction. | 0.900 | 0.884 | 11.51 | yes |
| near_miss | Antibiotic prophylaxis reduces surgical site infection rates. | 0.898 | 0.890 | 7.94 | yes |
