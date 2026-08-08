# Demo video script

Target runtime **4:30–5:30**. Five shots plus a close. The order is deliberate:
the system earns trust with a clean result, then shows it handles disagreement,
then shows it knows when to refuse — which is the most interesting thing it
does and lands hardest once the viewer has seen it succeed twice.

---

## Pre-recording checklist

Run every item. Most of these are failures that only surface on camera.

**T-15 minutes**

- [ ] **Start with `.\demo.ps1`, not `.\dev.ps1`.** `demo.ps1` runs uvicorn without `--reload`, which removes the reload watcher's second process and its restart cycles. A reload mid-take kills the backend for ~40 seconds while models reload.
- [ ] **Check `http://localhost:8000/health`.** Confirm all four:
  - `"status": "ok"` and `"pipeline_loaded": true`
  - `"stance_model_device": "cuda"` — **if this says `cpu`, stop.** Torch came from the CPU wheel index and every verification will take ~140 s instead of 2–4 s. Nothing else in the demo is recoverable from that.
  - `"passages_indexed": 446` — a 0 here means `data/index/` is missing; run `python -m app.indexing.build_index`.
  - `"rss_mb"` around 900–1000. Much lower usually means models have not finished loading.
- [ ] **Send one throwaway `/verify` call** before recording. The first request after startup pays for lazy CUDA context initialisation and is several seconds slower than every later one. Do not let the demo's first claim be that request.
- [ ] **Frontend up at `http://localhost:3000`** with the scope banner visible at the top.

**T-5 minutes**

- [ ] Browser at 100% zoom, bookmarks bar hidden, one tab only.
- [ ] Notifications silenced (Windows Focus Assist on).
- [ ] Downloads folder cleared, or at least the download shelf empty, so the report download is unambiguous on screen.
- [ ] Dark/light theme decided and fixed — the UI follows the system theme, and switching mid-recording changes every colour.
- [ ] Have the three claims copyable so no shot depends on typing accuracy.

**Claims used**

| Shot | Claim | Expect |
|---|---|---|
| 1 | `Mindfulness-based interventions significantly reduce ADHD symptoms.` | SUPPORTED, ~2.5 s |
| 2 | `Social media use causes increased depression and anxiety in adolescents.` | Directional, contested topic, ~2 s |
| 3 | `Eating chocolate cake improves stock market forecasting.` | INSUFFICIENT — NOT_COVERED_BY_CORPUS, ~10 ms |

> **Verify shot 2 before recording.** The conflicting-findings section only
> renders when retrieval returns both a supporting and a contradicting passage,
> and CONTRADICT recall is 52.9% — so which contested claim produces a visible
> conflict is not guaranteed. Run all five contested claims beforehand and use
> whichever actually yields `conflicting_pairs`. If none does, say so in the
> voiceover and show the section's absence honestly; do not imply a conflict
> that is not on screen.

---

## Shot 1 — A supported claim, end to end (~90 s)

**Screen:** empty results view, scope banner visible at top.

> "This is EvidenceAI. It checks a research claim against a corpus of 183
> mental health papers. Before anything else — the banner at the top stays
> there the whole time. This is a literature triage aid. It is not a diagnostic
> tool and it is not treatment advice."

**Action:** paste claim 1. Click **Verify claim**.

> "Verification runs three stages: retrieval, then a natural-language-inference
> pass per passage, then aggregation."

*Let the loading state play in full — it names the three stages and counts real
elapsed time. Do not cut it. It is ~2.5 seconds and it shows the system is not
faking progress.*

**Screen:** verdict banner resolves to SUPPORTED.

> "Supported, with GRADE certainty moderate. And critically — the wording.
> 'The retrieved evidence supports this claim.' Never 'you should do this.'
> Every verdict in this system is a statement about evidence, not a
> recommendation."

**Action:** scroll to counts, then to the evidence list. Stop on a passage with
highlighted text.

> "Three supporting passages, seven neutral, from seven papers. And here is the
> part that matters: the highlighted sentence is the exact sentence the
> classifier keyed on. That is not a summary — it is a readout of the model
> that made the call, quoted verbatim from the paper."

**Action:** point to the source footer — tier badge, year, journal, DOI link.

> "Every passage carries its source: study design, journal, year, a link to the
> paper. Preprints get an explicit 'not peer reviewed' badge."

---

## Shot 2 — A contested claim (~70 s)

**Action:** new claim, paste claim 2, verify.

> "Now something the field genuinely disagrees about. The corpus was built for
> this: five contested topics were each queried twice — once for terms
> suggesting an effect, once for null findings — so both sides are actually in
> the index. Nothing was tagged with a side; that would be pre-writing the
> answer."

**Screen:** scroll to the conflicting-findings section.

> "Conflicting findings get their own section, side by side — supporting on the
> left, contradicting on the right. They are deliberately not mixed into the
> main evidence list, because a disagreement between two papers is a finding in
> its own right. It survives even when one side is in the majority. A majority
> is not a consensus."

**Action:** scroll to the explanation panel and stop on the closing note.

> "And the explanation. This is assembled from the counts and from sentences
> quoted out of the passages. No language model writes any of it. That is
> deliberate: an explanation that reads well but was generated after the fact
> cannot be checked against the decision it claims to describe."

---

## Shot 3 — Correct abstention (~80 s) · *the important one*

> "Here is the failure mode this system was specifically fixed for."

**Action:** paste claim 3 — chocolate cake and stock market forecasting. Verify.

*Result is near-instant. Let the contrast land.*

> "That returned in about ten milliseconds, not two seconds — because it never
> ran the classifier at all."

**Screen:** the no-verdict banner — hatched, dashed border, 'No verdict issued'
above 'Not covered by this corpus'.

> "And look at how different this is. It is not a coloured answer badge in a
> different shade. It is a structurally different component: no verdict issued,
> and the reason is the headline."

> "Before this was fixed, that claim came back as a confident **CONTRADICTED**.
> So did 'left-handed pigeons navigate using prime numbers'. The relevance
> threshold was reading a score that gets normalised per query — which pushes
> the top hit of *every* query toward 1.0, including a query about nothing in
> the corpus. It was answering 'is this the best of what we found', which is
> not the same question as 'is any of this relevant'."

**Action:** point to the coverage readout — closest passage 0.211, top-10 mean.

> "The gate now uses raw, un-normalised cosine, which is comparable across
> queries, against thresholds calibrated on 40 claims. Closest passage scored
> 0.21 against a floor of 0.91."

> "And the wording is careful: 'This is not evidence the claim is false.' Only
> that this corpus cannot speak to it. The system also distinguishes this from
> 'evidence inconclusive', which means the corpus *does* cover the topic but
> nothing retrieved took a side. Different failures, reported differently."

---

## Shot 4 — Report download (~45 s)

**Action:** return to the supported result from shot 1. Scroll to the Export
section. Click **Download report (.docx)**. Open it in Word.

> "The whole analysis exports as a Word document."

**Action:** scroll the document — header, then stop on the disclaimer box.

> "Scope disclaimer directly under the header, before the verdict — not buried
> at the end, because a caveat that arrives after the conclusion is not doing
> its job."

**Action:** page through to the evidence table, then to limitations.

> "Verdict, counts, the evidence table with rationale sentences bolded in
> place, conflicting findings, and the full limitations — including the
> classifier's measured per-class performance. It states plainly that SUPPORT
> recall is 29%, so the support counts in the report are a floor, not a census.
> Then a methodology appendix covering retrieval, the coverage gate, and how
> certainty was derived."

---

## Shot 5 — Architecture walkthrough (~60 s)

**Screen:** `docs/diagrams/architecture.png`, full screen.

> "Briefly, the shape of it. Everything on the left runs once, offline —
> scraping, deduplication, OpenAlex enrichment for retraction status, chunking,
> embedding. Retracted papers are dropped at index time, so they can never be
> retrieved as evidence."

> "Everything on the right runs per request and makes no network calls at all.
> Models and index load once at startup. That is why it is two to four seconds
> and not two minutes."

**Screen:** switch to `verdict-decision-tree.png`.

> "And this is every path through the verdict logic. Three different routes
> reach 'insufficient evidence' — not covered by the corpus, too few qualifying
> passages, or nothing directional retrieved. They are distinguished because
> they mean different things. Abstention here is a decision tree you can read,
> not a fallback."

*Optional, if runtime allows:* mention the 140 s → 2–4 s GPU move and the
faiss/torch OpenMP segfault fix.

---

## Close (~20 s)

> "Explanations derived from computed values and quoted sentences, never
> generated. Retrieval ranking and corpus gating kept on separate signals
> because one score cannot answer both questions. A fine-tuning experiment that
> was run twice and refuted, with zero-shot retained on the evidence. And a
> system that can say it does not know."

---

## If something breaks mid-take

| Symptom | Cause | Recovery |
|---|---|---|
| Verification hangs 30 s+ | Model on CPU, or backend restarted | Check `/health`; restart with `demo.ps1` |
| "The API is not responding" | Backend died or was never started | The error state is honest and specific — you can leave it in and restart |
| Empty evidence on an in-domain claim | Index missing or stale | `python -m app.indexing.build_index` |
| Report download does nothing | Backend restarted since the verdict, losing nothing — verdicts persist in SQLite | Re-verify and retry; if it 404s, the store was reset |
| No conflicting-findings section | That claim produced no contradicting passage | Use a different contested claim — verified in advance per the checklist |
