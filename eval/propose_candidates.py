"""Draft candidate claims for the evaluation set and surface the corpus papers behind each.

Run from the repository root:

    python eval/propose_candidates.py
    python eval/propose_candidates.py --write-claims    # also fill eval/claims.csv skeleton

The candidates below were drafted against what this corpus actually contains —
its contested-topic tags, its meta-analyses and RCTs, and the topics its titles
cluster around — so that the evaluation set tests retrieval and stance on real
material rather than on claims the corpus was never going to have papers for.

**No expected verdict is produced by this script, deliberately.** Each candidate
carries a `slot`, which is a proposal about what kind of test the claim
provides, not an answer. Deciding what the retrieved evidence actually says is
the labelling job, and a candidate proposed as "well supported" turning out to
be inconclusive is a real finding about the corpus, not an error to be tidied
away before labelling starts.

For the same reason the worksheet shows no relevance scores, no coverage
signals, and no system verdict. Those are the pipeline's opinion, and a label
formed after reading them would no longer be independent of the thing it is
supposed to be scoring.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app._thread_limits  # noqa: F401,E402  (must precede the faiss/torch imports)

from app.pipeline.retrieval import RetrievalIndex  # noqa: E402

PAPERS_PER_CLAIM = 6
PASSAGES_RETRIEVED = 12


@dataclass(frozen=True)
class Candidate:
    claim_id: str
    claim_text: str
    domain_topic: str
    slot: str
    contested: bool
    off_corpus: bool
    rationale: str


SLOTS = {
    "well-supported": (
        "Proposed as well supported. The corpus should hold several papers "
        "pointing the same way; the question when labelling is whether they "
        "really do, and how strongly."
    ),
    "well-contradicted": (
        "Proposed as well contradicted. These are phrased so that the corpus's "
        "own evidence points against them — several are deliberate inversions "
        "of what a meta-analysis in the corpus reports."
    ),
    "contested": (
        "Drawn from the five topics the corpus was seeded with on both sides. "
        "'The field disagrees' is the interesting answer here, so a one-sided "
        "verdict on these is a failure worth catching."
    ),
    "inconclusive": (
        "Proposed as inconclusive: the corpus has something on the topic, but "
        "thin — often a single study, a case report, or a protocol — so there "
        "may be no direction to read off it."
    ),
    "off-corpus": (
        "Proposed as outside the corpus entirely. These test the coverage gate. "
        "None of them reuses a claim from the threshold calibration set, so "
        "they measure the gate rather than re-confirming its fit."
    ),
}


CANDIDATES: list[Candidate] = [
    # --- Contested: two per seeded contested topic -------------------------
    Candidate("EV-01", "Mindfulness-based interventions significantly reduce ADHD symptoms.",
              "mindfulness_adhd_symptoms", "contested", True, False,
              "The canonical form of the seeded topic; 15 papers carry this tag."),
    Candidate("EV-02", "Mindfulness training reduces ADHD symptoms in children more than an active control treatment.",
              "mindfulness_adhd_symptoms", "contested", True, False,
              "Narrower than EV-01 and aimed at the active-control trials (MYmind vs CBT), where the disagreement actually sits."),
    Candidate("EV-03", "Cannabis use increases the risk of psychosis and schizophrenia.",
              "cannabis_psychosis_risk", "contested", True, False,
              "Canonical form; 15 tagged papers including a causation analysis and a withdrawal-psychosis review."),
    Candidate("EV-04", "Cannabis use causally contributes to schizophrenia rather than being explained by confounding.",
              "cannabis_psychosis_risk", "contested", True, False,
              "Isolates the causal question from the association, which is where the literature splits."),
    Candidate("EV-05", "Social media use causes increased depression and anxiety in adolescents.",
              "social_media_adolescent_mental_health", "contested", True, False,
              "Canonical form; 16 tagged papers, the largest contested cluster."),
    Candidate("EV-06", "Reducing or stopping social media use improves adolescent mental health.",
              "social_media_adolescent_mental_health", "contested", True, False,
              "The intervention side of the same debate; the corpus has a social media detox cohort study."),
    Candidate("EV-07", "Single-session psychological debriefing after trauma prevents PTSD.",
              "psychological_debriefing_ptsd_prevention", "contested", True, False,
              "Canonical form; 14 tagged papers."),
    Candidate("EV-08", "Critical incident stress debriefing improves mental health outcomes in emergency service personnel.",
              "psychological_debriefing_ptsd_prevention", "contested", True, False,
              "The corpus holds both a CISD randomised trial and a systematic review of debriefing in rescue teams."),
    Candidate("EV-09", "SSRIs are more effective than placebo for mild-to-moderate depression.",
              "ssri_mild_depression_efficacy", "contested", True, False,
              "Canonical form; 14 tagged papers."),
    Candidate("EV-10", "Antidepressant efficacy over placebo increases with baseline depression severity.",
              "ssri_mild_depression_efficacy", "contested", True, False,
              "The severity-stratification question underneath the SSRI debate; the corpus has severity-and-efficacy and placebo-response papers."),

    # --- Proposed well-supported -------------------------------------------
    Candidate("EV-11", "Mindfulness-based interventions improve core ADHD symptoms in adults.",
              "adhd", "well-supported", False, False,
              "A systematic review and meta-analysis of MBIs in adult ADHD sits in the corpus, alongside several RCTs."),
    Candidate("EV-12", "Digital mental health interventions reduce depressive symptoms.",
              "digital_mental_health", "well-supported", False, False,
              "Several meta-analyses of digital interventions for depression are indexed."),
    Candidate("EV-13", "Smartphone applications produce measurable improvements in mental health symptoms.",
              "digital_mental_health", "well-supported", False, False,
              "The corpus holds a systematic review and meta-analysis of mental health apps."),
    Candidate("EV-14", "Digital technology interventions improve mental health outcomes in older adults.",
              "digital_mental_health", "well-supported", False, False,
              "Two separate meta-analyses on older adults are indexed, including one on digital tai chi."),
    Candidate("EV-15", "Psilocybin reduces symptoms of major depressive disorder.",
              "psychedelics", "well-supported", False, False,
              "A randomised clinical trial of psilocybin in major depression plus a phase IIa psychedelic trial."),
    Candidate("EV-16", "Esketamine is effective for treatment-resistant depression.",
              "psychedelics", "well-supported", False, False,
              "A registered-report systematic review of esketamine plus an RCT of oral esketamine."),
    Candidate("EV-17", "Mindfulness-based parent training reduces ADHD-related difficulties in children.",
              "adhd", "well-supported", False, False,
              "MindChamp, an MBSR-for-parents RCT, and a mindfulness-enhanced behavioural parent training RCT are all indexed."),
    Candidate("EV-18", "Digital interventions improve mental health outcomes in patients with cancer.",
              "digital_mental_health", "well-supported", False, False,
              "A dedicated systematic review and meta-analysis is in the corpus."),
    Candidate("EV-19", "Prenatal cannabis exposure is associated with neuropsychiatric problems in offspring.",
              "cannabis", "well-supported", False, False,
              "A systematic review and meta-analysis on prenatal exposure and offspring outcomes is indexed."),
    Candidate("EV-20", "Problematic social media use is associated with depression and anxiety symptoms in adolescents.",
              "social_media", "well-supported", False, False,
              "Association rather than causation — deliberately weaker than EV-05, to see whether the two are distinguished."),
    Candidate("EV-21", "Exercise interventions improve cognitive function in people with mild cognitive impairment.",
              "exercise", "well-supported", False, False,
              "A systematic review and network meta-analysis of exercise modalities in MCI is indexed."),
    Candidate("EV-22", "Digitally delivered cognitive behavioural therapy is effective for adults with ADHD.",
              "digital_mental_health", "well-supported", False, False,
              "A randomised trial of a CBT-based digital intervention for adult ADHD is indexed."),

    # --- Proposed well-contradicted ----------------------------------------
    Candidate("EV-23", "Debriefing after childbirth prevents psychological trauma in women.",
              "ptsd", "well-contradicted", False, False,
              "The corpus holds a review of debriefing interventions for exactly this population and outcome."),
    Candidate("EV-24", "Group critical incident stress debriefing reduces PTSD symptoms in emergency services personnel.",
              "ptsd", "well-contradicted", False, False,
              "A randomised controlled trial of group CISD with emergency services personnel is indexed."),
    Candidate("EV-25", "Mindfulness for youth outperforms group cognitive behavioural therapy for childhood ADHD.",
              "adhd", "well-contradicted", False, False,
              "The MYmind-versus-CBT randomised trial is indexed and compares exactly these two arms."),
    Candidate("EV-26", "Combat and operational stress control interventions prevent PTSD in military personnel.",
              "ptsd", "well-contradicted", False, False,
              "A systematic review and meta-analysis of these interventions is indexed."),
    Candidate("EV-27", "A single session of mindfulness training produces lasting reductions in core ADHD symptoms.",
              "adhd", "well-contradicted", False, False,
              "A single-session mindfulness trial measuring cardiac vagal control and core symptoms is indexed."),
    Candidate("EV-28", "Menopausal hormone therapy has no effect on depressive symptoms in perimenopausal women.",
              "depression", "well-contradicted", False, False,
              "Deliberate inversion: the indexed meta-analysis addresses efficacy for exactly this population."),
    Candidate("EV-29", "Population increases in cannabis use have been matched by proportional increases in schizophrenia incidence.",
              "cannabis", "well-contradicted", False, False,
              "The specific epidemiological objection raised in the cannabis causation literature the corpus holds."),
    Candidate("EV-30", "Lithium augmentation has been superseded and no longer has a role in treatment-resistant depression.",
              "depression", "well-contradicted", False, False,
              "The indexed paper is a reappraisal of lithium and other augmentation strategies."),
    Candidate("EV-31", "Baseline depression severity is unrelated to placebo response in major depressive disorder.",
              "depression", "well-contradicted", False, False,
              "An indexed trial analyses episode duration and severity as predictors of placebo response."),
    Candidate("EV-32", "Virtual reality interventions have no effect on emotional outcomes or suicidal ideation.",
              "digital_mental_health", "well-contradicted", False, False,
              "A randomised trial of VR-based positive psychotherapy measuring exactly these outcomes is indexed."),

    # --- Proposed inconclusive ---------------------------------------------
    Candidate("EV-33", "Yoga reduces anxiety severity in adults.",
              "anxiety", "inconclusive", False, False,
              "One randomised trial on yoga and anxiety; likely too thin to reach the source threshold."),
    Candidate("EV-34", "Neurofeedback is an effective treatment for insomnia.",
              "sleep", "inconclusive", False, False,
              "The only indexed material is a single-case experimental study in primary care."),
    Candidate("EV-35", "A ketogenic diet affects mood symptoms in people taking antidepressants.",
              "depression", "inconclusive", False, False,
              "Represented by a single case report, which is exactly the situation the source-count floor exists for."),
    Candidate("EV-36", "Changes in sleep duration predict later cognitive impairment in older adults.",
              "sleep", "inconclusive", False, False,
              "One cohort study on this association is indexed."),
    Candidate("EV-37", "Physical activity buffers the mental health harms of problematic social media use.",
              "social_media", "inconclusive", False, False,
              "One moderation study addresses this; the moderation question is narrower than the paper's headline."),
    Candidate("EV-38", "Virtual reality nature experiences improve wellbeing in long-term care residents.",
              "digital_mental_health", "inconclusive", False, False,
              "A mixed-methods feasibility study, which by design does not establish effect."),
    Candidate("EV-39", "Post-incident psychosocial interventions in the workplace improve staff mental health.",
              "workplace", "inconclusive", False, False,
              "A systematic review of current practice is indexed, which may describe provision rather than effect."),
    Candidate("EV-40", "Irregular social rhythms increase all-cause mortality.",
              "sleep", "inconclusive", False, False,
              "One association study, and an outcome well outside the corpus's centre of gravity."),

    # --- Proposed off-corpus ------------------------------------------------
    # None of these appears in eval/results/similarity_calibration.md. Reusing
    # the claims the thresholds were fitted on would measure fit, not skill.
    Candidate("EV-41", "Tranexamic acid reduces mortality in patients with traumatic haemorrhage.",
              "off_corpus_near_miss", "off-corpus", False, True,
              "Near miss: clinical, trial-shaped, and shares the word 'trauma' with the PTSD cluster."),
    Candidate("EV-42", "Helicobacter pylori eradication lowers the incidence of gastric cancer.",
              "off_corpus_near_miss", "off-corpus", False, True,
              "Near miss: a real biomedical claim from a field the corpus does not index."),
    Candidate("EV-43", "Inhaled corticosteroids reduce exacerbation frequency in COPD.",
              "off_corpus_near_miss", "off-corpus", False, True,
              "Near miss: drug-versus-outcome phrasing closely parallel to the antidepressant claims."),
    Candidate("EV-44", "Perioperative beta-blockade reduces cardiac events in non-cardiac surgery.",
              "off_corpus_near_miss", "off-corpus", False, True,
              "Near miss: a contested clinical question, but in cardiology."),
    Candidate("EV-45", "Metformin slows the progression of diabetic retinopathy.",
              "off_corpus_near_miss", "off-corpus", False, True,
              "Near miss: endocrine, and phrased in the same intervention-outcome shape as the in-domain claims."),
    Candidate("EV-46", "Graphene oxide membranes increase the throughput of seawater desalination.",
              "off_corpus_other_domain", "off-corpus", False, True,
              "Other domain: materials science, no shared vocabulary with the corpus."),
    Candidate("EV-47", "Attention heads in transformer models specialise into syntactic roles during pretraining.",
              "off_corpus_other_domain", "off-corpus", False, True,
              "Other domain: machine learning, but shares 'attention' with the ADHD cluster, which is the trap."),
    Candidate("EV-48", "Mycorrhizal fungal networks transfer carbon between neighbouring trees.",
              "off_corpus_other_domain", "off-corpus", False, True,
              "Other domain: forest ecology."),
    Candidate("EV-49", "Playing the accordion at high altitude improves shortwave radio reception.",
              "off_corpus_nonsense", "off-corpus", False, True,
              "Nonsense: the floor case, should be rejected on similarity alone."),
    Candidate("EV-50", "Sorting laundry by colour extends the half-life of tritium.",
              "off_corpus_nonsense", "off-corpus", False, True,
              "Nonsense: the floor case."),
]

CSV_COLUMNS = [
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


def gather_papers(index: RetrievalIndex, claim: str) -> list[dict]:
    """Top distinct papers behind a claim, with their best-matching passage.

    Scores are collected but not reported. What a labeller needs is the source
    text to read; the ranking numbers are the system's view of the claim and
    would anchor the judgement being collected.
    """
    result = index.search(claim, top_k=PASSAGES_RETRIEVED)
    by_paper: dict[str, dict] = {}
    for hit in result.passages:
        entry = by_paper.get(hit.paper.paper_id)
        if entry is None:
            by_paper[hit.paper.paper_id] = {
                "paper": hit.paper,
                "passage": hit.passage,
                "passage_count": 1,
            }
        else:
            entry["passage_count"] += 1
    return list(by_paper.values())[:PAPERS_PER_CLAIM]


def write_worksheet(index: RetrievalIndex, path: Path) -> None:
    lines: list[str] = []
    add = lines.append

    add("# Candidate claims for the evaluation set")
    add("")
    add(
        "50 candidates drafted against the contents of this corpus, with the "
        "papers each one retrieves listed underneath so they can be read before "
        "labelling. Generated by `python eval/propose_candidates.py`."
    )
    add("")
    add("## How to use this")
    add("")
    add(
        "1. Read the papers under a claim. 2. Decide what they actually say. "
        "3. Put your verdict in `eval/claims.csv` under `expected_verdict`, with "
        "`expected_grade` if you want certainty scored too. 4. Run "
        "`python eval/run_evaluation.py --sample 5` whenever you want to see "
        "where things stand — unlabelled rows are skipped, so you never have to "
        "finish all 50 before measuring anything."
    )
    add("")
    add(
        "**The `slot` on each candidate is a proposal, not an answer.** It says "
        "what kind of test the claim was drafted to provide. If a claim proposed "
        "as well supported turns out to be inconclusive against this corpus, "
        "that is a finding about the corpus and the label should say so — the "
        "slot is not a target to hit."
    )
    add("")
    add(
        "No relevance scores, coverage signals, or system verdicts appear below. "
        "They are the pipeline's opinion of these claims, and a label formed "
        "after reading them would no longer be independent of what it is meant "
        "to be scoring."
    )
    add("")
    add(
        "`contested` and `off_corpus` are pre-filled in `claims.csv` from how "
        "each candidate was sourced — `contested` where the claim comes from one "
        "of the five seeded topics, `off_corpus` where it was drafted from "
        "outside mental health. Both are worth checking rather than trusting: "
        "`off_corpus` in particular decides whether a claim is scored against "
        "`NOT_COVERED_BY_CORPUS` or `EVIDENCE_INCONCLUSIVE`."
    )
    add("")

    counts: dict[str, int] = {}
    for candidate in CANDIDATES:
        counts[candidate.slot] = counts.get(candidate.slot, 0) + 1
    add("| Proposed slot | Candidates |")
    add("|---|---|")
    for slot in SLOTS:
        add(f"| {slot} | {counts.get(slot, 0)} |")
    add(f"| **total** | **{len(CANDIDATES)}** |")
    add("")

    for slot, blurb in SLOTS.items():
        members = [c for c in CANDIDATES if c.slot == slot]
        if not members:
            continue
        add(f"## {slot} ({len(members)})")
        add("")
        add(blurb)
        add("")

        for candidate in members:
            add(f"### `{candidate.claim_id}` — {candidate.claim_text}")
            add("")
            add(f"- **Topic:** `{candidate.domain_topic}`")
            add(f"- **Contested:** {'yes' if candidate.contested else 'no'} · "
                f"**Off-corpus:** {'yes' if candidate.off_corpus else 'no'}")
            add(f"- **Why this candidate:** {candidate.rationale}")
            add("")

            entries = gather_papers(index, candidate.claim_text)
            if not entries:
                add("_Nothing retrieved._")
                add("")
                continue

            if candidate.off_corpus:
                add(
                    "_Nearest material the corpus holds. If none of it is "
                    "genuinely about the claim, that supports the off-corpus "
                    "label._"
                )
                add("")

            for entry in entries:
                paper = entry["paper"]
                passage = entry["passage"]
                tier = paper.publication_tier.value if paper.publication_tier else "not classified"
                flags = []
                if paper.is_preprint:
                    flags.append("preprint, not peer reviewed")
                if paper.openalex_checked and paper.is_retracted:
                    flags.append("RETRACTED")
                flag_text = f" · _{'; '.join(flags)}_" if flags else ""
                snippet = passage.text.strip().replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400].rsplit(" ", 1)[0] + "…"

                add(f"**{paper.title}**  ")
                add(
                    f"{paper.year} · {paper.journal or 'no journal recorded'} · "
                    f"{tier} · {entry['passage_count']} passage(s) retrieved{flag_text}  "
                )
                add(f"<{paper.source_url}>")
                add("")
                add(f"> {snippet}")
                add("")
            add("---")
            add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_claims_skeleton(path: Path) -> None:
    """Write claims.csv with everything filled in except the labels."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, CSV_COLUMNS)
        writer.writeheader()
        for candidate in CANDIDATES:
            writer.writerow(
                {
                    "claim_id": candidate.claim_id,
                    "claim_text": candidate.claim_text,
                    "domain_topic": candidate.domain_topic,
                    "expected_verdict": "",
                    "expected_grade": "",
                    "contested": str(candidate.contested).lower(),
                    "off_corpus": str(candidate.off_corpus).lower(),
                    "notes": "",
                    "labeller": "",
                    "labelled_date": "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draft evaluation candidates and surface the corpus papers behind each."
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "eval" / "candidate_worksheet.md"
    )
    parser.add_argument(
        "--write-claims",
        action="store_true",
        help="Also write eval/claims.csv with the candidates filled in and "
        "expected_verdict left blank. Overwrites the file.",
    )
    parser.add_argument("--claims", type=Path, default=REPO_ROOT / "eval" / "claims.csv")
    args = parser.parse_args()

    print("Loading retrieval index...", file=sys.stderr)
    index = RetrievalIndex()
    print(f"Retrieving papers for {len(CANDIDATES)} candidates...", file=sys.stderr)

    write_worksheet(index, args.out)
    print(f"Worksheet: {args.out}", file=sys.stderr)

    if args.write_claims:
        write_claims_skeleton(args.claims)
        print(f"Claims skeleton: {args.claims} ({len(CANDIDATES)} rows, unlabelled)", file=sys.stderr)
    else:
        print(
            "claims.csv not written. Re-run with --write-claims to fill it in.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
