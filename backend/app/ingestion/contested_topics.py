"""Targeted retrieval for claims where the literature genuinely disagrees, so
the corpus holds real material for the CONFLICTING verdict.

Each claim below is queried twice against the same population and intervention:
once with terms oriented toward a positive effect, once toward null findings.
That is a retrieval strategy rather than a label, so papers are tagged with
`contested_topic` but never with an assumed side.

Usage:
    python -m app.ingestion.contested_topics
    python -m app.ingestion.contested_topics --per-query-limit 6
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from app.config import settings
from app.ingestion.pubmed_client import DATE_FILTER, LANGUAGE_FILTER, PubMedClient
from app.models import Paper

DEFAULT_PER_QUERY_LIMIT = 8


@dataclass
class ContestedClaim:
    slug: str
    claim: str
    disagreement: str
    base_query: str
    effect_query_extra: str
    null_query_extra: str


CONTESTED_CLAIMS: list[ContestedClaim] = [
    ContestedClaim(
        slug="ssri_mild_depression_efficacy",
        claim="SSRIs are more effective than placebo for mild-to-moderate depression.",
        disagreement=(
            "Patient-level meta-analyses (Kirsch 2008; Fournier et al. 2010, JAMA) "
            "found the SSRI-vs-placebo effect falls below the threshold for "
            "clinical significance in mild-to-moderate depression, rising mainly "
            "at severe symptom levels. Larger network meta-analyses (Cipriani et "
            "al. 2018, Lancet) found all antidepressants statistically superior "
            "to placebo across severities. The debate concerns clinical "
            "meaningfulness of the effect size and publication bias in the "
            "underlying efficacy literature (Turner et al. 2008, NEJM)."
        ),
        # Broader SSRI+depression MeSH terms surfaced generic current antidepressant
        # literature, not papers actually engaging the severity-stratified
        # clinical-significance debate — narrowed to tiab terms specific to that
        # debate (network meta-analyses / severity-stratified efficacy vs the
        # clinical-significance / publication-bias critique).
        base_query='"Antidepressive Agents"[MeSH] AND placebo[tiab]',
        effect_query_extra='(("network meta-analysis"[tiab] OR efficacy[tiab]) AND severity[tiab])',
        null_query_extra=(
            '("clinical significance"[tiab] OR "publication bias"[tiab] '
            'OR "mild to moderate"[tiab])'
        ),
    ),
    ContestedClaim(
        slug="social_media_adolescent_mental_health",
        claim="Social media use causes increased depression/anxiety in adolescents.",
        disagreement=(
            "Some researchers (Twenge, Haidt) argue rising adolescent social "
            "media/smartphone use is a substantial contributor to increases in "
            "teen depression, anxiety, and self-harm. Others (Orben & Przybylski "
            "2019, Nature Human Behaviour; Odgers) argue effect sizes in large "
            "representative datasets are tiny and causality is not established, "
            "criticizing the methodology of studies claiming a strong effect."
        ),
        base_query=(
            '"Social Media"[MeSH] AND "Adolescent"[MeSH] '
            'AND ("Depression"[MeSH] OR "Anxiety"[MeSH])'
        ),
        effect_query_extra='(increase*[tiab] OR harmful[tiab] OR "associated with"[tiab])',
        null_query_extra=(
            '("no association"[tiab] OR "not associated"[tiab] OR "small effect"[tiab] '
            'OR "no significant"[tiab])'
        ),
    ),
    ContestedClaim(
        slug="psychological_debriefing_ptsd_prevention",
        claim="Single-session psychological debriefing after trauma prevents PTSD.",
        disagreement=(
            "Debriefing (e.g. Critical Incident Stress Debriefing) was widely "
            "adopted on the premise that processing a traumatic memory soon "
            "after the event reduces later PTSD risk. Cochrane systematic "
            "reviews (Rose, Bisson et al.) found no evidence it reduces PTSD "
            "symptoms, and some individual RCTs (Mayou et al. 2000, BMJ) found "
            "signs it may worsen outcomes for some individuals. Debate continues "
            "over specific protocols and populations."
        ),
        base_query=(
            '(debriefing[tiab] OR "critical incident stress"[tiab]) '
            'AND "Stress Disorders, Post-Traumatic"[MeSH]'
        ),
        effect_query_extra='(effective[tiab] OR reduce*[tiab] OR beneficial[tiab])',
        null_query_extra=(
            '("no benefit"[tiab] OR "not effective"[tiab] OR "no significant"[tiab] '
            'OR harmful[tiab])'
        ),
    ),
    ContestedClaim(
        slug="cannabis_psychosis_risk",
        claim="Cannabis use increases the risk of psychosis/schizophrenia.",
        disagreement=(
            "Longitudinal cohort studies (e.g. Dunedin study; Swedish conscript "
            "cohort) show a dose-response association between cannabis use "
            "(especially adolescent, high-THC) and later psychosis/schizophrenia "
            "diagnoses, read by some as evidence of a causal contribution. "
            "Critics argue reverse causation (prodromal symptoms driving "
            "cannabis use as self-medication) and unmeasured genetic/"
            "environmental confounding cannot be excluded, and note population "
            "cannabis-use increases haven't been matched by proportional rises "
            "in schizophrenia incidence."
        ),
        base_query='"Cannabis"[MeSH] AND ("Psychotic Disorders"[MeSH] OR "Schizophrenia"[MeSH])',
        effect_query_extra='(risk[tiab] AND (increase*[tiab] OR associated[tiab]))',
        null_query_extra=(
            '("no association"[tiab] OR confound*[tiab] OR "reverse causation"[tiab] '
            'OR "not causal"[tiab])'
        ),
    ),
    ContestedClaim(
        slug="mindfulness_adhd_symptoms",
        claim="Mindfulness-based interventions significantly reduce ADHD symptoms.",
        disagreement=(
            "Several RCTs and meta-analyses report significant reductions in "
            "inattention/hyperactivity symptoms in youth and adults with ADHD "
            "following mindfulness training. Other RCTs comparing mindfulness "
            "to an active control or treatment-as-usual find no statistically "
            "significant advantage, and reviews flag small sample sizes and "
            "high risk of bias in the positive studies."
        ),
        base_query=(
            '"Mindfulness"[MeSH] AND "Attention Deficit Disorder with Hyperactivity"[MeSH]'
        ),
        effect_query_extra='(effective[tiab] OR improv*[tiab] OR reduce*[tiab])',
        null_query_extra='("no significant"[tiab] OR "not superior"[tiab] OR "no effect"[tiab])',
    ),
]


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _load_existing() -> list[dict]:
    path = settings.data_dir / "pubmed_papers.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(papers: list[dict]) -> None:
    path = settings.data_dir / "pubmed_papers.json"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(papers)} total papers to {path}")


def run(per_query_limit: int) -> None:
    client = PubMedClient()
    existing = _load_existing()
    seen_dois = {p["doi"].strip().lower() for p in existing if p.get("doi")}
    seen_titles = {_normalize_title(p.get("title", "")) for p in existing}

    all_papers: list[Paper] = []
    report_lines: list[str] = []

    for claim in CONTESTED_CLAIMS:
        print(f"\n=== {claim.slug} ===")
        print(f"claim: {claim.claim}")
        side_results: dict[str, list[Paper]] = {"effect_query": [], "null_query": []}

        for side, extra in (("effect_query", claim.effect_query_extra), ("null_query", claim.null_query_extra)):
            query = f"({claim.base_query}) AND {extra} AND {LANGUAGE_FILTER} AND {DATE_FILTER}"
            try:
                results = client.search(query, per_query_limit)
            except Exception as exc:
                print(f"  ERROR [{side}]: {exc}", file=sys.stderr)
                continue

            kept = []
            for paper in results:
                doi_key = paper.doi.strip().lower() if paper.doi else None
                norm_title = _normalize_title(paper.title)
                if doi_key and doi_key in seen_dois:
                    continue
                if norm_title in seen_titles:
                    continue
                if doi_key:
                    seen_dois.add(doi_key)
                seen_titles.add(norm_title)
                tagged = paper.model_copy(update={"contested_topic": claim.slug})
                kept.append(tagged)

            side_results[side] = kept
            all_papers.extend(kept)
            print(f"  [{side}] +{len(kept)} new / {len(results)} fetched")

        report_lines.append(
            f"{claim.slug}: effect_query={len(side_results['effect_query'])} new, "
            f"null_query={len(side_results['null_query'])} new"
        )
        if not side_results["effect_query"] or not side_results["null_query"]:
            report_lines.append(f"  FLAG: only one side of this debate was found in fresh results.")

    merged = existing + [p.model_dump(mode="json") for p in all_papers]
    _write(merged)

    print("\n" + "=" * 70)
    print(f"Added {len(all_papers)} new contested-topic papers this run.")
    for line in report_lines:
        print(line)
    print("=" * 70)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Targeted retrieval for contested mental-health claims.")
    parser.add_argument("--per-query-limit", type=int, default=DEFAULT_PER_QUERY_LIMIT)
    args = parser.parse_args()
    run(args.per_query_limit)


if __name__ == "__main__":
    main()
