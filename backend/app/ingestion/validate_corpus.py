"""Validate data/corpus.json and report data-quality metrics.

Usage:
    python -m app.ingestion.validate_corpus

Prints a readable report to stdout and writes the same report to
eval/results/corpus_validation.txt (evidence for the final report).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.config import settings

MIN_TOTAL_PAPERS = 120
SHORT_ABSTRACT_THRESHOLD_CHARS = 200
FUZZY_TITLE_MATCH_THRESHOLD = 0.90

MIN_TIERS_REPRESENTED = 3  # hard PASS/FAIL gate: distinct evidence tiers with >=1 paper
TIER_FLAG_THRESHOLD = 10  # informational flag only, not a PASS/FAIL gate

TOPIC_FLAG_THRESHOLD = 10  # informational flag only
TOPIC_FAIL_THRESHOLD = 8  # hard PASS/FAIL gate

# Topic coverage is estimated from title/abstract/MeSH keyword matching, NOT from
# ingestion-time provenance — no scraper in this project tags which search query
# found a given paper, and pmc_scraper/preprint_scraper (Day 1) used a different,
# free-text topic list than pubmed_client's MeSH topics (Day 2) anyway. This is
# a content classification of what the corpus actually covers, not a record of
# how each paper was found. Keep that distinction in mind when reading it.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "depression": [r"depress"],
    "anxiety": [r"anxiety", r"anxious"],
    "mindfulness": [r"mindful"],
    "adhd": [r"adhd", r"attention.deficit", r"hyperactivity"],
    "sleep": [r"sleep", r"insomnia"],
    "social_media_mental_health": [r"social media", r"smartphone", r"screen time"],
    "cbt": [r"cognitive.behaviora?l therapy", r"\bcbt\b"],
    "exercise_mental_health": [r"exercise", r"physical activity"],
    "digital_mental_health": [
        r"digital (mental health|intervention)", r"internet-based intervention",
        r"telemedicine", r"telehealth", r"mhealth", r"\bapp-based\b",
    ],
}


@dataclass
class ValidationReport:
    lines: list[str] = field(default_factory=list)
    passed: bool = True

    def line(self, text: str = "") -> None:
        self.lines.append(text)

    def fail(self, reason: str) -> None:
        self.passed = False
        self.lines.append(f"  FAIL: {reason}")

    def flag(self, reason: str) -> None:
        """Informational only — does not affect PASS/FAIL."""
        self.lines.append(f"  FLAG: {reason}")

    def render(self) -> str:
        return "\n".join(self.lines)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def load_corpus() -> list[dict]:
    if not settings.corpus_path.exists():
        raise FileNotFoundError(f"No corpus found at {settings.corpus_path}")
    with open(settings.corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{settings.corpus_path} does not contain a JSON list")
    return data


def find_fuzzy_duplicates(papers: list[dict]) -> list[tuple[str, str, float]]:
    """Return (title_a, title_b, ratio) for title pairs that look like duplicates
    but weren't caught by exact-normalised-title dedup at ingestion time."""
    normalized = [(p.get("paper_id", "?"), _normalize_title(p.get("title", ""))) for p in papers]
    matches: list[tuple[str, str, float]] = []
    for i in range(len(normalized)):
        id_a, title_a = normalized[i]
        if not title_a:
            continue
        for j in range(i + 1, len(normalized)):
            id_b, title_b = normalized[j]
            if not title_b or title_a == title_b:
                continue
            ratio = SequenceMatcher(None, title_a, title_b).ratio()
            if ratio >= FUZZY_TITLE_MATCH_THRESHOLD:
                matches.append((f"{id_a}: {title_a[:70]}", f"{id_b}: {title_b[:70]}", ratio))
    return matches


def _paper_text(paper: dict) -> str:
    return " ".join(
        [
            paper.get("title") or "",
            paper.get("abstract") or "",
            " ".join(paper.get("mesh_terms") or []),
        ]
    ).lower()


def classify_topics(papers: list[dict]) -> dict[str, int]:
    """Estimate topic coverage via keyword matching — see TOPIC_KEYWORDS docstring
    note above. A paper may match zero, one, or several topics."""
    counts = {topic: 0 for topic in TOPIC_KEYWORDS}
    for paper in papers:
        text = _paper_text(paper)
        for topic, patterns in TOPIC_KEYWORDS.items():
            if any(re.search(pattern, text) for pattern in patterns):
                counts[topic] += 1
    return counts


def validate(papers: list[dict]) -> ValidationReport:
    r = ValidationReport()
    total = len(papers)

    r.line("=" * 70)
    r.line("EvidenceAI Corpus Validation Report")
    r.line("=" * 70)
    r.line(f"Corpus file: {settings.corpus_path}")
    r.line(f"Total papers: {total}")
    r.line()

    # --- per-source counts ---
    r.line("-- Papers per source_database (counts every source a merged paper was found in) --")
    source_counts: Counter = Counter()
    for p in papers:
        sources = p.get("all_source_databases") or (
            [p["source_database"]] if p.get("source_database") else ["MISSING"]
        )
        source_counts.update(sources)
    for source, count in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        r.line(f"  {source:<12} {count}")
    r.line()

    # --- missing fields ---
    r.line("-- Missing required fields --")
    missing_abstract = [p for p in papers if not (p.get("abstract") or "").strip()]
    missing_authors = [p for p in papers if not p.get("authors")]
    missing_year = [p for p in papers if not p.get("year")]
    missing_doi = [p for p in papers if not p.get("doi")]
    r.line(f"  missing abstract: {len(missing_abstract)}")
    r.line(f"  missing authors:  {len(missing_authors)}")
    r.line(f"  missing year:     {len(missing_year)}")
    r.line(f"  missing DOI:      {len(missing_doi)} (expected for many preprints/sources)")
    r.line()

    # --- abstract length distribution ---
    r.line("-- Abstract length distribution (characters) --")
    lengths = sorted(len((p.get("abstract") or "")) for p in papers)
    short_abstracts = [p for p in papers if len(p.get("abstract") or "") < SHORT_ABSTRACT_THRESHOLD_CHARS]
    if lengths:
        r.line(f"  min: {lengths[0]}  max: {lengths[-1]}  "
               f"median: {lengths[len(lengths) // 2]}  "
               f"mean: {sum(lengths) / len(lengths):.0f}")
    r.line(f"  flagged as likely truncated/failed parse (< {SHORT_ABSTRACT_THRESHOLD_CHARS} chars): "
           f"{len(short_abstracts)}")
    for p in short_abstracts:
        r.line(f"    - {p.get('paper_id', '?')}: {len(p.get('abstract') or '')} chars "
               f"({p.get('source_database', '?')})")
    r.line()

    # --- year distribution ---
    r.line("-- Year distribution --")
    years = [p.get("year") for p in papers if p.get("year")]
    if years:
        r.line(f"  range: {min(years)}-{max(years)}")
        year_counts = Counter(years)
        for year in sorted(year_counts):
            r.line(f"    {year}: {'#' * year_counts[year]} ({year_counts[year]})")
    r.line()

    # --- preprint vs peer-reviewed ---
    r.line("-- Preprint vs peer-reviewed --")
    preprint_count = sum(1 for p in papers if p.get("is_preprint"))
    peer_reviewed_count = total - preprint_count
    preprint_pct = preprint_count / total if total else 0
    r.line(f"  preprint (not peer reviewed): {preprint_count} ({preprint_pct:.0%})")
    r.line(f"  peer-reviewed:                {peer_reviewed_count} ({1 - preprint_pct:.0%})")
    r.line()

    # --- evidence tier distribution ---
    r.line(f"-- Evidence tier distribution (flag if any tier has < {TIER_FLAG_THRESHOLD}) --")
    tier_counts = Counter(p.get("publication_tier") or "UNCLASSIFIED" for p in papers)
    tiers_represented = [t for t in tier_counts if t != "UNCLASSIFIED"]
    for tier, count in sorted(tier_counts.items(), key=lambda kv: -kv[1]):
        r.line(f"  {tier:<20} {count}")
        if tier != "UNCLASSIFIED" and count < TIER_FLAG_THRESHOLD:
            r.flag(f"tier {tier} has only {count} papers (< {TIER_FLAG_THRESHOLD})")
    r.line(f"  ({len(tiers_represented)} distinct tiers represented, "
           f"excluding UNCLASSIFIED — UNCLASSIFIED papers predate publication_tier "
           f"or came from a source that doesn't populate it, e.g. Day 1 PMC/PsyArXiv)")
    r.line()

    # --- topic distribution (estimated — see TOPIC_KEYWORDS module docstring) ---
    r.line(f"-- Estimated topic coverage (keyword-based content classification, NOT "
           f"ingestion provenance — flag if < {TOPIC_FLAG_THRESHOLD}) --")
    topic_counts = classify_topics(papers)
    for topic, count in sorted(topic_counts.items(), key=lambda kv: -kv[1]):
        r.line(f"  {topic:<28} {count}")
        if count < TOPIC_FLAG_THRESHOLD:
            r.flag(f"topic {topic} has only {count} matching papers (< {TOPIC_FLAG_THRESHOLD})")
    r.line()

    # --- publication_type population rate ---
    r.line("-- publication_type population --")
    with_pub_type = [p for p in papers if p.get("publication_type")]
    r.line(f"  overall: {len(with_pub_type)}/{total} ({len(with_pub_type) / total:.0%})" if total else "  n/a")
    pubmed_papers = [
        p for p in papers
        if "pubmed" in (p.get("all_source_databases") or [p.get("source_database")])
    ]
    pubmed_with_type = [p for p in pubmed_papers if p.get("publication_type")]
    if pubmed_papers:
        r.line(f"  PubMed-sourced: {len(pubmed_with_type)}/{len(pubmed_papers)} "
               f"({len(pubmed_with_type) / len(pubmed_papers):.0%})")
        if len(pubmed_with_type) / len(pubmed_papers) < 0.95:
            r.flag("PubMed-sourced papers should be near 100% populated for publication_type")
    r.line()

    # --- retraction status ---
    r.line("-- Retraction status (OpenAlex) --")
    checked = [p for p in papers if p.get("openalex_checked")]
    retracted = [p for p in papers if p.get("is_retracted")]
    r.line(f"  checked against OpenAlex: {len(checked)}/{total}")
    r.line(f"  retracted:                {len(retracted)}")
    for p in retracted:
        r.line(f"    ** {p.get('paper_id', '?')}: {p.get('title', '')[:70]}")
    r.line()

    # --- contested-topic coverage ---
    r.line("-- Contested-topic coverage --")
    contested_counts = Counter(p.get("contested_topic") for p in papers if p.get("contested_topic"))
    if contested_counts:
        for slug, count in sorted(contested_counts.items()):
            r.line(f"  {slug:<40} {count} papers")
            if count < 2:
                r.flag(f"{slug} has only {count} paper(s) — can't represent both sides of a debate with fewer than 2")
        r.line("  (paper count only — whether both sides are ACTUALLY represented "
               "requires reading abstracts or the stance classifier; not re-verified "
               "automatically here to avoid asserting an unread paper's stance)")
    else:
        r.line("  none found")
    r.line()

    # --- suspected duplicates ---
    r.line(f"-- Suspected duplicates (fuzzy title match >= {FUZZY_TITLE_MATCH_THRESHOLD:.0%}) --")
    fuzzy_dupes = find_fuzzy_duplicates(papers)
    if fuzzy_dupes:
        for title_a, title_b, ratio in fuzzy_dupes:
            r.line(f"  {ratio:.0%}  {title_a}")
            r.line(f"        {title_b}")
    else:
        r.line("  none found")
    r.line()

    # --- PASS/FAIL verdict ---
    r.line("-- PASS/FAIL --")
    missing_url = [p for p in papers if not (p.get("source_url") or "").strip()]

    if total < MIN_TOTAL_PAPERS:
        r.fail(f"only {total} papers, need >= {MIN_TOTAL_PAPERS}")
    else:
        r.line(f"  OK: {total} papers >= {MIN_TOTAL_PAPERS}")

    if missing_abstract:
        r.fail(f"{len(missing_abstract)} papers have an empty abstract")
    else:
        r.line("  OK: all papers have a non-empty abstract")

    if missing_url:
        r.fail(f"{len(missing_url)} papers have no source_url")
    else:
        r.line("  OK: all papers have a source_url")

    if len(tiers_represented) < MIN_TIERS_REPRESENTED:
        r.fail(f"only {len(tiers_represented)} evidence tiers represented, "
               f"need >= {MIN_TIERS_REPRESENTED}")
    else:
        r.line(f"  OK: {len(tiers_represented)} evidence tiers represented "
               f"(>= {MIN_TIERS_REPRESENTED})")

    thin_topics = {t: c for t, c in topic_counts.items() if c < TOPIC_FAIL_THRESHOLD}
    if thin_topics:
        r.fail(f"topics below {TOPIC_FAIL_THRESHOLD} papers: "
               f"{', '.join(f'{t}={c}' for t, c in thin_topics.items())}")
    else:
        r.line(f"  OK: no topic has fewer than {TOPIC_FAIL_THRESHOLD} papers")

    r.line()
    r.line(f"VERDICT: {'PASS' if r.passed else 'FAIL'}")
    r.line("=" * 70)
    return r


def main() -> None:
    papers = load_corpus()
    report = validate(papers)
    print(report.render())

    output_path = settings.data_dir.parent / "eval" / "results" / "corpus_validation.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.render() + "\n", encoding="utf-8")
    print(f"\nReport written to {output_path}")

    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
