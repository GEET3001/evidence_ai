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

MIN_TOTAL_PAPERS = 50
SHORT_ABSTRACT_THRESHOLD_CHARS = 200
FUZZY_TITLE_MATCH_THRESHOLD = 0.90


@dataclass
class ValidationReport:
    lines: list[str] = field(default_factory=list)
    passed: bool = True

    def line(self, text: str = "") -> None:
        self.lines.append(text)

    def fail(self, reason: str) -> None:
        self.passed = False
        self.lines.append(f"  FAIL: {reason}")

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
    r.line("-- Papers per source_database --")
    source_counts = Counter(p.get("source_database", "MISSING") for p in papers)
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
    r.line(f"  preprint (not peer reviewed): {preprint_count}")
    r.line(f"  peer-reviewed:                {peer_reviewed_count}")
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
