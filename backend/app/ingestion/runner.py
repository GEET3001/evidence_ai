"""CLI to run all ingestion scrapers, dedupe, and write data/corpus.json.

Usage:
    python -m app.ingestion.runner
    python -m app.ingestion.runner --dry-run
    python -m app.ingestion.runner --topics "cbt depression" "adhd medication"

"attempted" / "succeeded" / "failed" / "skipped-needs-js" in the summary
count (topic, source) combinations, not individual papers: attempted is
every combination tried; succeeded returned at least one paper; failed
raised an exception; skipped-needs-js logged an entry to needs_js.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

from app.config import settings
from app.ingestion.cochrane_scraper import CochraneScraper
from app.ingestion.pmc_scraper import PMCScraper
from app.ingestion.preprint_scraper import MedrxivScraper, PsyArxivScraper
from app.models import Paper

DEFAULT_TOPICS = [
    "cognitive behavioral therapy depression",
    "mindfulness anxiety students",
    "social media use adolescent mental health",
    "ADHD medication efficacy",
    "sleep quality depression",
    "exercise intervention anxiety",
    "digital mental health interventions",
]

DEFAULT_MAX_PAPERS = 70
DEFAULT_PER_QUERY_LIMIT = 8
DRY_RUN_LIMIT = 2
DRY_RUN_TOPIC_COUNT = 2


@dataclass
class SourceStats:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_needs_js: int = 0


def _normalize_title(title: str) -> str:
    """Lowercase and strip punctuation/whitespace for duplicate detection."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _count_lines(path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def run(topics: list[str], per_query_limit: int, max_papers: int) -> list[Paper]:
    scrapers = {
        "pmc": PMCScraper(),
        "psyarxiv": PsyArxivScraper(),
        "medrxiv": MedrxivScraper(),
        "cochrane": CochraneScraper(),
    }
    stats = {name: SourceStats() for name in scrapers}

    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    papers: list[Paper] = []

    needs_js_path = settings.raw_dir / "needs_js.txt"

    for topic in topics:
        print(f"\n=== topic: {topic!r} ===")
        for name, scraper in scrapers.items():
            stats[name].attempted += 1
            lines_before = _count_lines(needs_js_path)
            try:
                results = scraper.search(topic, per_query_limit)
            except Exception as exc:
                print(f"  [{name}] ERROR: {exc}", file=sys.stderr)
                stats[name].failed += 1
                continue

            if _count_lines(needs_js_path) > lines_before:
                stats[name].skipped_needs_js += 1

            if not results:
                print(f"  [{name}] 0 results")
                continue
            stats[name].succeeded += 1

            added = 0
            for paper in results:
                norm_title = _normalize_title(paper.title)
                doi_key = paper.doi.strip().lower() if paper.doi else None

                if doi_key and doi_key in seen_dois:
                    continue
                if norm_title in seen_titles:
                    continue

                if doi_key:
                    seen_dois.add(doi_key)
                seen_titles.add(norm_title)
                papers.append(paper)
                added += 1

            print(
                f"  [{name}] +{added} new / {len(results)} fetched "
                f"-> running total: {len(papers)}"
            )

            if len(papers) >= max_papers:
                print(f"\nReached target of {max_papers} papers — stopping early.")
                _write_corpus(papers)
                _print_summary(stats)
                return papers

    _write_corpus(papers)
    _print_summary(stats)
    return papers


def _write_corpus(papers: list[Paper]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.corpus_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump(mode="json") for p in papers], f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(papers)} papers to {settings.corpus_path}")


def _print_summary(stats: dict[str, SourceStats]) -> None:
    print("\nPer-source summary (counts are topic x source attempts):")
    print(f"{'source':<10} {'attempted':>10} {'succeeded':>10} {'failed':>8} {'skip-js':>8}")
    for name, s in stats.items():
        print(f"{name:<10} {s.attempted:>10} {s.succeeded:>10} {s.failed:>8} {s.skipped_needs_js:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvidenceAI paper ingestion scrapers.")
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    parser.add_argument("--per-query-limit", type=int, default=DEFAULT_PER_QUERY_LIMIT)
    parser.add_argument("--max-papers", type=int, default=DEFAULT_MAX_PAPERS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch only 2 papers per source from 2 topics, for testing.",
    )
    args = parser.parse_args()

    topics = args.topics
    per_query_limit = args.per_query_limit
    max_papers = args.max_papers
    if args.dry_run:
        topics = topics[:DRY_RUN_TOPIC_COUNT]
        per_query_limit = DRY_RUN_LIMIT
        max_papers = 10_000  # let every source get exercised; don't stop early
        print(f"[dry-run] {len(topics)} topics, limit={DRY_RUN_LIMIT} per source")

    run(topics, per_query_limit, max_papers)


if __name__ == "__main__":
    main()
