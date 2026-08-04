"""Enrich corpus papers with quality signals from OpenAlex.

Retraction status is the field that matters most: a claim verifier must not
present a retracted paper as evidence. DOI lookups are authoritative and are
never backfilled by title search, since a wrong match would attach retraction
and citation signals to the wrong paper. Only papers with no DOI fall back to a
strict fuzzy title match, and anything below threshold is logged as unmatched
rather than guessed.

Usage:
    python -m app.ingestion.openalex_enrich
    python -m app.ingestion.openalex_enrich --input ../data/pubmed_papers.json
    python -m app.ingestion.openalex_enrich --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.ingestion.base import MAX_RETRY_ATTEMPTS, REQUEST_TIMEOUT_SECONDS, USER_AGENT

OPENALEX_BASE = "https://api.openalex.org"
MIN_REQUEST_INTERVAL_SECONDS = 0.12  # ~8 req/sec — polite even though OpenAlex allows more
TITLE_MATCH_THRESHOLD = 0.92  # stricter than the corpus dedup threshold (0.90): a false
# match here mislabels citation/retraction data on the wrong paper, not just a duplicate.
TOP_CONCEPTS = 5

CACHE_DIR = settings.raw_dir / "openalex"
DRY_RUN_LIMIT = 5


@dataclass
class EnrichStats:
    doi_match: int = 0
    title_match: int = 0
    unmatched: list[str] = field(default_factory=list)  # "paper_id: title"
    retracted: list[str] = field(default_factory=list)  # "paper_id: title (doi)"


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _safe_filename(identifier: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in identifier)[:150]


class OpenAlexClient:
    """Rate-limited, retrying, disk-cached client for the OpenAlex works API."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.mailto = settings.OPENALEX_MAILTO
        self._last_request_at = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_at = time.monotonic()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> requests.Response | None:
        self._throttle()
        full_params = {**(params or {}), "mailto": self.mailto}
        response = self.session.get(url, params=full_params, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 404:
            return None  # not found is not a transient error — don't retry it
        response.raise_for_status()
        return response

    def get_by_doi(self, doi: str) -> dict | None:
        """Look up a work by DOI, using and populating an on-disk cache."""
        cache_path = CACHE_DIR / f"doi_{_safe_filename(doi)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached if cached.get("_found") else None

        doi_clean = doi.strip().removeprefix("https://doi.org/").removeprefix("doi:")
        response = self._get(f"{OPENALEX_BASE}/works/https://doi.org/{doi_clean}")
        if response is None:
            cache_path.write_text(json.dumps({"_found": False}), encoding="utf-8")
            return None

        data = response.json()
        data["_found"] = True
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def search_by_title(self, title: str) -> list[dict]:
        """Return up to 5 title-search candidates, using and populating a cache."""
        cache_path = CACHE_DIR / f"title_{_safe_filename(_normalize_title(title))}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        response = self._get(f"{OPENALEX_BASE}/works", params={"search": title, "per_page": 5})
        results = response.json().get("results", []) if response is not None else []
        cache_path.write_text(json.dumps(results), encoding="utf-8")
        return results


def _ensure_baseline_fields(paper: dict) -> dict:
    """Guarantee openalex_checked/is_retracted are explicit bools on disk, never
    just absent — an absent key reads as None via plain dict .get() by anyone
    not going through the Paper model's defaults, which is exactly the silent
    "missing == not retracted" ambiguity this field exists to prevent."""
    return {
        "openalex_checked": paper.get("openalex_checked", False),
        "is_retracted": paper.get("is_retracted", False),
        **paper,
    }


def _map_fields(work: dict) -> dict:
    """Map an OpenAlex work object onto the Paper model's enrichment fields."""
    open_access = work.get("open_access") or {}
    concepts = sorted(work.get("concepts") or [], key=lambda c: c.get("score", 0), reverse=True)
    institutions: list[str] = []
    for authorship in work.get("authorships") or []:
        for inst in authorship.get("institutions") or []:
            name = inst.get("display_name")
            if name and name not in institutions:
                institutions.append(name)

    return {
        "openalex_checked": True,
        "is_retracted": bool(work.get("is_retracted", False)),
        "cited_by_count": work.get("cited_by_count"),
        "is_open_access": open_access.get("is_oa"),
        "open_access_url": open_access.get("oa_url"),
        "concepts": [
            {"display_name": c["display_name"], "score": c["score"]}
            for c in concepts[:TOP_CONCEPTS]
            if c.get("display_name") is not None and c.get("score") is not None
        ],
        "referenced_works_count": len(work.get("referenced_works") or []),
        "author_institutions": institutions,
    }


def enrich_paper(paper: dict, client: OpenAlexClient, stats: EnrichStats) -> dict:
    label = f"{paper.get('paper_id', '?')}: {paper.get('title', '')[:70]}"
    doi = paper.get("doi")

    if doi:
        work = client.get_by_doi(doi)
        if work is not None:
            stats.doi_match += 1
            enriched = {**paper, **_map_fields(work)}
            if enriched["is_retracted"]:
                stats.retracted.append(f"{label} ({doi})")
            return enriched
        stats.unmatched.append(f"{label} (doi not found in OpenAlex: {doi})")
        return _ensure_baseline_fields(paper)

    title = paper.get("title", "")
    candidates = client.search_by_title(title) if title else []
    norm_title = _normalize_title(title)
    best_work, best_ratio = None, 0.0
    for candidate in candidates:
        ratio = SequenceMatcher(None, norm_title, _normalize_title(candidate.get("title") or "")).ratio()
        if ratio > best_ratio:
            best_work, best_ratio = candidate, ratio

    if best_work is not None and best_ratio >= TITLE_MATCH_THRESHOLD:
        stats.title_match += 1
        enriched = {**paper, **_map_fields(best_work)}
        if enriched["is_retracted"]:
            stats.retracted.append(f"{label} (title match, ratio={best_ratio:.2f})")
        return enriched

    stats.unmatched.append(f"{label} (no DOI; best title-match ratio={best_ratio:.2f})")
    return _ensure_baseline_fields(paper)


def run(input_path: Path, output_path: Path, limit: int | None) -> None:
    with open(input_path, encoding="utf-8") as f:
        papers = json.load(f)

    client = OpenAlexClient()
    stats = EnrichStats()
    total = len(papers) if limit is None else min(limit, len(papers))

    enriched_papers = []
    for i, paper in enumerate(papers):
        if limit is not None and i >= limit:
            enriched_papers.append(_ensure_baseline_fields(paper))
            continue
        enriched_papers.append(enrich_paper(paper, client, stats))
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  processed {i + 1}/{total}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_papers, f, indent=2, ensure_ascii=False)

    matched = stats.doi_match + stats.title_match
    attempted = matched + len(stats.unmatched)
    match_rate = matched / attempted if attempted else 0.0

    print("\n" + "=" * 70)
    print("OpenAlex enrichment summary")
    print("=" * 70)
    print(f"Processed: {total} papers -> {output_path}")
    print(f"Enriched via DOI:   {stats.doi_match}")
    print(f"Enriched via title: {stats.title_match}")
    print(f"Unmatched:          {len(stats.unmatched)}")
    print(f"Match rate:         {match_rate:.0%}")

    if stats.unmatched:
        print("\nUnmatched (logged, not guessed):")
        for line in stats.unmatched:
            print(f"  - {line}")

    print(f"\nRetracted papers found: {len(stats.retracted)}")
    if stats.retracted:
        for line in stats.retracted:
            print(f"  ** RETRACTED ** {line}")
        print(
            "\nThese are still in the output, flagged with is_retracted=True — "
            "decide whether to exclude them or keep them as a retraction-check demo."
        )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich corpus papers with OpenAlex signals.")
    parser.add_argument("--input", default=str(settings.corpus_path))
    parser.add_argument("--output", default=None, help="Defaults to overwriting --input.")
    parser.add_argument("--dry-run", action="store_true", help=f"Only process the first {DRY_RUN_LIMIT} papers.")
    args = parser.parse_args()

    input_path = Path(args.input)
    limit = DRY_RUN_LIMIT if args.dry_run else None

    if args.output:
        output_path = Path(args.output)
    elif args.dry_run:
        # Don't let a dry run partially rewrite the real corpus file.
        output_path = input_path.with_name(f"{input_path.stem}_dryrun{input_path.suffix}")
    else:
        output_path = input_path

    if args.dry_run:
        print(f"[dry-run] processing only the first {DRY_RUN_LIMIT} papers -> {output_path}")

    run(input_path, output_path, limit)


if __name__ == "__main__":
    main()
