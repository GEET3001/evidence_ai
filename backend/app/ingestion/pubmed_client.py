"""PubMed (not PMC) client via NCBI E-utilities, with evidence-tier stratification.

Distinct from pmc_scraper.py: PMC only indexes the open-access full-text
subset, while PubMed covers essentially all indexed biomedical literature
(abstract + metadata only) and is where MeSH headings and PublicationType
live. This module targets db="pubmed" for that richer, broader metadata.

Later phases weight evidence by study design (meta-analysis > RCT > cohort >
cross-sectional) and map confidence onto GRADE. A corpus skewed to one tier
makes that weighting meaningless, so this client deliberately queries across
MeSH topics x publication-type tiers rather than taking whatever a single
broad search returns, and reports per-tier fill progress as it runs.

Same access policy as pmc_scraper.py: eutils is a documented API, not a
crawlable web site, so robots.txt is not enforced (see base.BaseScraper's
module docstring). Requests self-identify via tool/email per NCBI's usage
policy. Rate limit is 3 req/sec by default, 10 req/sec with a free NCBI API
key (see .env.example for how to get one) — this client detects the key and
adjusts automatically.

Usage:
    python -m app.ingestion.pubmed_client
    python -m app.ingestion.pubmed_client --dry-run
    python -m app.ingestion.pubmed_client --target 90
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import requests

from app.config import settings
from app.ingestion.base import (
    MAX_RETRY_ATTEMPTS,
    REQUEST_TIMEOUT_SECONDS,
    BaseScraper,
)
from app.models import Paper, PublicationTier, SourceDatabase

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

RATE_LIMIT_NO_KEY = 1.0 / 3
RATE_LIMIT_WITH_KEY = 1.0 / 10

LANGUAGE_FILTER = "english[lang]"
DATE_FILTER = "2010:3000[pdat]"

TOPICS = [
    '"Depressive Disorder"[MeSH]',
    '"Anxiety Disorders"[MeSH]',
    '"Mindfulness"[MeSH]',
    '"Attention Deficit Disorder with Hyperactivity"[MeSH]',
    '"Sleep Wake Disorders"[MeSH]',
    '"Social Media"[MeSH] AND "Mental Health"[MeSH]',
    '"Cognitive Behavioral Therapy"[MeSH]',
    '"Exercise"[MeSH] AND "Mental Health"[MeSH]',
]

DEFAULT_TARGET = 90
DRY_RUN_PER_QUERY_LIMIT = 2
DEFAULT_PER_QUERY_LIMIT = 10


@dataclass
class QueryTier:
    """A group of PubMed publication-type filters that share one fill target.

    `share` is this tier's fraction of the overall --target, based on the
    requested ~15 meta-analyses/systematic reviews : 45 RCTs : 30
    observational split (1 : 3 : 2 of a 90-paper default).
    """

    name: str
    pt_filters: list[str]
    share: int
    filled: int = 0


def _build_tiers(target: int) -> list[QueryTier]:
    """Split `target` ~1:3:2 across tiers, matching the requested 15:45:30 default."""
    return [
        QueryTier("high_evidence", ['"Meta-Analysis"[pt]', '"Systematic Review"[pt]'], round(target * 15 / 90)),
        QueryTier("rct", ['"Randomized Controlled Trial"[pt]'], round(target * 45 / 90)),
        QueryTier("observational", ['"Observational Study"[pt]'], round(target * 30 / 90)),
    ]


# --- publication-type / MeSH heading -> normalised tier ---


def _map_tier(pub_types: list[str], mesh_terms: list[str]) -> PublicationTier:
    """Best-effort mapping from PubMed's own vocab to our normalised tier.

    PublicationTypeList carries "Meta-Analysis", "Systematic Review",
    "Randomized Controlled Trial", and "Case Reports" directly. It does NOT
    carry "Cohort Study" / "Case-Control Study" / "Cross-Sectional Study" —
    those only appear as MeSH headings, so observational subtype detection
    falls back to MeshHeadingList. A generic "Observational Study" pub type
    with no matching MeSH heading is tagged OTHER rather than guessed.
    """
    pt_set = {p.lower() for p in pub_types}
    mesh_set = {m.lower() for m in mesh_terms}

    if "meta-analysis" in pt_set:
        return PublicationTier.META_ANALYSIS
    if "systematic review" in pt_set:
        return PublicationTier.SYSTEMATIC_REVIEW
    if "randomized controlled trial" in pt_set:
        return PublicationTier.RCT
    if "case reports" in pt_set:
        return PublicationTier.CASE_REPORT
    if "cohort studies" in mesh_set:
        return PublicationTier.COHORT
    if "case-control studies" in mesh_set:
        return PublicationTier.CASE_CONTROL
    if "cross-sectional studies" in mesh_set:
        return PublicationTier.CROSS_SECTIONAL
    return PublicationTier.OTHER


class PubMedClient(BaseScraper):
    """Retrieves PubMed records (abstract + metadata) matching a search query."""

    source_name = "pubmed"
    # eutils is a documented JSON/XML API, not a crawlable web site — see
    # base.BaseScraper's module docstring for why robots.txt doesn't apply.
    check_robots = False

    def __init__(self) -> None:
        super().__init__()
        self.tool = settings.NCBI_TOOL
        self.email = settings.NCBI_EMAIL
        self.api_key = settings.NCBI_API_KEY
        self.min_interval = RATE_LIMIT_WITH_KEY if self.api_key else RATE_LIMIT_NO_KEY
        self._last_eutils_request_at = 0.0

    def search(self, query: str, limit: int) -> list[Paper]:
        """Search PubMed for `query`, return up to `limit` parsed papers."""
        pmids = self._esearch(query, limit)
        if not pmids:
            return []
        return self._efetch(query, pmids)

    # --- rate-limited, retrying HTTP (own throttle: rate depends on API key) ---

    def _throttle_eutils(self) -> None:
        elapsed = time.monotonic() - self._last_eutils_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_eutils_request_at = time.monotonic()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        reraise=True,
    )
    def _eutils_get(self, url: str, params: dict) -> requests.Response:
        self._throttle_eutils()
        full_params = {"tool": self.tool, "email": self.email, **params}
        if self.api_key:
            full_params["api_key"] = self.api_key
        response = self.session.get(url, params=full_params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response

    def _esearch(self, query: str, limit: int) -> list[str]:
        params = {"db": "pubmed", "retmode": "json", "retmax": limit, "term": query}
        response = self._eutils_get(f"{EUTILS_BASE}/esearch.fcgi", params)
        self.save_raw(f"esearch_{query}", response.text, "json")
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, query: str, pmids: list[str]) -> list[Paper]:
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        response = self._eutils_get(f"{EUTILS_BASE}/efetch.fcgi", params)
        self.save_raw(f"efetch_{query}", response.text, "xml")

        soup = BeautifulSoup(response.text, "xml")
        papers: list[Paper] = []
        for article in soup.find_all("PubmedArticle"):
            paper = self._parse_article(article)
            if paper is not None:
                papers.append(paper)
        return papers

    def _parse_article(self, article) -> Paper | None:
        try:
            pmid_tag = article.find("PMID")
            if pmid_tag is None or not pmid_tag.text.strip():
                return None
            pmid = pmid_tag.text.strip()

            title_tag = article.find("ArticleTitle")
            title = self.clean_text(str(title_tag), "xml") if title_tag else ""
            if not title:
                return None

            abstract_parts = []
            for abstract_text in article.find_all("AbstractText"):
                text = self.clean_text(str(abstract_text), "xml")
                if not text:
                    continue
                label = abstract_text.get("Label")
                abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(abstract_parts)
            if not abstract:
                return None

            year = self._parse_year(article)
            if year is None:
                return None

            authors = []
            for author in article.find_all("Author"):
                last = author.find("LastName")
                if not last or not last.text.strip():
                    continue
                name = last.text.strip()
                fore = author.find("ForeName") or author.find("Initials")
                if fore and fore.text.strip():
                    name = f"{fore.text.strip()} {name}"
                authors.append(name)

            journal_tag = article.find("Journal")
            journal = None
            if journal_tag is not None:
                title_el = journal_tag.find("Title") or journal_tag.find("ISOAbbreviation")
                if title_el and title_el.text.strip():
                    journal = title_el.text.strip()

            doi = None
            for id_tag in article.find_all("ArticleId"):
                if id_tag.get("IdType") == "doi" and id_tag.text.strip():
                    doi = id_tag.text.strip()
                    break
            if doi is None:
                eloc = article.find("ELocationID", {"EIdType": "doi"})
                if eloc is not None and eloc.text.strip():
                    doi = eloc.text.strip()

            pub_types = [
                pt.text.strip() for pt in article.find_all("PublicationType") if pt.text.strip()
            ]
            mesh_terms = [
                d.text.strip()
                for d in article.find_all("DescriptorName")
                if d.text.strip()
            ]
            tier = _map_tier(pub_types, mesh_terms)
            # PubMed always lists the generic "Journal Article" type alongside
            # (or before) the informative one (e.g. "Meta-Analysis") — prefer
            # the specific type for the human-readable publication_type field.
            specific_types = [pt for pt in pub_types if pt != "Journal Article"]
            publication_type = specific_types[0] if specific_types else (
                pub_types[0] if pub_types else None
            )

            return Paper(
                paper_id=f"pmid_{pmid}",
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                source_database=SourceDatabase.PUBMED,
                publication_type=publication_type,
                publication_tier=tier,
                mesh_terms=mesh_terms,
                journal=journal,
                pmid=pmid,
                is_preprint=False,
                doi=doi,
                retrieved_at=datetime.now(timezone.utc),
            )
        except Exception:
            return None

    @staticmethod
    def _parse_year(article) -> int | None:
        pub_date = article.find("PubDate")
        if pub_date is not None:
            year_tag = pub_date.find("Year")
            if year_tag and year_tag.text.strip().isdigit():
                return int(year_tag.text.strip())
            medline_date = pub_date.find("MedlineDate")
            if medline_date and medline_date.text.strip():
                match = re.search(r"\d{4}", medline_date.text)
                if match:
                    return int(match.group())
        return None


# --- stratified collection across topics x tiers ---


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _tier_bucket(tier: PublicationTier) -> str:
    """Map a paper's derived tier back to one of the three progress buckets."""
    if tier in (PublicationTier.META_ANALYSIS, PublicationTier.SYSTEMATIC_REVIEW):
        return "high_evidence"
    if tier == PublicationTier.RCT:
        return "rct"
    return "observational"


def run(target: int, per_query_limit: int) -> list[Paper]:
    client = PubMedClient()
    tiers = _build_tiers(target)
    tiers_by_name = {t.name: t for t in tiers}

    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    papers: list[Paper] = []

    for tier in tiers:
        print(f"\n=== tier: {tier.name} (target {tier.share}) ===")
        for topic in TOPICS:
            if tier.filled >= tier.share:
                break
            for pt_filter in tier.pt_filters:
                if tier.filled >= tier.share:
                    break
                query = f"({topic}) AND {pt_filter} AND {LANGUAGE_FILTER} AND {DATE_FILTER}"
                try:
                    results = client.search(query, per_query_limit)
                except Exception as exc:
                    print(f"  ERROR [{topic[:40]}... / {pt_filter}]: {exc}", file=sys.stderr)
                    continue

                added = 0
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
                    papers.append(paper)
                    bucket = _tier_bucket(paper.publication_tier or PublicationTier.OTHER)
                    tiers_by_name[bucket].filled += 1
                    added += 1

                print(
                    f"  [{pt_filter}] {topic[:45]!r}: +{added} new / {len(results)} fetched"
                )
                _print_progress(tiers)

    _write_output(papers)
    _print_progress(tiers, final=True)
    return papers


def _print_progress(tiers: list[QueryTier], final: bool = False) -> None:
    prefix = "\nFinal per-tier fill:" if final else "    progress:"
    print(prefix, "  ".join(f"{t.name}={t.filled}/{t.share}" for t in tiers))


def _write_output(papers: list[Paper]) -> None:
    out_path = settings.data_dir / "pubmed_papers.json"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump(mode="json") for p in papers], f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(papers)} papers to {out_path}")
    print("(data/corpus.json was not touched — merge pubmed_papers.json into it separately.)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified PubMed collection across MeSH topics x evidence tiers."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Total papers to collect, split ~1:3:2 across "
        "meta-analysis/systematic-review : RCT : observational (default 90).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch only 2 results per query, to exercise every tier x topic combo cheaply.",
    )
    args = parser.parse_args()

    per_query_limit = DRY_RUN_PER_QUERY_LIMIT if args.dry_run else DEFAULT_PER_QUERY_LIMIT
    target = args.target
    if args.dry_run:
        print(f"[dry-run] limit={DRY_RUN_PER_QUERY_LIMIT} per query, target={target}")

    run(target, per_query_limit)


if __name__ == "__main__":
    main()
