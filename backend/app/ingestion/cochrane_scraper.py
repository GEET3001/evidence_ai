"""Cochrane Library systematic review abstract scraper.

Search is attempted as static HTML; pages that turn out to need JS rendering
are logged to data/raw/needs_js.txt and skipped. Parsed records are tagged
publication_type="systematic_review" and is_preprint=False, since that is what
the Cochrane Database of Systematic Reviews publishes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bs4 import BeautifulSoup

from app.ingestion.base import BaseScraper
from app.models import Paper, SourceDatabase

COCHRANE_SEARCH_URL = "https://www.cochranelibrary.com/search"


class CochraneScraper(BaseScraper):
    """Retrieves Cochrane systematic review abstracts matching a query."""

    source_name = "cochrane"
    check_robots = True
    RESULT_SELECTOR = "div.search-result-item, li.search-results-item"

    def search(self, query: str, limit: int) -> list[Paper]:
        url = COCHRANE_SEARCH_URL
        try:
            response = self._get(url, params={"q": query})
        except Exception as exc:
            self.log_needs_js(f"{url}?q={query}", note=f"fetch failed: {exc}")
            return []

        self.save_raw(query, response.text, "html")
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select(self.RESULT_SELECTOR)
        if not results:
            self.log_needs_js(f"{url}?q={query}", note="0 result blocks found in static HTML")
            return []

        papers: list[Paper] = []
        for result in results[:limit]:
            paper = self._parse_result(result)
            if paper is not None:
                papers.append(paper)
        return papers

    def _parse_result(self, result) -> Paper | None:
        try:
            title_tag = result.select_one("a.result-title, h3 a")
            title = self.clean_text(str(title_tag)) if title_tag else ""
            if not title:
                return None

            href = title_tag.get("href") if title_tag else None
            if not href:
                return None
            source_url = (
                href if href.startswith("http") else f"https://www.cochranelibrary.com{href}"
            )

            doi = None
            if "/doi/" in source_url:
                doi = source_url.split("/doi/", 1)[1].split("/full")[0].split("?")[0]

            authors_tag = result.select_one(".search-result-authors, .authors")
            authors: list[str] = []
            if authors_tag:
                authors = [
                    a.strip() for a in self.clean_text(str(authors_tag)).split(",") if a.strip()
                ]

            year = None
            date_tag = result.select_one(".search-result-date, .publish-date")
            if date_tag:
                digits = "".join(c for c in self.clean_text(str(date_tag)) if c.isdigit())
                if len(digits) >= 4:
                    year = int(digits[-4:])
            if year is None:
                return None

            abstract_tag = result.select_one(".search-result-abstract, .abstract")
            abstract = self.clean_text(str(abstract_tag)) if abstract_tag else ""
            if not abstract:
                return None

            paper_id = "cochrane_" + source_url.rstrip("/").rsplit("/", 1)[-1]

            return Paper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                source_url=source_url,
                source_database=SourceDatabase.COCHRANE,
                publication_type="systematic_review",
                is_preprint=False,
                doi=doi,
                retrieved_at=datetime.now(timezone.utc),
            )
        except Exception:
            return None
