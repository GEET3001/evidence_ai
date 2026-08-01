"""PsyArXiv (via the OSF API) and medRxiv (static HTML) preprint scrapers.

Every record from these sources has is_preprint=True: preprints are not
peer reviewed and must be flagged as such downstream.
"""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote

from bs4 import BeautifulSoup

from app.ingestion.base import BaseScraper
from app.models import Paper, SourceDatabase

OSF_PREPRINTS_URL = "https://api.osf.io/v2/preprints/"
MEDRXIV_SEARCH_URL = "https://www.medrxiv.org/search/{query}"

# A hit's title+description must contain at least one query term this long
# or longer to survive client-side relevance filtering (see PsyArxivScraper).
MIN_RELEVANCE_TERM_LENGTH = 4


class PsyArxivScraper(BaseScraper):
    """Retrieves PsyArXiv preprints via the public OSF v2 REST API (JSON).

    PsyArXiv is hosted on OSF, which publishes a documented, versioned
    REST API — used here instead of scraping the (JS-rendered) OSF search
    UI. check_robots is off for the same reason as PMC's eutils: this is
    a dedicated API host, not a crawlable web site.
    """

    source_name = "psyarxiv"
    check_robots = False

    def search(self, query: str, limit: int) -> list[Paper]:
        params = {
            "filter[provider]": "psyarxiv",
            "q": query,
            "page[size]": limit,
            "embed": "contributors",
        }
        response = self._get(OSF_PREPRINTS_URL, params=params)
        self.save_raw(query, response.text, "json")
        data = response.json()

        query_terms = {t.lower() for t in query.split() if len(t) >= MIN_RELEVANCE_TERM_LENGTH}
        papers: list[Paper] = []
        for item in data.get("data", []):
            paper = self._parse_preprint(item, query_terms)
            if paper is not None:
                papers.append(paper)
            if len(papers) >= limit:
                break
        return papers

    def _parse_preprint(self, item: dict, query_terms: set[str]) -> Paper | None:
        try:
            attrs = item.get("attributes", {})
            title = (attrs.get("title") or "").strip()
            abstract = (attrs.get("description") or "").strip()
            if not title or not abstract:
                return None

            # OSF's top-level `q` search is not guaranteed to be strictly
            # precision-filtered server-side; drop obviously irrelevant
            # hits client-side rather than trust it blindly.
            haystack = f"{title} {abstract}".lower()
            if query_terms and not any(term in haystack for term in query_terms):
                return None

            date_published = attrs.get("date_published")
            if not date_published or len(date_published) < 4:
                return None
            year = int(date_published[:4])

            authors = []
            contrib_data = item.get("embeds", {}).get("contributors", {}).get("data", [])
            for contrib in contrib_data:
                user = contrib.get("embeds", {}).get("users", {}).get("data", {})
                full_name = user.get("attributes", {}).get("full_name")
                if full_name:
                    authors.append(full_name)

            preprint_id = item.get("id", "")
            links = item.get("links", {})
            source_url = links.get("html") or links.get("self") or f"https://osf.io/{preprint_id}"
            doi = attrs.get("doi") or None

            return Paper(
                paper_id=f"psyarxiv_{preprint_id}",
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                source_url=source_url,
                source_database=SourceDatabase.PSYARXIV,
                publication_type=None,
                is_preprint=True,
                doi=doi,
                retrieved_at=datetime.now(timezone.utc),
            )
        except Exception:
            return None


class MedrxivScraper(BaseScraper):
    """Retrieves medRxiv preprints by scraping static search-result HTML.

    medRxiv's search results have historically been server-rendered
    (Highwire/Silverchair platform). This is attempted as static HTML
    first; if the page turns out to require JS rendering — a successful
    fetch that yields zero recognisable result blocks, or a fetch that
    fails outright (e.g. bot-protection blocking non-browser clients) —
    the search URL is logged to data/raw/needs_js.txt and skipped rather
    than adding a Selenium fallback (explicitly out of scope for now).
    """

    source_name = "medrxiv"
    check_robots = True
    RESULT_SELECTOR = "li.search-result, div.highwire-article-citation"

    def search(self, query: str, limit: int) -> list[Paper]:
        url = MEDRXIV_SEARCH_URL.format(query=quote(query.strip()))
        try:
            response = self._get(url)
        except Exception as exc:
            self.log_needs_js(url, note=f"fetch failed: {exc}")
            return []

        self.save_raw(query, response.text, "html")
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select(self.RESULT_SELECTOR)
        if not results:
            self.log_needs_js(url, note="0 result blocks found in static HTML")
            return []

        papers: list[Paper] = []
        for result in results[:limit]:
            paper = self._parse_result(result)
            if paper is not None:
                papers.append(paper)
        return papers

    def _parse_result(self, result) -> Paper | None:
        try:
            title_tag = result.select_one(
                ".highwire-cite-title, a.highwire-cite-linked-title"
            )
            title = self.clean_text(str(title_tag)) if title_tag else ""
            if not title:
                return None

            link_tag = result.select_one("a[href]")
            href = link_tag["href"] if link_tag else None
            if not href:
                return None
            source_url = href if href.startswith("http") else f"https://www.medrxiv.org{href}"

            doi = None
            if "/content/10." in source_url:
                doi = "10." + source_url.split("/content/10.", 1)[1].split("v")[0]

            authors_tag = result.select_one(".highwire-citation-authors")
            authors: list[str] = []
            if authors_tag:
                author_spans = authors_tag.select(".highwire-citation-author")
                if author_spans:
                    authors = [self.clean_text(str(a)) for a in author_spans]
                else:
                    authors = [self.clean_text(str(authors_tag))]

            year = None
            date_tag = result.select_one(".highwire-cite-metadata-date")
            if date_tag:
                digits = "".join(c for c in self.clean_text(str(date_tag)) if c.isdigit())
                if len(digits) >= 4:
                    year = int(digits[-4:])
            if year is None:
                return None

            abstract_tag = result.select_one(".highwire-cite-snippet")
            abstract = self.clean_text(str(abstract_tag)) if abstract_tag else ""
            if not abstract:
                return None

            paper_id = "medrxiv_" + source_url.rstrip("/").rsplit("/", 1)[-1]

            return Paper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                source_url=source_url,
                source_database=SourceDatabase.MEDRXIV,
                publication_type=None,
                is_preprint=True,
                doi=doi,
                retrieved_at=datetime.now(timezone.utc),
            )
        except Exception:
            return None
