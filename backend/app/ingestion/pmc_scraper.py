"""PubMed Central (PMC) scraper via NCBI E-utilities.

PMC's search/article HTML is not scraped directly: NCBI publishes a free,
documented API (E-utilities) that is the sanctioned way to access PMC
records programmatically and is far more stable than parsing HTML whose
layout changes. Requests are self-identified with `tool`/`email` params
per NCBI's E-utilities usage policy. Raw XML is still saved to
data/raw/pmc/ before parsing, per the project's raw-first policy.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bs4 import BeautifulSoup

from app.ingestion.base import BaseScraper
from app.models import Paper, SourceDatabase

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL = "evidenceai"
NCBI_EMAIL = "geetnatu73@gmail.com"


class PMCScraper(BaseScraper):
    """Retrieves PMC open-access articles matching a search query."""

    source_name = "pmc"
    # eutils is a documented JSON/XML API, not a crawlable web site — see
    # base.BaseScraper's module docstring for why robots.txt doesn't apply.
    check_robots = False

    def search(self, query: str, limit: int) -> list[Paper]:
        pmc_ids = self._esearch(query, limit)
        if not pmc_ids:
            return []
        return self._efetch(query, pmc_ids)

    def _esearch(self, query: str, limit: int) -> list[str]:
        params = {
            "db": "pmc",
            "retmode": "json",
            "retmax": limit,
            "term": query,
            "tool": NCBI_TOOL,
            "email": NCBI_EMAIL,
        }
        response = self._get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
        self.save_raw(f"esearch_{query}", response.text, "json")
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, query: str, pmc_ids: list[str]) -> list[Paper]:
        params = {
            "db": "pmc",
            "id": ",".join(pmc_ids),
            "rettype": "full",
            "retmode": "xml",
            "tool": NCBI_TOOL,
            "email": NCBI_EMAIL,
        }
        response = self._get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        self.save_raw(f"efetch_{query}", response.text, "xml")

        soup = BeautifulSoup(response.text, "xml")
        papers: list[Paper] = []
        for article in soup.find_all("article"):
            paper = self._parse_article(article)
            if paper is not None:
                papers.append(paper)
        return papers

    def _parse_article(self, article) -> Paper | None:
        try:
            # NCBI's efetch XML labels the PMC id "pmcaid" (bare numeric id,
            # matching esearch's idlist) or "pmcid" (prefixed, e.g. "PMC123").
            pmcid_tag = article.find("article-id", {"pub-id-type": "pmcaid"})
            if pmcid_tag is not None and pmcid_tag.text.strip():
                pmcid = pmcid_tag.text.strip()
            else:
                pmcid_tag = article.find("article-id", {"pub-id-type": "pmcid"})
                if pmcid_tag is None or not pmcid_tag.text.strip():
                    return None
                pmcid = pmcid_tag.text.strip().removeprefix("PMC")
            if not pmcid:
                return None

            title_tag = article.find("article-title")
            title = self.clean_text(str(title_tag), "xml") if title_tag else ""
            if not title:
                return None

            authors = []
            for contrib in article.find_all("contrib", {"contrib-type": "author"}):
                surname = contrib.find("surname")
                given = contrib.find("given-names")
                if surname and surname.text.strip():
                    name = surname.text.strip()
                    if given and given.text.strip():
                        name = f"{given.text.strip()} {name}"
                    authors.append(name)

            year = None
            pub_date = article.find("pub-date")
            if pub_date is not None:
                year_tag = pub_date.find("year")
                if year_tag and year_tag.text.strip().isdigit():
                    year = int(year_tag.text.strip())
            if year is None:
                return None

            abstract_tag = article.find("abstract")
            abstract = self.clean_text(str(abstract_tag), "xml") if abstract_tag else ""
            if not abstract:
                return None

            doi_tag = article.find("article-id", {"pub-id-type": "doi"})
            doi = doi_tag.text.strip() if doi_tag and doi_tag.text.strip() else None

            publication_type = None
            subj_group = article.find("subj-group", {"subj-group-type": "heading"})
            if subj_group is not None:
                subject = subj_group.find("subject")
                if subject and subject.text.strip():
                    publication_type = subject.text.strip()

            return Paper(
                paper_id=f"pmc_{pmcid}",
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                source_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                source_database=SourceDatabase.PMC,
                publication_type=publication_type,
                is_preprint=False,
                doi=doi,
                retrieved_at=datetime.now(timezone.utc),
            )
        except Exception:
            return None
