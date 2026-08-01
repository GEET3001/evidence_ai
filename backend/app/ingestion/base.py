"""Shared scraping infrastructure: rate limiting, robots.txt checks, retries.

robots.txt is enforced for sources fetched as web pages (medRxiv, Cochrane).
It is deliberately NOT enforced for dedicated JSON/XML APIs (NCBI eutils,
the OSF REST API) — those hosts blanket-disallow "/" for all user agents in
robots.txt because the exclusion protocol targets crawlers indexing HTML,
not documented API clients. Access to those APIs is governed by their own
published usage policies (rate limits, required tool/email identification),
which this module follows instead. Set `check_robots = False` on a subclass
to opt out; the default is True (strict).
"""

from __future__ import annotations

import time
import urllib.robotparser
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import Paper

USER_AGENT = (
    "EvidenceAI-ResearchBot/0.1 "
    "(academic student project; explainable research claim verification "
    "for mental health literature; contact: geetnatu73@gmail.com)"
)

MIN_REQUEST_INTERVAL_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRY_ATTEMPTS = 4


class RobotsDisallowedError(RuntimeError):
    """Raised when robots.txt disallows fetching a URL."""


class BaseScraper(ABC):
    """Abstract base for source-specific paper scrapers.

    Subclasses implement `search()` and get a shared rate-limited,
    retrying HTTP session plus helpers for robots.txt checks, persisting
    raw responses, and cleaning HTML to text.
    """

    source_name: str  # e.g. "pmc", "psyarxiv", "medrxiv", "cochrane"
    check_robots: bool = True

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_at: dict[str, float] = {}
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    @abstractmethod
    def search(self, query: str, limit: int) -> list[Paper]:
        """Search this source for papers matching `query`, return up to `limit`."""

    # --- robots.txt ---

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{host}/robots.txt")
            try:
                rp.read()
            except Exception:
                # Unreachable robots.txt: treat as "no restrictions found"
                # rather than failing closed, since RobotFileParser has no
                # entries to enforce in that case anyway.
                pass
            self._robots_cache[host] = rp
        return self._robots_cache[host]

    def _check_allowed(self, url: str) -> None:
        if not self.check_robots:
            return
        rp = self._robots_for(url)
        if not rp.can_fetch(USER_AGENT, url):
            raise RobotsDisallowedError(f"robots.txt disallows fetching: {url}")

    # --- rate limiting ---

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self._last_request_at.get(host)
        if last is not None:
            elapsed = time.monotonic() - last
            if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_at[host] = time.monotonic()

    # --- HTTP with retry ---

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        self._check_allowed(url)
        self._throttle(url)
        response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response

    # --- raw response persistence ---

    def save_raw(self, identifier: str, content: str, extension: str) -> Path:
        """Persist a raw response to data/raw/<source_name>/ before parsing."""
        out_dir = settings.raw_dir / self.source_name
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in identifier)[:150]
        path = out_dir / f"{safe_id}.{extension}"
        path.write_text(content, encoding="utf-8")
        return path

    def log_needs_js(self, url: str, note: str = "") -> None:
        """Record a URL that appears to need JS rendering, for a future pass."""
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        path = settings.raw_dir / "needs_js.txt"
        suffix = f"  # {note}" if note else ""
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{self.source_name}] {url}{suffix}\n")

    # --- HTML helpers ---

    @staticmethod
    def clean_text(html_or_text: str, parser: str = "html.parser") -> str:
        """Strip tags and normalise whitespace."""
        soup = BeautifulSoup(html_or_text, parser)
        return " ".join(soup.get_text(separator=" ").split())
