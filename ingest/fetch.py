"""
Tier 1, stage 2 — fetch article bodies for discovered URLs.

    from ingest.discover import discover
    from ingest.fetch import fetch_all

    for record in fetch_all(discover(limit=50)):
        ...   # same dict shape as ingest.clean.load_csv yields

Takes `FeedEntry` objects from `ingest.discover` and returns records ready for
`ingest.chunk.to_documents`. One HTTP request per article, so this is the
expensive half of the crawl and the half that must behave itself.

WHY TRAFILATURA AND NOT BEAUTIFULSOUP
-------------------------------------
The v1 scrapers used hand-written CSS selectors per publisher. That approach
has a failure mode nobody notices: when a site changes its template the
selector stops matching, extraction returns a nav bar or an empty string, and
the pipeline records a "successful" scrape of junk. There were 130 domains to
keep selectors for.

trafilatura does boilerplate removal generically — it scores DOM nodes by text
density and link ratio rather than matching a class name — so a template change
degrades quality slightly instead of silently producing garbage. It also
returns None rather than an empty string when it cannot find an article, which
is a real signal this module acts on.

BEING A POLITE CRAWLER
----------------------
This hits ~130 real news domains that owe us nothing. So:

  * `robots.txt` is honoured, fetched once per host and cached.
  * Requests to the same host are spaced by `delay` seconds; the crawl-delay
    a site declares in robots.txt wins if it is longer than ours.
  * The User-Agent identifies the project and links to the repository, so an
    administrator seeing it in a log can find out what it is.
  * Non-HTML responses are dropped without reading the body.
  * Failures are logged and skipped, never retried in a tight loop.

None of that is optional politeness theatre — a crawler that ignores robots.txt
and hammers a host is the reason IP ranges get blocked.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from core.text import sha256_id
from ingest.clean import MIN_TEXT_CHARS
from ingest.discover import USER_AGENT, FeedEntry

logger = logging.getLogger(__name__)

# Anything larger than this is not an article. Guards against accidentally
# pulling a video page or a mis-linked archive.
MAX_BYTES = 4 * 1024 * 1024

OK_CONTENT = ("text/html", "application/xhtml")

# Below this, trafilatura found no article — it found a scrap.
#
# Distinct from MIN_TEXT_CHARS (200), and the two answer different questions.
# This one is structural: "is there an article on this page at all?" A page of
# pure navigation yields a few characters once the fallback extractor has
# scraped the bottom of the barrel, and calling that a successful extraction
# makes the failure counters lie. MIN_TEXT_CHARS is the separate editorial
# question of whether an article we *did* find is substantial enough to be
# evidence.
MIN_EXTRACT_CHARS = 50


@dataclass
class FetchStats:
    """Reported by the CLI and folded into the manifest's clean_stats."""

    attempted: int = 0
    fetched: int = 0
    blocked_by_robots: int = 0
    http_error: int = 0
    wrong_content_type: int = 0
    extraction_failed: int = 0
    too_short: int = 0
    duplicate: int = 0
    kept: int = 0
    with_reliable_date: int = 0
    domains: dict[str, int] = field(default_factory=dict)


def summarise(stats: FetchStats) -> str:
    top = sorted(stats.domains.items(), key=lambda kv: -kv[1])[:5]
    coverage = stats.with_reliable_date / max(stats.kept, 1)
    return (
        f"  attempted           {stats.attempted:,}\n"
        f"  fetched             {stats.fetched:,}\n"
        f"  blocked by robots   {stats.blocked_by_robots:,}\n"
        f"  http errors         {stats.http_error:,}\n"
        f"  wrong content type  {stats.wrong_content_type:,}\n"
        f"  extraction failed   {stats.extraction_failed:,}\n"
        f"  too short           {stats.too_short:,}\n"
        f"  duplicate text      {stats.duplicate:,}\n"
        f"  kept                {stats.kept:,}\n"
        f"  real publish dates  {stats.with_reliable_date:,}"
        f" ({coverage:.0%})   <- 4% on the CSV corpus\n"
        f"  top domains         {', '.join(f'{d} ({n})' for d, n in top)}"
    )


# ── robots.txt ───────────────────────────────────────────────────────────────

class RobotsCache:
    """
    One robots.txt lookup per host, remembered for the run.

    A host whose robots.txt cannot be fetched is treated as **allowing** the
    crawl. That is the convention the standard describes: absence of a policy is
    not a prohibition. A host that returns a policy we cannot parse is also
    allowed, for the same reason.
    """

    def __init__(self, user_agent: str = USER_AGENT) -> None:
        self.user_agent = user_agent
        self._parsers: dict[str, RobotFileParser | None] = {}

    def _parser(self, url: str) -> RobotFileParser | None:
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host in self._parsers:
            return self._parsers[host]

        parser = RobotFileParser()
        parser.set_url(f"{host}/robots.txt")
        try:
            parser.read()
        except Exception as exc:
            logger.debug("robots.txt unreadable for %s (%s) — allowing",
                         host, type(exc).__name__)
            parser = None
        self._parsers[host] = parser
        return parser

    def allowed(self, url: str) -> bool:
        parser = self._parser(url)
        if parser is None:
            return True
        try:
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def crawl_delay(self, url: str) -> float:
        parser = self._parser(url)
        if parser is None:
            return 0.0
        try:
            value = parser.crawl_delay(self.user_agent)
            return float(value) if value else 0.0
        except Exception:
            return 0.0


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_article(html: str, url: str = "") -> str | None:
    """
    Pull the article body out of a page, or None if there isn't one.

    `favor_precision` biases towards dropping borderline blocks. For evidence
    retrieval that is the right trade: a comment thread or a "related stories"
    rail embedded into a chunk becomes a retrievable passage attributed to a
    credible outlet, which is worse than a slightly shorter article.

    The fallback extractors stay enabled — they recover real articles from odd
    templates — but they will happily return a scrap from a page that has no
    article at all. A nav-only page yields a single character. So the result is
    held to `MIN_EXTRACT_CHARS` before being called a success; otherwise this
    returns None and the caller counts it as an extraction failure, which is
    what it is.
    """
    import trafilatura

    try:
        text = trafilatura.extract(
            html,
            url=url or None,
            favor_precision=True,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
    except Exception as exc:
        logger.debug("trafilatura raised on %s (%s)", url, type(exc).__name__)
        return None

    if not text or len(text.strip()) < MIN_EXTRACT_CHARS:
        return None
    return text


# ── Fetch ────────────────────────────────────────────────────────────────────

def fetch_all(
    entries: Iterable[FeedEntry],
    *,
    stats: FetchStats | None = None,
    delay: float = 1.0,
    timeout: float = 20.0,
    obey_robots: bool = True,
) -> Iterator[dict]:
    """
    Fetch each entry and yield a pipeline record.

    Yields the same dict shape as `ingest.clean.load_csv`, so the downstream
    stages cannot tell the two apart — which is what allows a crawl and the CSV
    to be merged with no special-casing anywhere after this point.

    De-duplicates on body text as well as URL: the same wire story republished
    under three URLs must become one document, or the report reads a single
    source as three-way corroboration.
    """
    import requests

    stats = stats if stats is not None else FetchStats()
    robots = RobotsCache() if obey_robots else None

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    last_hit: dict[str, float] = {}
    seen_text: set[str] = set()
    seen_url: set[str] = set()

    for entry in entries:
        if entry.url in seen_url:
            continue
        seen_url.add(entry.url)
        stats.attempted += 1

        if robots is not None and not robots.allowed(entry.url):
            stats.blocked_by_robots += 1
            logger.debug("robots.txt disallows %s", entry.url)
            continue

        host = urlparse(entry.url).netloc.lower()
        # A site asking for more space than we planned to give it gets it.
        host_delay = max(delay, robots.crawl_delay(entry.url) if robots else 0.0)
        wait = host_delay - (time.monotonic() - last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)

        try:
            response = session.get(entry.url, timeout=timeout, stream=True)
            last_hit[host] = time.monotonic()

            content_type = response.headers.get("content-type", "").lower()
            if not any(ok in content_type for ok in OK_CONTENT):
                stats.wrong_content_type += 1
                response.close()
                continue

            # Read with a ceiling rather than trusting content-length, which is
            # absent on chunked responses and lies often enough elsewhere.
            body = response.raw.read(MAX_BYTES + 1, decode_content=True)
            response.close()
            if len(body) > MAX_BYTES:
                stats.wrong_content_type += 1
                continue

            response.raise_for_status()
            html = body.decode(response.encoding or "utf-8", errors="replace")
            stats.fetched += 1
        except Exception as exc:
            stats.http_error += 1
            logger.debug("Fetch failed %s (%s)", entry.url, type(exc).__name__)
            continue

        text = extract_article(html, entry.url)
        if not text:
            stats.extraction_failed += 1
            continue

        text = text.strip()
        if len(text) < MIN_TEXT_CHARS:
            stats.too_short += 1
            continue

        key = sha256_id(text)
        if key in seen_text:
            stats.duplicate += 1
            continue
        seen_text.add(key)

        stats.kept += 1
        if entry.date_reliable:
            stats.with_reliable_date += 1
        stats.domains[entry.domain] = stats.domains.get(entry.domain, 0) + 1

        yield entry.as_record(text)

    logger.info("Fetched %s/%s, kept %s",
                f"{stats.fetched:,}", f"{stats.attempted:,}", f"{stats.kept:,}")
