"""
Tier 1, stage 1 — discover article URLs from RSS/Atom feeds.

    from ingest.discover import discover
    for entry in discover(limit=50):
        print(entry.url, entry.publish_date)

Reads `ingest/sources.yaml` (363 feeds, 130 domains) and yields one `FeedEntry`
per article found. **No article bodies are fetched here** — that is
`ingest.fetch`. Discovery is cheap, one request per feed; fetching is expensive,
one request per article. Keeping them separate means a discovery run can be
inspected, filtered or cached before committing to thousands of article fetches.

WHY THIS EXISTS: THE DATE PROBLEM
---------------------------------
1,622 of the 1,687 articles in the CSV corpus (96%) carry placeholder January-1
dates. The original scraper fabricated them with a URL-parsing heuristic when it
could not find a real date, and they are baked into the raw scrape — re-cleaning
cannot recover them. Because recency is 30% of the credibility weight,
`core/credibility.py` currently drops that term entirely and reweights to
domain 0.7 / type 0.3.

**RSS entries carry a real `published` field, supplied by the publisher.** That
is the entire reason this module exists. A crawled corpus has genuine dates, so
recency becomes a usable signal again instead of a disclosed defect.

DATE RELIABILITY IS JUDGED DIFFERENTLY HERE
-------------------------------------------
`ingest.clean.parse_publish_date` treats any January-1 date as a placeholder.
That is correct for the CSV, where 1,551 of 1,687 rows are Jan-1 stamps. It is
*wrong* for feed data, where a January-1 date is simply an article published on
New Year's Day.

So this module does not reuse that heuristic. A feed date is trusted when
feedparser could parse it into a real timestamp and it falls in a plausible
window; otherwise the entry carries no date and `date_reliable=False`. The
distinction is deliberate and is why `FeedEntry` sets the flag itself rather
than deferring to the cleaner.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SOURCES = Path(__file__).resolve().parent / "sources.yaml"

# A feed date outside this window is a parser artifact, not a publication date.
# Seen in the wild: epoch-zero stamps (1970) from misconfigured generators, and
# dates years in the future from templating bugs.
EARLIEST = date(1995, 1, 1)
FUTURE_TOLERANCE = timedelta(days=2)      # timezone skew, not a time machine

# Identify the crawler honestly. A blank or spoofed User-Agent is what gets a
# crawler blocked, and rightly.
USER_AGENT = (
    "VerifAI/2.0 (fact-verification research project; "
    "+https://github.com/nitish2k27/multi-hop-claim--verification1)"
)


@dataclass(frozen=True)
class FeedEntry:
    """
    One article, as advertised by a feed. No body text yet.

    `publish_date` is an ISO date string or None. `date_reliable` says whether
    the publisher actually supplied it — the whole point of crawling.
    """

    url: str
    title: str
    publish_date: str | None
    date_reliable: bool
    domain: str
    source: str
    feed_url: str
    summary: str = ""
    language: str = "en"

    def as_record(self, text: str) -> dict:
        """
        Combine with fetched body text into the record shape the rest of the
        pipeline already consumes.

        Deliberately identical to what `ingest.clean.load_csv` yields, so
        `ingest.chunk.to_documents` needs no knowledge of where a record came
        from and the two sources can be merged without special-casing.
        """
        return {
            "url": self.url,
            "text": text,
            "title": self.title,
            "source": self.source,
            "domain": self.domain,
            "publish_date": self.publish_date,
            "date_reliable": self.date_reliable,
            "language": self.language,
            "category": "",
            "corpus_tier": "news",
        }


# ── Feed list ────────────────────────────────────────────────────────────────

def load_feeds(path: Path | None = None) -> list[str]:
    """
    Read the feed URLs out of sources.yaml.

    Parsed by hand rather than with PyYAML: the file is a flat list of scalars
    under one `feeds:` key, and a 12-line reader is a smaller dependency
    surface than a YAML engine for a format this shape. If sources.yaml ever
    grows nested structure, swap this for `yaml.safe_load` and delete the note.
    """
    path = path or SOURCES
    if not path.exists():
        raise FileNotFoundError(f"Feed list not found: {path}")

    feeds: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and "://" in stripped:
            feeds.append(stripped[2:].strip())
    return feeds


# ── Date handling ────────────────────────────────────────────────────────────

def _entry_date(entry) -> tuple[str | None, bool]:
    """
    Pull a trustworthy publication date off a feedparser entry.

    feedparser normalises RFC-822, RFC-3339 and several malformed variants into
    `published_parsed`, a UTC time.struct_time. If it could not parse the field
    at all, the attribute is absent — which is the honest signal that no date is
    available, and far better than guessing one from the URL.
    """
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if not parsed:
            continue
        try:
            value = datetime(*parsed[:6], tzinfo=timezone.utc).date()
        except (TypeError, ValueError):
            continue

        today = datetime.now(timezone.utc).date()
        if value < EARLIEST or value > today + FUTURE_TOLERANCE:
            logger.debug("Implausible feed date %s — discarding", value)
            continue

        # Note: no January-1 check. See the module docstring — that heuristic
        # belongs to the CSV path and would wrongly discard genuine New Year's
        # Day articles here.
        return value.isoformat(), True

    return None, False


def _domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.").split(":")[0]


# ── Discovery ────────────────────────────────────────────────────────────────

def parse_feed(raw: str | bytes, feed_url: str) -> list[FeedEntry]:
    """
    Turn one feed document into entries.

    Takes the feed *content*, not a URL, so this is unit-testable without a
    network call — `discover()` does the fetching.
    """
    import feedparser

    parsed = feedparser.parse(raw)

    # bozo means the document was malformed. Not fatal: feedparser recovers from
    # most real-world breakage, and half the feeds on the open web are slightly
    # invalid. Worth logging, not worth discarding a feed over.
    if getattr(parsed, "bozo", 0):
        logger.debug("Feed %s is malformed (%s) — using what parsed",
                     feed_url, getattr(parsed, "bozo_exception", "?"))

    feed_title = (getattr(parsed, "feed", {}) or {}).get("title", "")
    feed_language = (getattr(parsed, "feed", {}) or {}).get("language", "en")
    if feed_language:
        feed_language = str(feed_language).split("-")[0].lower()[:2] or "en"

    entries: list[FeedEntry] = []
    for item in getattr(parsed, "entries", []):
        url = (getattr(item, "link", "") or "").strip()
        if not url or "://" not in url:
            continue

        publish_date, reliable = _entry_date(item)
        domain = _domain_of(url)

        entries.append(FeedEntry(
            url=url,
            title=(getattr(item, "title", "") or "").strip(),
            publish_date=publish_date,
            date_reliable=reliable,
            domain=domain,
            # Prefer the feed's own title over the bare domain: "BBC News" reads
            # better in a report than "feeds.bbci.co.uk".
            source=feed_title.strip() or domain,
            feed_url=feed_url,
            summary=(getattr(item, "summary", "") or "").strip()[:500],
            language=feed_language or "en",
        ))
    return entries


def discover(
    feeds: Iterable[str] | None = None,
    *,
    limit: int | None = None,
    per_feed: int | None = None,
    delay: float = 1.0,
    timeout: float = 20.0,
) -> Iterator[FeedEntry]:
    """
    Fetch every feed and yield the articles it advertises, de-duplicated by URL.

    `delay` is seconds to wait between requests to the *same host*. Feeds from
    one publisher are grouped, so this throttles per-publisher rather than
    globally — 130 domains means the crawl is not serialised behind one slow
    site while still never hammering anyone.

    A feed that errors is logged and skipped. One dead feed out of 363 must not
    abort a discovery run.
    """
    import requests

    feed_list = list(feeds) if feeds is not None else load_feeds()
    logger.info("Discovering across %d feeds", len(feed_list))

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    last_hit: dict[str, float] = {}
    seen: set[str] = set()
    yielded = 0
    failed = 0

    # Group by host so the per-host delay does not serialise the whole run.
    for feed_url in sorted(feed_list, key=lambda u: (_domain_of(u), u)):
        if limit is not None and yielded >= limit:
            break

        host = _domain_of(feed_url)
        wait = delay - (time.monotonic() - last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)

        try:
            response = session.get(feed_url, timeout=timeout)
            last_hit[host] = time.monotonic()
            response.raise_for_status()
            entries = parse_feed(response.content, feed_url)
        except Exception as exc:
            failed += 1
            logger.warning("Feed failed: %s (%s)", feed_url, type(exc).__name__)
            continue

        kept = 0
        for entry in entries:
            if per_feed is not None and kept >= per_feed:
                break
            if entry.url in seen:
                continue
            seen.add(entry.url)
            kept += 1
            yielded += 1
            yield entry
            if limit is not None and yielded >= limit:
                break

        logger.debug("%s -> %d new", feed_url, kept)

    logger.info("Discovered %d unique articles (%d feeds failed)", yielded, failed)


def date_coverage(entries: list[FeedEntry]) -> dict:
    """
    How many discovered entries carry a real publisher date.

    This is the number that justifies the whole crawl path: on the CSV corpus
    it is 4%. Reported by `ingest.run --crawl` so the improvement is visible
    rather than asserted.
    """
    total = len(entries)
    reliable = sum(1 for e in entries if e.date_reliable)
    return {
        "entries": total,
        "with_reliable_date": reliable,
        "coverage": round(reliable / total, 4) if total else 0.0,
    }
