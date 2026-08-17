"""
Tier 1, stage 3 — clean and normalise source records.

Scope note: this is the **from-CSV** path. `data/processed/news_articles_rag.csv`
has already been through the old `src/data_processing/clean_news_data.py`
(text normalisation, source standardisation, category mapping), so this module
does not repeat that work. What it adds is what the old cleaner got wrong or
never did:

  1. Domain normalisation, so credibility scoring can key off it at all.
  2. Publish-date reliability flagging (see below).
  3. Quality filtering and dedup, re-applied because we don't trust the CSV to
     be free of near-duplicates.

The full crawl-cleaning port lands with the `--crawl` path in M5.

THE DATE PROBLEM
----------------
1,231 of 1,687 rows carry `2024-01-01`, another 320 carry `2026-01-01`; only 15
distinct dates exist in the whole corpus. These are placeholders that the old
scraper's `extract_year_from_url` fabricated when it could not find a real date,
and they are already present in `data/raw/scraped_news.csv` — so they happened at
scrape time and re-cleaning cannot recover them.

Recency is 30% of the credibility weight. Letting a fabricated date drive that is
worse than having no date at all, so every record carries `date_reliable`, and
`core/credibility.py` rebalances to domain 0.7 / type 0.3 when it is false.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import pandas as pd
from dateutil import parser as dateparser

from core.text import sha256_id

logger = logging.getLogger(__name__)

# Text shorter than this is a stub, a paywall notice, or a nav fragment —
# never usable evidence.
MIN_TEXT_CHARS = 200


@dataclass
class CleanStats:
    """Reported by the CLI and folded into the manifest."""
    rows_in: int = 0
    dropped_short: int = 0
    dropped_dupe_url: int = 0
    dropped_dupe_content: int = 0
    dates_unreliable: int = 0
    rows_out: int = 0
    sources: dict[str, int] = field(default_factory=dict)


def normalise_domain(domain: str | None, url: str | None = None) -> str:
    """
    Reduce a domain to the form `core/credibility.py`'s table is keyed on.

    Falls back to parsing the URL when the `domain` column is missing. Uses
    `removeprefix`, not `.replace("www.", "")` — the old scorer's replace stripped
    "www." from anywhere in the string, so `bbc.com/newswww.x` would mangle.
    """
    d = (domain or "").strip().lower()
    if not d and url:
        from urllib.parse import urlparse

        d = urlparse(url).netloc.lower()
    d = d.removeprefix("www.")
    return d.split(":")[0]                      # drop any :port


def parse_publish_date(raw) -> tuple[str | None, bool]:
    """
    Return (ISO date string, is_reliable).

    Uses dateutil rather than `datetime.fromisoformat` — RSS emits RFC-822
    ("Mon, 10 Mar 2026 14:22:00 GMT"), which fromisoformat rejects. The old
    scorer swallowed that exception and returned a neutral 0.5, so a whole class
    of parseable dates was silently discarded.

    A January 1st date is treated as a placeholder. That misflags articles
    genuinely published on New Year's Day; with 1,551 of 1,687 rows being Jan-1
    stamps, accepting that false-positive rate is clearly the right trade.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, False
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None, False

    try:
        dt = dateparser.parse(text)
    except (ValueError, OverflowError, TypeError):
        return None, False
    if dt is None:
        return None, False

    reliable = not (dt.month == 1 and dt.day == 1)
    return dt.date().isoformat(), reliable


def load_csv(path, stats: CleanStats) -> Iterator[dict]:
    """
    Stream cleaned records from the processed-news CSV.

    Yields plain dicts rather than a DataFrame so the chunking stage can consume
    them lazily — the whole corpus is only ~7MB today, but the crawl path in M5
    will not be.
    """
    df = pd.read_csv(path)
    stats.rows_in = len(df)
    logger.info("Loaded %s rows from %s", f"{len(df):,}", path)

    seen_urls: set[str] = set()
    seen_content: set[str] = set()

    for row in df.itertuples(index=False):
        text = str(getattr(row, "text", "") or "").strip()
        url = str(getattr(row, "url", "") or "").strip()

        if len(text) < MIN_TEXT_CHARS:
            stats.dropped_short += 1
            continue

        if url:
            if url in seen_urls:
                stats.dropped_dupe_url += 1
                continue
            seen_urls.add(url)

        # Content-level dedup catches the same wire story republished under
        # different URLs. It matters more here than in a general search corpus:
        # three copies of one Reuters piece would otherwise look like three
        # independent sources corroborating a claim.
        content_key = sha256_id(text)
        if content_key in seen_content:
            stats.dropped_dupe_content += 1
            continue
        seen_content.add(content_key)

        publish_date, date_reliable = parse_publish_date(
            getattr(row, "publish_date", None)
        )
        if not date_reliable:
            stats.dates_unreliable += 1

        source = str(getattr(row, "source", "") or "unknown").strip()
        stats.sources[source] = stats.sources.get(source, 0) + 1
        stats.rows_out += 1

        yield {
            "url": url or f"urn:sha256:{content_key[:16]}",
            "text": text,
            "title": str(getattr(row, "title", "") or "").strip(),
            "source": source,
            "domain": normalise_domain(
                getattr(row, "domain", None), url
            ),
            "publish_date": publish_date,
            "date_reliable": date_reliable,
            "language": str(getattr(row, "language", "en") or "en").strip(),
            "category": str(getattr(row, "category", "") or "").strip(),
            "corpus_tier": "news",
        }


def summarise(stats: CleanStats) -> str:
    top = sorted(stats.sources.items(), key=lambda kv: -kv[1])[:5]
    return (
        f"  rows in            {stats.rows_in:,}\n"
        f"  dropped (short)    {stats.dropped_short:,}\n"
        f"  dropped (dup url)  {stats.dropped_dupe_url:,}\n"
        f"  dropped (dup text) {stats.dropped_dupe_content:,}\n"
        f"  rows out           {stats.rows_out:,}\n"
        f"  unreliable dates   {stats.dates_unreliable:,}"
        f" ({stats.dates_unreliable / max(stats.rows_out, 1):.0%})\n"
        f"  top sources        {', '.join(f'{s} ({n})' for s, n in top)}"
    )
