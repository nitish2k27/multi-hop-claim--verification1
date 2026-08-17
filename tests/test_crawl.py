"""
Crawl pipeline tests.

**No network.** Every test drives the parsing and merging logic with fixture
documents held in this file, so the suite stays fast, deterministic, and does
not hit 130 news domains every time someone runs `pytest`. The parts that do
make requests (`discover()`, `fetch_all()`) are deliberately thin wrappers
around `parse_feed()` and `extract_article()`, which are what is tested here.

The bugs these guard against are the silent kind, which is the theme of the
rest of the suite too:

  * a feed date that parses but is nonsense (epoch zero, or next year) being
    written into the index as though a publisher supplied it,
  * the January-1 heuristic from the CSV path leaking into the feed path and
    discarding genuine New Year's Day articles,
  * a merge that de-duplicates on URL but not on body text, letting one wire
    story become three "independent" sources.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from ingest.clean import CleanStats, merge_records
from ingest.discover import FeedEntry, date_coverage, load_feeds, parse_feed


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _rss(items: str, title: str = "Example News") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>{title}</title>
  <link>https://example.com</link>
  <language>en-GB</language>
  {items}
</channel></rss>"""


ITEM_GOOD = """
  <item>
    <title>India's software exports reach 222 billion dollars</title>
    <link>https://example.com/tech/software-exports-222bn</link>
    <pubDate>Mon, 10 Mar 2025 14:22:00 GMT</pubDate>
    <description>Exports of computer software and services rose 11%.</description>
  </item>"""

ITEM_NO_DATE = """
  <item>
    <title>An article with no publication date</title>
    <link>https://example.com/news/undated</link>
  </item>"""

ITEM_NEW_YEARS_DAY = """
  <item>
    <title>Genuinely published on New Year's Day</title>
    <link>https://example.com/news/new-year</link>
    <pubDate>Wed, 01 Jan 2025 09:00:00 GMT</pubDate>
  </item>"""

ITEM_EPOCH = """
  <item>
    <title>Misconfigured feed generator</title>
    <link>https://example.com/news/epoch</link>
    <pubDate>Thu, 01 Jan 1970 00:00:00 GMT</pubDate>
  </item>"""


def _future_item() -> str:
    ahead = datetime.now(timezone.utc) + timedelta(days=400)
    return f"""
  <item>
    <title>Templating bug put this in the future</title>
    <link>https://example.com/news/future</link>
    <pubDate>{ahead.strftime("%a, %d %b %Y %H:%M:%S GMT")}</pubDate>
  </item>"""


ARTICLE_HTML = """<!doctype html><html><head><title>T</title></head><body>
<nav><a href="/">Home</a><a href="/tech">Tech</a><a href="/sport">Sport</a></nav>
<article>
  <h1>India's software exports reach 222 billion dollars</h1>
  <p>Exports of computer software and services, including ITeS and BPO,
  increased by 11 percent on a year-over-year basis to 222 billion dollars in
  2024-25, according to the Electronics and Computer Software Export Promotion
  Council. The council said the growth was driven by demand for digital
  engineering and cloud migration work from clients in North America and
  Europe, which together account for the large majority of India's services
  export earnings in this category.</p>
  <p>Industry bodies expect the pace to hold through the coming year, though
  they cautioned that currency movement and visa policy remain the two largest
  sources of uncertainty for the sector over that horizon.</p>
</article>
<footer><p>Copyright Example News. All rights reserved. Subscribe now!</p></footer>
</body></html>"""


# ── Feed parsing ─────────────────────────────────────────────────────────────

def test_real_publisher_date_is_kept_and_trusted():
    """The entire justification for crawling: dates come from the publisher."""
    entries = parse_feed(_rss(ITEM_GOOD), "https://example.com/feed")

    assert len(entries) == 1
    entry = entries[0]
    assert entry.publish_date == "2025-03-10"
    assert entry.date_reliable is True
    assert entry.domain == "example.com"
    assert entry.source == "Example News"      # feed title, not the bare domain
    assert entry.language == "en"              # "en-GB" narrowed to "en"


def test_missing_date_is_absent_not_invented():
    """
    No date must mean no date.

    The v1 scraper guessed one from the URL when the feed did not supply it,
    which is how 96% of the CSV corpus ended up with fabricated January-1
    stamps. Absent-and-flagged is strictly better than present-and-wrong.
    """
    entries = parse_feed(_rss(ITEM_NO_DATE), "https://example.com/feed")

    assert entries[0].publish_date is None
    assert entries[0].date_reliable is False


def test_new_years_day_is_trusted_from_a_feed():
    """
    The CSV path treats any Jan-1 date as a placeholder, and is right to:
    1,551 of 1,687 rows are Jan-1 stamps.

    That heuristic must NOT leak into the feed path. Here a January-1 date came
    from the publisher, so discarding it would throw away good data. This test
    exists because reusing clean.parse_publish_date would have been the obvious
    shortcut and would have been wrong.
    """
    entries = parse_feed(_rss(ITEM_NEW_YEARS_DAY), "https://example.com/feed")

    assert entries[0].publish_date == "2025-01-01"
    assert entries[0].date_reliable is True


@pytest.mark.parametrize("item,label", [
    (ITEM_EPOCH, "epoch-zero"),
    (_future_item(), "far-future"),
])
def test_implausible_dates_are_discarded(item, label):
    """
    A date that parses is not automatically a date that is true.

    Epoch-zero stamps come from misconfigured generators and far-future ones
    from templating bugs. Both would parse cleanly and then drive the recency
    term of credibility scoring, which is exactly the failure the CSV corpus
    already has.
    """
    entries = parse_feed(_rss(item), "https://example.com/feed")

    assert entries[0].publish_date is None, f"{label} date was accepted"
    assert entries[0].date_reliable is False


def test_entry_without_a_link_is_skipped():
    entries = parse_feed(
        _rss("<item><title>No link here</title></item>"),
        "https://example.com/feed",
    )
    assert entries == []


def test_malformed_feed_still_yields_what_parsed():
    """Half the feeds on the open web are slightly invalid. Recover, don't bail."""
    broken = _rss(ITEM_GOOD).replace("</channel></rss>", "")
    entries = parse_feed(broken, "https://example.com/feed")
    assert len(entries) == 1


# ── Record shape ─────────────────────────────────────────────────────────────

def test_record_shape_matches_the_csv_path_exactly():
    """
    The crawl and CSV records must be indistinguishable downstream.

    `ingest.chunk.to_documents` reads specific keys. If the crawl produced a
    different shape, chunks would silently lose metadata that credibility
    scoring and the report renderer depend on — a missing `domain` scores every
    source at the unknown-domain default without raising anything.
    """
    from ingest.chunk import to_documents

    entry = parse_feed(_rss(ITEM_GOOD), "https://example.com/feed")[0]
    record = entry.as_record("x" * 900)

    expected = {"url", "text", "title", "source", "domain", "publish_date",
                "date_reliable", "language", "category", "corpus_tier"}
    assert set(record) == expected

    (parent, chunks), = list(to_documents([record]))
    assert parent.metadata["domain"] == "example.com"
    assert parent.metadata["date_reliable"] is True
    assert parent.metadata["publish_date"] == "2025-03-10"
    assert chunks and all(c.metadata["parent_id"] == parent.metadata["parent_id"]
                          for c in chunks)


# ── Merge semantics ──────────────────────────────────────────────────────────

def _record(url: str, text: str, *, reliable: bool = True) -> dict:
    return {
        "url": url, "text": text, "title": "t", "source": "s",
        "domain": "example.com",
        "publish_date": "2025-03-10" if reliable else None,
        "date_reliable": reliable, "language": "en", "category": "",
        "corpus_tier": "news",
    }


def test_merge_prefers_the_first_stream_on_a_url_collision():
    """
    ingest.run passes the crawl stream first precisely so it wins here — the
    crawled copy has a real date and its CSV twin has a placeholder.
    """
    crawl = iter([_record("https://example.com/a", "crawled body", reliable=True)])
    csv = iter([_record("https://example.com/a", "csv body", reliable=False)])

    merged = list(merge_records(crawl, csv))

    assert len(merged) == 1
    assert merged[0]["text"] == "crawled body"
    assert merged[0]["date_reliable"] is True


def test_merge_deduplicates_on_body_text_across_streams():
    """
    The same wire story under different URLs must collapse to one document.

    Without this, three copies of one Reuters piece appear to the report as
    three independent sources corroborating a claim — the exact failure this
    system is built to avoid.
    """
    body = "identical wire copy " * 40
    crawl = iter([_record("https://a.com/x", body)])
    csv = iter([_record("https://b.com/y", body)])

    merged = list(merge_records(crawl, csv))
    assert len(merged) == 1


def test_merge_keeps_genuinely_distinct_records():
    crawl = iter([_record("https://a.com/x", "first article body")])
    csv = iter([_record("https://b.com/y", "second, different body")])

    assert len(list(merge_records(crawl, csv))) == 2


def test_merge_counts_what_it_dropped():
    stats = CleanStats()
    body = "shared body text " * 30
    list(merge_records(
        iter([_record("https://a.com/x", body)]),
        iter([_record("https://a.com/x", "different text entirely")]),
        iter([_record("https://c.com/z", body)]),
        stats=stats,
    ))
    assert stats.dropped_dupe_url == 1
    assert stats.dropped_dupe_content == 1


# ── Extraction ───────────────────────────────────────────────────────────────

def test_extraction_returns_the_article_and_drops_the_furniture():
    """
    trafilatura earns its place by removing nav and footer generically, rather
    than through a per-publisher CSS selector that breaks on a template change
    and silently returns a nav bar.
    """
    from ingest.fetch import extract_article

    text = extract_article(ARTICLE_HTML, "https://example.com/a")

    assert text is not None
    assert "222 billion" in text
    assert "Subscribe now" not in text
    assert "Home" not in text.split("\n")[0]


def test_extraction_returns_none_when_there_is_no_article():
    """
    None, not "". An empty string would flow onward and be dropped later as
    "too short", which reports the wrong reason for the failure.
    """
    from ingest.fetch import extract_article

    assert extract_article("<html><body><nav>a</nav></body></html>") is None


# ── Feed list ────────────────────────────────────────────────────────────────

def test_sources_yaml_loads_and_is_substantial():
    feeds = load_feeds()
    assert len(feeds) > 300
    assert all(f.startswith("http") for f in feeds)
    assert len(set(feeds)) == len(feeds), "sources.yaml contains duplicates"


def test_date_coverage_reports_the_fraction():
    entries = [
        FeedEntry("u1", "t", "2025-03-10", True, "d", "s", "f"),
        FeedEntry("u2", "t", None, False, "d", "s", "f"),
    ]
    assert date_coverage(entries) == {
        "entries": 2, "with_reliable_date": 1, "coverage": 0.5,
    }


def test_date_coverage_handles_an_empty_run():
    assert date_coverage([])["coverage"] == 0.0
