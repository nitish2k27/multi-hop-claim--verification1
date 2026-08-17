"""
Source credibility scoring.

Ported from `src/rag/credibility_scorer.py` with three substantive changes:

1. **Keys off `metadata['domain']`, not a source slug.** The old table mixed real
   domains (`reuters.com`) with scraper slugs (`economic_times_tech`, `ndtv`),
   so a document whose `source` happened to be spelled the other way silently
   fell through to the 0.50 default. Every chunk in `index/` carries a real
   `domain` — that is now the only lookup key.

2. **`removeprefix('www.')`, not `.replace('www.', '')`.** The old version
   mangled any domain containing that substring anywhere.

3. **Honours `date_reliable`.** 96% of this corpus has placeholder Jan-1 dates
   (see BUILD_PLAN §3). Letting a fabricated date drive 30% of a credibility
   score is worse than not scoring recency at all, so when the date is not
   trustworthy the weights collapse to domain 0.7 / type 0.3.

The scale is deliberately narrow (0.3–0.95). It expresses "how much should this
source move the verdict", not "is this true" — a low score means *treat with
caution and say so in the report*, which is exactly what the prompt asks the
model to do.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as date_parser

logger = logging.getLogger(__name__)

# Score assigned to a domain that is not in the table below. Deliberately
# mid-scale: unknown is not the same as untrustworthy, and the report prompt
# treats anything under 0.50 as a flag-explicitly case.
UNKNOWN_DOMAIN = 0.50

# ── Domain authority ─────────────────────────────────────────────────────────
#
# Hand-curated, and covering the domains that actually appear in this corpus —
# an authority table full of `reuters.com` entries is decoration when the corpus
# contains no Reuters. Keys are stored without a `www.` prefix.
#
# This is a judgement call encoded as data, not a measurement. It is reviewable
# and adjustable in one place, which is the point.
DOMAIN_SCORES: dict[str, float] = {
    # Fact-checkers — the only sources whose entire purpose is verification
    "snopes.com":                  0.90,
    "factcheck.org":               0.93,
    "politifact.com":              0.91,
    "boomlive.in":                 0.88,
    "altnews.in":                  0.87,

    # International wires and public broadcasters
    "reuters.com":                 0.95,
    "apnews.com":                  0.94,
    "bbc.com":                     0.93,
    "bbc.co.uk":                   0.93,
    "npr.org":                     0.92,
    "dw.com":                      0.83,
    "france24.com":                0.82,
    "aljazeera.com":               0.82,

    # Newspapers of record
    "theguardian.com":             0.87,
    "nytimes.com":                 0.88,
    "wsj.com":                     0.88,
    "washingtonpost.com":          0.86,
    "economist.com":               0.87,
    "ft.com":                      0.87,
    "bloomberg.com":               0.86,
    "spiegel.de":                  0.84,
    "elpais.com":                  0.84,
    "scmp.com":                    0.78,
    "politico.eu":                 0.82,
    "middleeasteye.net":           0.72,

    # Indian press
    "thehindu.com":                0.84,
    "indianexpress.com":           0.83,
    "livemint.com":                0.81,
    "economictimes.indiatimes.com": 0.80,
    "timesofindia.indiatimes.com": 0.76,
    "ndtv.com":                    0.76,
    "ndtvprofit.com":              0.76,
    "indiatoday.in":               0.74,
    "news18.com":                  0.72,
    "zeenews.india.com":           0.66,
    "amarujala.com":               0.66,
    "medianama.com":               0.74,

    # US broadcast
    "cnn.com":                     0.78,
    "edition.cnn.com":             0.78,

    # Research institutes and academia
    "atlanticcouncil.org":         0.82,
    "csis.org":                    0.82,
    "med.stanford.edu":            0.90,
    "technologyreview.com":        0.83,

    # Technology press — competent in their lane, and that is most of this corpus
    "arstechnica.com":             0.80,
    "wired.com":                   0.80,
    "theverge.com":                0.76,
    "techcrunch.com":              0.74,
    "engadget.com":                0.72,
    "mashable.com":                0.66,
    "tech.eu":                     0.70,
    "inc42.com":                   0.68,
    "techstartups.com":            0.60,
    "xataka.com":                  0.68,
    "hipertextual.com":            0.64,

    # Security reporting — narrow scope, strong track record inside it
    "krebsonsecurity.com":         0.85,
    "threatpost.com":              0.78,

    # Vendor and project blogs: accurate about their own product, and
    # promotional about everything else. Scored as first-party sources.
    "aws.amazon.com":              0.70,
    "azure.microsoft.com":         0.70,
    "devblogs.microsoft.com":      0.70,
    "aka.ms":                      0.65,
    "github.blog":                 0.70,
    "github.com":                  0.60,
    "stackoverflow.blog":          0.68,
    "developers.cloudflare.com":   0.70,
    "ziglang.org":                 0.70,
    "lwn.net":                     0.80,

    # Community and personal publishing — unreviewed by construction
    "dev.to":                      0.45,
    "towardsdatascience.com":      0.50,
    "news.ycombinator.com":        0.40,
    "hume.ai":                     0.55,

    # Entertainment and celebrity press — trade papers rate above tabloids
    "variety.com":                 0.68,
    "hollywoodreporter.com":       0.68,
    "deadline.com":                0.66,
    "billboard.com":               0.64,
    "rollingstone.com":            0.62,
    "pitchfork.com":               0.60,
    "elle.com":                    0.50,
    "bollywoodhungama.com":        0.48,
    "usmagazine.com":              0.40,
    "pagesix.com":                 0.38,
    "tmz.com":                     0.35,
}

# ── Source type ──────────────────────────────────────────────────────────────

TYPE_SCORES: dict[str, float] = {
    "fact_checker":      0.90,
    "academic":          0.90,
    "news_agency":       0.88,
    "international_org": 0.88,
    "government":        0.85,
    "newspaper":         0.80,
    "trade_press":       0.70,
    "vendor_blog":       0.60,
    "user_upload":       0.50,
    "blog":              0.50,
    "tabloid":           0.40,
    "social_media":      0.30,
    "unknown":           0.50,
}

# Domains whose type cannot be inferred from the suffix alone.
_EXPLICIT_TYPES: dict[str, str] = {
    "snopes.com":        "fact_checker",
    "factcheck.org":     "fact_checker",
    "politifact.com":    "fact_checker",
    "boomlive.in":       "fact_checker",
    "altnews.in":        "fact_checker",
    "reuters.com":       "news_agency",
    "apnews.com":        "news_agency",
    "bbc.com":           "news_agency",
    "bbc.co.uk":         "news_agency",
    "dw.com":            "news_agency",
    "france24.com":      "news_agency",
    "atlanticcouncil.org": "academic",
    "csis.org":          "academic",
    "technologyreview.com": "academic",
    "tmz.com":           "tabloid",
    "pagesix.com":       "tabloid",
    "usmagazine.com":    "tabloid",
    "bollywoodhungama.com": "tabloid",
    "dev.to":            "blog",
    "towardsdatascience.com": "blog",
    "news.ycombinator.com": "social_media",
    "aws.amazon.com":    "vendor_blog",
    "azure.microsoft.com": "vendor_blog",
    "devblogs.microsoft.com": "vendor_blog",
    "github.blog":       "vendor_blog",
    "stackoverflow.blog": "vendor_blog",
    "developers.cloudflare.com": "vendor_blog",
    "aka.ms":            "vendor_blog",
    "ziglang.org":       "vendor_blog",
    "arstechnica.com":   "trade_press",
    "wired.com":         "trade_press",
    "theverge.com":      "trade_press",
    "techcrunch.com":    "trade_press",
    "engadget.com":      "trade_press",
    "mashable.com":      "trade_press",
    "tech.eu":           "trade_press",
    "inc42.com":         "trade_press",
    "techstartups.com":  "trade_press",
    "xataka.com":        "trade_press",
    "hipertextual.com":  "trade_press",
    "krebsonsecurity.com": "trade_press",
    "threatpost.com":    "trade_press",
    "lwn.net":           "trade_press",
    "variety.com":       "trade_press",
    "hollywoodreporter.com": "trade_press",
    "deadline.com":      "trade_press",
    "billboard.com":     "trade_press",
    "rollingstone.com":  "trade_press",
    "pitchfork.com":     "trade_press",
    "elle.com":          "trade_press",
    "medianama.com":     "trade_press",
}


def normalise_domain(domain: str | None) -> str:
    """Lowercase, strip a leading `www.`. Returns `'unknown'` for empty input."""
    if not domain:
        return "unknown"
    return domain.strip().lower().removeprefix("www.")


def infer_source_type(domain: str) -> str:
    """
    Best-effort source type from the domain.

    Explicit table first, then suffix heuristics for the long tail — a `.gov`
    or `.edu` address is a reliable signal in a way that `.com` never is.
    """
    if domain in _EXPLICIT_TYPES:
        return _EXPLICIT_TYPES[domain]
    if domain.endswith((".gov", ".gov.uk", ".gov.in", ".mil")):
        return "government"
    if domain.endswith((".edu", ".ac.uk")):
        return "academic"
    if domain.endswith(".org"):
        return "international_org"
    if domain in DOMAIN_SCORES:
        return "newspaper"
    return "unknown"


def _recency_score(publish_date: str | None) -> float:
    """
    Linear decay from 1.0 today to a 0.2 floor, halving at one year.

    Only ever consulted when the date is flagged reliable, so a parse failure
    here means the metadata lied — fall back to neutral rather than guessing.
    """
    if not publish_date:
        return 0.50
    try:
        published = date_parser.parse(str(publish_date))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - published).days
        return max(0.2, min(1.0, 1.0 - (age_days / 365.0) * 0.5))
    except (ValueError, OverflowError, TypeError):
        logger.debug("Unparseable publish_date: %r", publish_date)
        return 0.50


def tier_of(score: float) -> str:
    """Human-readable band, used in the report and the CLI output."""
    if score >= 0.85:
        return "HIGH"
    if score >= 0.70:
        return "MEDIUM"
    if score >= 0.50:
        return "LOW"
    return "VERY_LOW"


def score_document(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Credibility for one retrieved chunk, from its metadata alone.

    Returns the components as well as the total — a single opaque number is
    impossible to argue with, and the report needs to explain *why* a source was
    down-weighted.
    """
    domain = normalise_domain(metadata.get("domain"))
    domain_score = DOMAIN_SCORES.get(domain, UNKNOWN_DOMAIN)

    source_type = metadata.get("source_type") or infer_source_type(domain)
    type_score = TYPE_SCORES.get(source_type, TYPE_SCORES["unknown"])

    date_reliable = bool(metadata.get("date_reliable", False))

    if date_reliable:
        recency = _recency_score(metadata.get("publish_date"))
        total = domain_score * 0.5 + recency * 0.3 + type_score * 0.2
        weights = "domain 0.5 / recency 0.3 / type 0.2"
    else:
        # The date is a placeholder. Redistribute its weight onto the two
        # signals that are actually grounded rather than scoring a fiction.
        recency = None
        total = domain_score * 0.7 + type_score * 0.3
        weights = "domain 0.7 / type 0.3 (recency dropped — date not reliable)"

    total = round(min(1.0, max(0.0, total)), 4)

    # A user-supplied document carries a hard ceiling. Applied last, after all
    # the normal weighting, so it is a cap rather than an input to the formula —
    # an upload from a domain that happens to score well must still not become
    # authoritative. Without this, uploading a file asserting X would make the
    # system confirm X, which is the most obvious way to fool a fact-checker.
    cap = metadata.get("credibility_cap")
    capped = False
    if cap is not None and total > float(cap):
        total = round(float(cap), 4)
        capped = True

    return {
        "total":         total,
        "capped":        capped,
        "domain":        domain,
        "domain_score":  domain_score,
        "source_type":   source_type,
        "type_score":    type_score,
        "recency_score": recency,
        "date_reliable": date_reliable,
        "weighting":     weights,
        "tier":          tier_of(total),
        "known_domain":  domain in DOMAIN_SCORES,
    }
