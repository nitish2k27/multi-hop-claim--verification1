"""
Text helpers shared by both tiers.

This module exists for one specific reason: **the BM25 tokenizer must be
importable from `serve/` even though the index is built in `ingest/`.**

`BM25Retriever` stores its `preprocess_func` by reference when pickled, so the
function has to live at a stable, importable path that does not violate the tier
boundary (`serve/` may not import from `ingest/`). `core/` is the only shared
surface, so it lives here.

A lambda or a function defined inside `ingest/` would either fail to pickle or
force `serve/` to import `ingest/` at load time.
"""

from __future__ import annotations

import re

# Loaded once at import, not per call. The old sparse_retrieval.py called
# nltk's stopwords.words('english') inside the per-document tokenize loop —
# 1,687 corpus reads per index build, and again per query.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself me more most my myself no nor
not of off on once only or other ought our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why with would you your yours yourself yourselves
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def bm25_tokenize(text: str) -> list[str]:
    """
    Lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens.

    Deliberately simple and dependency-free — no NLTK download step, nothing to
    go stale between the machine that builds the index and the one that serves
    it. Must stay stable: changing it silently invalidates a pickled BM25 index
    built by an earlier version.
    """
    return [
        t for t in _TOKEN.findall(text.lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


def sha256_id(*parts: str) -> str:
    """Stable content-addressed id. Same inputs always produce the same id."""
    import hashlib

    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x00")          # separator, so ("ab","c") != ("a","bc")
    return h.hexdigest()
