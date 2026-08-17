"""
Query the OLD index — `data/chroma_db/` — for the before/after comparison.

This is why CLAUDE.md lists "`data/chroma_db/` stays untouched" as an invariant.
It is the only surviving artifact of the pre-rebuild system, and it cannot be
regenerated: the bug it demonstrates is that text was never embedded, and there
is no way to un-embed the new index to recreate that state.

WHAT IS ACTUALLY IN THERE
-------------------------
    collection      news_articles
    items           1,687          <- one row per ARTICLE, not per chunk
    embedding dim   384            <- same all-MiniLM-L6-v2
    ids             the article URL
    metadata        language, title, domain, source, publish_date

Verified against the source CSV: the stored `documents` total 7,338,249
characters, byte-for-byte the same as the CSV's `text` column. So the old index
stored every article **in full** and embedded only the first ~256 tokens of each.
The information loss was entirely at embed time, which is exactly why it was
invisible — inspecting the database shows complete articles.

Two things make the comparison fair:

* Same embedding model and dimension, so query vectors live in the same space.
* Queried through the raw `chromadb` client rather than `langchain_chroma`, so
  ids come back explicitly and the gold match is an exact URL comparison rather
  than a guess based on page content.
"""

from __future__ import annotations

import logging
from typing import Any

# Native-library import order still applies here — see core.init_native_libs.
from core import init_native_libs

init_native_libs()

import chromadb  # noqa: E402

from core import models  # noqa: E402
from core.config import ROOT  # noqa: E402

logger = logging.getLogger(__name__)

BASELINE_PATH = ROOT / "data" / "chroma_db"
BASELINE_COLLECTION = "news_articles"


class BaselineUnavailable(RuntimeError):
    """The old index is missing. The A/B cannot run without it."""


class BaselineIndex:
    """Read-only handle on the pre-rebuild Chroma index."""

    def __init__(self) -> None:
        if not (BASELINE_PATH / "chroma.sqlite3").exists():
            raise BaselineUnavailable(
                f"Old index not found at {BASELINE_PATH}\n"
                f"  This is the before-baseline for the M3 A/B and cannot be "
                f"rebuilt — the bug it demonstrates is missing embeddings.\n"
                f"  Without it, run the eval with --skip-baseline."
            )

        # Explicitly read-only. Nothing in the eval should ever be able to
        # modify the one artifact that cannot be regenerated.
        self._client = chromadb.PersistentClient(path=str(BASELINE_PATH))
        self._collection = self._client.get_collection(BASELINE_COLLECTION)
        self._embedder = models.embedder()

        logger.info(
            "Baseline index: %s items in %r",
            f"{self._collection.count():,}", BASELINE_COLLECTION,
        )

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """
        Top-k articles by dense similarity, with their URLs.

        The embedding is computed with the *same* model object the new index
        uses, so any difference in results comes from what was indexed, not from
        how the query was encoded.
        """
        vector = self._embedder.embed_query(query)
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for rank, doc_id in enumerate(ids):
            metadata = metadatas[rank] if rank < len(metadatas) else {}
            hits.append({
                "rank": rank + 1,
                # The id IS the URL in this collection — that is what makes a
                # gold match an exact string comparison.
                "url": doc_id,
                "distance": distances[rank] if rank < len(distances) else None,
                "domain": (metadata or {}).get("domain"),
                "title": (metadata or {}).get("title"),
            })
        return hits


_INSTANCE: BaselineIndex | None = None


def get_baseline() -> BaselineIndex:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = BaselineIndex()
    return _INSTANCE
