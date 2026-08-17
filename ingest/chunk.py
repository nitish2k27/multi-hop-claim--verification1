"""
Tier 1, stage 4 — split articles into embeddable chunks.

THE BUG THIS FIXES
------------------
`all-MiniLM-L6-v2` has `max_seq_length: 256` tokens — roughly 1,000 characters.
The old pipeline passed whole articles to `encode()`, and SentenceTransformer
truncated them silently. With a median article of 2,996 characters, about 65% of
the median document was never in the vector at all, and no error was raised
anywhere.

Worse, the old BM25 index was built over the *full* text, so dense and sparse
retrieval disagreed about what a document even contained, and RRF was fusing two
rankings over different corpora.

700 characters is ~180 tokens — comfortably inside the cap with room for the
model's special tokens.

SEARCH ON CHUNKS, REASON ON PARENTS
-----------------------------------
Every chunk carries `parent_id`, which keys the parent docstore Tier 2's
`MultiVectorRetriever` reads. Retrieval gets chunk-level precision; stance
detection gets the whole article for context.

`parent_id` is a *hash of the URL*, not the URL itself: `LocalFileStore` uses
keys directly as filenames, and a URL's `/ ? : &` are illegal in Windows paths.
The readable URL survives as its own metadata field.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import cfg
from core.text import sha256_id

logger = logging.getLogger(__name__)


def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Separator order matters: prefer paragraph breaks, then lines, then sentence
    ends, and only fall back to a hard character cut when nothing else fits. A
    chunk that ends mid-sentence embeds noticeably worse than one that ends on a
    boundary.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _clean_metadata(meta: dict) -> dict:
    """
    Chroma accepts only str / int / float / bool in metadata.

    A `None` raises at insert time and a list is silently dropped, so normalise
    here rather than discovering it 12,000 rows into a build.
    """
    out = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def to_documents(records: Iterable[dict]) -> Iterator[tuple[Document, list[Document]]]:
    """
    Yield `(parent_document, chunks)` for each source record.

    The parent is stored whole in the docstore; only the chunks are embedded.
    """
    splitter = build_splitter()

    for record in records:
        text = record["text"]
        url = record["url"]
        parent_id = sha256_id(url)

        # Shared by the parent and every one of its chunks, so a retrieved chunk
        # carries everything credibility scoring and the report renderer need
        # without a second lookup.
        base = _clean_metadata({
            "parent_id":     parent_id,
            "url":           url,
            "title":         record.get("title"),
            "source":        record.get("source"),
            "domain":        record.get("domain"),      # credibility keys off THIS
            "publish_date":  record.get("publish_date"),
            "date_reliable": record.get("date_reliable", False),
            "language":      record.get("language", "en"),
            "category":      record.get("category"),
            "corpus_tier":   record.get("corpus_tier", "news"),
        })

        parent = Document(page_content=text, metadata=dict(base))

        chunks: list[Document] = []
        for i, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if not piece:
                continue
            meta = dict(base)
            meta["chunk_index"] = i
            chunks.append(Document(page_content=piece, metadata=meta))

        if chunks:
            yield parent, chunks
