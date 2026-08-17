"""
Tier 2's read side of the contract.

Opens whatever `index/CURRENT` points at, verifies it is safe to query, and
assembles the retriever stack once at startup.

**This module reads files. It imports nothing from `ingest/`.** Even the Chroma
collection name comes out of `manifest.json` rather than a shared constant —
that is the tier boundary being real rather than merely declared.

THE MANIFEST INTERLOCK
----------------------
Indexing with one embedding model and querying with another is the classic
silent RAG failure: the dimensions often match, no exception is raised anywhere,
and retrieval quietly returns plausible nonsense forever. `check_compatible()`
compares the manifest's model name and dimension against the live embedder and
refuses to start on a mismatch. It is about fifteen lines and it is the highest
value-per-line code in the project.

RETRIEVAL SHAPE
---------------
    Chroma(k=20)  --+
                    +--> EnsembleRetriever (RRF)  -->  ContextualCompressionRetriever
    BM25(k=20)    --+                                    |
                                                         +- ScoringCrossEncoderReranker
                                                         +- RelevanceFloor         <- abstain
                                                         +- CredibilityScorer
                                                              |
                                                              v
                                                    expand_to_parents()

Dense catches paraphrase, sparse catches exact names and numbers a 384-dim
vector smooths away. Reciprocal rank fusion needs no score calibration between
the two, which is what makes them combinable at all.

DEVIATION FROM BUILD_PLAN §6.6 — deliberate, and the reason matters
------------------------------------------------------------------
The plan wires `MultiVectorRetriever` as the dense leg, so parents come back
*before* reranking. That reintroduces the exact bug M1 exists to fix: the
cross-encoder truncates at 512 tokens, and the median parent article is 2,996
characters, so it would rerank on the first third of each document and score the
rest blind. Worse, an `EnsembleRetriever` over a parent-returning dense leg and
a chunk-level BM25 leg fuses two rankings of different things.

So the ensemble runs entirely at chunk level, where both legs agree on what a
document is and the cross-encoder sees text that fits, and parents are expanded
*after* compression — the same "search on chunks, reason on parents" idea, in
the only order where each model gets input it can handle. The docstore is still
LangChain's `create_kv_docstore(LocalFileStore(...))`, exactly as written by
Tier 1.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

# Must precede the langchain_chroma import — see core.init_native_libs.
from core import init_native_libs

init_native_libs()

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever  # noqa: E402
from langchain.retrievers.document_compressors import DocumentCompressorPipeline  # noqa: E402
from langchain.storage import LocalFileStore, create_kv_docstore  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from core import models  # noqa: E402
from core.compressors import (  # noqa: E402
    CredibilityScorer,
    RelevanceFloor,
    ScoringCrossEncoderReranker,
)
from core.config import cfg  # noqa: E402

logger = logging.getLogger(__name__)


class IndexUnavailable(RuntimeError):
    """No usable index. Carries the command that fixes it."""


class IndexIncompatible(RuntimeError):
    """The index exists but was built with different settings than we query with."""


# ── Loading ──────────────────────────────────────────────────────────────────

def load_manifest() -> dict[str, Any]:
    """Read the published index's manifest, or explain how to create one."""
    build = cfg.active_index
    path = cfg.manifest_path(build)
    if not path.exists():
        raise IndexUnavailable(
            f"No index published at {build}\n"
            f"  Build one first:  python -m ingest.run --from-csv"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    logger.info(
        "Index %s: %s docs, %s chunks, %s (dim %d)",
        manifest.get("version"), f"{manifest.get('documents', 0):,}",
        f"{manifest.get('chunks', 0):,}", manifest.get("embed_model"),
        manifest.get("embed_dim", -1),
    )
    return manifest


def check_compatible(manifest: dict[str, Any]) -> None:
    """
    Refuse to serve an index built with a different embedder.

    Checks the name first (cheap, catches the common case of editing
    `EMBED_MODEL` in `.env`), then actually embeds a probe string to compare
    dimensions — because a renamed or re-pinned model can keep its name and
    change its output, and that failure is invisible from the config alone.
    """
    indexed_model = manifest.get("embed_model")
    if indexed_model != cfg.embed_model:
        raise IndexIncompatible(
            f"Embedding model mismatch — refusing to serve.\n"
            f"  index built with : {indexed_model}\n"
            f"  configured now   : {cfg.embed_model}\n"
            f"  Querying an index with a different model returns plausible "
            f"nonsense and raises nothing.\n"
            f"  Fix: set EMBED_MODEL={indexed_model} in .env, or rebuild with "
            f"python -m ingest.run --from-csv --force"
        )

    live_dim = len(models.embedder().embed_query("dimension probe"))
    indexed_dim = manifest.get("embed_dim")
    if live_dim != indexed_dim:
        raise IndexIncompatible(
            f"Embedding dimension mismatch — refusing to serve.\n"
            f"  index built with : {indexed_dim}\n"
            f"  live model gives : {live_dim}\n"
            f"  Same model name, different output. Rebuild the index."
        )

    logger.info("Manifest check passed (%s, dim %d)", indexed_model, live_dim)


def _load_vectorstore(build: Path, manifest: dict[str, Any]) -> Chroma:
    """Collection name comes from the manifest, never from an ingest import."""
    return Chroma(
        collection_name=manifest.get("collection", "evidence"),
        embedding_function=models.embedder(),
        persist_directory=str(cfg.chroma_path(build)),
    )


def _load_bm25(build: Path) -> BaseRetriever:
    """
    Unpickle the sparse index Tier 1 built.

    The pickle references `core.text.bm25_tokenize` by module path, which is why
    that function lives in `core/` and not in `ingest/` — unpickling here must
    not drag Tier 1 into Tier 2's import graph.
    """
    path = cfg.bm25_path(build)
    if not path.exists():
        raise IndexUnavailable(f"Missing sparse index: {path}")
    with open(path, "rb") as fh:
        bm25 = pickle.load(fh)
    bm25.k = cfg.candidate_k
    return bm25


def _load_parent_store(build: Path):
    """Whole articles, keyed by `parent_id`. Written by Tier 1's index build."""
    return create_kv_docstore(LocalFileStore(str(cfg.parents_path(build))))


# ── Assembly ─────────────────────────────────────────────────────────────────

class EvidenceRetriever:
    """
    The assembled read path, built once and reused across every verification.

    Everything here is expensive to construct (model loads, an unpickle of
    13,101 chunks) and cheap to call. The old code rebuilt the BM25 index inside
    each request handler; this is that fix made structural.
    """

    def __init__(self) -> None:
        self.manifest = load_manifest()
        check_compatible(self.manifest)

        build = cfg.active_index
        self.build_dir = build
        self.vectorstore = _load_vectorstore(build, self.manifest)
        self.parents = _load_parent_store(build)

        dense = self.vectorstore.as_retriever(
            search_kwargs={"k": cfg.candidate_k}
        )
        sparse = _load_bm25(build)

        # Equal weights: neither leg has earned a thumb on the scale until M3
        # measures one. Another number to calibrate, not guess.
        self.base = EnsembleRetriever(
            retrievers=[dense, sparse], weights=[0.5, 0.5]
        )

        self.pipeline = DocumentCompressorPipeline(transformers=[
            ScoringCrossEncoderReranker(
                model=models.cross_encoder(), top_n=cfg.top_k
            ),
            RelevanceFloor(floor=cfg.relevance_floor),
            CredibilityScorer(),
        ])

        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.pipeline,
            base_retriever=self.base,
        )

        logger.info(
            "Retriever ready — candidates %d, top_k %d, floor %.2f",
            cfg.candidate_k, cfg.top_k, cfg.relevance_floor,
        )

    # ── Query ────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> tuple[list[Document], dict[str, Any]]:
        """
        Run the full read path.

        Returns `(documents, stats)`. An empty list is a legitimate, expected
        result — it is what routes the graph to the abstain path — so it is
        never an error, and the stats explain which stage emptied it.
        """
        candidates = self.base.invoke(query)
        kept = self.pipeline.compress_documents(candidates, query)
        expanded = self.expand_to_parents(list(kept))

        stats = {
            "candidates": len(candidates),
            "after_compression": len(kept),
            "after_parent_expansion": len(expanded),
            "top_relevance": (
                max((d.metadata.get("relevance_score", 0.0) for d in kept),
                    default=0.0)
            ),
            # What the best *rejected* candidate scored. When a claim abstains
            # this is the number that says whether it was a near miss or
            # nowhere close, and it is the signal M3's floor sweep needs.
            "best_rejected": self._best_rejected(candidates, kept, query),
            "floor": cfg.relevance_floor,
        }
        logger.info(
            "Retrieved %d candidates -> %d chunks above floor %.2f "
            "-> %d articles (top relevance %.4f)",
            stats["candidates"], stats["after_compression"],
            cfg.relevance_floor, stats["after_parent_expansion"],
            stats["top_relevance"],
        )
        return expanded, stats

    def rank_chunks(self, query: str, k: int, *, rerank: bool = True) -> list[Document]:
        """
        Top-k chunks by rank, with **no relevance floor and no parent expansion**.

        For the eval harness only. Recall@k measures whether the right document
        is findable at all, which is a property of the ranking; applying the
        floor first would conflate that with the abstention threshold, and the
        floor sweep needs to vary that threshold independently. Parent expansion
        is skipped so `k` counts retrieved units rather than post-dedup articles.

        `rerank=False` gives the raw fused ensemble order, which isolates how
        much the cross-encoder is contributing.
        """
        candidates = self.base.invoke(query)
        if not rerank:
            return list(candidates)[:k]
        reranker = ScoringCrossEncoderReranker(
            model=models.cross_encoder(), top_n=k
        )
        return list(reranker.compress_documents(candidates, query))

    def dense_only(self, query: str, k: int) -> list[Document]:
        """
        Dense retrieval against the new index, with nothing else applied.

        This is the like-for-like counterpart to the old index, which was also
        dense-only Chroma over the same embedding model. Comparing it against
        `BaselineIndex.search` isolates the chunking change from the later
        additions (BM25 fusion, reranking) that the old system never had.
        """
        return self.vectorstore.similarity_search(query, k=k)

    def _best_rejected(self, candidates, kept, query: str) -> float:
        """Highest relevance score among candidates the floor dropped."""
        if kept:
            return 0.0
        if not candidates:
            return 0.0
        # Nothing survived, so rerank without the floor purely to report the
        # ceiling. This is a second cross-encoder pass over the same candidates
        # — the pipeline runs as one object, so the first pass's scores are not
        # reachable from out here once the floor has emptied the list. Accepted
        # because it only ever runs on the abstain path, which makes no LLM
        # call, and because M3's floor sweep needs this number.
        reranker = ScoringCrossEncoderReranker(
            model=models.cross_encoder(), top_n=1
        )
        scored = reranker.compress_documents(candidates, query)
        return scored[0].metadata["relevance_score"] if scored else 0.0

    def expand_to_parents(self, chunks: list[Document]) -> list[Document]:
        """
        Swap each surviving chunk for its whole article.

        Retrieval ranked chunks because that is what embeds well; the LLM and
        the stance model reason better over a full article than a 700-character
        window that may begin mid-sentence.

        Rank order is preserved and parents are de-duplicated: three chunks from
        one article must become one piece of evidence, not three, or the report
        will read a single source as three-way corroboration. The chunk's own
        scores ride along on the parent so nothing downstream loses them.
        """
        if not chunks:
            return []

        ordered_ids: list[str] = []
        best_chunk: dict[str, Document] = {}
        for chunk in chunks:
            pid = chunk.metadata.get("parent_id")
            if not pid:
                continue
            if pid not in best_chunk:
                ordered_ids.append(pid)
                best_chunk[pid] = chunk  # first == highest ranked

        fetched = self.parents.mget(ordered_ids)

        out: list[Document] = []
        for pid, parent in zip(ordered_ids, fetched):
            chunk = best_chunk[pid]
            if parent is None:
                # Docstore miss: serve the chunk rather than dropping evidence.
                logger.warning("No parent for %s — falling back to chunk", pid)
                out.append(chunk)
                continue
            out.append(Document(
                page_content=parent.page_content,
                metadata={
                    **parent.metadata,
                    "relevance_score": chunk.metadata.get("relevance_score"),
                    "rerank_logit": chunk.metadata.get("rerank_logit"),
                    "credibility": chunk.metadata.get("credibility"),
                    "credibility_tier": chunk.metadata.get("credibility_tier"),
                    "credibility_detail": chunk.metadata.get("credibility_detail"),
                    # Kept for the report: the passage that actually matched is
                    # far better citation material than a 3,000-character article.
                    "matched_chunk": chunk.page_content,
                },
            ))

        if len(out) < len(chunks):
            logger.debug(
                "Parent expansion collapsed %d chunks into %d articles",
                len(chunks), len(out),
            )
        return out


_INSTANCE: EvidenceRetriever | None = None


def get_retriever() -> EvidenceRetriever:
    """Process-wide singleton. First call pays the startup cost."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = EvidenceRetriever()
    return _INSTANCE
