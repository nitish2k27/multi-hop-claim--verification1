"""
Tier 1, stage 5 — build the artifact set that is the contract with Tier 2.

A build writes four things into `index/v{n}/`:

    chroma/        vector index over chunks
    parents/       whole articles, keyed by parent_id (MultiVectorRetriever)
    bm25.pkl       sparse index over the same chunks
    manifest.json  what was built, with what, and whether to trust the dates

...validates them, and only then points `index/CURRENT` at the new version.
Nothing is mutated in place, so a failed or half-finished build can never take
down a serving index — and rolling back is a one-line file write.

WHY BM25 IS BUILT HERE
----------------------
`BM25Retriever.from_documents()` loads every chunk into memory and tokenises it.
The old code did that inside the request handler, on every single verification,
over all 1,687 documents. It belongs in the build, pickled once, loaded once at
Tier 2 startup.

WHY THE MANIFEST IS A SAFETY INTERLOCK
--------------------------------------
It records the embedding model and dimension. Tier 2 refuses to boot on a
mismatch. Indexing with one model and querying with another produces plausible
garbage and raises nothing anywhere — this is the cheapest possible guard
against the single most common silent RAG failure.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

# MUST run before langchain_chroma pulls in chromadb — see core.init_native_libs.
# Importing chromadb first hard-crashes the process (0xC0000005) when
# sentence_transformers is imported afterwards. Do not reorder these two lines.
from core import init_native_libs

init_native_libs()

from langchain.embeddings import CacheBackedEmbeddings  # noqa: E402
from langchain.storage import LocalFileStore, create_kv_docstore  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from core.config import cfg  # noqa: E402
from core.text import bm25_tokenize, sha256_id  # noqa: E402

logger = logging.getLogger(__name__)

COLLECTION = "evidence"

# Chroma rejects oversized single inserts; 500 stays well inside every backend
# limit while keeping the number of round trips small.
UPSERT_BATCH = 500


# ── Embeddings ────────────────────────────────────────────────────────────────

def build_embedder() -> CacheBackedEmbeddings:
    """
    MiniLM, wrapped in a persistent cache.

    The cache is namespaced by model name, so switching `EMBED_MODEL` cannot
    serve vectors computed by a different model. Re-running a build skips
    embedding for every chunk seen before, which is what makes an incremental
    run take seconds instead of minutes.
    """
    underlying = HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        model_kwargs={"device": cfg.device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    cfg.embed_cache.mkdir(parents=True, exist_ok=True)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying,
        LocalFileStore(str(cfg.embed_cache)),
        namespace=cfg.embed_model,
    )


# ── Fingerprint (what makes a re-run a no-op) ─────────────────────────────────

def corpus_fingerprint(parents: list[Document]) -> str:
    """
    One hash covering the input set *and* every parameter that changes the
    output. If this matches the live manifest, rebuilding would reproduce the
    same artifacts byte-for-byte, so the run can exit immediately.
    """
    return sha256_id(
        *sorted(p.metadata["parent_id"] for p in parents),
        cfg.embed_model,
        str(cfg.chunk_size),
        str(cfg.chunk_overlap),
    )


def live_fingerprint() -> str | None:
    """Fingerprint recorded by the currently-published index, if any."""
    manifest = cfg.manifest_path(cfg.active_index)
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("fingerprint")
    except (json.JSONDecodeError, OSError):
        return None


# ── Build ─────────────────────────────────────────────────────────────────────

def build(
    parents: list[Document],
    chunks: list[Document],
    *,
    clean_stats: dict | None = None,
) -> Path:
    """
    Write a complete, validated index into a fresh version directory.

    Returns the build directory. Does **not** publish it — the caller decides
    whether to move the pointer, so a validation failure leaves the live index
    untouched.
    """
    version = cfg.next_version()
    build_dir = cfg.index_root / version
    build_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Building %s", build_dir)

    try:
        embedder = build_embedder()
        dim = len(embedder.embed_query("dimension probe"))
        logger.info("Embedding model %s -> %d dimensions", cfg.embed_model, dim)

        # ── Vector index ──────────────────────────────────────────────────
        # Chunk ids are content hashes, so the build is deterministic and a
        # re-run upserts identical rows. Identical text collapsing to one row is
        # intentional: syndicated wire copy republished on three sites must not
        # look like three independent sources corroborating a claim.
        store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embedder,
            persist_directory=str(cfg.chroma_path(build_dir)),
        )

        seen: set[str] = set()
        unique: list[Document] = []
        ids: list[str] = []
        for chunk in chunks:
            cid = sha256_id(chunk.page_content)
            if cid in seen:
                continue
            seen.add(cid)
            unique.append(chunk)
            ids.append(cid)

        collapsed = len(chunks) - len(unique)
        if collapsed:
            logger.info("Collapsed %s duplicate chunks (syndicated copy)", f"{collapsed:,}")

        logger.info("Embedding and indexing %s chunks...", f"{len(unique):,}")
        for start in range(0, len(unique), UPSERT_BATCH):
            end = start + UPSERT_BATCH
            store.add_documents(unique[start:end], ids=ids[start:end])
            logger.info("  %s / %s", f"{min(end, len(unique)):,}", f"{len(unique):,}")

        # ── Parent docstore ───────────────────────────────────────────────
        # Persistent, not in-memory: it has to survive the handoff to Tier 2,
        # which is exactly why MultiVectorRetriever's default InMemoryStore
        # would not work across the tier boundary.
        parent_store = create_kv_docstore(
            LocalFileStore(str(cfg.parents_path(build_dir)))
        )
        parent_store.mset([(p.metadata["parent_id"], p) for p in parents])
        logger.info("Stored %s parent documents", f"{len(parents):,}")

        # ── Sparse index ──────────────────────────────────────────────────
        # preprocess_func must be an importable module-level function; a lambda
        # pickles as a reference that Tier 2 cannot resolve.
        bm25 = BM25Retriever.from_documents(unique, preprocess_func=bm25_tokenize)
        bm25.k = cfg.candidate_k
        with open(cfg.bm25_path(build_dir), "wb") as fh:
            pickle.dump(bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Pickled BM25 index (%s chunks)", f"{len(unique):,}")

        # ── Manifest ──────────────────────────────────────────────────────
        unreliable = sum(
            1 for p in parents if not p.metadata.get("date_reliable", False)
        )
        manifest = {
            "version":        version,
            "built_at":       datetime.now(timezone.utc).isoformat(),
            "fingerprint":    corpus_fingerprint(parents),
            "embed_model":    cfg.embed_model,
            "embed_dim":      dim,
            "chunk_size":     cfg.chunk_size,
            "chunk_overlap":  cfg.chunk_overlap,
            "collection":     COLLECTION,
            "documents":      len(parents),
            "chunks":         len(unique),
            "chunks_collapsed": collapsed,
            # Disclosed, not hidden. core/credibility.py drops the recency term
            # for any document whose own date_reliable is false; this flag tells
            # a reader of the index that the corpus as a whole has the defect.
            "dates_reliable": unreliable == 0,
            "dates_unreliable_count": unreliable,
            "tiers": _count(parents, "corpus_tier"),
            "sources": _count(parents, "source"),
            "clean_stats": clean_stats or {},
        }
        cfg.manifest_path(build_dir).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        validate(build_dir, store)
        return build_dir

    except Exception:
        # A partial build is worse than none — it would satisfy a naive
        # existence check while returning nothing useful.
        logger.error("Build failed; removing %s", build_dir)
        shutil.rmtree(build_dir, ignore_errors=True)
        raise


def _count(docs: list[Document], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in docs:
        value = str(d.metadata.get(key) or "unknown")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ── Validate & publish ───────────────────────────────────────────────────────

def validate(build_dir: Path, store: Chroma | None = None) -> None:
    """
    Prove the build is usable before anything points at it.

    Checks presence, non-emptiness, and — critically — that a real query returns
    real hits. An index that exists but retrieves nothing is the failure mode
    that would otherwise surface as "the model can't answer anything".
    """
    manifest = json.loads(
        cfg.manifest_path(build_dir).read_text(encoding="utf-8")
    )

    for path in (
        cfg.chroma_path(build_dir),
        cfg.parents_path(build_dir),
        cfg.bm25_path(build_dir),
    ):
        if not path.exists():
            raise RuntimeError(f"Build incomplete — missing {path.name}")

    if manifest["chunks"] < 1:
        raise RuntimeError("Build produced zero chunks")

    if store is None:
        store = Chroma(
            collection_name=manifest["collection"],
            embedding_function=build_embedder(),
            persist_directory=str(cfg.chroma_path(build_dir)),
        )

    hits = store.similarity_search("economy growth government", k=3)
    if not hits:
        raise RuntimeError("Smoke query returned no results — index is unusable")

    # Confirm the pickled sparse index survives a round trip *and* that its
    # tokenizer resolves. If bm25_tokenize ever moved out of core/, this is
    # where it would surface — at build time, not in a request handler.
    with open(cfg.bm25_path(build_dir), "rb") as fh:
        bm25 = pickle.load(fh)
    if not bm25.invoke("economy growth government"):
        raise RuntimeError("BM25 index returned no results — pickle is unusable")

    logger.info("Validation passed (%s chunks, dim %d)",
                f"{manifest['chunks']:,}", manifest["embed_dim"])


def publish(build_dir: Path) -> None:
    """Point CURRENT at a validated build. Single atomic file write."""
    (cfg.index_root / cfg.POINTER).write_text(build_dir.name, encoding="utf-8")
    logger.info("Published %s -> index/%s", cfg.POINTER, build_dir.name)
