"""
Web search fallback — the second chance before abstaining.

Fires **only** when the index returned nothing above the relevance floor. A
claim the corpus can answer never triggers a search, so this costs nothing on
the normal path and the Tavily quota is spent only where it might help.

    retrieve --no evidence--> web --no evidence--> abstain
                               |
                               +--evidence--> stance -> generate

WHY THIS DOES NOT DELETE THE ABSTAIN PATH
-----------------------------------------
The obvious risk of adding web search to a system whose whole point is
abstention: the web has *something* about everything, so the system stops
abstaining and the differentiator disappears.

Two things prevent that:

1. **Web results go through the same compressor pipeline as index results** —
   reranker, then the same relevance floor, then credibility. Search returning
   ten links is not evidence; clearing 0.25 against the claim is. A search for
   "the moon is square in shape" returns pages about the Moon, and pages that
   *refute* the claim score mid-range and are correctly kept — the LLM then
   returns FALSE with citations, which is a better answer than abstaining.
   Genuinely off-topic results score near zero and are dropped exactly as
   corpus chunks are.

2. **Abstain remains reachable** and is the routed destination whenever the
   floor empties the web results too, when no key is configured, and when the
   search errors.

CREDIBILITY OF WEB RESULTS
--------------------------
Scored by the same `core.credibility` table as corpus documents, keyed on the
result's domain — `reuters.com` found via search is as credible as
`reuters.com` found in the index, which is the correct behaviour. Unknown
domains get the 0.50 default. `date_reliable` is set False unless Tavily
returns a published date, so credibility falls back to domain 0.7 / type 0.3
rather than scoring a date nobody verified.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from core.config import cfg
from serve.schemas import VerifyState

logger = logging.getLogger(__name__)

# Content farms and aggregators that mostly restate other sources. Excluded at
# the query so the search budget goes to primary reporting.
EXCLUDE_DOMAINS = [
    "pinterest.com",
    "quora.com",
    "answers.com",
    "ask.com",
]

_WARNED_NO_KEY = False


def _tavily_unavailable() -> str | None:
    """Return a reason string if web search cannot run, else None."""
    if not cfg.web_search_enabled:
        return "disabled (WEB_SEARCH_ENABLED=false)"
    if not cfg.tavily_api_key:
        return "no TAVILY_API_KEY configured"
    try:
        import tavily  # noqa: F401
    except ImportError:
        return "tavily-python not installed (pip install -e '.[search]')"
    return None


def _build_retriever():
    """
    Tavily as a LangChain `BaseRetriever`.

    Deliberately `langchain_community`'s wrapper rather than the standalone
    `langchain-tavily` package: that one requires langchain-core>=1.0 and
    upgrading to it moves `langchain.retrievers`, which breaks every retriever
    import in `serve/`. See the comment in pyproject.toml.
    """
    from langchain_community.retrievers import TavilySearchAPIRetriever

    return TavilySearchAPIRetriever(
        api_key=cfg.tavily_api_key,
        k=cfg.web_search_k,
        search_depth=cfg.web_search_depth,
        # The full page text, not Tavily's short snippet. The reranker needs
        # enough text to judge relevance, and a two-line snippet is not enough
        # to support a verdict either.
        include_raw_content=True,
        # Tavily's own LLM-generated answer is deliberately NOT requested. It
        # would be an unsourced assertion entering an evidence pipeline, and
        # every citation in the report has to resolve to a real document.
        include_generated_answer=False,
        exclude_domains=EXCLUDE_DOMAINS,
    )


def _to_evidence(documents: list[Document]) -> list[Document]:
    """
    Normalise Tavily documents onto the same metadata shape as index chunks.

    Everything downstream — credibility, the prompt's evidence block, the
    renderer, citation validation — reads `url` / `domain` / `title`. Tavily
    puts the URL in `source`, so without this remapping web evidence would show
    up as `unknown` with a 0.50 default credibility regardless of publisher.
    """
    out: list[Document] = []
    for doc in documents:
        meta = doc.metadata
        url = meta.get("source") or meta.get("url") or ""
        domain = urlparse(url).netloc.lower() if url else "unknown"

        text = (doc.page_content or "").strip()
        if not text:
            continue

        out.append(Document(
            page_content=text,
            metadata={
                "url": url,
                "domain": domain,
                "title": meta.get("title") or "",
                "source": domain,
                # Marks provenance for the report and the eval harness. The
                # renderer surfaces this so a reader can tell corpus evidence
                # from live web evidence.
                "corpus_tier": "web",
                "retrieved_from": "web_search",
                # No verified publication date, so credibility drops the
                # recency term rather than trusting whatever the page claims.
                "date_reliable": False,
                "publish_date": meta.get("published_date") or "",
                # Tavily's own relevance score, kept for diagnostics. NOT used
                # as `relevance_score` — that field belongs to our cross-encoder
                # so the floor compares like with like across both sources.
                "tavily_score": meta.get("score"),
            },
        ))
    return out


def web_search(state: VerifyState) -> dict:
    """
    Search the live web, then hold the results to the same bar as the index.

    Returns the same shape as `retrieve`, so the graph's downstream nodes
    cannot tell which source the evidence came from — only the metadata says.
    """
    global _WARNED_NO_KEY

    claim = state.get("claim_english") or state.get("claim", "")
    stats = dict(state.get("retrieval_stats") or {})

    reason = _tavily_unavailable()
    if reason:
        if not _WARNED_NO_KEY:
            logger.warning("Web search unavailable — %s. Routing to abstain.", reason)
            _WARNED_NO_KEY = True
        stats["web_skipped"] = reason
        return {"evidence": [], "evidence_source": "none", "retrieval_stats": stats}

    logger.info("Index found nothing — searching the web for %r", claim[:60])

    try:
        raw = _build_retriever().invoke(claim)
    except Exception as exc:
        # A search failure must not take down a verification. Abstaining is a
        # correct, already-implemented outcome; crashing is not.
        logger.error("Web search failed (%s) — routing to abstain", exc)
        stats["web_error"] = str(exc)
        return {"evidence": [], "evidence_source": "none", "retrieval_stats": stats}

    candidates = _to_evidence(raw)
    stats["web_results"] = len(candidates)

    if not candidates:
        logger.info("Web search returned nothing usable")
        return {"evidence": [], "evidence_source": "none", "retrieval_stats": stats}

    # The same compressor pipeline the index results go through. This is the
    # line that keeps abstention meaningful: search hits are candidates, not
    # evidence, until they clear the floor.
    from serve.retriever import get_retriever

    kept = list(get_retriever().pipeline.compress_documents(candidates, claim))

    stats["web_after_floor"] = len(kept)
    stats["web_top_relevance"] = max(
        (d.metadata.get("relevance_score", 0.0) for d in kept), default=0.0
    )

    logger.info(
        "Web: %d results -> %d above floor %.2f (top %.4f)",
        len(candidates), len(kept), cfg.relevance_floor,
        stats["web_top_relevance"],
    )

    return {
        "evidence": kept,
        "evidence_source": "web" if kept else "none",
        "retrieval_stats": stats,
    }


web_node = RunnableLambda(web_search, name="web_search")
