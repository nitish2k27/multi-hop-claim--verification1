"""
Evidence retrieval against the published index.

Thin by design: `serve.retriever` owns the retriever stack and the manifest
interlock, and this node's only job is to run it and put the result in the
graph state. Keeping assembly out of the node is what stops the old bug where
the BM25 index was rebuilt over all 1,687 documents inside every request.

Returning an empty `evidence` list is a normal outcome, not a failure — it is
the signal the graph routes on. See `core.compressors.RelevanceFloor`.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableLambda

from serve.retriever import get_retriever
from serve.schemas import VerifyState

logger = logging.getLogger(__name__)


def retrieve_evidence(state: VerifyState) -> dict:
    """Query the index and record both the evidence and why there was so little."""
    query = state.get("claim_english") or state.get("claim", "")

    retriever = get_retriever()
    documents, stats = retriever.retrieve(query)

    # An attached document joins the evidence, scored by the *same* pipeline —
    # so an upload that has nothing to do with the claim is dropped by the floor
    # rather than padding the list. Its credibility is capped at 0.6 upstream
    # (see serve.nodes.adapt._as_upload_evidence), which is what stops a user
    # from making their own assertion true by attaching a file that states it.
    uploads = state.get("upload_evidence") or []
    if uploads:
        scored = list(retriever.pipeline.compress_documents(uploads, query))
        stats["upload_documents"] = len(uploads)
        stats["upload_kept"] = len(scored)
        if scored:
            # Placed first: it is the user's own document and they expect to see
            # it considered. Ranking is not authority — the 0.6 cap and the
            # "unverified submission" label carry that.
            documents = scored + documents
            logger.info("Attached document kept as evidence (relevance %.4f, "
                        "credibility %.2f)",
                        scored[0].metadata.get("relevance_score", 0.0),
                        scored[0].metadata.get("credibility", 0.0))
        else:
            logger.info("Attached document dropped — not relevant to the claim")

    if not documents:
        logger.info(
            "No evidence above floor %.2f (best candidate scored %.4f)",
            stats["floor"], stats["best_rejected"],
        )

    source = "none"
    if documents:
        source = "upload" if uploads and stats.get("upload_kept") else "index"

    return {
        "evidence": documents,
        "evidence_source": source,
        "retrieval_stats": stats,
    }


retrieve_node = RunnableLambda(retrieve_evidence, name="retrieve_evidence")
