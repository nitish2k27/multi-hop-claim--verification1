"""
Post-retrieval document compressors.

These are `BaseDocumentCompressor`s, so they chain inside a
`DocumentCompressorPipeline` and the whole stack is a single LangChain object
the retriever knows how to call.

    ScoringCrossEncoderReranker  ->  reorder by true relevance, stamp the score
    RelevanceFloor               ->  drop anything below the floor (ABSTAIN)
    CredibilityScorer            ->  annotate with source credibility

Order matters and is not arbitrary. Reranking must come first because the floor
reads the score it stamps; credibility comes last because it should annotate
only the documents that survived.

WHY A CUSTOM RERANKER
---------------------
LangChain's stock `CrossEncoderReranker` **discards the scores it computes** —
it sorts by them, slices `top_n`, and returns bare documents:

    scores = self.model.score(...)
    result = sorted(zip(documents, scores), key=itemgetter(1), reverse=True)
    return [doc for doc, _ in result[:self.top_n]]        # scores dropped

BUILD_PLAN §6.5's `RelevanceFloor` reads `metadata['relevance_score']`, which
that reranker never sets. Chained as written, every document would score the
`0.0` default, fall below the floor, and be dropped — so *every* claim would
abstain, and the abstain test would have passed for entirely the wrong reason.
Subclassing to record the score is the smallest fix that keeps the pipeline
composition intact.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from core.credibility import score_document

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Squash a cross-encoder logit into (0, 1). Overflow-safe at both ends."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class ScoringCrossEncoderReranker(CrossEncoderReranker):
    """
    `CrossEncoderReranker` that records what it computed.

    Writes two metadata fields on every document it returns:

        rerank_logit     raw cross-encoder output, roughly [-11, +7]
        relevance_score  sigmoid of that logit, in (0, 1)

    THE SCALE MATTERS. `ms-marco-MiniLM-L-6-v2` has `num_labels=1` and
    sentence-transformers leaves its activation as `Identity()`, so `.score()`
    returns unbounded logits — measured on this corpus:

        claim vs. matching article       +7.10  ->  0.9992
        claim vs. unrelated article     -11.14  ->  0.0000
        claim vs. article that REFUTES   -0.06  ->  0.4850

    A floor expressed against the raw logit would be unreadable and would move
    with the model. Against the sigmoid it is a probability-shaped number in
    (0, 1) that survives swapping the reranker.

    That third row is the case worth protecting: an article stating the Moon is
    spherical *refutes* "the moon is square" and must not be filtered out —
    contradicting evidence is relevant evidence. It lands mid-scale, well clear
    of a 0.15 floor, while genuinely off-topic text lands at zero.
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        scores = list(self.model.score([(query, d.page_content) for d in documents]))

        ranked = sorted(
            zip(documents, scores), key=lambda pair: float(pair[1]), reverse=True
        )

        out: list[Document] = []
        for doc, raw in ranked[: self.top_n]:
            logit = float(raw)
            # Copy rather than mutate: these Documents come out of the vector
            # store and BM25 index, and stamping query-specific scores onto a
            # cached object would leak between requests.
            out.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "rerank_logit": round(logit, 4),
                        "relevance_score": round(_sigmoid(logit), 6),
                    },
                )
            )

        logger.debug(
            "Reranked %d -> %d (top score %.4f)",
            len(documents), len(out),
            out[0].metadata["relevance_score"] if out else 0.0,
        )
        return out


class RelevanceFloor(BaseDocumentCompressor):
    """
    Drop documents the reranker scored below `floor`.

    **This is the abstention mechanism.** When it empties the list the graph
    routes to web search and then to `abstain` — no LLM call is made, so an
    unanswerable claim costs nothing and returns "insufficient evidence"
    instead of a confident essay about whatever happened to rank highest.

    Retrieval always returns *something*; similarity has no notion of "nothing
    here matches". This is the component that supplies one.

    `floor` is a placeholder until M3 sweeps it against the adversarial set.
    """

    floor: float = 0.15

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        kept = [
            d for d in documents
            if d.metadata.get("relevance_score", 0.0) >= self.floor
        ]
        if len(kept) != len(documents):
            logger.info(
                "Relevance floor %.2f: kept %d of %d (best dropped: %.4f)",
                self.floor, len(kept), len(documents),
                max((d.metadata.get("relevance_score", 0.0)
                     for d in documents[len(kept):]), default=0.0),
            )
        return kept


class CredibilityScorer(BaseDocumentCompressor):
    """
    Annotate each surviving document with a source credibility assessment.

    Filters nothing — a low-credibility source is still evidence, and the report
    prompt requires the model to *flag* it rather than ignore it. Silently
    dropping tabloids would hide a judgement the report should be making out
    loud.
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        out: list[Document] = []
        for doc in documents:
            assessment: dict[str, Any] = score_document(doc.metadata)
            out.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "credibility": assessment["total"],
                        "credibility_tier": assessment["tier"],
                        "credibility_detail": assessment,
                    },
                )
            )
        return out
