"""
Tests for the failures that produce no error.

This file is deliberately small and deliberately not "coverage". Every test here
exists because the failure it catches is **invisible** — it produces plausible
output, raises nothing, and would ship. The old `tests/` directory had 13 files
and zero `assert` statements; these nine are worth more than all of it.

Each test names the bug it prevents. Run with:

    pytest -q

Most need neither an LLM nor a network, so they stay runnable when the Groq
daily quota is gone.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from tests.conftest import needs_index, needs_models


# ── 1. Index/embedder mismatch ───────────────────────────────────────────────

@needs_index
def test_manifest_mismatch_refuses_to_serve(monkeypatch):
    """
    Indexing with one embedding model and querying with another is the classic
    silent RAG failure: dimensions often still match, nothing raises, and
    retrieval returns confident nonsense forever. Tier 2 must refuse to boot.
    """
    from core.config import cfg
    from serve.retriever import IndexIncompatible, check_compatible, load_manifest

    manifest = load_manifest()
    check_compatible(manifest)  # the real config must pass

    monkeypatch.setattr(cfg, "embed_model", "sentence-transformers/all-mpnet-base-v2")
    with pytest.raises(IndexIncompatible) as exc:
        check_compatible(manifest)

    # The message has to be actionable, not just correct.
    assert "EMBED_MODEL" in str(exc.value)
    assert manifest["embed_model"] in str(exc.value)


@needs_index
def test_manifest_dimension_mismatch_is_caught(monkeypatch):
    """
    A model can keep its name and change its output. Checking the name alone
    would pass; the dimension probe is what actually catches it.
    """
    from serve.retriever import IndexIncompatible, check_compatible, load_manifest

    manifest = dict(load_manifest())
    manifest["embed_dim"] = 768  # index claims 768, live model gives 384

    with pytest.raises(IndexIncompatible, match="dimension"):
        check_compatible(manifest)


# ── 2. Ingest idempotency ────────────────────────────────────────────────────

@needs_index
def test_reingest_is_a_noop():
    """
    Re-running ingestion must not rebuild. The fingerprint covers the input set
    *and* every parameter that changes the output, so an unchanged corpus has to
    reproduce the live fingerprint exactly.

    Without this, a re-run silently produces index/v2, v3, v4... each a full
    13-minute embed of identical content.
    """
    from ingest.chunk import to_documents
    from ingest.clean import CleanStats, load_csv
    from ingest.index import corpus_fingerprint, live_fingerprint
    from core.config import cfg

    parents = [p for p, _ in to_documents(load_csv(cfg.source_csv, CleanStats()))]
    assert corpus_fingerprint(parents) == live_fingerprint()


@needs_index
def test_fingerprint_changes_with_chunk_size(monkeypatch):
    """
    The fingerprint must cover build *parameters*, not just the documents.
    If it only hashed the corpus, changing chunk_size would be a no-op and the
    index would silently keep the old chunking.
    """
    from core.config import cfg
    from ingest.index import corpus_fingerprint

    parents = [Document(page_content="x", metadata={"parent_id": "a"})]
    before = corpus_fingerprint(parents)

    monkeypatch.setattr(cfg, "chunk_size", cfg.chunk_size + 100)
    assert corpus_fingerprint(parents) != before


# ── 3 & 4. The relevance floor and the score it depends on ───────────────────

def test_relevance_floor_filters_on_stamped_score(doc):
    """The floor is the abstention mechanism. It must actually filter."""
    from core.compressors import RelevanceFloor

    documents = [
        doc("a", relevance_score=0.97),
        doc("b", relevance_score=0.26),
        doc("c", relevance_score=0.24),
        doc("d"),  # never scored
    ]
    kept = RelevanceFloor(floor=0.25).compress_documents(documents, "q")

    assert [d.page_content for d in kept] == ["a", "b"]


@needs_index
def test_reranker_stamps_relevance_score(retriever, doc):
    """
    THE BUG THIS CATCHES, and why it is worth a dedicated test:

    LangChain's stock `CrossEncoderReranker` computes scores, sorts by them, and
    **throws them away** — it never writes `relevance_score` to metadata. Wired
    naively, `RelevanceFloor` would then read the 0.0 default for every
    document, drop everything, and *every claim would abstain*.

    That failure looks like success: the abstain test passes, the pipeline
    "works", and the system silently answers nothing. `ScoringCrossEncoderReranker`
    exists solely to prevent it, so the stamping is asserted directly.
    """
    from core.compressors import ScoringCrossEncoderReranker
    from core import models

    reranker = ScoringCrossEncoderReranker(model=models.cross_encoder(), top_n=2)
    documents = [
        doc("India's economy expanded 8.2 percent in the 2024 fiscal year."),
        doc("Rashmika Mandanna wore intricate bridal henna at the ceremony."),
    ]
    ranked = reranker.compress_documents(documents, "India GDP growth 2024")

    assert all("relevance_score" in d.metadata for d in ranked)
    # And it must be a probability-shaped number, not the raw logit — the floor
    # is expressed in (0,1) and a raw logit ranges roughly -11..+7.
    assert all(0.0 <= d.metadata["relevance_score"] <= 1.0 for d in ranked)
    assert ranked[0].metadata["relevance_score"] > ranked[1].metadata["relevance_score"]


# ── 5. Citation grounding ────────────────────────────────────────────────────

def test_citation_validator_rejects_fabricated_index():
    """
    `with_structured_output` guarantees `evidence_index` is an integer. It cannot
    guarantee the integer points at a document that was retrieved — a model
    citing [9] for a 3-item context produces a perfectly valid object containing
    a fabricated citation. That is the exact failure this project exists to
    prevent, so it is enforced in code.
    """
    from serve.nodes.generate import _drop_bad_citations, _validate_citations
    from serve.schemas import Citation, EvidenceAssessment, VerificationReport

    report = VerificationReport(
        claim="c", claim_type="other",
        evidence_analysis=[
            EvidenceAssessment(evidence_index=1, source="a", stance="SUPPORTS",
                               directly_relevant=True, reasoning="r"),
            EvidenceAssessment(evidence_index=9, source="b", stance="NEUTRAL",
                               directly_relevant=False, reasoning="r"),
        ],
        contradictions=[], verdict="TRUE", confidence=90,
        key_findings=[
            Citation(evidence_index=2, finding="real", quote=""),
            Citation(evidence_index=7, finding="fabricated", quote=""),
        ],
        limitations=[], conclusion="c",
    )

    problems = _validate_citations(report, evidence_count=3)
    assert len(problems) == 2
    assert _validate_citations(report, evidence_count=9) == []

    cleaned = _drop_bad_citations(report, evidence_count=3)
    assert [f.finding for f in cleaned.key_findings] == ["real"]
    assert cleaned.verdict == "TRUE"          # the verdict itself is untouched
    assert any("removed automatically" in limit for limit in cleaned.limitations)


# ── 6. Credibility with a fabricated date ────────────────────────────────────

def test_credibility_drops_recency_when_date_unreliable():
    """
    96% of this corpus carries placeholder Jan-1 dates. Letting a fabricated
    date drive 30% of a credibility score is worse than not scoring recency at
    all, so the weights must collapse to domain 0.7 / type 0.3.

    Silent because a wrong credibility number still looks like a number.
    """
    from core.credibility import score_document

    unreliable = score_document({"domain": "www.bbc.com", "date_reliable": False,
                                 "publish_date": "2024-01-01"})
    reliable = score_document({"domain": "www.bbc.com", "date_reliable": True,
                               "publish_date": "2026-08-01"})

    assert unreliable["recency_score"] is None
    assert reliable["recency_score"] is not None
    assert unreliable["total"] == pytest.approx(0.93 * 0.7 + 0.88 * 0.3, abs=1e-6)

    # `.replace('www.', '')` would mangle a domain containing it anywhere;
    # removeprefix is the correct operation.
    assert unreliable["domain"] == "bbc.com"
    assert score_document({"domain": "nobody.example"})["domain_score"] == 0.50


# ── 7 & 8. Routing — the three exits ─────────────────────────────────────────

def test_abstain_routing_and_zero_cost():
    """
    No evidence must route to `abstain` (via `web`), and abstaining must cost
    nothing. A regression here is invisible: the system still returns a report,
    it just starts paying for and fabricating analyses of irrelevant documents —
    which is precisely the behaviour of the system this replaced.
    """
    from core.config import cfg
    from serve.graph import route_after_retrieval, route_after_web
    from serve.nodes.terminal import abstain_report

    assert route_after_retrieval({"evidence": [Document(page_content="x")]}) == "stance"
    assert route_after_web({"evidence": []}) == "abstain"

    # With the web fallback on, an empty index result goes to `web` first; with
    # it off, straight to `abstain`. Both must reach abstain eventually.
    assert route_after_retrieval({"evidence": []}) == (
        "web" if cfg.web_search_enabled else "abstain"
    )

    result = abstain_report({"claim": "the moon is square in shape",
                             "retrieval_stats": {"floor": 0.25, "candidates": 37,
                                                 "best_rejected": 0.0},
                             "diagnostics": {"llm_calls": 0}})
    assert result["outcome"] == "abstained"
    assert result["report"].verdict == "UNVERIFIABLE"
    assert result["report"].confidence == 0
    assert result["diagnostics"]["llm_calls"] == 0


def test_reject_routing_and_zero_cost():
    """A non-claim must never reach retrieval or an LLM."""
    from serve.graph import route_after_gate
    from serve.nodes.terminal import reject_report

    assert route_after_gate({"is_claim": True}) == "retrieve"
    assert route_after_gate({"is_claim": False}) == "reject"

    result = reject_report({"reject_reason": "not a claim",
                            "claim_confidence": 0.0002,
                            "diagnostics": {"llm_calls": 0}})
    assert result["outcome"] == "rejected"
    assert result["report"] is None          # a verification of a non-claim is a category error
    assert result["diagnostics"]["llm_calls"] == 0


@needs_models
def test_gate_separates_claims_from_questions():
    """
    The gate must pass a false-but-checkable claim and reject a question.
    Rejecting "the moon is square" would be the wrong kind of correct — falsity
    is the verdict's job, not the gate's.
    """
    from serve.nodes.gate import claim_gate

    assert claim_gate({"claim": "India's GDP grew 8% in 2024"})["is_claim"] is True
    assert claim_gate({"claim": "the moon is square in shape"})["is_claim"] is True
    assert claim_gate({"claim": "what time is it"})["is_claim"] is False
    # Too short to verify: rejected before the model is even loaded.
    assert claim_gate({"claim": "hi"})["is_claim"] is False


# ── 9. The tier boundary ─────────────────────────────────────────────────────

def test_serve_does_not_import_ingest():
    """
    The two-tier split is the architectural story: ingestion and serving share
    *files*, not imports. A stray import would break that silently — everything
    would still run, and the claim in the README would become false.
    """
    import re
    from pathlib import Path

    from core.config import ROOT

    offenders = [
        path.name
        for path in (ROOT / "serve").rglob("*.py")
        if re.search(r"^\s*(from|import)\s+ingest\b",
                     path.read_text(encoding="utf-8"), re.M)
    ]
    assert offenders == [], f"serve/ imports ingest/: {offenders}"


@needs_index
def test_manifest_records_the_known_data_defect():
    """
    The corpus has fabricated publish dates. That must be recorded in the
    manifest rather than quietly ignored — a reader of the index needs to know,
    and `core.credibility` depends on the per-document flag.
    """
    from core.config import cfg

    manifest = json.loads(
        cfg.manifest_path(cfg.active_index).read_text(encoding="utf-8")
    )
    assert "dates_reliable" in manifest
    assert manifest["dates_unreliable_count"] > 0
    assert manifest["dates_reliable"] is False
