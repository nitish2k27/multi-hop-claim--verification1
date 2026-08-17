"""
The UNVERIFIABLE -> web escalation, and its loop guard.

THE GAP THIS CLOSES
-------------------
The relevance floor catches claims the corpus knows nothing about; those
retrieve zero evidence and reach `web` directly. It cannot catch the harder
case — a claim about a subject the corpus covers *well*, asserting a fact the
corpus does not contain.

"India's GDP grew 8% in 2024" retrieves six on-topic articles at 0.97
relevance. They clear the floor easily, so the pre-escalation graph never
considered searching, and the run ended at UNVERIFIABLE with a configured web
fallback that was never asked. Measured after the change: the same claim
escalates and returns MOSTLY_FALSE, having found that the 8% figure belongs to
FY 2025-26 rather than calendar 2024.

These are routing tests. Every function under test is a pure predicate over
state, so none of this touches the network or an LLM.
"""

from __future__ import annotations

import pytest

from core.config import cfg
from serve.graph import route_after_generate, route_after_web
from serve.schemas import VerificationReport


def _report(verdict: str = "UNVERIFIABLE") -> VerificationReport:
    return VerificationReport(
        claim="a claim",
        claim_type="statistical",
        evidence_analysis=[],
        verdict=verdict,
        confidence=30,
        conclusion="c",
    )


@pytest.fixture
def web_on(monkeypatch):
    monkeypatch.setattr(cfg, "web_search_enabled", True)


@pytest.fixture
def web_off(monkeypatch):
    monkeypatch.setattr(cfg, "web_search_enabled", False)


# ── The escalation itself ────────────────────────────────────────────────────

def test_unverifiable_escalates_to_web(web_on):
    """The whole point: an UNVERIFIABLE verdict gets one more source to try."""
    state = {"report": _report("UNVERIFIABLE")}
    assert route_after_generate(state) == "web"


@pytest.mark.parametrize("verdict", ["TRUE", "MOSTLY_TRUE", "MOSTLY_FALSE", "FALSE"])
def test_a_reached_verdict_is_not_second_guessed(web_on, verdict):
    """
    Only UNVERIFIABLE escalates.

    The other four mean the model reached a conclusion from the evidence it
    had. Re-searching would spend a second LLM call for the chance to unsettle
    a sound verdict, which is a worse trade than accepting it.
    """
    assert route_after_generate({"report": _report(verdict)}) == "render"


def test_escalation_respects_web_search_disabled(web_off):
    """
    --no-web and an unconfigured Tavily key must both keep the old behaviour.

    This is what keeps the committed eval numbers valid: the system eval runs
    with web disabled, so the escalation cannot fire during it and the measured
    20/20 abstention still describes the same system.
    """
    assert route_after_generate({"report": _report("UNVERIFIABLE")}) == "render"


def test_no_report_does_not_escalate(web_on):
    """Defensive: `generate` always sets a report, but never crash if it didn't."""
    assert route_after_generate({}) == "render"


# ── The loop guard ───────────────────────────────────────────────────────────

def test_second_unverifiable_does_not_escalate_again(web_on):
    """
    THE LOOP GUARD.

    web -> stance -> generate -> web is a cycle in the graph. If the web
    results still leave the model unable to decide, `web_attempted` is what
    stops the run going round forever, burning an LLM call and a Tavily search
    on every lap.
    """
    state = {"report": _report("UNVERIFIABLE"), "web_attempted": True}
    assert route_after_generate(state) == "render"


def test_web_node_marks_attempted_even_when_it_finds_nothing(monkeypatch):
    """
    The flag must be set on *every* exit from the web node, not just the
    successful one.

    A skipped or failed search that left the flag unset would route straight
    back to `web` on the next pass, which is the same infinite loop reached by
    a different door.
    """
    from serve.nodes import websearch

    monkeypatch.setattr(websearch, "_tavily_unavailable", lambda: "no API key")

    out = websearch.web_search({"claim": "x", "evidence": [], "retrieval_stats": {}})
    assert out["web_attempted"] is True


# ── Where the web node returns to ────────────────────────────────────────────

def test_empty_web_with_no_report_abstains():
    """First pass: the index found nothing and neither did the web."""
    assert route_after_web({"evidence": []}) == "abstain"


def test_empty_web_with_a_report_keeps_the_verdict():
    """
    Second pass: we already have an UNVERIFIABLE report.

    Abstaining here would discard a verdict the user already paid an LLM call
    for and pretend nothing was retrieved. "I looked and could not confirm
    this" is both more honest and more useful than "I have nothing".
    """
    assert route_after_web({"evidence": [], "report": _report()}) == "render"


def test_web_with_evidence_always_reasons_over_it():
    doc = object()
    assert route_after_web({"evidence": [doc]}) == "stance"
    assert route_after_web({"evidence": [doc], "report": _report()}) == "stance"


# ── Evidence merging ─────────────────────────────────────────────────────────

def test_web_results_are_added_to_index_evidence_not_swapped_for_it(monkeypatch):
    """
    On an escalation the index evidence must survive.

    It was topical — it simply was not sufficient. Dropping it would throw away
    context the model may need to combine with the search results, and would
    renumber the citations the previous report already made.
    """
    from langchain_core.documents import Document

    from serve.nodes import websearch

    index_docs = [Document(page_content="from the index", metadata={"n": 1})]
    web_docs = [Document(page_content="from the web", metadata={"n": 2})]

    monkeypatch.setattr(websearch, "_tavily_unavailable", lambda: None)
    monkeypatch.setattr(websearch, "_build_retriever",
                        lambda: type("R", (), {"invoke": lambda self, q: web_docs})())
    monkeypatch.setattr(websearch, "_to_evidence", lambda raw: web_docs)

    class _Pipeline:
        def compress_documents(self, docs, query):
            return docs

    monkeypatch.setattr(
        "serve.retriever.get_retriever",
        lambda: type("X", (), {"pipeline": _Pipeline()})(),
    )

    out = websearch.web_search({
        "claim": "a claim",
        "evidence": index_docs,
        "evidence_source": "index",
        "retrieval_stats": {},
    })

    assert len(out["evidence"]) == 2
    # Index evidence keeps position 1, so citations made against it still hold.
    assert out["evidence"][0].page_content == "from the index"
    assert out["evidence"][1].page_content == "from the web"
    assert out["evidence_source"] == "mixed"


def test_first_pass_web_replaces_nothing_because_there_was_nothing(monkeypatch):
    """With no prior evidence, append and replace are the same operation."""
    from langchain_core.documents import Document

    from serve.nodes import websearch

    web_docs = [Document(page_content="from the web", metadata={})]

    monkeypatch.setattr(websearch, "_tavily_unavailable", lambda: None)
    monkeypatch.setattr(websearch, "_build_retriever",
                        lambda: type("R", (), {"invoke": lambda self, q: web_docs})())
    monkeypatch.setattr(websearch, "_to_evidence", lambda raw: web_docs)

    class _Pipeline:
        def compress_documents(self, docs, query):
            return docs

    monkeypatch.setattr(
        "serve.retriever.get_retriever",
        lambda: type("X", (), {"pipeline": _Pipeline()})(),
    )

    out = websearch.web_search({"claim": "c", "evidence": [], "retrieval_stats": {}})

    assert len(out["evidence"]) == 1
    assert out["evidence_source"] == "web"


# ── The graph actually wires it ──────────────────────────────────────────────

def test_the_compiled_graph_contains_the_cycle():
    """
    Guards against the routers being correct while the edge is not registered.

    `generate -> web` is the only cycle in the graph; if a refactor drops the
    conditional edge, every routing test above still passes and the feature is
    silently gone.
    """
    from serve.graph import mermaid

    diagram = mermaid()
    assert "generate" in diagram and "web" in diagram
    assert "generate -.-> web;" in diagram, (
        "the generate -> web escalation edge is missing from the compiled graph"
    )
