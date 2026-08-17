"""
The LangGraph wiring.

    adapt ──► translate ──► gate ──not a claim──────────► reject ──┐
     text                          │                               │
     voice  (Whisper)              claim                            │
     image  (OCR)                  ▼                               │
     pdf/docx               retrieve ──nothing──► web ──nothing──► abstain ─┤
                                  │                │                       │
                              evidence         evidence                    │
                                  ▼                ▼                       │
    stance ──────────► generate ─────────────────────────────────────► render ──► END

**Three exits, one renderer.** Every path ends in a rendered response; none can
return a bare state or a traceback. Two of the three never call an LLM.

Why a graph rather than a chain: the interesting behaviour here *is* the
branching. Expressing "no evidence means stop" as an edge makes abstention a
visible property of the architecture instead of an `if` buried in a request
handler — and `get_graph().draw_mermaid()` then renders a diagram that cannot
drift from the code, which is what M6 puts in the README.

**Abstain is still reachable**, and that matters enough to state: a web fallback
that always finds *something* would quietly delete the behaviour this project is
built around. Web results are held to the same relevance floor as index results,
so search hits that do not bear on the claim are dropped and the graph continues
to `abstain`.

`adapt` and `translate` sit ahead of the gate rather than inside it because both
can fail in ways the user needs to hear about specifically — "no readable text
in that image" is a different message from "that is not a claim", and collapsing
them would lose the distinction.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from serve.nodes.adapt import adapt_input
from serve.nodes.gate import claim_gate
from serve.nodes.language import to_english
from serve.nodes.generate import generate_report
from serve.nodes.render import render_output
from serve.nodes.retrieve import retrieve_evidence
from serve.nodes.stance import stance_and_credibility
from serve.nodes.terminal import abstain_report, reject_report
from serve.nodes.websearch import web_search
from serve.schemas import VerifyState

from core.config import cfg

logger = logging.getLogger(__name__)


# ── Routing ──────────────────────────────────────────────────────────────────
# Plain predicates over state, kept as named functions so the compiled mermaid
# diagram labels the branches with something meaningful.

def route_after_adapt(state: VerifyState) -> str:
    """
    Unreadable input -> reject, with the reason.

    An OCR failure, a scanned PDF with no text layer, an unsupported file type:
    all are things the user must be told about specifically, and none should
    reach translation or retrieval. Routing rather than raising is what keeps
    "every path renders a real response" true.
    """
    return "reject" if state.get("input_error") else "translate"


def route_after_gate(state: VerifyState) -> str:
    """Not a claim -> reject, without touching the index."""
    return "retrieve" if state.get("is_claim") else "reject"


def route_after_retrieval(state: VerifyState) -> str:
    """
    Evidence in the index -> reason about it. Nothing -> try the live web.

    Routing to `web` rather than straight to `abstain` is the whole of M4. The
    node itself is a no-op when web search is disabled or unconfigured, and
    returns no evidence in that case, so `route_after_web` then sends it to
    `abstain` — the pre-M4 behaviour, reached one hop later.
    """
    if state.get("evidence"):
        return "stance"
    return "web" if cfg.web_search_enabled else "abstain"


def route_after_web(state: VerifyState) -> str:
    """
    Last gate before abstention.

    Reached only when the index already failed. If the web produced nothing
    that cleared the relevance floor either, there is genuinely no evidence and
    the honest answer is to decline.
    """
    return "stance" if state.get("evidence") else "abstain"


def build_graph():
    """Compile the app. Cheap — nodes hold no state and load models lazily."""
    graph = StateGraph(VerifyState)

    graph.add_node("adapt", adapt_input)
    graph.add_node("translate", to_english)
    graph.add_node("gate", claim_gate)
    graph.add_node("retrieve", retrieve_evidence)
    graph.add_node("web", web_search)
    graph.add_node("stance", stance_and_credibility)
    graph.add_node("generate", generate_report)
    graph.add_node("abstain", abstain_report)
    graph.add_node("reject", reject_report)
    graph.add_node("render", render_output)

    graph.set_entry_point("adapt")
    graph.add_conditional_edges(
        "adapt", route_after_adapt,
        {"translate": "translate", "reject": "reject"},
    )
    graph.add_edge("translate", "gate")

    # Explicit path maps rather than bare callables: they document the reachable
    # set at the call site and make the rendered diagram legible.
    graph.add_conditional_edges(
        "gate", route_after_gate, {"retrieve": "retrieve", "reject": "reject"}
    )
    graph.add_conditional_edges(
        "retrieve", route_after_retrieval,
        {"stance": "stance", "web": "web", "abstain": "abstain"},
    )
    graph.add_conditional_edges(
        "web", route_after_web,
        {"stance": "stance", "abstain": "abstain"},
    )

    graph.add_edge("stance", "generate")

    for terminal in ("generate", "abstain", "reject"):
        graph.add_edge(terminal, "render")

    graph.add_edge("render", END)

    return graph.compile()


_APP = None


def get_app():
    """Process-wide compiled graph."""
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def mermaid() -> str:
    """The architecture diagram, generated from the real graph. Used at M6."""
    return get_app().get_graph().draw_mermaid()
