"""
Tier 2 data contracts.

Two unrelated things live here, and the distinction matters:

* **`VerifyState`** — the LangGraph state. A plain `TypedDict`, mutated by
  nodes returning partial dicts. Internal, never serialised to a user.
* **`VerificationReport`** — the schema the LLM is *constrained* to produce via
  `with_structured_output`. This replaces the regex verdict parsers in the old
  `report_exporter.py`, which is where a real bug lived: the exported markdown
  and the JSON API response could disagree about the verdict, because each
  scraped the prose separately. There is now one parse, done by the model
  itself against a schema, and everything downstream reads the same object.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field

# The five verdict labels. English even in a translated report, so that
# downstream code and the eval harness never have to parse localised text.
Verdict = Literal["TRUE", "MOSTLY_TRUE", "UNVERIFIABLE", "MOSTLY_FALSE", "FALSE"]
Stance = Literal["SUPPORTS", "REFUTES", "NEUTRAL"]
Outcome = Literal["verified", "abstained", "rejected"]


# ── What the LLM is constrained to return ────────────────────────────────────

class Citation(BaseModel):
    """A finding tied to a specific retrieved document."""

    evidence_index: int = Field(
        description="1-based index of the evidence item this finding rests on, "
                    "exactly as numbered in the context."
    )
    finding: str = Field(description="The finding, in one sentence.")
    quote: str = Field(
        default="",
        description="Short verbatim span from that evidence item supporting the "
                    "finding. Must appear in the evidence text.",
    )


class EvidenceAssessment(BaseModel):
    """The model's read of one piece of evidence."""

    evidence_index: int = Field(description="1-based index of the evidence item.")
    source: str = Field(description="Publication or domain the evidence came from.")
    stance: Stance = Field(
        description="Does this evidence support, refute, or stay neutral on the claim?"
    )
    directly_relevant: bool = Field(
        description="True only if this evidence bears on the claim itself, not "
                    "merely on the same topic."
    )
    reasoning: str = Field(description="Why, citing the evidence text.")


class VerificationReport(BaseModel):
    """
    The full structured verdict.

    Every field the old prose report had, now typed. `key_findings` is capped at
    five to stop the model padding, and each entry carries an `evidence_index`
    that `serve.nodes.generate` checks against the documents actually retrieved
    — the prompt *requests* grounded citations; that check *enforces* them.
    """

    claim: str = Field(description="The claim as verified, restated plainly.")
    claim_type: Literal[
        "statistical", "event", "policy", "scientific", "biographical", "other"
    ] = Field(description="What kind of claim this is.")

    evidence_analysis: list[EvidenceAssessment] = Field(
        description="One entry per evidence item provided, in order."
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Conflicts *between* evidence items. Empty list if none.",
    )

    verdict: Verdict = Field(description="One of the five labels.")
    confidence: int = Field(
        ge=0, le=100,
        description="0-100. Reflect genuine uncertainty; do not default to a "
                    "high number when the evidence is thin.",
    )

    key_findings: list[Citation] = Field(
        default_factory=list, max_length=5,
        description="At most five findings, most important first.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="What is missing, assumed, or caveated.",
    )
    conclusion: str = Field(
        description="Two to three sentences for a non-expert. No new information."
    )


# ── The graph state ──────────────────────────────────────────────────────────

class VerifyState(TypedDict, total=False):
    """
    State threaded through the graph.

    `total=False` because nodes fill it in progressively — the entry node sets
    `claim`, and `evidence` does not exist until `retrieve` runs. Every node
    returns only the keys it changed; LangGraph merges.
    """

    # Input, as given
    raw_input: str
    input_kind: Literal["text", "voice", "image", "document"]
    source_file: str

    # After adapt (M5)
    input_error: str         # set when the input could not be read at all
    extracted_text: str      # OCR / transcription / document text
    transcript: str          # voice only — kept separately so the UI can show it
    upload_evidence: list[Document]   # an attached document, as capped evidence

    # After translate.
    #   claim          what the user said, in their language -> report is in this
    #   claim_english  translated                            -> used for retrieval
    claim: str
    language: str
    claim_english: str

    # Gate
    is_claim: bool
    claim_confidence: float
    reject_reason: str

    # Retrieval
    evidence: list[Document]
    evidence_source: Literal["index", "web", "upload", "none"]
    retrieval_stats: dict[str, Any]

    # Stance
    stances: list[dict[str, Any]]

    # Output
    report: VerificationReport | None
    outcome: Outcome
    rendered: str
    artifacts: dict[str, str]   # format -> path, for html/docx/mp3 exports
    diagnostics: dict[str, Any]


def new_state(
    raw: str,
    *,
    input_kind: str | None = None,
    claim: str | None = None,
) -> VerifyState:
    """
    Seed a run. Keeps entry-point construction in one place.

    `raw` is whatever the user supplied — a claim, or a path to an audio file,
    image or document. `adapt` works out which.

    `claim` is only meaningful alongside a document: it is the claim to verify
    *against* that document. Supplied alone it is redundant with `raw`.
    """
    state: VerifyState = {
        "raw_input": raw,
        "diagnostics": {"llm_calls": 0},
    }
    if input_kind:
        state["input_kind"] = input_kind  # type: ignore[typeddict-item]
    if claim:
        state["claim"] = claim.strip()
    return state
