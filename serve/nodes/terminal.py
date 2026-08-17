"""
The two non-LLM exits: abstain and reject.

**Neither node calls an LLM.** That is the entire point. Together they are the
difference between this system and the one it replaces, where every input —
including "the moon is square in shape" — produced a ~700 word analysis built
out of whatever documents happened to rank highest, in that case an article
about bridal henna and a foldable phone review.

    reject   — the input was never a checkable claim (gate said no)
    abstain  — it was a claim, but no evidence cleared the relevance floor

Abstention emits a real `VerificationReport` with the UNVERIFIABLE verdict and
zero confidence, rather than a special-cased error shape. One output type means
the API, the renderer and M3's eval harness all read the same object, and
`state["outcome"]` distinguishes an honest abstention from an LLM that read the
evidence and concluded UNVERIFIABLE — two different things that must not be
scored as one.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableLambda

from serve.schemas import VerificationReport, VerifyState

logger = logging.getLogger(__name__)


def abstain_report(state: VerifyState) -> dict:
    """
    No evidence cleared the floor. Say so, and say how close it came.

    The near-miss number is deliberately surfaced to the user: "nothing scored
    above 0.15, the best was 0.03" is a meaningfully different statement from
    "the best was 0.14", and it tells a reader whether the corpus nearly had an
    answer or was nowhere near one.
    """
    claim = state.get("claim", "")
    stats = state.get("retrieval_stats") or {}
    floor = stats.get("floor", 0.0)
    best = stats.get("best_rejected", 0.0)
    candidates = stats.get("candidates", 0)

    logger.info(
        "ABSTAINED — %d candidates, none above floor %.2f (best %.4f)",
        candidates, floor, best,
    )

    detail = (
        f"Searched the indexed corpus and retrieved {candidates} candidate "
        f"passage(s), but none were relevant enough to support a verdict "
        f"(best relevance {best:.3f} against a floor of {floor:.2f})."
    )

    # After M4 an abstention means BOTH the corpus and the live web came up
    # short. Saying which is not pedantry — "my corpus doesn't cover this" and
    # "the open web doesn't either" are very different statements about a claim,
    # and the reader deserves to know which one they are being told.
    web_detail = None
    if "web_results" in stats:
        web_detail = (
            f"Also searched the live web and retrieved "
            f"{stats['web_results']} result(s); "
            f"{stats.get('web_after_floor', 0)} cleared the relevance floor."
        )
    elif stats.get("web_skipped"):
        web_detail = (
            f"Web search fallback did not run ({stats['web_skipped']}), so this "
            f"reflects the indexed corpus only."
        )
    elif stats.get("web_error"):
        web_detail = (
            f"The web search fallback failed ({stats['web_error']}), so this "
            f"reflects the indexed corpus only."
        )

    report = VerificationReport(
        claim=claim,
        claim_type="other",
        evidence_analysis=[],
        contradictions=[],
        verdict="UNVERIFIABLE",
        # Not a low number — the absence of one. There is no evidence to be
        # confident about, and inventing a percentage here would be exactly the
        # false precision this path exists to avoid.
        confidence=0,
        key_findings=[],
        limitations=[
            "No relevant evidence was found in the indexed corpus.",
            "This is a statement about the available evidence, not about the "
            "claim — the claim may well be true or false; this system simply "
            "has nothing to say about it.",
            detail,
            *([web_detail] if web_detail else []),
        ],
        conclusion=(
            "There is not enough evidence available to verify this claim. No "
            "verdict is offered rather than a guess."
        ),
    )

    return {
        "report": report,
        "outcome": "abstained",
        "diagnostics": {
            **state.get("diagnostics", {}),
            "abstained_because": "no_evidence_above_floor",
            "web_fallback_ran": "web_results" in stats,
            "best_rejected_score": best,
            # Proves the property the milestone is judged on.
            "llm_calls": state.get("diagnostics", {}).get("llm_calls", 0),
        },
    }


def reject_report(state: VerifyState) -> dict:
    """
    The gate said this is not a claim. Return the reason, nothing else.

    No `VerificationReport` is constructed: there is no claim, so a verification
    of it would be a category error. `render` handles the `None`.
    """
    reason = state.get("reject_reason") or "Input is not a verifiable claim."
    logger.info("REJECTED — %s", reason.split(".")[0])

    return {
        "report": None,
        "outcome": "rejected",
        "diagnostics": {
            **state.get("diagnostics", {}),
            "rejected_because": "not_a_claim",
            "claim_confidence": state.get("claim_confidence", 0.0),
        },
    }


abstain_node = RunnableLambda(abstain_report, name="abstain_report")
reject_node = RunnableLambda(reject_report, name="reject_report")
