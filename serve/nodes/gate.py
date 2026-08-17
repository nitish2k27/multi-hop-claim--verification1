"""
Claim gate — the first of the three exits.

Runs the trained binary claim detector and routes anything that is not a
checkable assertion straight to `reject`, before any retrieval or LLM call.
`"what time is it"` should cost nothing.

WHAT THIS MODEL ACTUALLY DOES
-----------------------------
Its reported test accuracy is 1.00, which is not a good sign — a perfect score
on 280k examples means the task was separable, not that the model is perfect.
Probing it by hand shows what it learned:

    India's GDP grew 8% in 2024              0.9999  claim
    the moon is square in shape              0.9992  claim      <- correct: false, but a claim
    The Eiffel Tower is located in Paris.    0.9999  claim
    what time is it                          0.0002  not a claim
    What is the GDP growth rate?             0.0001  not a claim
    I think the economy is doing well        0.0020  not a claim
    hello                                    0.0004  not a claim
    asdkjh askjdh                            0.9969  claim      <- WRONG

It separates declarative sentences from questions and opinions. It does not
judge verifiability, and it will wave gibberish through. That is an honest
description of a useful-but-narrow filter, and it is the right shape for this
job: the gate's purpose is to reject *interactions* that are not claims, and
downstream retrieval handles input that is a well-formed sentence about nothing
— gibberish retrieves nothing and abstains, which is the correct outcome anyway.

Note the second row. "The moon is square" is a claim and must pass this gate.
Rejecting false statements here would be the wrong kind of correct — falsity is
the verdict's job, not the gate's.
"""

from __future__ import annotations

import logging

import torch
from langchain_core.runnables import RunnableLambda

from core import models
from serve.schemas import VerifyState

logger = logging.getLogger(__name__)

# Below this many characters there is nothing to verify and the model's output
# is noise. Checked before the model so trivial input costs no inference.
MIN_CLAIM_CHARS = 8


def _score_claim(text: str) -> float:
    """P(LABEL_1) — the trained detector's probability that this is a claim."""
    tokenizer, model = models.claim_detector()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1)[0][1].item()


def claim_gate(state: VerifyState) -> dict:
    """
    Decide whether this input is worth verifying.

    Returns `is_claim` plus, when rejecting, a `reject_reason` written for the
    user rather than for a log — it is the entire content of the rejection
    response, so it has to say what would work instead.
    """
    text = (state.get("claim_english") or state.get("claim") or "").strip()

    if len(text) < MIN_CLAIM_CHARS:
        return {
            "is_claim": False,
            "claim_confidence": 0.0,
            "reject_reason": (
                "Input is too short to verify. Give a complete factual "
                "statement, for example: \"India's GDP grew 8% in 2024\"."
            ),
        }

    confidence = _score_claim(text)
    is_claim = confidence >= cfg_threshold()

    logger.info("Gate: %s (p=%.4f) %r", "PASS" if is_claim else "REJECT",
                confidence, text[:60])

    if is_claim:
        return {"is_claim": True, "claim_confidence": confidence}

    return {
        "is_claim": False,
        "claim_confidence": confidence,
        "reject_reason": (
            "This does not look like a verifiable factual claim — it reads as a "
            "question, an opinion, or conversational text. Fact-checking needs "
            "an assertion that could be true or false, for example: "
            "\"India's GDP grew 8% in 2024\"."
        ),
    }


def cfg_threshold() -> float:
    """Read at call time so a `.env` change does not need a code change."""
    from core.config import cfg

    return cfg.claim_threshold


# Exposed as a Runnable so the node composes with the rest of the LangChain
# surface (`.invoke`, `.batch`, tracing in LangSmith) rather than being an
# opaque function the graph happens to call.
gate_node = RunnableLambda(claim_gate, name="claim_gate")
