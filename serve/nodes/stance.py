"""
Stance detection — does each piece of evidence support or refute the claim?

Runs the locally trained FEVER model as a sentence *pair* classifier. The pair
encoding is the part that matters and the part the old code got right:

    tokenizer(claim, evidence)      ->  [CLS] claim [SEP] evidence [SEP]

Concatenating the two strings instead produces a single segment, drops the
`token_type_ids` the model was trained with, and quietly degrades accuracy in a
way nothing surfaces. That is preserved verbatim from `stance_detection.py`.

WHY THIS RUNS LOCALLY AT ALL
----------------------------
Asking the LLM for a stance per evidence item would cost eight extra API calls
per verification. This is one 110M-parameter forward pass per item on CPU,
batched, and the labels are only ever a *hint* to the LLM — which makes a 74%
model useful without making it load-bearing.

ITS ACCURACY IS 73.6% AND IT SHOWS
----------------------------------
Probed by hand against a claim of 8% GDP growth:

    "India's economy expanded 8.2 percent..."     SUPPORTS  0.739   correct
    "India's economy contracted by 2 percent..."  SUPPORTS  0.519   WRONG
    "Rashmika Mandanna wore bridal henna..."      NOT ENOUGH INFO 0.847  correct

It misread a direct contradiction, though at barely-above-chance confidence.
Two consequences, both implemented here:

* the confidence travels with the label, and anything under
  `STANCE_LOW_CONFIDENCE` is tagged so the prompt can tell the model to discount
  it;
* nothing downstream derives the verdict from these labels. `core.prompts`
  instructs the model to read the evidence text and follow it wherever the two
  disagree.

Aggregate counts are computed for the report's context and for the eval harness,
never as a verdict.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from langchain_core.runnables import RunnableLambda

from core import models
from core.config import cfg
from serve.schemas import VerifyState

logger = logging.getLogger(__name__)

# FEVER's third label is "not enough info", which is a statement about the
# evidence rather than a stance. NEUTRAL is the same idea in the report's
# vocabulary; the raw label is kept alongside for the eval harness.
_TO_STANCE = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NEUTRAL",
    "NEUTRAL": "NEUTRAL",
}

# The model was trained at 256 tokens. Evidence here is a whole article, so the
# tail is truncated — acceptable because the passage that actually matched
# retrieval is used instead when it is available.
MAX_PAIR_TOKENS = 256


def _classify(claim: str, evidences: list[str]) -> list[dict[str, Any]]:
    """One batched forward pass over all (claim, evidence) pairs."""
    tokenizer, model, id2label = models.stance_detector()

    inputs = tokenizer(
        [claim] * len(evidences),
        evidences,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PAIR_TOKENS,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)

    results: list[dict[str, Any]] = []
    for row in probabilities:
        predicted = int(row.argmax().item())
        raw_label = id2label[predicted]
        confidence = float(row[predicted].item())
        results.append({
            "stance": _TO_STANCE.get(raw_label, "NEUTRAL"),
            "raw_label": raw_label,
            "confidence": round(confidence, 4),
            "low_confidence": confidence < cfg.stance_low_confidence,
            "label_scores": {
                id2label[i]: round(float(row[i].item()), 4)
                for i in range(len(id2label))
            },
        })
    return results


def stance_and_credibility(state: VerifyState) -> dict:
    """
    Label every retrieved document's stance towards the claim.

    Credibility was already attached by the compressor pipeline during
    retrieval; this node only adds stance, then summarises both for the prompt.
    """
    claim = state.get("claim_english") or state.get("claim", "")
    documents = state.get("evidence") or []

    if not documents:
        return {"stances": []}

    # Prefer the chunk that actually matched over the full article: it is what
    # the retriever scored, it fits inside 256 tokens, and it is far less likely
    # to be a paragraph about something else entirely from later in the piece.
    texts = [
        doc.metadata.get("matched_chunk") or doc.page_content[:2000]
        for doc in documents
    ]

    predictions = _classify(claim, texts)

    stances: list[dict[str, Any]] = []
    for i, (doc, prediction) in enumerate(zip(documents, predictions), start=1):
        stances.append({
            # 1-based, matching core.prompts.format_evidence and the citation
            # validator in serve.nodes.generate. One numbering, three consumers.
            "evidence_index": i,
            "source": doc.metadata.get("domain") or doc.metadata.get("source", "unknown"),
            "credibility": doc.metadata.get("credibility", 0.5),
            "relevance_score": doc.metadata.get("relevance_score", 0.0),
            **prediction,
        })

    counts = {
        "SUPPORTS": sum(1 for s in stances if s["stance"] == "SUPPORTS"),
        "REFUTES":  sum(1 for s in stances if s["stance"] == "REFUTES"),
        "NEUTRAL":  sum(1 for s in stances if s["stance"] == "NEUTRAL"),
    }
    low = sum(1 for s in stances if s["low_confidence"])
    logger.info(
        "Stance over %d items: %s (%d low-confidence)",
        len(stances), counts, low,
    )

    return {
        "stances": stances,
        "diagnostics": {
            **state.get("diagnostics", {}),
            "stance_counts": counts,
            "stance_low_confidence": low,
        },
    }


stance_node = RunnableLambda(stance_and_credibility, name="stance_and_credibility")
