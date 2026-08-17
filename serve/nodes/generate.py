"""
Report generation — the only node that calls an LLM.

    prompt | llm.with_structured_output(VerificationReport)

`with_structured_output` is the reason this is a two-line chain. The old path
asked for markdown headers and then regex-parsed the prose for a verdict, in two
different places, which could and did disagree with each other. The verdict is
now a `Literal` field on a validated Pydantic model: unparseable output is a
retry, not a silently wrong answer.

WHAT THE SCHEMA STILL CANNOT ENFORCE
------------------------------------
It guarantees `evidence_index` is an integer. It cannot guarantee the integer
points at a document we actually retrieved — a model that invents `[9]` for a
seven-item context produces a perfectly valid `VerificationReport` containing a
fabricated citation. That is the exact failure this project exists to prevent,
so it is checked in code, here, after generation.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableLambda

from core.llm import get_llm
from core.prompts import build_prompt, format_evidence, language_instruction
from serve.schemas import VerificationReport, VerifyState

logger = logging.getLogger(__name__)


class CitationError(ValueError):
    """A generated report cited evidence that does not exist."""


# Structured output is a constraint on the *request*, not a guarantee about the
# generation. The model still emits the JSON itself, and it can emit invalid
# JSON — observed in the wild from `openai/gpt-oss-120b`:
#
#     "confidence":  thirty,
#
# The word, not the number. Groq's tool-call parser rejects the whole call with
# a 400 `tool_use_failed`, and without a retry that one token kills a run that
# had already paid for retrieval, stance detection and a web search.
#
# Sampling is stochastic, so a second attempt almost always parses. This is a
# different failure from a bad citation — that one needs the specific indices
# quoted back, this one just needs another roll.
GENERATION_ATTEMPTS = 3

_MALFORMED = ("tool_use_failed", "failed to parse", "invalid_request_error",
              "json", "tool call")


def _is_malformed_generation(exc: Exception) -> bool:
    """Is this the model emitting unparseable JSON, rather than a real error?"""
    text = str(exc).lower()
    return any(marker in text for marker in _MALFORMED)


def _generate_once(chain, inputs: dict) -> VerificationReport:
    """
    Invoke the chain, retrying a malformed generation.

    Only retries the parse failure. A quota error, an auth error or a retired
    model must surface immediately — retrying those burns time and tells the
    user nothing, and `core.llm` already classifies them.
    """
    last: Exception | None = None
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            if not _is_malformed_generation(exc) or attempt == GENERATION_ATTEMPTS:
                raise
            last = exc
            logger.warning(
                "Model emitted unparseable structured output (attempt %d/%d) — "
                "retrying. %s", attempt, GENERATION_ATTEMPTS, str(exc)[:120],
            )
    raise last  # unreachable; keeps type checkers honest


def _validate_citations(
    report: VerificationReport, evidence_count: int
) -> list[str]:
    """
    Check every index the model emitted against what was actually retrieved.

    Returns a list of human-readable problems; empty means the report is
    grounded. Both `key_findings` and `evidence_analysis` are checked — a
    fabricated index in the analysis section is just as wrong, and is the more
    likely place for the model to drift when the evidence list is long.
    """
    problems: list[str] = []
    valid = range(1, evidence_count + 1)

    for i, finding in enumerate(report.key_findings, start=1):
        if finding.evidence_index not in valid:
            problems.append(
                f"key_findings[{i}] cites evidence [{finding.evidence_index}], "
                f"but only [1]-[{evidence_count}] exist"
            )

    for i, assessment in enumerate(report.evidence_analysis, start=1):
        if assessment.evidence_index not in valid:
            problems.append(
                f"evidence_analysis[{i}] cites evidence "
                f"[{assessment.evidence_index}], but only [1]-[{evidence_count}] exist"
            )

    return problems


def _drop_bad_citations(
    report: VerificationReport, evidence_count: int
) -> VerificationReport:
    """
    Last resort after a failed retry: remove the ungrounded parts, keep the rest.

    Deliberately not a hard failure. A report whose verdict and conclusion are
    sound but which over-reached on one bullet is still worth returning — with
    the bad bullet gone and the removal disclosed in `limitations`, which is
    visible to the reader rather than buried in a log.
    """
    valid = range(1, evidence_count + 1)

    kept_findings = [
        f for f in report.key_findings if f.evidence_index in valid
    ]
    kept_analysis = [
        a for a in report.evidence_analysis if a.evidence_index in valid
    ]
    dropped = (len(report.key_findings) - len(kept_findings)) + (
        len(report.evidence_analysis) - len(kept_analysis)
    )

    limitations = list(report.limitations)
    if dropped:
        limitations.append(
            f"{dropped} generated statement(s) referenced evidence that was not "
            f"retrieved and were removed automatically."
        )

    return report.model_copy(update={
        "key_findings": kept_findings,
        "evidence_analysis": kept_analysis,
        "limitations": limitations,
    })


def generate_report(state: VerifyState) -> dict:
    """
    Produce the structured verdict from the retrieved evidence.

    Only ever reached with non-empty evidence — the graph routes to `abstain`
    otherwise, which is what makes "no evidence" cost zero tokens.
    """
    documents = state.get("evidence") or []
    stances = state.get("stances") or []
    claim = state.get("claim_english") or state.get("claim", "")
    language = state.get("language", "en")

    if not documents:  # defensive: the graph should have routed elsewhere
        raise RuntimeError("generate_report reached with no evidence")

    chain = build_prompt() | get_llm().with_structured_output(VerificationReport)

    inputs = {
        "claim": claim,
        "evidence_count": len(documents),
        "evidence_block": format_evidence(documents, stances),
        "language_instruction": language_instruction(language),
    }

    diagnostics = dict(state.get("diagnostics", {}))
    calls = diagnostics.get("llm_calls", 0)

    report: VerificationReport = _generate_once(chain, inputs)
    calls += 1

    problems = _validate_citations(report, len(documents))

    if problems:
        # One retry, with the specific failures quoted back. Telling the model
        # exactly which index was wrong works far better than re-asking.
        logger.warning("Ungrounded citations, retrying once: %s", problems)
        correction = (
            "\n\nYOUR PREVIOUS ANSWER WAS REJECTED. It cited evidence that does "
            "not exist:\n  - " + "\n  - ".join(problems) +
            f"\nEvery evidence_index must be an integer from 1 to "
            f"{len(documents)} inclusive, matching the numbered items above. "
            "Re-answer using only those."
        )
        report = _generate_once(chain, {
            **inputs, "evidence_block": inputs["evidence_block"] + correction
        })
        calls += 1

        problems = _validate_citations(report, len(documents))
        if problems:
            logger.error("Still ungrounded after retry; dropping: %s", problems)
            report = _drop_bad_citations(report, len(documents))
            diagnostics["citations_dropped"] = len(problems)

    diagnostics["llm_calls"] = calls
    diagnostics["citation_retry"] = calls > 1

    logger.info(
        "Verdict %s (%d%% confident) from %d evidence items, %d LLM call(s)",
        report.verdict, report.confidence, len(documents), calls,
    )

    return {"report": report, "outcome": "verified", "diagnostics": diagnostics}


generate_node = RunnableLambda(generate_report, name="generate_report")
