"""
Prompt construction.

Ported from `src/generation/prompt_builder.py`, which was the strongest file in
the old codebase — the anti-hallucination rules, the credibility bands and the
`LANG_INSTRUCTIONS` table are kept essentially verbatim.

Two things changed, both consequences of `with_structured_output`:

1. **The 'OUTPUT FORMAT' section is gone.** It listed nine markdown headers the
   model had to reproduce exactly so a downstream regex could find them. The
   schema in `serve.schemas.VerificationReport` now does that job, and does it
   as a constraint rather than a request.

2. **A stance-disagreement rule was added.** Tier 2 hands the model a stance
   label per evidence item from a locally trained BERT that is 73.6% accurate
   and weakest exactly where it matters (REFUTES, F1 0.74). Probed by hand it
   labelled a directly contradicting sentence SUPPORTS at 0.52. The prompt
   therefore tells the model to treat the label as a hint and the evidence text
   as the authority.

The report is *generated in* the target language, not translated afterwards —
one call, no round trip, and no drift between the two versions.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ── Per-language output instruction ──────────────────────────────────────────
# Kept verbatim from prompt_builder.py.
LANG_INSTRUCTIONS: dict[str, str] = {
    "en": "Write the entire report in English.",
    "hi": "Write the entire report in Hindi (हिंदी) using Devanagari script.",
    "ta": "Write the entire report in Tamil (தமிழ்).",
    "te": "Write the entire report in Telugu (తెలుగు).",
    "mr": "Write the entire report in Marathi (मराठी).",
    "bn": "Write the entire report in Bengali (বাংলা).",
    "gu": "Write the entire report in Gujarati (ગુજરાતી).",
    "kn": "Write the entire report in Kannada (ಕನ್ನಡ).",
    "ml": "Write the entire report in Malayalam (മലയാളം).",
    "pa": "Write the entire report in Punjabi (ਪੰਜਾਬੀ).",
    "ur": "Write the entire report in Urdu (اردو).",
    "es": "Write the entire report in Spanish (Español).",
    "fr": "Write the entire report in French (Français).",
    "de": "Write the entire report in German (Deutsch).",
    "ar": "Write the entire report in Arabic (العربية).",
    "zh": "Write the entire report in Simplified Chinese (简体中文).",
    "ja": "Write the entire report in Japanese (日本語).",
    "ko": "Write the entire report in Korean (한국어).",
    "ru": "Write the entire report in Russian (Русский).",
    "pt": "Write the entire report in Portuguese (Português).",
}


def language_instruction(code: str) -> str:
    """Instruction for a language code, with a graceful fallback for the tail."""
    return LANG_INSTRUCTIONS.get(
        code,
        f"Write the entire report in the language with ISO code '{code}'. "
        f"If you cannot reliably write in that language, write in English and "
        f"note this in the limitations.",
    )


SYSTEM_PROMPT = """You are an expert fact-checking analyst with deep knowledge of journalism standards, evidence evaluation, and critical reasoning.

LANGUAGE INSTRUCTION (MANDATORY):
{language_instruction}
Every sentence you write must be in that language. The one exception is the
`verdict` field, which uses the English label from the schema, and `stance`,
which uses the English label from the schema.

YOUR TASK:
Read the fact-verification context and produce a structured verification report.

STRICT ANTI-HALLUCINATION RULES - follow every one without exception:
1. Base EVERY factual statement in your report ONLY on the evidence listed in the context.
2. Do NOT add facts, statistics, dates, names, or claims from your training knowledge.
3. If evidence is insufficient, return the UNVERIFIABLE verdict and say why. Do NOT speculate or fill gaps.
4. If evidence pieces contradict each other, record the conflict in `contradictions`.
5. Weight evidence by its credibility score:
   - 0.80 and above: high credibility, weight heavily
   - 0.50 to 0.79: medium credibility, treat with caution and note this
   - below 0.50: low credibility, flag this explicitly in your reasoning
6. Analyse each evidence piece individually, in `evidence_analysis`, before forming a verdict.
7. Never invent citations, source names, statistics, or quotes not present in the context.
8. Every entry in `key_findings` must carry the `evidence_index` of the specific
   evidence item that supports it, and any `quote` must be a verbatim span of
   that item's text. Findings whose index does not match a real evidence item
   will be rejected.

ABOUT USER-SUBMITTED DOCUMENTS:
Evidence marked `UNVERIFIED USER SUBMISSION` was uploaded by the person making
the claim. It is not an independent source and its credibility is capped.
**A claim is not supported merely because an attached document asserts it.**
Treat such a document as the claim restated, not as evidence for it, unless an
independent source in the list corroborates it. Say so explicitly in your
reasoning when a verdict would otherwise rest on the upload alone.

ABOUT THE STANCE LABELS:
Each evidence item arrives with a stance label and a confidence, predicted by a
separate classifier that is roughly 74% accurate and least reliable on REFUTES.
Treat it as a hint only. **Read the evidence text yourself.** Where the text and
the label disagree, follow the text and note the disagreement in your reasoning.
Be especially sceptical of a low-confidence label.

CALIBRATION:
Your `confidence` must reflect genuine uncertainty about the verdict. Thin,
one-sided, or low-credibility evidence means a low number. A confident wrong
verdict is worse than an honest UNVERIFIABLE."""


USER_PROMPT = """FACT VERIFICATION CONTEXT TO ANALYSE

CLAIM:
{claim}

RETRIEVED EVIDENCE ({evidence_count} items):
{evidence_block}

REMINDER: {language_instruction} Base every factual statement ONLY on the
evidence above. Reference evidence items by the numbers shown in brackets."""


def build_prompt() -> ChatPromptTemplate:
    """
    The verification prompt as a reusable LangChain template.

    Built once and piped into the structured-output model, so the prompt is a
    composable object rather than a string assembled inside a request handler.
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])


def format_evidence(documents: list[Document], stances: list[dict]) -> str:
    """
    Render retrieved documents as the numbered evidence block.

    **These indices are the citation contract.** They are 1-based and match the
    order of `documents`, which is the order `serve.nodes.generate` validates
    `evidence_index` against. Changing the numbering here without changing the
    validator there silently breaks citation checking.

    Text is truncated to keep a 20-item context inside the model's window while
    leaving each item long enough to actually judge.
    """
    by_index = {s["evidence_index"]: s for s in stances}
    blocks: list[str] = []

    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        stance = by_index.get(i, {})
        cred = meta.get("credibility", 0.5)

        tier = meta.get("corpus_tier")
        origin = ""
        if tier == "user_upload":
            # Stated in the header, in capitals, because it changes how the
            # model must weigh the item — a cap alone is invisible to it.
            origin = "  ⚠ UNVERIFIED USER SUBMISSION — not an independent source"
        elif tier == "web":
            origin = "  (live web search result, not from the curated corpus)"

        header = (
            f"[{i}] source: {meta.get('domain') or meta.get('source') or 'unknown'}"
            f" | credibility: {cred:.2f} ({meta.get('credibility_tier', 'UNKNOWN')})"
            f" | relevance: {meta.get('relevance_score', 0.0):.3f}"
            f"{origin}"
        )
        if stance:
            header += (
                f"\n    predicted stance: {stance.get('stance', 'NEUTRAL')}"
                f" (confidence {stance.get('confidence', 0.0):.2f}"
                f"{', LOW - be sceptical' if stance.get('low_confidence') else ''})"
            )

        # Only surface a date the pipeline believes. Printing a placeholder
        # would invite the model to reason about a fabricated timeline.
        if meta.get("date_reliable"):
            header += f"\n    published: {meta.get('publish_date')}"
        else:
            header += "\n    published: date unknown (not reliably recorded)"

        if meta.get("title"):
            header += f"\n    title: {meta['title']}"

        text = doc.page_content.strip()
        if len(text) > 1800:
            text = text[:1800].rsplit(" ", 1)[0] + " [...]"

        blocks.append(f"{header}\n    text: {text}")

    return "\n\n".join(blocks) if blocks else "  No evidence retrieved."
