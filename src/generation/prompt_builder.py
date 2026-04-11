"""
src/generation/prompt_builder.py
──────────────────────────────────
Builds the full LLM prompt from NLP+RAG pipeline output.

Key design decisions:
  1. Output language — report is written in the user's detected language
  2. Evidence grounding — model must ONLY use provided evidence, no external facts
  3. Verdict constraint — exactly 5 labels: TRUE/MOSTLY TRUE/UNVERIFIABLE/MOSTLY FALSE/FALSE
  4. Anti-hallucination — every claim must be traceable to a specific evidence piece
  5. Credibility weighting — sources scored < 0.5 must be flagged explicitly
  6. Structured output — fixed section headers so report_exporter can parse them reliably
"""

from typing import Dict, Any

# ── Per-language output instruction ──────────────────────────────────────────
LANG_INSTRUCTIONS = {
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

SYSTEM_PROMPT_TEMPLATE = """You are an expert fact-checking analyst with deep knowledge of journalism standards, evidence evaluation, and critical reasoning.

LANGUAGE INSTRUCTION (MANDATORY):
{language_instruction}
Every section header, every sentence, every word in your report must be in that language.

YOUR TASK:
Read the fact-verification context provided and write a detailed professional verification report.

STRICT ANTI-HALLUCINATION RULES — follow every one without exception:
1. Base EVERY factual claim in your report ONLY on the evidence listed in the context.
2. Do NOT add facts, statistics, dates, names, or claims from your training knowledge.
3. If evidence is insufficient, say so clearly — do NOT speculate or fill gaps.
4. If evidence pieces contradict each other, highlight the contradiction explicitly.
5. Weight evidence by credibility score:
   - Score >= 0.80 → high credibility, weight heavily
   - Score 0.50-0.79 → medium credibility, treat with caution and note this
   - Score < 0.50 → low credibility, flag this explicitly in your analysis
6. Analyse each evidence piece individually before forming your verdict.
7. Your verdict MUST be exactly one of these five labels (use the English label even in other languages):
   TRUE | MOSTLY TRUE | UNVERIFIABLE | MOSTLY FALSE | FALSE
8. Never invent citations, source names, or statistics not present in the context.
9. Every bullet point in Key Findings must reference which specific evidence piece supports it.

OUTPUT FORMAT — use exactly these section headers in the exact order shown:

## Claim
State the exact claim being verified.

## Initial Assessment
What type of claim is this? (statistical/event/policy/scientific/biographical)
What evidence would be needed to verify it properly?

## Evidence Analysis
For each evidence piece in the context:
- Source name and credibility score
- Stance: SUPPORTS / REFUTES / NEUTRAL — with specific reasoning
- How directly relevant it is to the claim
- Any limitations of this specific piece

## Contradictions
Are any evidence pieces in conflict with each other?
If yes: explain the conflict and its implication for the verdict.
If no: state that no contradictions were found.

## Verdict
**[VERDICT: TRUE / MOSTLY TRUE / UNVERIFIABLE / MOSTLY FALSE / FALSE]**
Confidence: [0-100]%
Reasoning: 2-3 sentences citing specific evidence pieces.

## Key Findings
- Each bullet must cite a specific evidence piece
- Most important findings first
- Maximum 5 bullets

## Limitations
- Missing evidence that could change the verdict
- Assumptions that were necessary
- Caveats that apply to this verdict

## Conclusion
2-3 sentences for a non-expert reader.
Do NOT introduce new information — only summarise what the evidence showed."""


def build_system_prompt(user_language: str = "en") -> str:
    """Build system prompt with the correct language instruction."""
    lang_instruction = LANG_INSTRUCTIONS.get(
        user_language,
        f"Write the entire report in the language with ISO code '{user_language}'. "
        f"If you cannot reliably write in that language, write in English and note this."
    )
    return SYSTEM_PROMPT_TEMPLATE.format(language_instruction=lang_instruction)


def build_user_message(llm_context: str, user_language: str = "en") -> str:
    """Build the user message containing the context."""
    lang_instruction = LANG_INSTRUCTIONS.get(user_language, "Write in English.")
    return (
        "FACT VERIFICATION CONTEXT TO ANALYSE:\n\n"
        + llm_context
        + f"\n\nREMINDER: {lang_instruction} "
        + "Base every factual statement ONLY on the evidence above — no external knowledge."
    )


def build_groq_messages(llm_context: str, user_language: str = "en"):
    """Return the messages list for Groq API (OpenAI format)."""
    return [
        {"role": "system", "content": build_system_prompt(user_language)},
        {"role": "user",   "content": build_user_message(llm_context, user_language)},
    ]


def build_mistral_prompt(llm_context: str, user_language: str = "en") -> str:
    """Return the formatted prompt string for Mistral/Colab inference."""
    system = build_system_prompt(user_language)
    user   = build_user_message(llm_context, user_language)
    return f"<s>[INST] {system}\n\n{user} [/INST]"


def extract_llm_context(pipeline_output: Dict[str, Any]) -> str:
    """
    Extract the llm_context string from pipeline output.
    Handles both the standard format (has llm_context key) and
    fallback reconstruction from raw fields.
    """
    if isinstance(pipeline_output, str):
        return pipeline_output

    if "llm_context" in pipeline_output and pipeline_output["llm_context"]:
        return pipeline_output["llm_context"]

    # Fallback: reconstruct from raw fields
    claim    = pipeline_output.get("claim", "Unknown claim")
    verdict  = pipeline_output.get("verdict", "UNVERIFIABLE")
    conf     = pipeline_output.get("confidence", 0.0)
    evidence = pipeline_output.get("evidence", [])
    agg      = pipeline_output.get("aggregation", {})

    ev_lines = []
    for i, ev in enumerate(evidence, 1):
        cred = ev.get("credibility", {})
        score = cred.get("total_score", 0.5) if isinstance(cred, dict) else float(cred or 0.5)
        ev_lines.append(
            f"  [{i}] [{ev.get('stance','NEUTRAL')}] "
            f"(source: {ev.get('source','unknown')}, credibility: {score:.2f})\n"
            f"      {str(ev.get('document', ev.get('text', '')))[:300]}"
        )

    agg_lines = []
    if agg:
        agg_lines = [
            f"  - Preliminary verdict:  {agg.get('verdict', verdict)}",
            f"  - Support score:        {agg.get('support_percentage', 0):.1f}%",
            f"  - Refute score:         {agg.get('refute_percentage', 0):.1f}%",
            f"  - Neutral score:        {agg.get('neutral_percentage', 0):.1f}%",
            f"  - Evidence pieces:      {agg.get('num_evidence', len(evidence))}",
            f"  - Supports:             {agg.get('num_supports', 0)}",
            f"  - Refutes:              {agg.get('num_refutes', 0)}",
        ]

    return (
        "=== FACT VERIFICATION CONTEXT ===\n\n"
        f"CLAIM:\n  {claim}\n\n"
        f"RETRIEVED EVIDENCE:\n"
        + ("\n".join(ev_lines) if ev_lines else "  No evidence retrieved")
        + "\n\nEVIDENCE AGGREGATION:\n"
        + ("\n".join(agg_lines) if agg_lines else f"  Preliminary verdict: {verdict} ({conf:.1f}%)")
        + "\n\nTASK:\nProvide a detailed fact-verification analysis.\n"
        + "=== END CONTEXT ==="
    )