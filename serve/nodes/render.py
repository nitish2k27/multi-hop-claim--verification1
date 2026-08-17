"""
Render the final state as markdown.

Replaces `src/generation/report_exporter.py` minus its regex verdict parsers —
those existed to recover a verdict from prose, and there is no prose to recover
from any more. This module only formats an already-validated
`VerificationReport`; it never decides anything.

That separation is the fix for a real bug in the old system: the exporter's
regex and the API's own parse could disagree, so the downloaded report could
state a different verdict from the one the API returned for the same run.

All three outcomes render here — the graph has no exit that returns a bare
state or a traceback. M5 adds html/docx/mp3 alongside this.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from core.config import cfg
from serve.schemas import VerificationReport, VerifyState

# Rendered next to the verdict so the reader sees the system's own confidence
# in the same glance, without needing to interpret a bare integer.
_VERDICT_LABEL = {
    "TRUE":         "TRUE",
    "MOSTLY_TRUE":  "MOSTLY TRUE",
    "UNVERIFIABLE": "UNVERIFIABLE",
    "MOSTLY_FALSE": "MOSTLY FALSE",
    "FALSE":        "FALSE",
}

# Section headings, translated for the languages this project actually targets.
# The report *body* is generated in the user's language by the LLM; leaving the
# structure in English around it reads like a half-finished translation.
#
# The verdict label itself stays English on purpose — downstream code, the eval
# harness and the API all key on those five tokens, and localising them would
# mean parsing localised text to recover a verdict.
_HEADINGS = {
    "en": ["Verification report", "Insufficient evidence", "Claim", "Verdict",
           "Key findings", "Evidence", "Contradictions", "Limitations",
           "Conclusion", "Claim type", "confidence"],
    "hi": ["सत्यापन रिपोर्ट", "अपर्याप्त साक्ष्य", "दावा", "निर्णय",
           "मुख्य निष्कर्ष", "साक्ष्य", "विरोधाभास", "सीमाएँ",
           "निष्कर्ष", "दावे का प्रकार", "विश्वास"],
    "bn": ["যাচাই প্রতিবেদন", "অপর্যাপ্ত প্রমাণ", "দাবি", "রায়",
           "মূল ফলাফল", "প্রমাণ", "বৈপরীত্য", "সীমাবদ্ধতা",
           "উপসংহার", "দাবির ধরন", "আত্মবিশ্বাস"],
    "ta": ["சரிபார்ப்பு அறிக்கை", "போதிய சான்று இல்லை", "கூற்று", "தீர்ப்பு",
           "முக்கிய கண்டுபிடிப்புகள்", "சான்று", "முரண்பாடுகள்", "வரம்புகள்",
           "முடிவு", "கூற்று வகை", "நம்பிக்கை"],
    "te": ["ధృవీకరణ నివేదిక", "సరిపడని ఆధారాలు", "వాదన", "తీర్పు",
           "ముఖ్య అంశాలు", "ఆధారాలు", "వైరుధ్యాలు", "పరిమితులు",
           "ముగింపు", "వాదన రకం", "విశ్వాసం"],
    "mr": ["पडताळणी अहवाल", "अपुरा पुरावा", "दावा", "निर्णय",
           "मुख्य निष्कर्ष", "पुरावा", "विरोधाभास", "मर्यादा",
           "निष्कर्ष", "दाव्याचा प्रकार", "विश्वास"],
    "es": ["Informe de verificación", "Pruebas insuficientes", "Afirmación",
           "Veredicto", "Hallazgos clave", "Pruebas", "Contradicciones",
           "Limitaciones", "Conclusión", "Tipo de afirmación", "confianza"],
    "fr": ["Rapport de vérification", "Preuves insuffisantes", "Affirmation",
           "Verdict", "Constats clés", "Preuves", "Contradictions",
           "Limites", "Conclusion", "Type d'affirmation", "confiance"],
    "de": ["Prüfbericht", "Unzureichende Belege", "Behauptung", "Urteil",
           "Wichtigste Erkenntnisse", "Belege", "Widersprüche",
           "Einschränkungen", "Fazit", "Art der Behauptung", "Konfidenz"],
    "pt": ["Relatório de verificação", "Provas insuficientes", "Alegação",
           "Veredito", "Principais constatações", "Provas", "Contradições",
           "Limitações", "Conclusão", "Tipo de alegação", "confiança"],
    "ar": ["تقرير التحقق", "أدلة غير كافية", "الادعاء", "الحكم",
           "النتائج الرئيسية", "الأدلة", "التناقضات", "القيود",
           "الخلاصة", "نوع الادعاء", "الثقة"],
    "ru": ["Отчёт о проверке", "Недостаточно доказательств", "Утверждение",
           "Вердикт", "Основные выводы", "Доказательства", "Противоречия",
           "Ограничения", "Заключение", "Тип утверждения", "уверенность"],
}

_KEYS = ["report_title", "abstain_title", "claim", "verdict", "key_findings",
         "evidence", "contradictions", "limitations", "conclusion",
         "claim_type", "confidence"]


def _headings(language: str) -> dict[str, str]:
    """Section labels for a language, falling back to English."""
    values = _HEADINGS.get(language, _HEADINGS["en"])
    return dict(zip(_KEYS, values))


def _render_rejected(state: VerifyState) -> str:
    return "\n".join([
        "# Not a verifiable claim",
        "",
        f"> {state.get('claim', '')}",
        "",
        state.get("reject_reason", "Input is not a verifiable claim."),
        "",
        f"*Claim-detector confidence: {state.get('claim_confidence', 0.0):.3f}*",
        "",
        "*No evidence was retrieved and no language model was called.*",
    ])


def _render_report(state: VerifyState, report: VerificationReport) -> str:
    outcome = state.get("outcome", "verified")
    documents = state.get("evidence") or []
    stances = {s["evidence_index"]: s for s in (state.get("stances") or [])}

    h = _headings(state.get("language", "en"))

    lines: list[str] = [
        f"# {h['report_title'] if outcome == 'verified' else h['abstain_title']}",
        "",
        f"## {h['claim']}",
        f"> {report.claim}",
        "",
        f"## {h['verdict']}",
        f"**{_VERDICT_LABEL.get(report.verdict, report.verdict)}** "
        f"— {report.confidence}% {h['confidence']}",
        "",
    ]

    if outcome == "abstained":
        lines += [
            "This system found no evidence relevant enough to judge the claim, "
            "so it is declining to offer a verdict.",
            "",
        ]
    else:
        lines += [f"{h['claim_type']}: {report.claim_type}", ""]
        # Where the evidence came from changes how much weight a reader should
        # give it: the indexed corpus is fixed and inspectable, live web results
        # are neither. Stating it is more honest than silently mixing them.
        source = state.get("evidence_source")
        if source == "web":
            lines += [
                "> **Evidence source: live web search.** The indexed corpus had "
                "nothing relevant to this claim, so the system fell back to "
                "searching the web. These sources were not curated in advance.",
                "",
            ]

    if report.key_findings:
        lines += [f"## {h['key_findings']}", ""]
        for finding in report.key_findings:
            source = "unknown"
            index = finding.evidence_index
            if 1 <= index <= len(documents):
                meta = documents[index - 1].metadata
                source = meta.get("domain") or meta.get("source") or "unknown"
            lines.append(f"- {finding.finding}  \n  *[{index}] {source}*")
            if finding.quote:
                lines.append(f"  > {finding.quote}")
        lines.append("")

    if report.evidence_analysis:
        lines += [f"## {h['evidence']}", ""]
        for assessment in report.evidence_analysis:
            index = assessment.evidence_index
            meta = (
                documents[index - 1].metadata
                if 1 <= index <= len(documents) else {}
            )
            credibility = meta.get("credibility", 0.0)
            tier = meta.get("credibility_tier", "UNKNOWN")
            relevance = meta.get("relevance_score", 0.0)
            predicted = stances.get(index, {})

            tier = meta.get("corpus_tier")
            origin = ""
            if tier == "web":
                origin = " · via web search"
            elif tier == "user_upload":
                # Flagged loudly: this is the user's own document, and a reader
                # must not mistake it for an independent source.
                origin = " · **your upload, unverified**"
            lines.append(
                f"**[{index}] {assessment.source}** — "
                f"credibility {credibility:.2f} ({tier}), "
                f"relevance {relevance:.3f}{origin}"
            )
            lines.append(
                f"- Stance: **{assessment.stance}**"
                + (f" (model predicted {predicted.get('stance')} at "
                   f"{predicted.get('confidence', 0.0):.2f})"
                   if predicted and predicted.get("stance") != assessment.stance
                   else "")
            )
            lines.append(
                f"- Directly relevant: {'yes' if assessment.directly_relevant else 'no'}"
            )
            lines.append(f"- {assessment.reasoning}")
            if meta.get("url"):
                lines.append(f"- Source: {meta['url']}")
            lines.append("")

    if report.contradictions:
        lines += [f"## {h['contradictions']}", ""]
        lines += [f"- {c}" for c in report.contradictions]
        lines.append("")

    if report.limitations:
        lines += [f"## {h['limitations']}", ""]
        lines += [f"- {limit}" for limit in report.limitations]
        lines.append("")

    lines += [f"## {h['conclusion']}", "", report.conclusion, ""]

    diagnostics = state.get("diagnostics", {})
    stats = state.get("retrieval_stats", {})
    lines += [
        "---",
        "",
        f"*outcome: `{outcome}` · evidence: {len(documents)} · "
        f"candidates: {stats.get('candidates', 0)} · "
        f"LLM calls: {diagnostics.get('llm_calls', 0)}*",
    ]
    return "\n".join(lines)


def render_output(state: VerifyState) -> dict:
    """
    Format whichever of the three exits we arrived at, and export any extras.

    Markdown is always produced. Audio is produced only when the input was
    voice — the principle from the plan is that you get an answer back in the
    form you asked the question in, and generating an mp3 nobody asked for
    would be a slow network round trip on every text verification.
    """
    report = state.get("report")
    rendered = (_render_rejected(state) if report is None
                else _render_report(state, report))

    update: dict = {"rendered": rendered}

    if state.get("input_kind") == "voice" and cfg.voice_output and report:
        from serve.nodes.export import export

        artifacts = export(state, ["mp3"])
        if artifacts:
            update["artifacts"] = {**state.get("artifacts", {}), **artifacts}

    return update


render_node = RunnableLambda(render_output, name="render_output")
