#!/usr/bin/env python3
"""
test_my_claim.py
────────────────
Interactive NLP Pipeline Tester
Type any claim and get the full NLP + RAG pipeline output.

Usage:
    python test_my_claim.py
    python test_my_claim.py "India GDP grew 8% in 2024"
"""

import sys
import os
import json
import textwrap

# ── Silence noisy TF/oneDNN warnings ─────────────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ── Helpers ───────────────────────────────────────────────────────────────────

W  = "\033[0m"       # reset
B  = "\033[1m"       # bold
CY = "\033[96m"      # cyan
GR = "\033[92m"      # green
YE = "\033[93m"      # yellow
RE = "\033[91m"      # red
BL = "\033[94m"      # blue
MA = "\033[95m"      # magenta

VERDICT_COLORS = {
    "TRUE":         GR,
    "FALSE":        RE,
    "MOSTLY TRUE":  GR,
    "MOSTLY FALSE": RE,
    "UNVERIFIABLE": YE,
    "MIXED":        YE,
}


def hr(char="─", width=68):
    print(char * width)


def section(title):
    print()
    hr("═")
    print(f"{B}{CY}  {title}{W}")
    hr("═")


def sub(title):
    print(f"\n{B}{BL}  ▸ {title}{W}")
    hr("─")


def ok(msg):   print(f"  {GR}✓{W}  {msg}")
def warn(msg): print(f"  {YE}⚠{W}  {msg}")
def err(msg):  print(f"  {RE}✗{W}  {msg}")
def info(msg): print(f"     {msg}")


def wrap(text, width=60, indent=5):
    """Wrap long text with indent."""
    lines = textwrap.wrap(str(text), width=width)
    prefix = " " * indent
    return ("\n" + prefix).join(lines)


def print_claim_detection(cd: dict):
    sub("STEP 1 — Claim Detection")
    is_claim = cd.get("is_claim", False)
    conf     = cd.get("confidence", 0.0)
    label    = cd.get("label", "unknown")

    flag = f"{GR}YES — This is a verifiable factual claim{W}" if is_claim \
           else f"{YE}NO  — This looks like an opinion / question{W}"
    ok(f"Is claim : {flag}")
    ok(f"Label    : {B}{label}{W}")

    bar_len = int(conf * 30)
    bar     = f"{GR}{'█' * bar_len}{'░' * (30 - bar_len)}{W}"
    ok(f"Confidence: [{bar}] {conf:.1%}")


def print_entities(ents: dict):
    sub("STEP 2 — Named Entity Recognition (NER)")
    total = ents.get("total_entities", 0)
    types = ents.get("entity_types", [])
    by_type = ents.get("entities", {})

    if total == 0:
        warn("No named entities found")
        return

    ok(f"Total entities : {B}{total}{W}")
    ok(f"Entity types   : {', '.join(types) or 'none'}")

    type_labels = {
        "PER": "Person", "ORG": "Organisation", "LOC": "Location",
        "MISC": "Miscellaneous", "GPE": "Geo-political",
    }
    for etype, items in by_type.items():
        if not items:
            continue
        label = type_labels.get(etype, etype)
        names = ", ".join(str(i.get("text", i)) if isinstance(i, dict) else str(i)
                          for i in items)
        info(f"  {MA}{label:20s}{W} {names}")


def print_entity_linking(el: dict):
    sub("STEP 3 — Entity Linking (Wikidata)")
    linked = el.get("linked_entities", [])
    if not linked:
        warn("No entities linked to knowledge base")
        return
    ok(f"Linked {len(linked)} entit{'y' if len(linked)==1 else 'ies'} to Wikidata")
    for e in linked[:5]:  # show max 5
        name   = e.get("text", e.get("entity", "?"))
        qid    = e.get("wikidata_id", e.get("qid", ""))
        conf_e = e.get("confidence", 0.0)
        info(f"  {B}{name:30s}{W}  QID: {CY}{qid or '—'}{W}  conf: {conf_e:.2f}")


def print_temporal(temp: dict):
    sub("STEP 4 — Temporal Extraction")
    dates = temp.get("dates", [])
    if not dates:
        warn("No temporal expressions found")
        return
    ok(f"Found {len(dates)} temporal expression(s)")
    for d in dates:
        expr   = d.get("text", d.get("expression", "?"))
        dtype  = d.get("type", "?")
        norm   = d.get("normalized", d.get("value", ""))
        info(f"  '{YE}{expr}{W}'  →  {dtype}  ({norm})")


def print_stance(stance_info: dict):
    sub("STEP 5 — Stance Detection")
    stance = stance_info.get("stance", "UNKNOWN")
    conf   = stance_info.get("confidence", 0.0)
    colors = {"SUPPORTS": GR, "REFUTES": RE, "NOT ENOUGH INFO": YE, "NEUTRAL": YE}
    c = colors.get(stance, W)
    bar_len = int(conf * 30)
    bar     = f"{c}{'█' * bar_len}{'░' * (30 - bar_len)}{W}"
    ok(f"Stance     : {c}{B}{stance}{W}")
    ok(f"Confidence : [{bar}] {conf:.1%}")


def print_rag_evidence(rag: dict):
    sub("STEP 6 — RAG Evidence Retrieval")

    evidence = rag.get("evidence", [])
    verdict  = rag.get("verdict", "UNVERIFIABLE")
    conf     = rag.get("confidence", 0.0)
    scores   = rag.get("scores", {})

    vc = VERDICT_COLORS.get(verdict, YE)
    ok(f"Preliminary verdict : {vc}{B}{verdict}{W}")
    ok(f"Confidence          : {conf:.1f}%")
    ok(f"Evidence pieces     : {len(evidence)}")

    s_pct = scores.get("support_percentage", 0)
    r_pct = scores.get("refute_percentage", 0)
    n_pct = scores.get("neutral_percentage", 0)
    info(f"  Support: {GR}{s_pct:.1f}%{W}   Refute: {RE}{r_pct:.1f}%{W}   Neutral: {YE}{n_pct:.1f}%{W}")

    if not evidence:
        warn("No evidence retrieved from knowledge base")
        return

    print(f"\n  {B}Evidence breakdown:{W}")
    for i, ev in enumerate(evidence, 1):
        stance = ev.get("stance", "?")
        sc     = ev.get("stance_confidence", 0.0)
        source = ev.get("source", ev.get("metadata", {}).get("source", "unknown"))
        cred   = ev.get("credibility", {}).get("total_score", 0.0)
        tier   = ev.get("credibility", {}).get("tier", "?")
        doc    = ev.get("document", "")

        sc_color = {"SUPPORTS": GR, "REFUTES": RE}.get(stance, YE)

        print(f"\n  {B}Evidence #{i}{W}")
        info(f"Stance  : {sc_color}{stance}{W}  ({sc:.2f})")
        info(f"Source  : {CY}{source}{W}  |  Credibility: {cred:.2f}  ({tier})")
        snippet = wrap(doc[:200] + ("…" if len(doc) > 200 else ""), width=60, indent=9)
        info(f"Snippet : {snippet}")


def detect_and_translate(claim: str):
    """Detect language and translate to English. Returns (claim_en, user_lang, lang_name)."""
    try:
        from langdetect import detect
        user_lang = detect(claim)
    except Exception:
        user_lang = "en"

    LANG_NAMES = {
        "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
        "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada",
        "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "es": "Spanish",
        "fr": "French", "de": "German", "ar": "Arabic", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "ru": "Russian", "pt": "Portuguese",
    }
    lang_name = LANG_NAMES.get(user_lang, user_lang.upper())

    if user_lang == "en":
        return claim, user_lang, lang_name

    print(f"\n  {BL}Detected language:{W} {B}{lang_name}{W} — translating to English…")
    try:
        from deep_translator import GoogleTranslator
        claim_en = GoogleTranslator(source=user_lang, target="en").translate(claim)
        ok(f"Translated: {YE}{claim_en}{W}")
        return claim_en, user_lang, lang_name
    except Exception as e:
        warn(f"Translation failed ({e}) — running NLP on original text")
        return claim, user_lang, lang_name


def run_pipeline(claim: str):
    section(f"CLAIM: {claim}")

    # ── Detect language + translate to English ────────────────────────────────
    claim_en, user_lang, lang_name = detect_and_translate(claim)

    # ── NLP ──────────────────────────────────────────────────────────────────
    print(f"\n{B}Loading NLP pipeline… (first run takes 30–60s){W}")
    try:
        from src.nlp.nlp_pipeline import NLPPipeline
        nlp        = NLPPipeline()
        nlp_result = nlp.analyze(claim_en)   # always English input
        analysis   = nlp_result.get("analysis", {})

        print_claim_detection(analysis.get("claim_detection", {}))
        print_entities(analysis.get("entities", {}))
        print_entity_linking(analysis.get("entity_linking", {}))
        print_temporal(analysis.get("temporal", {}))
        if "stance" in analysis:
            print_stance(analysis["stance"])

    except Exception as e:
        err(f"NLP pipeline failed: {e}")
        import traceback; traceback.print_exc()
        return

    # ── RAG ──────────────────────────────────────────────────────────────────
    print(f"\n{B}Loading RAG pipeline…{W}")
    try:
        from src.rag.vector_database import VectorDatabase
        from src.rag.rag_pipeline    import RAGPipeline

        vdb   = VectorDatabase()
        stats = vdb.get_collection_stats("news_articles")

        if stats.get("exists") and stats.get("count", 0) > 0:
            rag        = RAGPipeline(vdb, "news_articles",
                                     nlp_model_manager=nlp.model_manager)
            rag_result = rag.verify_claim(claim, top_k=5)
            print_rag_evidence(rag_result)
        else:
            warn("ChromaDB is empty — RAG evidence skipped")

    except Exception as e:
        err(f"RAG pipeline failed: {e}")
        import traceback; traceback.print_exc()

    # ── Groq LLM Report ───────────────────────────────────────────────────────
    groq_report  = None
    groq_verdict = None
    groq_conf    = None

    if "rag_result" in locals():
        sub("STEP 7 — Groq LLM Report (llama-3.3-70b)")
        try:
            # Read Groq key
            key_path = os.path.join("configs", "groq_token.txt")
            groq_key = None
            if os.path.exists(key_path):
                with open(key_path, "r", encoding="utf-8") as f:
                    groq_key = f.read().strip()
            groq_key = groq_key or os.environ.get("GROQ_API_KEY")

            if not groq_key:
                warn("No Groq API key found in configs/groq_token.txt")
                warn("Skipping LLM report — RAG verdict shown above is still valid")
            else:
                print(f"  {BL}Calling Groq API…{W}  (usually 3–8 seconds)")
                from src.generation.report_generator_groq import ReportGeneratorGroq
                gen          = ReportGeneratorGroq(api_key=groq_key)
                groq_report  = gen.generate(rag_result, user_language=user_lang)

                # Extract verdict / confidence line from report
                for line in groq_report.splitlines():
                    ll = line.upper()
                    for v in ("TRUE", "FALSE", "MOSTLY TRUE", "MOSTLY FALSE",
                              "UNVERIFIABLE", "MIXED"):
                        if v in ll:
                            groq_verdict = v
                            break
                    if groq_verdict:
                        break
                groq_conf = rag_result.get("confidence", 0.0)

                vc = VERDICT_COLORS.get(groq_verdict or "", YE)
                ok(f"Groq verdict : {vc}{B}{groq_verdict or 'see report'}{W}  "
                   f"({groq_conf:.1f}% confidence)")
                print()

                # Print the full report, word-wrapped
                hr("·")
                for line in groq_report.splitlines():
                    if line.strip():
                        print(f"  {wrap(line, width=64, indent=2)}")
                    else:
                        print()
                hr("·")

        except Exception as e:
            err(f"Groq generation failed: {e}")
            import traceback; traceback.print_exc()

    # ── HTML Report export ────────────────────────────────────────────────────
    html_path = None
    if groq_report and "rag_result" in locals():
        sub("STEP 8 — HTML Report Export")
        try:
            from src.generation.report_exporter import ReportExporter
            exporter  = ReportExporter(output_dir="data/reports")
            safe_name = claim[:80].replace("\n", " ").replace("\r", " ").strip()
            exports   = exporter.export_all(groq_report, claim=safe_name,
                                            rag_result=rag_result)
            html_path = exports.get("html")
            pdf_path  = exports.get("pdf")
            docx_path = exports.get("docx")
            if html_path and not str(html_path).startswith("SKIPPED"):
                ok(f"HTML report : {GR}{html_path}{W}")
            else:
                warn("HTML export failed")
            if pdf_path and not str(pdf_path).startswith("SKIPPED"):
                ok(f"PDF  report : {GR}{pdf_path}{W}")
            else:
                warn(f"PDF  skipped (WeasyPrint GTK libs not installed on Windows)")
            if docx_path and not str(docx_path).startswith("SKIPPED"):
                ok(f"DOCX report : {GR}{docx_path}{W}")
        except Exception as e:
            warn(f"Report export failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("SUMMARY")
    ok(f"Input language : {B}{lang_name}{W}{' → translated to English for NLP' if user_lang != 'en' else ''}")
    try:
        cd = nlp_result["analysis"]["claim_detection"]
        ok(f"Claim detected : {'YES' if cd['is_claim'] else 'NO'} "
           f"({cd['confidence']:.1%} confidence)")
    except Exception:
        pass

    if "rag_result" in locals():
        rv = rag_result.get("verdict", "?")
        rc = rag_result.get("confidence", 0.0)
        vc = VERDICT_COLORS.get(rv, YE)
        ok(f"RAG verdict    : {vc}{B}{rv}{W}  ({rc:.1f}%)")

    if groq_verdict:
        vc = VERDICT_COLORS.get(groq_verdict, YE)
        ok(f"Groq verdict   : {vc}{B}{groq_verdict}{W}  ({groq_conf:.1f}%)")

    if html_path and not str(html_path).startswith("SKIPPED"):
        ok(f"HTML saved     : {html_path}")

    print(f"\n  {GR}Done!{W}  Full JSON saved to  {B}data/test_output.json{W}\n")

    # Save full JSON
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/test_output.json", "w", encoding="utf-8") as f:
            combined = {
                "claim":       claim,
                "nlp_result":  nlp_result  if "nlp_result"  in locals() else {},
                "rag_result":  rag_result  if "rag_result"  in locals() else {},
                "groq_report": groq_report if groq_report else "",
            }
            json.dump(combined, f, indent=2, default=str)
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    hr("═")
    print(f"{B}{CY}  FACT VERIFICATION — NLP PIPELINE TESTER{W}")
    hr("═")

    # Claim from command-line arg
    if len(sys.argv) > 1:
        claim = " ".join(sys.argv[1:])
        print(f"\n  Using claim from argument: {YE}{claim}{W}\n")
    else:
        print(f"""
  {B}How to use:{W}
    • Type your claim and press Enter
    • Type {CY}quit{W} or press Ctrl+C to exit
    • Or pass claim directly:
      {GR}python test_my_claim.py "India GDP grew 8% in 2024"{W}
""")
        while True:
            try:
                claim = input(f"  {B}Enter claim:{W} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n  {YE}Bye!{W}\n")
                sys.exit(0)

            if not claim:
                warn("Please enter a claim")
                continue
            if claim.lower() in ("quit", "exit", "q"):
                print(f"\n  {YE}Bye!{W}\n")
                sys.exit(0)
            break

    run_pipeline(claim)


if __name__ == "__main__":
    main()