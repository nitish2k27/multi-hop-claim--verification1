"""
scripts/test_groq_pipeline.py

End-to-end test using Groq API instead of Colab server.
Completely separate from test_full_pipeline.py (Colab approach).

SETUP:
    1. Get free API key at console.groq.com
    2. python scripts/setup_groq.py "gsk_your_key_here"
    3. python scripts/test_groq_pipeline.py "Your claim here"

DIFFERENCES from Colab approach:
    - No Colab notebook needed
    - No ngrok URL to manage
    - Uses Llama-3-70B (not your fine-tuned adapter)
    - Much faster (~5 seconds vs ~60 seconds)
    - Free tier: 500 requests/day, 14400 tokens/minute
"""

import sys
import logging
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def run_groq_test(claim):
    print("\n" + "="*60)
    print("GROQ PIPELINE TEST")
    print("="*60)
    print("Claim: " + claim + "\n")

    # ── Step 1: NLP + RAG locally ────────────────────────────
    print("[1/4] Running NLP + RAG pipeline locally...")
    rag_result = None

    try:
        from src.nlp.nlp_pipeline import NLPPipeline
        from src.rag.vector_database import VectorDatabase
        from src.rag.rag_pipeline import RAGPipeline

        nlp       = NLPPipeline()
        vector_db = VectorDatabase()
        rag       = RAGPipeline(
            vector_db         = vector_db,
            collection_name   = "news_articles",
            nlp_model_manager = nlp.model_manager,
        )

        rag_result = rag.verify_claim(claim=claim, top_k=5)

        print("     Verdict:         " + rag_result.get("verdict", "?")
              + " (" + str(round(rag_result.get("confidence", 0), 1)) + "%)")
        print("     Evidence pieces: " + str(len(rag_result.get("evidence", []))))
        print("     llm_context:     " + str(len(rag_result.get("llm_context", ""))) + " chars")

    except Exception as e:
        print("     [ERROR] Pipeline failed: " + str(e))
        traceback.print_exc()
        print("\n     Falling back to mock output...")
        rag_result = _mock_output(claim)

    # ── Step 2: Generate report via Groq API ─────────────────
    print("\n[2/4] Generating report via Groq API...")
    try:
        from src.generation.report_generator_groq import ReportGeneratorGroq
        generator = ReportGeneratorGroq()
        report    = generator.generate(rag_result)

    except Exception as e:
        print("     [ERROR] Groq generation failed: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: Export files ──────────────────────────────────
    print("\n[3/4] Exporting report files...")
    files = {}

    gen_dir = project_root / "src" / "generation"
    if (gen_dir / "report_exporter.py").exists():
        try:
            from src.generation.report_exporter import ReportExporter
            exporter = ReportExporter(output_dir="data/reports_groq")

            files["markdown"] = exporter.to_markdown(report, claim=claim)
            print("     markdown : " + files["markdown"])

            files["html"] = exporter.to_html(report, claim=claim)
            print("     html     : " + files["html"])

            try:
                files["docx"] = exporter.to_docx(report, claim=claim)
                print("     docx     : " + files["docx"])
            except ImportError:
                print("     docx     : SKIPPED - pip install python-docx")

            try:
                files["pdf"] = exporter.to_pdf(report, claim=claim)
                print("     pdf      : " + files["pdf"])
            except ImportError:
                print("     pdf      : SKIPPED - pip install weasyprint")

        except Exception as e:
            print("     [ERROR] Export failed: " + str(e))
            traceback.print_exc()
    else:
        print("     [SKIP] report_exporter.py not found")

    # ── Step 4: Results ───────────────────────────────────────
    print("\n[4/4] Results")
    print("="*60)
    print("Claim      : " + claim)
    print("Verdict    : " + rag_result.get("verdict", "UNVERIFIABLE"))
    print("Confidence : " + str(round(rag_result.get("confidence", 0.0), 1)) + "%")
    print("\nFull report:")
    print("-"*60)
    print(report)
    print("-"*60)

    if files:
        print("\nFiles saved to data/reports_groq/:")
        for fmt, path in files.items():
            print("  " + fmt.ljust(10) + ": " + path)

    print("\n[OK] Groq pipeline test complete")
    return report


def _mock_output(claim):
    return {
        "claim":       claim,
        "verdict":     "UNVERIFIABLE",
        "confidence":  0.0,
        "evidence":    [],
        "ready_for_llm": True,
        "metadata":    {"original_claim": claim},
        "llm_context": (
            "=== FACT VERIFICATION CONTEXT ===\n"
            "CLAIM:\n  " + claim + "\n\n"
            "RETRIEVED EVIDENCE:\n"
            "  No evidence retrieved (mock mode)\n\n"
            "EVIDENCE AGGREGATION:\n"
            "  - Preliminary verdict: UNVERIFIABLE\n"
            "  - Evidence pieces: 0\n"
            "=== END CONTEXT ==="
        ),
    }


if __name__ == "__main__":
    claim = sys.argv[1] if len(sys.argv) > 1 else "India's GDP grew 8% in 2024"
    run_groq_test(claim)