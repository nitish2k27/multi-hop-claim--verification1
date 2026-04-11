"""
scripts/test_full_pipeline.py
──────────────────────────────
End-to-end test: Local NLP+RAG → Colab LLM → Report → Export

Usage:
    python scripts/test_full_pipeline.py
    python scripts/test_full_pipeline.py "Your custom claim here"
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def run_test(claim: str):
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)
    print(f"\nClaim: {claim}\n")

    # ── Step 1: NLP analysis (runs locally) ──────────────────
    print("[1/4] Running NLP + RAG pipeline locally...")
    rag_result = None

    try:
        # ── 1a: NLP pipeline ─────────────────────────────────
        from src.nlp.nlp_pipeline import NLPPipeline
        nlp        = NLPPipeline()
        nlp_result = nlp.analyze(claim)
        print(f"     NLP done. Entities: {nlp_result['analysis']['entities']['total_entities']}")

        # ── 1b: RAG pipeline ──────────────────────────────────
        # RAGPipeline.__init__ signature:
        #   def __init__(self, vector_db, collection_name="news_articles",
        #                nlp_model_manager=None)
        # verify_claim signature:
        #   def verify_claim(self, claim, top_k=5, user_context_docs=None)
        # Note: verify_claim takes only the claim string — NOT the nlp_result dict
        from src.rag.vector_database import VectorDatabase
        from src.rag.rag_pipeline import RAGPipeline

        vector_db = VectorDatabase()
        rag       = RAGPipeline(
            vector_db        = vector_db,
            collection_name  = "news_articles",
            nlp_model_manager = nlp.model_manager,   # reuse already-loaded models
        )

        # verify_claim(claim_string, top_k, user_context_docs)
        rag_result = rag.verify_claim(claim=claim, top_k=5)

        verdict        = rag_result.get("verdict", "UNVERIFIABLE")
        confidence     = rag_result.get("confidence", 0.0)
        evidence_count = len(rag_result.get("evidence", []))
        context_len    = len(rag_result.get("llm_context", ""))

        print(f"     Verdict:         {verdict} ({confidence:.1f}%)")
        print(f"     Evidence pieces: {evidence_count}")
        print(f"     llm_context:     {context_len} chars")

    except Exception as e:
        print(f"     [ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n     Falling back to mock output — LLM will still be called")
        rag_result = _mock_nlp_output(claim)

    # ── Step 2: Send to Colab LLM ─────────────────────────────
    print("\n[2/4] Sending context to Colab inference server...")
    try:
        from src.generation.report_generator import ReportGenerator
        generator = ReportGenerator()
        report    = generator.generate(rag_result)
        words     = len(report.split())
        print(f"     Report received: {words} words / {len(report)} chars")

    except Exception as e:
        print(f"     [ERROR] LLM generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: Export files ──────────────────────────────────
    print("\n[3/4] Exporting report files...")
    files = {}

    gen_dir = project_root / "src" / "generation"
    if not (gen_dir / "report_exporter.py").exists():
        print(f"     [SKIP] report_exporter.py not found at {gen_dir}")
        print(f"     Copy the report_exporter.py file there first")
    else:
        try:
            from src.generation.report_exporter import ReportExporter
            exporter = ReportExporter(output_dir="data/reports")

            files["markdown"] = exporter.to_markdown(report, claim=claim)
            print(f"     markdown : {files['markdown']}")

            files["html"] = exporter.to_html(report, claim=claim)
            print(f"     html     : {files['html']}")

            try:
                files["docx"] = exporter.to_docx(report, claim=claim)
                print(f"     docx     : {files['docx']}")
            except ImportError:
                print("     docx     : SKIPPED — pip install python-docx")

            try:
                files["pdf"] = exporter.to_pdf(report, claim=claim)
                print(f"     pdf      : {files['pdf']}")
            except ImportError:
                print("     pdf      : SKIPPED — pip install weasyprint")

        except Exception as e:
            print(f"     [ERROR] Export failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Step 4: Print results ─────────────────────────────────
    verdict    = rag_result.get("verdict", "UNVERIFIABLE")
    confidence = rag_result.get("confidence", 0.0)

    print("\n[4/4] Results")
    print("="*60)
    print(f"Claim      : {claim}")
    print(f"Verdict    : {verdict}")
    print(f"Confidence : {confidence:.1f}%")
    print(f"\nFull report:")
    print("-"*60)
    print(report)
    print("-"*60)

    if files:
        print("\nDownloadable files saved:")
        for fmt, path in files.items():
            print(f"  {fmt:10s}: {path}")

    print("\n[OK] Pipeline test complete")
    return report


def _mock_nlp_output(claim: str) -> dict:
    """Mock NLP output matching exact format of rag_pipeline.verify_claim()"""
    return {
        "claim":       claim,
        "verdict":     "UNVERIFIABLE",
        "confidence":  0.0,
        "evidence":    [],
        "ready_for_llm": True,
        "metadata":    {"original_claim": claim},
        "llm_context": f"""=== FACT VERIFICATION CONTEXT ===

CLAIM:
  {claim}

CLAIM ANALYSIS:
  Is verifiable claim: True
  Claim confidence:    0.85

NAMED ENTITIES:
  - Extracted from claim text

RETRIEVED EVIDENCE:
  No evidence retrieved (mock mode — NLP pipeline unavailable)

EVIDENCE AGGREGATION:
  - Preliminary verdict:  UNVERIFIABLE
  - Support score:        0.0%
  - Refute score:         0.0%
  - Evidence pieces:      0

TASK:
Based on the claim, entities, temporal context, and retrieved evidence above,
provide a detailed fact-verification analysis with:
1. Final verdict (TRUE / FALSE / MOSTLY TRUE / MOSTLY FALSE / UNVERIFIABLE)
2. Confidence score (0-100%)
3. Key evidence supporting your verdict
4. Any conflicting evidence
5. Important caveats or limitations
=== END CONTEXT ===""",
    }


if __name__ == "__main__":
    claim = sys.argv[1] if len(sys.argv) > 1 else "India's GDP grew 8% in 2024"
    run_test(claim)