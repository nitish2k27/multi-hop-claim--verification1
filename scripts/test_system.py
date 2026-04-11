"""
scripts/test_system.py
──────────────────────
Complete system test. Run this after placing all the new files.
Tests every component in isolation first, then the full pipeline.

Run from project root:
    python scripts/test_system.py

Expected result: all checks pass, full pipeline produces a report.
"""

import sys
import os
import logging
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

OK   = "[OK]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check(label, fn):
    try:
        result = fn()
        print(f"  {OK}   {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  {FAIL} {label} — {e}")
        return False


print("\n" + "="*60)
print("SYSTEM READINESS TEST")
print("="*60)

# ── 1. Critical imports ────────────────────────────────────────────────────────
print("\n[1] Critical imports")

check("src.rag import (must be from rag_pipeline)",
      lambda: __import__("src.rag", fromlist=["RAGPipeline"]).RAGPipeline.__module__)

check("src.generation import",
      lambda: str(__import__("src.generation", fromlist=["ReportExporter"])))

check("src.document_processing import",
      lambda: str(__import__("src.document_processing.document_handler",
                              fromlist=["DocumentHandler"])))

check("src.fact_verification_service import",
      lambda: str(__import__("src.fact_verification_service",
                              fromlist=["FactVerificationService"])))

check("src.enhanced_main_pipeline import (backward compat)",
      lambda: str(__import__("src.enhanced_main_pipeline",
                              fromlist=["EnhancedFactVerificationPipeline"])))

check("src.preprocessing.input_processor import (must be instant)",
      lambda: str(__import__("src.preprocessing.input_processor",
                              fromlist=["InputProcessor"])))

# ── 2. RAG __init__ fix ────────────────────────────────────────────────────────
print("\n[2] RAG __init__ imports correct class")

def check_rag_class():
    from src.rag import RAGPipeline
    assert hasattr(RAGPipeline, "verify_claim"), \
        "RAGPipeline has no verify_claim — still importing from retrieval.py!"
    return "verify_claim() present"
check("RAGPipeline.verify_claim exists", check_rag_class)

# ── 3. InputProcessor is lazy (no startup cost) ───────────────────────────────
print("\n[3] InputProcessor lazy loading")

def check_input_processor_lazy():
    import time
    from src.preprocessing.input_processor import InputProcessor
    t0 = time.time()
    ip = InputProcessor()
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"InputProcessor took {elapsed:.1f}s — still eager loading!"
    return f"loaded in {elapsed:.2f}s"
check("InputProcessor instantiates in <5s", check_input_processor_lazy)

# ── 4. DocumentHandler uses correct method ────────────────────────────────────
print("\n[4] DocumentHandler.add_to_rag uses add_documents")

def check_doc_handler():
    import inspect
    from src.document_processing.document_handler import DocumentHandler
    src = inspect.getsource(DocumentHandler.add_to_rag)
    assert "add_texts" not in src, "Still using add_texts (broken)"
    assert "add_documents" in src, "add_documents not found"
    return "add_documents() confirmed"
check("add_to_rag uses add_documents()", check_doc_handler)

# ── 5. Generation module ──────────────────────────────────────────────────────
print("\n[5] Generation module")

def check_groq_model():
    from src.generation.report_generator_groq import DEFAULT_MODEL
    assert "3.3" in DEFAULT_MODEL or "3-3" in DEFAULT_MODEL.replace(".", "-"), \
        f"Wrong model: {DEFAULT_MODEL} — should be llama-3.3-70b-versatile"
    return DEFAULT_MODEL
check("Groq model is llama-3.3-70b-versatile", check_groq_model)

check("prompt_builder importable",
      lambda: str(__import__("src.generation.prompt_builder",
                              fromlist=["build_groq_messages"])))

def check_prompt_multilingual():
    from src.generation.prompt_builder import build_groq_messages
    msgs = build_groq_messages("test context", user_language="hi")
    assert any("Hindi" in m["content"] or "हिंदी" in m["content"] for m in msgs), \
        "Hindi instruction not found in prompt"
    return "Hindi instruction injected"
check("prompt_builder injects language instruction", check_prompt_multilingual)

# ── 6. enhanced_main_pipeline voice path ──────────────────────────────────────
print("\n[6] enhanced_main_pipeline voice path fixed")

def check_enhanced_voice():
    import inspect
    from src.enhanced_main_pipeline import EnhancedFactVerificationPipeline
    src = inspect.getsource(EnhancedFactVerificationPipeline)
    assert "voice_processing" not in src, \
        "Still has src.voice_processing — wrong path"
    return "No voice_processing reference"
check("enhanced_main_pipeline has no voice_processing", check_enhanced_voice)

# ── 7. ChromaDB + RAG available ───────────────────────────────────────────────
print("\n[7] Vector database")

def check_chromadb():
    from src.rag.vector_database import VectorDatabase
    vdb   = VectorDatabase()
    stats = vdb.get_collection_stats("news_articles")
    if not stats["exists"]:
        return "collection not found (run ingest_to_rag.py)"
    return f"{stats['count']:,} articles indexed"
check("ChromaDB news_articles collection", check_chromadb)

# ── 8. NLP pipeline quick load ────────────────────────────────────────────────
print("\n[8] NLP pipeline (takes 30-60s first run)")
print("  [..] Loading NLP pipeline — please wait...")

def check_nlp():
    from src.nlp.nlp_pipeline import NLPPipeline
    nlp    = NLPPipeline()
    result = nlp.analyze("India's GDP grew 8% in 2024")
    cd     = result["analysis"]["claim_detection"]
    ents   = result["analysis"]["entities"]["total_entities"]
    return f"claim={cd['is_claim']} conf={cd['confidence']:.2f} entities={ents}"
check("NLPPipeline.analyze runs", check_nlp)

# ── 9. Groq API key present ───────────────────────────────────────────────────
print("\n[9] Groq API key")

def check_groq_key():
    key_file = Path("configs/groq_token.txt")
    if not key_file.exists():
        return "configs/groq_token.txt missing — run: python scripts/setup_groq.py KEY"
    key = key_file.read_text().strip()
    if not key:
        return "groq_token.txt is empty"
    return f"key present: {key[:8]}..."
check("Groq key in configs/groq_token.txt", check_groq_key)

# ── 10. Full pipeline end-to-end ──────────────────────────────────────────────
print("\n[10] Full pipeline end-to-end (Groq)")
print("  [..] Running complete pipeline — please wait...")

def check_full_pipeline():
    from src.fact_verification_service import FactVerificationService
    svc    = FactVerificationService()
    result = svc.verify(
        claim    = "India's GDP grew 8% in 2024",
        llm_mode = "groq",
    )
    if result.get("error"):
        raise RuntimeError(result["error"])
    verdict = result["verdict"]
    conf    = result["confidence"]
    words   = len(result["report"].split()) if result["report"] else 0
    html    = result["html_path"]
    assert words > 50, f"Report too short: {words} words"
    return f"verdict={verdict} conf={conf:.1f}% words={words} html={bool(html)}"

check("FactVerificationService.verify() end-to-end", check_full_pipeline)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("All checks done. Fix any [FAIL] items before running the UI.")
print("="*60 + "\n")