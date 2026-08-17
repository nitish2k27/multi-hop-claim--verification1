"""
src/fact_verification_service.py
──────────────────────────────────
The single entry point for the entire fact verification system.
Replaces the broken enhanced_main_pipeline.py.

Key design decisions:
  1. ALL components are lazy-loaded — no startup crash, no 2-minute wait
  2. Handles all input types: text, image/screenshot, pdf, docx, audio
  3. Translates non-English input to English before NLP
  4. Adds uploaded documents to the RAG vector DB
  5. Runs full NLP + RAG pipeline locally
  6. Calls Groq API or Colab for final report generation
  7. Translates report back to user's original language
  8. For voice input: generates audio output via gTTS
  9. Exports HTML + PDF + DOCX reports

Usage:
    from src.fact_verification_service import FactVerificationService

    svc    = FactVerificationService()
    result = svc.verify(claim="India GDP grew 8%", llm_mode="groq")
    print(result["report"])
    print(result["html_path"])
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator

logger = logging.getLogger(__name__)


# ── Lazy component cache ──────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}


def _get_nlp():
    if "nlp" not in _cache:
        from src.nlp.nlp_pipeline import NLPPipeline
        _cache["nlp"] = NLPPipeline()
    return _cache["nlp"]


def _get_vector_db():
    if "vdb" not in _cache:
        from src.rag.vector_database import VectorDatabase
        _cache["vdb"] = VectorDatabase()
    return _cache["vdb"]


def _get_rag(collection_name: str = "news_articles"):
    key = f"rag_{collection_name}"
    if key not in _cache:
        from src.rag.rag_pipeline import RAGPipeline
        vdb = _get_vector_db()
        nlp = _get_nlp()
        _cache[key] = RAGPipeline(
            vector_db=vdb,
            collection_name=collection_name,
            nlp_model_manager=nlp.model_manager,
        )
    return _cache[key]


def _get_translator():
    if "translator" not in _cache:
        try:
            from src.multilingual.translator import Translator
            _cache["translator"] = Translator(backend="google")
        except Exception as e:
            logger.warning(f"Translator unavailable: {e}")
            _cache["translator"] = None
    return _cache["translator"]


def _get_doc_handler():
    if "doc_handler" not in _cache:
        from src.document_processing.document_handler import DocumentHandler
        _cache["doc_handler"] = DocumentHandler()
    return _cache["doc_handler"]


def _get_speech_handler():
    if "speech" not in _cache:
        try:
            from src.voice_processing.speech_handler import SpeechHandler
            _cache["speech"] = SpeechHandler(stt_backend="whisper", tts_backend="gtts")
        except Exception as e:
            logger.warning(f"Speech handler unavailable: {e}")
            _cache["speech"] = None
    return _cache["speech"]


# ── Language helpers ──────────────────────────────────────────────────────────

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "es": "Spanish",
    "fr": "French", "de": "German", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "pt": "Portuguese",
}


def _detect_language(text: str) -> str:
    """Fast language detection using langdetect (no FastText startup cost)."""
    if not text or len(text.strip()) < 5:
        return "en"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        pass
    # Script-based fallback
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F: return "hi"
        if 0x0600 <= cp <= 0x06FF: return "ar"
        if 0x0B80 <= cp <= 0x0BFF: return "ta"
        if 0x0C00 <= cp <= 0x0C7F: return "te"
        if 0x4E00 <= cp <= 0x9FFF: return "zh"
    return "en"


def _translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English. Returns original if translation unavailable."""
    if source_lang == "en" or source_lang == "unknown":
        return text
    translator = _get_translator()
    if translator is None:
        logger.warning("Translator not available — using original text")
        return text
    try:
        return translator.to_english(text, source_lang)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


def _translate_from_english(text: str, target_lang: str) -> str:
    """Translate English text back to target language."""
    if target_lang == "en" or target_lang == "unknown":
        return text
    translator = _get_translator()
    if translator is None:
        return text
    try:
        return translator.from_english(text, target_lang)
    except Exception as e:
        logger.warning(f"Back-translation failed: {e}")
        return text


# ── Input extraction ──────────────────────────────────────────────────────────

def _extract_text_from_image(file_path: str) -> str:
    """OCR on image/screenshot using easyocr. Falls back to pytesseract."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(file_path, detail=0)
        return " ".join(results).strip()
    except ImportError:
        pass
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(file_path)
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        raise ImportError(
            "No OCR library available.\n"
            "Install: pip install easyocr   OR   pip install pytesseract pillow"
        )


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF. Tries PyPDF2, falls back to pdfplumber."""
    try:
        import PyPDF2
        pages = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t: pages.append(t)
        return "\n\n".join(pages).strip()
    except ImportError:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
    except ImportError:
        raise ImportError("pip install PyPDF2   OR   pip install pdfplumber")


def _extract_text_from_docx(file_path: str) -> str:
    """Extract text from Word document."""
    try:
        import docx
        doc = docx.Document(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except ImportError:
        raise ImportError("pip install python-docx")


def _transcribe_audio(file_path: str) -> Dict[str, str]:
    """
    Transcribe audio using Whisper.
    Returns dict with 'text' and 'language'.
    """
    speech = _get_speech_handler()
    if speech is None:
        raise RuntimeError(
            "Speech handler not available.\n"
            "Install: pip install openai-whisper"
        )
    result = speech.speech_to_text(str(file_path))
    return {
        "text":     result.get("text", ""),
        "language": result.get("language", "en"),
    }


def _generate_audio_output(text: str, language: str, output_dir: str = "data/audio") -> Optional[str]:
    """Generate audio file from text using gTTS. Returns path or None."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        from gtts import gTTS
        ts   = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(Path(output_dir) / f"response_{ts}.mp3")
        # gTTS needs a 2-char lang code; map common ones
        lang_map = {"zh": "zh-CN", "pa": "pa"}
        gtts_lang = lang_map.get(language, language)
        tts = gTTS(text=text[:3000], lang=gtts_lang, slow=False)
        tts.save(path)
        logger.info(f"Audio output saved: {path}")
        return path
    except Exception as e:
        logger.warning(f"Audio generation failed: {e}")
        return None


# ── Core pipeline ─────────────────────────────────────────────────────────────

class FactVerificationService:
    """
    Unified entry point for the complete fact verification pipeline.

    All components lazy-load on first use — no startup delay.
    """

    def verify(
        self,
        # ── Input ──────────────────────────────────────────────────────────
        claim:          Optional[str]  = None,
        file_path:      Optional[str]  = None,
        input_type:     str            = "text",
        # ── Options ────────────────────────────────────────────────────────
        llm_mode:       str            = "groq",
        groq_api_key:   Optional[str]  = None,
        colab_url:      Optional[str]  = None,
        top_k:          int            = 5,
        generate_audio: bool           = False,
        output_dir:     str            = "data/reports",
    ) -> Dict[str, Any]:
        """
        Run the full verification pipeline.

        input_type:
            "text"        — claim is a plain text string
            "image"       — file_path is a screenshot/image, OCR extracts claim
            "pdf"         — file_path is a PDF, text extracted and added to RAG
            "docx"        — file_path is a Word doc, extracted and added to RAG
            "audio"       — file_path is an audio file, Whisper transcribes
            "pdf_claim"   — claim is text, file_path is PDF used as evidence context
            "docx_claim"  — claim is text, file_path is DOCX used as evidence context

        llm_mode: "groq" or "colab"

        Returns dict with keys:
            claim, claim_english, user_language, verdict, confidence,
            report, report_translated, html_path, pdf_path, docx_path,
            audio_path (if generate_audio=True), rag_result, nlp_result,
            steps (list of step descriptions for UI display)
        """

        steps  = []
        result = {
            "claim": claim, "claim_english": None, "user_language": "en",
            "verdict": "UNVERIFIABLE", "confidence": 0.0,
            "report": "", "report_translated": "",
            "html_path": None, "pdf_path": None, "docx_path": None,
            "audio_path": None, "rag_result": None, "nlp_result": None,
            "steps": steps, "error": None,
        }

        try:
            # ── Step 1: Extract text from input ──────────────────────────────
            steps.append("Step 1/7  Extracting text from input...")
            user_context_docs = []

            if input_type == "text":
                if not claim:
                    raise ValueError("No claim provided")
                raw_text    = claim
                user_lang   = _detect_language(raw_text)

            elif input_type == "image":
                if not file_path:
                    raise ValueError("No image file provided")
                steps[-1] = "Step 1/7  Running OCR on image..."
                raw_text  = _extract_text_from_image(file_path)
                user_lang = _detect_language(raw_text)
                claim     = raw_text

            elif input_type == "audio":
                if not file_path:
                    raise ValueError("No audio file provided")
                steps[-1] = "Step 1/7  Transcribing audio with Whisper..."
                transcription = _transcribe_audio(file_path)
                raw_text      = transcription["text"]
                user_lang     = transcription["language"]
                claim         = raw_text
                generate_audio = True   # voice in → audio out

            elif input_type == "pdf":
                if not file_path:
                    raise ValueError("No PDF file provided")
                steps[-1] = "Step 1/7  Extracting text from PDF..."
                raw_text  = _extract_text_from_pdf(file_path)
                user_lang = _detect_language(raw_text)
                claim     = raw_text[:500]  # first 500 chars as the claim summary

            elif input_type == "docx":
                if not file_path:
                    raise ValueError("No DOCX file provided")
                steps[-1] = "Step 1/7  Extracting text from Word document..."
                raw_text  = _extract_text_from_docx(file_path)
                user_lang = _detect_language(raw_text)
                claim     = raw_text[:500]

            elif input_type == "pdf_claim":
                # Claim is the text, PDF is the evidence context
                if not claim:
                    raise ValueError("No claim provided")
                if not file_path:
                    raise ValueError("No PDF context file provided")
                steps[-1] = "Step 1/7  Loading claim + PDF evidence context..."
                raw_text  = claim
                user_lang = _detect_language(raw_text)
                # Add PDF to RAG as priority context
                pdf_text  = _extract_text_from_pdf(file_path)
                pdf_chunks = [{"text": pdf_text[i:i+500], "chunk_id": i//500}
                              for i in range(0, len(pdf_text), 500)]
                user_context_docs = [{"text": c["text"],
                                       "metadata": {"document_name": Path(file_path).name}}
                                      for c in pdf_chunks]

            elif input_type == "docx_claim":
                if not claim:
                    raise ValueError("No claim provided")
                if not file_path:
                    raise ValueError("No DOCX context file provided")
                steps[-1] = "Step 1/7  Loading claim + DOCX evidence context..."
                raw_text  = claim
                user_lang = _detect_language(raw_text)
                docx_text = _extract_text_from_docx(file_path)
                docx_chunks = [{"text": docx_text[i:i+500], "chunk_id": i//500}
                               for i in range(0, len(docx_text), 500)]
                user_context_docs = [{"text": c["text"],
                                       "metadata": {"document_name": Path(file_path).name}}
                                      for c in docx_chunks]
            else:
                raise ValueError(f"Unknown input_type: {input_type}")

            result["user_language"] = user_lang
            result["claim"]         = claim
            lang_name = LANG_NAMES.get(user_lang, user_lang.upper())
            steps[-1] = f"Step 1/7  Input extracted — language detected: {lang_name}"

            # ── Step 2: Upload document to RAG (if file provided) ─────────────
            if file_path and input_type in ("pdf", "docx"):
                steps.append("Step 2/7  Adding document to knowledge base (RAG)...")
                try:
                    handler = _get_doc_handler()
                    vdb     = _get_vector_db()
                    doc_data = handler.process_upload(file_path)
                    handler.add_to_rag(doc_data, vdb, "uploaded_documents")
                    steps[-1] = (f"Step 2/7  Document added to RAG: "
                                 f"{len(doc_data['chunks'])} chunks indexed")
                except Exception as e:
                    steps[-1] = f"Step 2/7  Document indexing skipped: {e}"
            else:
                steps.append("Step 2/7  No document upload required")

            # ── Step 3: Translate to English ──────────────────────────────────
            steps.append("Step 3/7  Translating to English for NLP...")
            claim_en = _translate_to_english(raw_text, user_lang)
            result["claim_english"] = claim_en
            if user_lang == "en":
                steps[-1] = "Step 3/7  Already English — no translation needed"
            else:
                steps[-1] = (f"Step 3/7  Translated from {lang_name} → English: "
                             f'"{claim_en[:80]}..."')

            # ── Step 4: NLP pipeline ──────────────────────────────────────────
            steps.append("Step 4/7  Running NLP pipeline (BERT, NER, temporal)...")
            nlp        = _get_nlp()
            nlp_result = nlp.analyze(claim_en)
            result["nlp_result"] = nlp_result

            analysis  = nlp_result["analysis"]
            cd        = analysis["claim_detection"]
            ents      = analysis["entities"]
            temporal  = analysis.get("temporal", {}).get("dates", [])
            ent_str   = ", ".join(
                f"{k}:{len(v)}" for k, v in ents["entities"].items() if v
            ) or "none"
            steps[-1] = (
                f"Step 4/7  NLP done — "
                f"claim={'YES' if cd['is_claim'] else 'NO'} "
                f"({cd['confidence']:.2f}), "
                f"entities: {ent_str}, "
                f"temporal: {len(temporal)}"
            )

            # ── Step 5: RAG retrieval + evidence scoring ───────────────────────
            steps.append("Step 5/7  Searching knowledge base (RAG + credibility)...")
            rag = _get_rag("news_articles")

            # Pass user context docs if we have them (pdf_claim / docx_claim)
            rag_result = rag.verify_claim(
                claim=claim_en,
                top_k=top_k,
                user_context_docs=user_context_docs if user_context_docs else None,
            )
            result["rag_result"] = rag_result

            ev_count = len(rag_result.get("evidence", []))
            verdict  = rag_result.get("verdict", "UNVERIFIABLE")
            conf     = rag_result.get("confidence", 0.0)
            ctx_len  = len(rag_result.get("llm_context", ""))
            steps[-1] = (
                f"Step 5/7  RAG done — {ev_count} evidence pieces, "
                f"preliminary verdict: {verdict} ({conf:.1f}%), "
                f"LLM context: {ctx_len} chars"
            )
            result["verdict"]    = verdict
            result["confidence"] = conf

            # ── Step 6: LLM generation ─────────────────────────────────────────
            steps.append(f"Step 6/7  Generating report via {llm_mode.upper()}...")
            report = _generate_report(rag_result, llm_mode, user_lang,
                                       groq_api_key, colab_url)
            result["report"] = report

            # Translate report back to user language if needed
            if user_lang != "en":
                steps[-1] = (f"Step 6/7  Report generated — "
                             f"translating back to {lang_name}...")
                report_translated = _translate_from_english(report, user_lang)
                result["report_translated"] = report_translated
                steps[-1] = (f"Step 6/7  Report generated and translated "
                             f"to {lang_name} — {len(report.split())} words")
            else:
                result["report_translated"] = report
                steps[-1] = f"Step 6/7  Report generated — {len(report.split())} words"

            # ── Step 7: Export files ───────────────────────────────────────────
            steps.append("Step 7/7  Exporting HTML, PDF, DOCX reports...")
            try:
                from src.generation.report_exporter import ReportExporter
                exporter = ReportExporter(output_dir=output_dir)
                # Sanitise claim for use as filename
                safe_claim = claim[:80].replace("\n", " ").replace("\r", " ").strip()
                exports  = exporter.export_all(report, claim=safe_claim, rag_result=rag_result)
            except Exception as export_err:
                logger.warning(f"Full export failed ({export_err}), falling back to HTML only")
                # Minimal HTML fallback — always works, no system libs required
                try:
                    from src.generation.report_exporter import ReportExporter
                    exporter = ReportExporter(output_dir=output_dir)
                    safe_claim = claim[:80].replace("\n", " ").replace("\r", " ").strip()
                    html_path = exporter.to_html(report, safe_claim, rag_result)
                    exports = {"html": html_path,
                               "pdf":  f"SKIPPED: {export_err}",
                               "docx": f"SKIPPED: {export_err}"}
                except Exception as html_err:
                    logger.error(f"HTML export also failed: {html_err}")
                    exports = {"html": None,
                               "pdf":  f"SKIPPED: {export_err}",
                               "docx": f"SKIPPED: {export_err}"}

            result["html_path"] = exports.get("html")
            result["pdf_path"]  = exports.get("pdf")
            result["docx_path"] = exports.get("docx")
            steps[-1] = (
                f"Step 7/7  Reports saved: "
                f"HTML={bool(result['html_path'])}, "
                f"PDF={not str(exports.get('pdf', '')).startswith('SKIPPED')}, "
                f"DOCX={not str(exports.get('docx', '')).startswith('SKIPPED')}"
            )

            # ── Optional: audio output ────────────────────────────────────────
            if generate_audio:
                steps.append("Generating audio response via gTTS...")
                # Use translated report excerpt for audio
                audio_text = (result["report_translated"] or report)[:800]
                audio_path = _generate_audio_output(audio_text, user_lang)
                result["audio_path"] = audio_path
                steps[-1] = (f"Audio output: "
                             f"{'saved to ' + audio_path if audio_path else 'failed'}")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result["error"] = str(e)
            steps.append(f"[ERROR] {e}")

        return result

    # ── Streaming version (for UI live display) ───────────────────────────────

    def verify_streaming(self, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Same as verify() but yields intermediate results after each step,
        so the UI can update progressively (like Claude's thinking display).

        Usage in UI:
            for partial in svc.verify_streaming(claim="...", llm_mode="groq"):
                update_ui(partial["steps"][-1], partial.get("report"))
        """
        # We run the full pipeline but yield a snapshot after each major step.
        # Since Python generators can't easily pause mid-function, we implement
        # this by running in a thread and yielding from a queue.
        import threading, queue

        q = queue.Queue()
        result_holder = [None]

        def run():
            res = self.verify(**kwargs)
            result_holder[0] = res
            q.put(None)  # sentinel

        t = threading.Thread(target=run, daemon=True)
        t.start()

        # Poll the queue every 0.5s and yield the current result snapshot
        import time
        while t.is_alive() or not q.empty():
            try:
                sentinel = q.get(timeout=0.5)
                if sentinel is None:
                    break
            except Exception:
                pass
            if result_holder[0]:
                yield dict(result_holder[0])
            else:
                yield {"steps": ["Processing..."], "report": "", "error": None}

        if result_holder[0]:
            yield dict(result_holder[0])


# ── LLM dispatch ─────────────────────────────────────────────────────────────

def _generate_report(
    rag_result: Dict[str, Any],
    llm_mode: str,
    user_language: str,
    groq_api_key: Optional[str] = None,
    colab_url: Optional[str] = None,
) -> str:
    if llm_mode == "groq":
        from src.generation.report_generator_groq import ReportGeneratorGroq
        gen = ReportGeneratorGroq(api_key=groq_api_key)
        return gen.generate(rag_result, user_language=user_language)

    elif llm_mode == "colab":
        from src.generation.report_generator import ReportGenerator
        gen = ReportGenerator(inference_url=colab_url)
        return gen.generate(rag_result, user_language=user_language)

    else:
        raise ValueError(f"Unknown llm_mode: {llm_mode}. Use 'groq' or 'colab'.")