"""
app.py
──────
FastAPI backend for the VerifAI fact verification UI.

Solves the Streamlit timeout problem:
  - /verify/stream   → SSE endpoint, pushes each pipeline step live
  - /upload          → file upload, returns temp path
  - /download        → serve HTML/DOCX reports
  - /health          → health check for UI demo mode detection

Run:
    pip install fastapi uvicorn python-multipart
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then open ui/index.html in your browser.
"""

import os
import json
import asyncio
import tempfile
import traceback
from pathlib import Path
from typing import Optional, AsyncGenerator

os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="VerifAI", version="1.0")

# Allow browser requests from file:// or any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI
UI_DIR = Path(__file__).parent / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

UPLOAD_DIR = Path("data/uploads_temp")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Input type detection ───────────────────────────────────────
EXT_TO_TYPE = {
    ".pdf": "pdf", ".docx": "docx", ".doc": "docx",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".webm": "audio",
}


def detect_input_type(filename: str, has_claim: bool) -> str:
    ext = Path(filename).suffix.lower()
    base = EXT_TO_TYPE.get(ext, "text")
    if has_claim and base in ("pdf", "docx"):
        return base + "_claim"   # claim + doc as evidence context
    return base


# ── SSE helpers ────────────────────────────────────────────────
def _json_default(obj):
    """Handle numpy/torch scalar types that json.dumps can't serialize."""
    import numpy as np
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):      return int(obj)
    if isinstance(obj, np.ndarray):                             return obj.tolist()
    if isinstance(obj, np.bool_):                               return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def sse(event_dict: dict) -> str:
    return f"data: {json.dumps(event_dict, default=_json_default)}\n\n"


def step_start(step_id: str) -> str:
    return sse({"type": "step_start", "step_id": step_id})


def step_done(step_id: str, detail: str = "") -> str:
    return sse({"type": "step_done", "step_id": step_id, "detail": detail})


def step_error(step_id: str, detail: str = "") -> str:
    return sse({"type": "step_error", "step_id": step_id, "detail": detail})


# ── Health ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "VerifAI"}


# ── File upload ────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), claim: str = Form("")):
    suffix = Path(file.filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=str(UPLOAD_DIR)
    )
    content = await file.read()
    tmp.write(content)
    tmp.close()

    input_type = detect_input_type(file.filename, bool(claim.strip()))
    return {"path": tmp.name, "input_type": input_type, "filename": file.filename}


# ── Streaming verify endpoint ──────────────────────────────────
class VerifyRequest(BaseModel):
    claim:      Optional[str] = ""
    llm_mode:   str = "groq"
    file_path:  Optional[str] = None
    input_type: Optional[str] = "text"
    colab_url:  Optional[str] = None


@app.post("/verify/stream")
async def verify_stream(req: VerifyRequest):
    return StreamingResponse(
        _run_pipeline(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":       "no-cache",
            "X-Accel-Buffering":   "no",
            "Connection":          "keep-alive",
        },
    )


async def _run_pipeline(req: VerifyRequest) -> AsyncGenerator[str, None]:
    """
    Runs the full pipeline in a thread and streams SSE events.
    Uses asyncio.run_in_executor so the blocking NLP/RAG work
    doesn't block the event loop — the SSE connection stays alive
    for up to 150 seconds (2.5 min).
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run_sync():
        """Blocking pipeline — runs in a thread pool."""
        try:
            _pipeline_with_callbacks(req, queue, loop)
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                sse({"type": "error", "message": str(e)})
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    asyncio.ensure_future(loop.run_in_executor(None, run_sync))

    # Yield SSE events as they arrive
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=300.0)
        except asyncio.TimeoutError:
            yield sse({"type": "error", "message": "Pipeline timeout (150s)"})
            break
        if item is None:
            break
        yield item


def emit(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, event: str):
    """Thread-safe SSE emit."""
    loop.call_soon_threadsafe(queue.put_nowait, event)


def _pipeline_with_callbacks(req: VerifyRequest, queue, loop):
    """
    Full pipeline with SSE callbacks after each step.
    Each step emits step_start → (work) → step_done/step_error.
    NLP and RAG intermediate results are emitted as separate events.
    """
    e = lambda ev: emit(queue, loop, ev)
    claim = req.claim or ""
    file_path = req.file_path
    input_type = req.input_type or "text"
    llm_mode = req.llm_mode
    colab_url = req.colab_url
    whisper_lang = None   # set when Whisper transcribes audio

    # ── Step 1: Input ─────────────────────────────────────────
    e(step_start("input"))
    try:
        if file_path:
            # Detect from actual file extension if not set
            if input_type == "text":
                input_type = detect_input_type(file_path, bool(claim.strip()))

            # ── Extract text from non-text file types ──────────
            if input_type == "image":
                e(sse({"type": "step_progress", "step_id": "input",
                       "detail": "Running OCR on image..."}))
                from src.fact_verification_service import _extract_text_from_image
                claim = _extract_text_from_image(file_path)

            elif input_type == "audio":
                e(sse({"type": "step_progress", "step_id": "input",
                       "detail": "Transcribing audio with Whisper (may take ~30s)..."}))
                from src.fact_verification_service import _transcribe_audio
                tr = _transcribe_audio(file_path)
                claim = tr["text"]
                whisper_lang = tr.get("language", None)

            elif input_type == "pdf" and not claim.strip():
                e(sse({"type": "step_progress", "step_id": "input",
                       "detail": "Extracting text from PDF..."}))
                from src.fact_verification_service import _extract_text_from_pdf
                claim = _extract_text_from_pdf(file_path)[:600]

            elif input_type == "docx" and not claim.strip():
                e(sse({"type": "step_progress", "step_id": "input",
                       "detail": "Extracting text from DOCX..."}))
                from src.fact_verification_service import _extract_text_from_docx
                claim = _extract_text_from_docx(file_path)[:600]

        if not claim and not file_path:
            raise ValueError("No claim or file provided")
        if not claim:
            raise ValueError(f"Could not extract text from {input_type} file")
        e(step_done("input", f"type={input_type} · {len(claim)} chars"))
    except Exception as ex:
        e(step_error("input", str(ex)))
        return

    # ── Step 2: Translation check ─────────────────────────────
    e(step_start("translate"))
    claim_english = claim or ""
    lang = "en"
    try:
        from src.fact_verification_service import _detect_language, _translate_to_english, LANG_NAMES
        text_for_lang = claim or ""
        # Whisper already detected language for audio — reuse it; else detect
        lang = whisper_lang if whisper_lang else (
            _detect_language(text_for_lang) if text_for_lang else "en"
        )
        lang_name = LANG_NAMES.get(lang, lang.upper())
        if lang != "en" and claim:
            claim_english = _translate_to_english(claim, lang)
            e(step_done("translate", f"{lang_name} → English: \"{claim_english[:60]}\""))
        else:
            e(step_done("translate", f"detected: {lang_name} (no translation needed)"))
    except Exception as ex:
        e(step_done("translate", "language detection skipped"))
        lang = "en"

    # ── Step 3: NLP (always on English text) ──────────────────
    e(step_start("nlp"))
    nlp_result = None
    nlp = None
    try:
        from src.nlp.nlp_pipeline import NLPPipeline
        nlp = NLPPipeline()
        claim_for_nlp = claim_english  # always English → accurate NER + temporal
        nlp_result = nlp.analyze(claim_for_nlp)
        analysis = nlp_result.get("analysis", {})
        cd = analysis.get("claim_detection", {})
        ents = analysis.get("entities", {})
        total = ents.get("total_entities", 0)
        conf = cd.get("confidence", 0)
        e(step_done("nlp", f"claim={cd.get('is_claim')} conf={conf:.2f} entities={total}"))
        # Emit NLP result to UI
        e(sse({"type": "nlp_result", "payload": nlp_result}))
    except Exception as ex:
        e(step_error("nlp", str(ex)[:120]))
        logger.error(traceback.format_exc())
        return

    # ── Step 4: RAG retrieval ─────────────────────────────────
    e(step_start("rag"))
    rag_result = None
    try:
        from src.rag.vector_database import VectorDatabase
        from src.rag.rag_pipeline import RAGPipeline
        vdb = VectorDatabase()
        rag = RAGPipeline(
            vector_db=vdb,
            collection_name="news_articles",
            nlp_model_manager=nlp.model_manager if nlp else None,
        )

        # Handle user document context — for both evidence-context and standalone doc
        user_context_docs = None
        if file_path and input_type in ("pdf_claim", "docx_claim", "pdf", "docx"):
            try:
                if "pdf" in input_type:
                    from src.fact_verification_service import _extract_text_from_pdf
                    doc_text = _extract_text_from_pdf(file_path)
                else:
                    from src.fact_verification_service import _extract_text_from_docx
                    doc_text = _extract_text_from_docx(file_path)
                user_context_docs = [
                    {"text": doc_text[i:i+500],
                     "metadata": {"document_name": Path(file_path).name}}
                    for i in range(0, len(doc_text), 500)
                ]
            except Exception as ex:
                logger.warning(f"Doc extraction for context failed: {ex}")

        rag_result = rag.verify_claim(
            claim=claim_english,  # English claim → better vector search
            top_k=5,
            user_context_docs=user_context_docs,
        )
        ev_count = len(rag_result.get("evidence", []))
        verdict = rag_result.get("verdict", "UNVERIFIABLE")
        conf = rag_result.get("confidence", 0)
        e(step_done("rag", f"{ev_count} evidence · {verdict} ({conf:.1f}%)"))
        e(sse({"type": "rag_result", "payload": rag_result}))
    except Exception as ex:
        e(step_error("rag", str(ex)[:120]))
        logger.error(traceback.format_exc())
        return

    # ── Step 5: Credibility scoring (already done inside RAG) ─
    e(step_start("credibility"))
    try:
        agg = rag_result.get("aggregation", {})
        sup = agg.get("support_percentage", 0)
        ref = agg.get("refute_percentage", 0)
        e(step_done("credibility", f"support={sup:.1f}% refute={ref:.1f}%"))
    except Exception:
        e(step_done("credibility", "scoring complete"))

    # ── Step 6: LLM generation ────────────────────────────────
    e(step_start("generation"))
    report = ""
    try:
        if llm_mode == "groq":
            from src.generation.report_generator_groq import ReportGeneratorGroq
            gen = ReportGeneratorGroq()
            report = gen.generate(rag_result, user_language=lang)
        elif llm_mode == "colab" and colab_url:
            from src.generation.report_generator import ReportGenerator
            gen = ReportGenerator(inference_url=colab_url)
            report = gen.generate(rag_result, user_language=lang)
        else:
            raise ValueError(f"Unknown llm_mode or missing colab_url: {llm_mode}")
        e(step_done("generation", f"{len(report.split())} words generated"))
    except Exception as ex:
        e(step_error("generation", str(ex)[:120]))
        logger.error(traceback.format_exc())
        return

    # ── Step 7: Export ────────────────────────────────────────
    e(step_start("export"))
    html_path = None
    docx_path = None
    try:
        from src.generation.report_exporter import ReportExporter
        exporter = ReportExporter(output_dir="data/reports")
        safe_claim = (claim or "report")[:80].replace("\n", " ").replace("\r", " ").strip()
        exports = {}
        try:
            exports = exporter.export_all(report, claim=safe_claim, rag_result=rag_result)
        except Exception as exp_err:
            logger.warning(f"Full export failed: {exp_err}, trying HTML only")
            try:
                html_path = exporter.to_html(report, safe_claim, rag_result)
                exports = {"html": html_path, "pdf": None, "docx": None}
            except Exception:
                pass

        html_path = exports.get("html")
        docx_path = exports.get("docx") if not str(exports.get("docx","")).startswith("SKIPPED") else None
        e(step_done("export", f"html={bool(html_path)} docx={bool(docx_path)}"))
    except Exception as ex:
        e(step_done("export", f"export skipped: {str(ex)[:60]}"))

    # ── Step 8 (optional): Audio output for voice input ───────
    audio_path = None
    if input_type == "audio":
        e(step_start("audio_out"))
        try:
            from src.fact_verification_service import _generate_audio_output
            audio_text = (report or "")[:800]
            audio_path = _generate_audio_output(audio_text, lang)
            e(step_done("audio_out", f"saved: {audio_path}"))
        except Exception as ex:
            e(step_done("audio_out", f"skipped: {str(ex)[:60]}"))

    # ── Emit final report event ───────────────────────────────
    e(sse({
        "type":       "report",
        "payload": {
            "claim":         claim,
            "verdict":       rag_result.get("verdict", "UNVERIFIABLE"),
            "confidence":    rag_result.get("confidence", 0.0),
            "report":        report,
            "html_path":     str(html_path) if html_path else None,
            "docx_path":     str(docx_path) if docx_path else None,
            "audio_path":    str(audio_path) if audio_path else None,
            "user_language": lang,
        }
    }))


# ── Download endpoint ──────────────────────────────────────────
@app.get("/download")
async def download_file(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File not found")
    # Security: only allow files in our data/ directory
    try:
        p.resolve().relative_to(Path("data").resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    media_map = {
        ".html": "text/html",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".mp3":  "audio/mpeg",
        ".wav":  "audio/wav",
    }
    media_type = media_map.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), media_type=media_type, filename=p.name)


# ── Static UI fallback ─────────────────────────────────────────
@app.get("/")
async def root():
    idx = Path("ui/index.html")
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"status": "VerifAI API running", "docs": "/docs"})


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*56)
    print("  VerifAI — Fact Verification System")
    print("  UI:  open ui/index.html in your browser")
    print("  API: http://127.0.0.1:8000/docs")
    print("="*56 + "\n")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # reload=True causes mid-request restarts on file changes
        log_level="warning",
        timeout_keep_alive=160, # keep SSE connection alive for 2.5 min
    )