"""
HTTP API.

    uvicorn serve.api:app --reload
    open http://127.0.0.1:8000

Replaces the old `app.py`, whose streaming was a hand-rolled queue plus a thread
pool plus `loop.call_soon_threadsafe` to push progress out of a synchronous
pipeline. None of that is needed: LangGraph's `astream(stream_mode="updates")`
already yields one event per completed node, so the SSE endpoint is a loop over
the graph and the node names *are* the step IDs.

Scope is deliberately a single-user local demo — no auth, no rate limiting, no
CORS lockdown. That is a stated non-goal in BUILD_PLAN §0, not an oversight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from core.config import ROOT, cfg
from serve.schemas import new_state

logger = logging.getLogger(__name__)

app = FastAPI(title="VerifAI", version="2.0.0")

# Uploads land here for the life of the request. Kept out of the repo tree so a
# stray file can never be mistaken for corpus data.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "verifai-uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

UI_FILE = ROOT / "ui" / "index.html"        # no-build fallback
REACT_DIST = ROOT / "ui" / "dist"           # written by `npm run build`
REACT_INDEX = REACT_DIST / "index.html"

# Accepted upload types, mirroring serve.nodes.adapt. Anything else is refused
# at the door rather than after it has been written to disk.
ALLOWED = {
    ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".mpga",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    ".pdf", ".docx", ".txt", ".md",
}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> Path:
    """Persist an upload under a generated name."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            400, f"Unsupported file type {suffix!r}. "
                 f"Allowed: {', '.join(sorted(ALLOWED))}"
        )

    # Never reuse the client-supplied name for the path — a filename is
    # attacker-controlled and `../` in it would write outside the directory.
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with open(target, "wb") as handle:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "File exceeds the 25 MB limit.")
            handle.write(chunk)
    return target


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _summarise(node: str, update: dict) -> dict:
    """
    Per-node detail for the progress stream.

    Structured rather than a prose string so the UI can render each step
    differently — the same information `serve.cli --trace` prints.
    """
    if node == "adapt":
        if update.get("input_error"):
            return {"error": update["input_error"]}
        return {
            "kind": update.get("input_kind", "text"),
            "extracted": (update.get("extracted_text") or "")[:400],
            "transcript": update.get("transcript", ""),
        }
    if node == "translate":
        return {"language": update.get("language", "en"),
                "english": update.get("claim_english", "")[:200]}
    if node == "gate":
        return {"is_claim": update.get("is_claim"),
                "confidence": round(update.get("claim_confidence", 0.0), 4),
                "reason": update.get("reject_reason", "")}
    if node in ("retrieve", "web"):
        stats = update.get("retrieval_stats", {})
        return {"evidence": len(update.get("evidence") or []),
                "candidates": stats.get("candidates", stats.get("web_results", 0)),
                "floor": stats.get("floor")}
    if node == "stance":
        return update.get("diagnostics", {}).get("stance_counts", {})
    if node == "generate":
        report = update.get("report")
        return ({"verdict": report.verdict, "confidence": report.confidence}
                if report else {})
    if node in ("abstain", "reject"):
        return {"outcome": update.get("outcome")}
    if node == "render":
        return {"chars": len(update.get("rendered", "")),
                "artifacts": list((update.get("artifacts") or {}).keys())}
    return {}


def _final_payload(state: dict, elapsed: float) -> dict:
    report = state.get("report")
    return {
        "claim": state.get("claim", ""),
        "language": state.get("language", "en"),
        "input_kind": state.get("input_kind", "text"),
        "extracted_text": state.get("extracted_text", ""),
        "transcript": state.get("transcript", ""),
        "outcome": state.get("outcome", "unknown"),
        "report": report.model_dump() if report else None,
        "rendered": state.get("rendered", ""),
        "evidence": [
            {
                "index": i,
                "url": d.metadata.get("url"),
                "domain": d.metadata.get("domain"),
                "title": d.metadata.get("title"),
                "credibility": d.metadata.get("credibility"),
                "relevance": d.metadata.get("relevance_score"),
                "origin": d.metadata.get("corpus_tier", "news"),
            }
            for i, d in enumerate(state.get("evidence") or [], start=1)
        ],
        "artifacts": {
            name: f"/download/{Path(path).name}"
            for name, path in (state.get("artifacts") or {}).items()
        },
        "diagnostics": state.get("diagnostics", {}),
        "elapsed_seconds": round(elapsed, 2),
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """
    Serve the React build if it exists, else the single-file fallback UI.

    `frontend/` builds into `ui/dist/`. That directory is only present after
    `npm run build`, so a clone with no Node toolchain still gets a working
    interface from `ui/index.html` — the app is usable without ever installing
    npm, and the React build is an upgrade rather than a requirement.
    """
    if REACT_INDEX.exists():
        return REACT_INDEX.read_text(encoding="utf-8")
    if UI_FILE.exists():
        return UI_FILE.read_text(encoding="utf-8")
    return "<h1>VerifAI</h1><p>No UI built. The API is at /verify.</p>"


@app.get("/health")
async def health() -> dict:
    """Startup check — is an index published and does it match our embedder?"""
    from serve.retriever import IndexIncompatible, IndexUnavailable, load_manifest

    try:
        manifest = load_manifest()
        status, detail = "ok", None
    except (IndexUnavailable, IndexIncompatible) as exc:
        manifest, status, detail = {}, "error", str(exc)

    return {
        "status": status,
        "detail": detail,
        "index": manifest.get("version"),
        "documents": manifest.get("documents"),
        "chunks": manifest.get("chunks"),
        "embed_model": manifest.get("embed_model"),
        "relevance_floor": cfg.relevance_floor,
        "web_search": bool(cfg.tavily_api_key) and cfg.web_search_enabled,
        "capabilities": _capabilities(),
    }


def _capabilities() -> dict:
    """What this deployment can actually accept. Drives the UI's controls."""
    from serve.nodes.adapt import ocr_unavailable

    ocr = ocr_unavailable()
    return {
        "text": True,
        "voice": bool(cfg.groq_api_key),
        "image": ocr is None,
        "image_reason": ocr,
        "document": True,
        "web_search": bool(cfg.tavily_api_key) and cfg.web_search_enabled,
    }


@app.post("/verify")
async def verify(
    claim: str = Form(default=""),
    file: UploadFile | None = File(default=None),
) -> dict:
    """Verify and return the whole result. Use /verify/stream for progress."""
    import time

    from serve.graph import get_app

    state = _build_state(claim, file)
    started = time.monotonic()
    final = await get_app().ainvoke(state)
    return _final_payload(final, time.monotonic() - started)


@app.post("/verify/stream")
async def verify_stream(
    claim: str = Form(default=""),
    file: UploadFile | None = File(default=None),
):
    """
    Server-sent events, one per completed graph node.

    This is the whole reason the graph exists as a graph: node names become step
    IDs for free, and adding a node to the pipeline adds a step to the UI with
    no plumbing on either side.
    """
    import time

    from serve.graph import get_app

    state = _build_state(claim, file)

    async def stream():
        started = time.monotonic()
        final = dict(state)
        yield _sse({"type": "start"})
        try:
            async for chunk in get_app().astream(state, stream_mode="updates"):
                for node, update in chunk.items():
                    final.update(update)
                    yield _sse({"type": "step", "step": node,
                                "detail": _summarise(node, update)})
                    # Hand control back so the event actually flushes rather
                    # than buffering until the graph completes.
                    await asyncio.sleep(0)
            yield _sse({"type": "done",
                        "result": _final_payload(final, time.monotonic() - started)})
        except Exception as exc:
            logger.exception("Verification failed")
            message = str(exc)
            if "rate_limit" in message or "429" in message:
                message = ("Groq quota reached — retrieval completed, but the "
                           "report could not be generated. Try again later.")
            yield _sse({"type": "error", "message": message})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # Without this, a proxy or the browser may buffer the whole stream and
        # the progress UI arrives all at once at the end.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _build_state(claim: str, file: UploadFile | None) -> dict:
    """
    Turn a form submission into graph state.

    Three shapes, and the distinction matters:
      claim only          -> verify the text
      file only           -> extract a claim from it, then verify
      file + claim        -> verify the claim, with the file as capped evidence
    """
    claim = (claim or "").strip()

    if file is not None and file.filename:
        path = _save_upload(file)
        return dict(new_state(str(path), claim=claim or None))

    if not claim:
        raise HTTPException(400, "Provide a claim, a file, or both.")
    return dict(new_state(claim))


@app.get("/download/{name}")
async def download(name: str):
    """Serve a generated export."""
    # Resolve and confirm containment: `name` comes from the client, and a
    # traversal would otherwise read arbitrary files. Checked against the
    # configured output directory, not the working directory.
    target = (cfg.output_dir / name).resolve()
    if not target.is_file() or cfg.output_dir.resolve() not in target.parents:
        raise HTTPException(404, "Not found")
    return FileResponse(target, filename=target.name)


@app.post("/export")
async def export_report(payload: dict) -> dict:
    """Not implemented — exports are produced during /verify via `formats`."""
    raise HTTPException(501, "Use the artifacts returned by /verify.")


# The React build's hashed JS/CSS live under /assets. Mounted only when the
# build exists so a clone without Node does not fail at import.
if REACT_DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=REACT_DIST / "assets"),
              name="assets")


# CORS, for running the Vite dev server directly against this API.
#
# Not needed for the normal flow — vite.config.js proxies /verify, /health and
# /download to this port, so the browser sees one origin and never sends a
# preflight. This is here for anyone who bypasses the proxy. Localhost only;
# opening it wider would be pointless for a single-user local tool.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _warm() -> None:
    """
    Load the models at startup rather than during the first request.

    Cold start is ~10 s across four CPU models; paying it here means the first
    verification is as fast as the tenth, which matters for a live demo.
    """
    logging.getLogger("serve").setLevel(logging.INFO)
    try:
        from serve.retriever import get_retriever

        await asyncio.to_thread(get_retriever)
        logger.info("Models and index loaded")
    except Exception as exc:
        # A missing index must not stop the server: /health then reports the
        # problem and the UI can show it, which is more useful than a crash.
        logger.error("Startup warm-up failed: %s", exc)
