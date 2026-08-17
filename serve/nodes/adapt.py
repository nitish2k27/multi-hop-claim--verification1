"""
Input adaptation — turn whatever the user gave us into a claim string.

Four input types, one output contract: `claim` (text to verify), plus whatever
provenance the later nodes need.

    text        typed or pasted                  -> used as-is
    voice       mp3/wav/m4a/ogg/webm             -> Groq whisper-large-v3
    image       png/jpg screenshot               -> pytesseract OCR
    document    pdf/docx                         -> PyPDF2 / python-docx

URL input was in the original plan and is deliberately **not** built: a URL is
just a slower way to get text, and every URL a user would paste is either an
article (in which case the claim is ambiguous) or a social post (screenshot it).

THE IMAGE CASE IS THE INTERESTING ONE
-------------------------------------
The workflow it serves: see a claim in an Instagram or WhatsApp post, screenshot
it, paste it in. OCR gives us the sentence, and then the *claim gate* answers the
question the user actually has — "is this even a checkable factual claim, or is
it opinion / a joke / engagement bait?" That gate already exists; images just
make its answer worth surfacing on its own.

OCR degrades rather than failing. `pytesseract` is a thin wrapper around the
Tesseract **system binary**, which is not a pip install. If it is absent, image
input returns an actionable message and every other input type keeps working —
same pattern as the Tavily fallback in M4.

DOCUMENTS CAN ARRIVE TWO WAYS
-----------------------------
    document + claim    the document becomes EVIDENCE for that claim
    document alone      a claim is extracted from the document

The first case is the one with a safety property attached: an uploaded document
must not be able to make its own assertion true. It enters evidence capped at
0.6 credibility and labelled an unverified submission — see `_as_upload_evidence`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from core.config import cfg
from serve.schemas import VerifyState

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".mpga"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DOC_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

# Groq's audio endpoint rejects files above 25 MB on the free tier.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Enough of a document to find a claim in, without shipping a novel to the LLM.
MAX_DOC_CHARS = 20_000

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


class InputError(RuntimeError):
    """The input could not be read. Message is written for the user."""


# ── Type detection ───────────────────────────────────────────────────────────

def detect_kind(raw: str) -> str:
    """
    Decide what we were handed.

    A path that exists on disk wins over treating the string as a claim, because
    "report.pdf" is a plausible thing to type but a terrible claim.
    """
    candidate = Path(raw)
    if not candidate.exists() or not candidate.is_file():
        return "text"

    suffix = candidate.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return "voice"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in DOC_SUFFIXES:
        return "document"
    raise InputError(
        f"Unsupported file type: {suffix or '(none)'}\n"
        f"  Supported: audio {sorted(AUDIO_SUFFIXES)}, "
        f"images {sorted(IMAGE_SUFFIXES)}, documents {sorted(DOC_SUFFIXES)}"
    )


# ── Voice ────────────────────────────────────────────────────────────────────

def transcribe(path: Path) -> str:
    """
    Groq `whisper-large-v3`.

    Remote on purpose. Running Whisper locally would mean a 1.5 GB model on a
    machine that is already holding four; the API costs audio-seconds rather
    than tokens, so it does not compete with the daily token budget either.
    """
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise InputError(
            f"Audio file is {path.stat().st_size / 1e6:.1f} MB; the limit is 25 MB. "
            f"Trim the clip and try again."
        )

    from groq import Groq

    client = Groq(api_key=cfg.require_groq())
    logger.info("Transcribing %s with %s", path.name, cfg.groq_whisper_model)

    with open(path, "rb") as handle:
        result = client.audio.transcriptions.create(
            file=(path.name, handle.read()),
            model=cfg.groq_whisper_model,
            # No `language` hint: auto-detection is the point. A Hindi clip has
            # to come back as Hindi so the report is generated in Hindi.
            response_format="json",
        )

    text = (result.text or "").strip()
    if not text:
        raise InputError("Transcription returned nothing — is the audio silent?")
    logger.info("Transcribed %d chars", len(text))
    return text


# ── Image ────────────────────────────────────────────────────────────────────

def ocr_unavailable() -> str | None:
    """Reason OCR cannot run, or None. Checked before any image work."""
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        return f"{exc.name} not installed — pip install pytesseract pillow"

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return (
            "the Tesseract engine is not installed (pytesseract is only a "
            "wrapper).\n"
            "    Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "    macOS:   brew install tesseract\n"
            "    Linux:   apt install tesseract-ocr\n"
            "    Then either add it to PATH or set TESSERACT_CMD in .env."
        )
    return None


def extract_image_text(path: Path) -> str:
    """
    OCR a screenshot.

    Tuned for the actual use case — a screenshot of a social post, which is
    clean rendered text on a flat background rather than a photographed
    document. `--psm 6` ("assume a single uniform block of text") reads that
    far better than the default page-segmentation mode, which hunts for columns
    that are not there.
    """
    reason = ocr_unavailable()
    if reason:
        raise InputError(f"Cannot read text from images: {reason}")

    import pytesseract
    from PIL import Image

    if cfg.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

    with Image.open(path) as image:
        # Greyscale helps on the light-grey-on-white that social apps love.
        text = pytesseract.image_to_string(image.convert("L"), config="--psm 6")

    text = _tidy_ocr(text)
    if not text:
        raise InputError(
            "No readable text found in that image. If the text is small, try a "
            "larger screenshot; if it is a photo, try cropping to just the text."
        )
    logger.info("OCR extracted %d chars from %s", len(text), path.name)
    return text


def _tidy_ocr(text: str) -> str:
    """
    Clean the wreckage OCR leaves on social screenshots.

    Drops the interface furniture — like counts, handles, timestamps — that
    surrounds the sentence we actually care about, then rejoins hard-wrapped
    lines so the claim arrives as one sentence rather than four fragments.
    """
    junk = re.compile(
        r"^\s*(\d+[\d,.km]*\s*(likes?|views?|comments?|shares?|retweets?)"
        r"|view all.*|see more|see translation|follow|følg|\d+[hdwm] ago"
        r"|liked by.*|replying to.*)\s*$",
        re.IGNORECASE,
    )
    lines = [
        line.strip() for line in text.splitlines()
        if line.strip() and not junk.match(line.strip())
    ]
    joined = " ".join(lines)
    joined = re.sub(r"\s{2,}", " ", joined)
    # OCR routinely turns "l" into "|" and inserts spaces before punctuation.
    joined = joined.replace("|", "l")
    joined = re.sub(r"\s+([.,;:!?])", r"\1", joined)
    return joined.strip()


# ── Documents ────────────────────────────────────────────────────────────────

def extract_document_text(path: Path) -> str:
    """Text out of a PDF, DOCX, or plain text file."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")

    elif suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise InputError("PDF support needs: pip install -e '.[docs]'") from None
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)

    elif suffix == ".docx":
        try:
            import docx
        except ImportError:
            raise InputError("DOCX support needs: pip install -e '.[docs]'") from None
        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)

    else:
        raise InputError(f"Cannot read {suffix} documents")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise InputError(
            f"No extractable text in {path.name}. If it is a scanned PDF, the "
            f"pages are images — screenshot the text and use image input."
        )
    return text[:MAX_DOC_CHARS]


def extract_claim_from(text: str) -> str:
    """
    Pick the most claim-like sentence out of a document.

    Uses the trained claim detector rather than an LLM call: it is already
    loaded, it costs a forward pass, and picking the best of ~40 sentences is
    exactly the binary judgement it was fine-tuned for. Falls back to the first
    substantial sentence if nothing scores well, so a document always yields
    *something* to verify.
    """
    sentences = [
        s.strip() for s in _SENTENCE_END.split(text)
        if 40 <= len(s.strip()) <= 400 and len(_WORD.findall(s)) >= 6
    ][:40]

    if not sentences:
        return text[:300].strip()

    import torch

    from core import models

    tokenizer, model = models.claim_detector()
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True,
                       max_length=128, padding=True)
    with torch.no_grad():
        scores = torch.softmax(model(**inputs).logits, dim=1)[:, 1]

    best = int(scores.argmax().item())
    logger.info("Extracted claim from document (p=%.3f of %d candidates): %r",
                scores[best].item(), len(sentences), sentences[best][:70])
    return sentences[best]


def _as_upload_evidence(text: str, name: str) -> list[Document]:
    """
    Turn an uploaded document into evidence — deliberately weak evidence.

    THE SAFETY PROPERTY: a user can upload a document asserting anything. If it
    entered the evidence pool at face value, the system would confirm any claim
    its own attachment made, which is the single most obvious way to fool a
    fact-checker.

    Three guards, all visible downstream:
      * `source_type: user_upload` -> 0.50 in the credibility type table
      * `credibility_cap: 0.6`     -> enforced in core.credibility.score_document
      * `corpus_tier: user_upload` -> the prompt and renderer both label it

    It is still ranked and floored like everything else, so an irrelevant upload
    is dropped rather than padding the evidence list.
    """
    return [Document(
        page_content=text,
        metadata={
            "url": f"upload://{name}",
            "domain": "user upload",
            "source": name,
            "title": name,
            "corpus_tier": "user_upload",
            "source_type": "user_upload",
            "retrieved_from": "user_upload",
            "credibility_cap": cfg.upload_credibility_cap,
            "date_reliable": False,
            "publish_date": "",
        },
    )]


# ── The node ─────────────────────────────────────────────────────────────────

def adapt_input(state: VerifyState) -> dict:
    """
    Normalise any supported input into a claim, and note where it came from.

    Text passes straight through, so the common path costs nothing.

    **Never raises.** An unreadable input is routed to `reject` with the reason
    attached, because "I could not read your file" is a legitimate outcome the
    user needs to see — not a stack trace. That keeps the guarantee that every
    path through the graph renders a real response.
    """
    try:
        return _adapt(state)
    except InputError as exc:
        logger.info("Input rejected: %s", str(exc).splitlines()[0])
        return {
            "input_error": str(exc),
            "is_claim": False,
            "claim": state.get("claim") or state.get("raw_input", ""),
            "reject_reason": str(exc),
        }


def _adapt(state: VerifyState) -> dict:
    raw = (state.get("raw_input") or "").strip()
    if not raw:
        raise InputError("No input given.")

    kind = state.get("input_kind") or detect_kind(raw)

    if kind == "text":
        return {"input_kind": "text", "claim": raw}

    path = Path(raw)
    update: dict = {"input_kind": kind, "source_file": path.name}

    if kind == "voice":
        transcript = transcribe(path)
        update["transcript"] = transcript
        update["extracted_text"] = transcript
        update["claim"] = transcript

    elif kind == "image":
        text = extract_image_text(path)
        update["extracted_text"] = text
        # The whole sentence goes to the gate, which then answers the user's
        # actual question: is this a checkable claim at all?
        update["claim"] = text

    elif kind == "document":
        text = extract_document_text(path)
        update["extracted_text"] = text

        supplied = (state.get("claim") or "").strip()
        if supplied and supplied != raw:
            # Document + claim: verify the claim, with the document as evidence.
            update["claim"] = supplied
            update["upload_evidence"] = _as_upload_evidence(text, path.name)
            logger.info("Document attached as evidence for a supplied claim")
        else:
            # Document alone: find a claim inside it.
            update["claim"] = extract_claim_from(text)

    logger.info("Adapted %s input -> claim: %r", kind, update["claim"][:70])
    return update


adapt_node = RunnableLambda(adapt_input, name="adapt_input")
