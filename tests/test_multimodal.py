"""
M5 tests — input adaptation, language handling, and the upload safety property.

The important one is `test_upload_credibility_is_capped`. Everything else here
guards ordinary breakage; that one guards the way a user could deliberately fool
the system, by attaching a document that asserts whatever they want confirmed.

Fixtures live in `tests/fixtures/` and are generated, not recorded — the audio
clips are produced by gTTS, so the voice test is a genuine round trip
(text → speech → Whisper → text) with nothing to download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import needs_models

FIXTURES = Path(__file__).parent / "fixtures"
needs_fixtures = pytest.mark.skipif(
    not FIXTURES.exists() or not any(FIXTURES.iterdir()),
    reason="fixtures not generated — see tests/fixtures/README",
)


# ── The safety property ──────────────────────────────────────────────────────

def test_upload_credibility_is_capped():
    """
    A user-supplied document must never become authoritative.

    Without the cap, uploading a file that asserts X and then asking "is X
    true?" would return TRUE with the upload as its own evidence — the most
    obvious way to fool a fact-checker, and the reason BUILD_PLAN §M5 makes it
    an explicit acceptance criterion.

    The cap is applied *after* normal weighting, so it is a ceiling rather than
    an input to the formula: an upload cannot inherit authority from a
    high-scoring domain either.
    """
    from core.config import cfg
    from core.credibility import score_document

    # A document claiming to be from the most credible domain in the table.
    hostile = score_document({
        "domain": "reuters.com",
        "source_type": "user_upload",
        "credibility_cap": cfg.upload_credibility_cap,
        "date_reliable": True,
        "publish_date": "2026-08-01",
    })
    assert hostile["total"] <= cfg.upload_credibility_cap
    assert hostile["capped"] is True

    # The same metadata without the cap scores far higher — proving the cap is
    # what is doing the work, not the source_type alone.
    uncapped = score_document({
        "domain": "reuters.com",
        "source_type": "user_upload",
        "date_reliable": True,
        "publish_date": "2026-08-01",
    })
    assert uncapped["total"] > cfg.upload_credibility_cap


def test_upload_evidence_is_labelled_for_the_prompt():
    """
    The cap is invisible to the LLM unless the evidence block says so.

    A number in metadata means nothing to a language model; the prompt has to
    state that the item is a user submission, or the model will weigh it like
    any other source.
    """
    from core.prompts import format_evidence
    from serve.nodes.adapt import _as_upload_evidence

    documents = _as_upload_evidence("Exports collapsed to 40 billion.", "my.docx")
    documents[0].metadata.update({"credibility": 0.5, "credibility_tier": "LOW",
                                  "relevance_score": 0.99})

    block = format_evidence(documents, [])
    assert "UNVERIFIED USER SUBMISSION" in block
    assert "not an independent source" in block


def test_upload_metadata_carries_the_cap():
    from core.config import cfg
    from serve.nodes.adapt import _as_upload_evidence

    meta = _as_upload_evidence("text", "report.pdf")[0].metadata
    assert meta["credibility_cap"] == cfg.upload_credibility_cap
    assert meta["corpus_tier"] == "user_upload"
    assert meta["source_type"] == "user_upload"
    assert meta["date_reliable"] is False


# ── Language ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("भारत का सॉफ्टवेयर निर्यात 222 अरब डॉलर तक पहुंच गया", "hi"),
    ("India's software exports reached 222 billion dollars in 2024-25", "en"),
    ("Les exportations de logiciels ont atteint 222 milliards de dollars", "fr"),
    ("ভারতের সফটওয়্যার রপ্তানি বেড়েছে", "bn"),
    ("இந்தியாவின் மென்பொருள் ஏற்றுமதி அதிகரித்தது", "ta"),
    ("Россия увеличила экспорт программного обеспечения", "ru"),
])
def test_language_detection(text, expected):
    from serve.nodes.language import detect_language

    assert detect_language(text) == expected


def test_language_detection_is_deterministic():
    """
    `langdetect` seeds its RNG from the clock unless told otherwise, so the same
    string can detect differently between runs. That would make the eval harness
    non-reproducible and produce reports in a randomly varying language.
    """
    from serve.nodes.language import detect_language

    text = "Les exportations de logiciels ont atteint 222 milliards de dollars"
    assert len({detect_language(text) for _ in range(8)}) == 1


def test_english_claims_skip_translation():
    """The common path must not pay for an LLM call it does not need."""
    from serve.nodes.language import to_english

    state = {"claim": "India's software exports reached 222 billion dollars",
             "diagnostics": {"llm_calls": 0}}
    result = to_english(state)

    assert result["language"] == "en"
    assert result["claim_english"] == state["claim"]
    assert "diagnostics" not in result      # untouched -> no call was counted


# ── Input adaptation ─────────────────────────────────────────────────────────

@needs_fixtures
@pytest.mark.parametrize("name,kind", [
    ("claim_hi.mp3", "voice"),
    ("insta_post.png", "image"),
    ("false_report.docx", "document"),
])
def test_detect_kind(name, kind):
    from serve.nodes.adapt import detect_kind

    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    assert detect_kind(str(path)) == kind


def test_plain_text_is_not_mistaken_for_a_path():
    from serve.nodes.adapt import detect_kind

    assert detect_kind("India's GDP grew 8% in 2024") == "text"
    assert detect_kind("report.pdf") == "text"      # does not exist on disk


def test_unreadable_input_rejects_instead_of_raising():
    """
    Every path through the graph must render a response. An OCR failure or an
    unsupported file type is something the user needs told, not a traceback —
    so `adapt` converts it into a rejection with the reason attached.
    """
    from serve.nodes.adapt import adapt_input
    from serve.graph import route_after_adapt

    result = adapt_input({"raw_input": ""})
    assert result["input_error"]
    assert result["is_claim"] is False
    assert route_after_adapt(result) == "reject"

    # A readable input routes onward.
    assert route_after_adapt(adapt_input({"raw_input": "a real claim here"})) \
        == "translate"


def test_ocr_absence_is_detected_not_crashed_on():
    """
    pytesseract is a wrapper around a system binary. Importing it succeeds even
    when Tesseract itself is missing, so the check has to actually invoke it.
    """
    from serve.nodes.adapt import ocr_unavailable

    reason = ocr_unavailable()
    assert reason is None or isinstance(reason, str)
    if reason:
        # The message must tell the user how to fix it, not just that it broke.
        assert "tesseract" in reason.lower()


@needs_models
def test_claim_extraction_picks_the_claim_like_sentence():
    """
    A document alone yields a claim to check. The trained detector is reused
    rather than an LLM call — picking the most assertion-like of N sentences is
    exactly its binary task.
    """
    from serve.nodes.adapt import extract_claim_from

    text = (
        "Welcome to our quarterly newsletter. "
        "We hope you are all having a wonderful week so far. "
        "India's software exports reached 222 billion dollars in 2024-25, "
        "an eleven percent rise over the previous year. "
        "Thanks for reading and see you next month."
    )
    assert "222 billion" in extract_claim_from(text)


@needs_fixtures
def test_docx_text_extraction():
    from serve.nodes.adapt import extract_document_text

    path = FIXTURES / "false_report.docx"
    if not path.exists():
        pytest.skip("fixture not generated")
    text = extract_document_text(path)
    assert "40 billion" in text


def test_ocr_tidy_strips_social_furniture():
    """
    OCR of a screenshot returns the interface as well as the text. Like counts
    and "View all 302 comments" are not part of the claim and would otherwise
    be embedded and searched as if they were.
    """
    from serve.nodes.adapt import _tidy_ocr

    raw = (
        "factsdaily_official\n"
        "India's software exports reached 222 billion\n"
        "dollars in 2024-25.\n"
        "12,431 likes\n"
        "View all 302 comments\n"
        "2 days ago\n"
    )
    cleaned = _tidy_ocr(raw)
    assert "222 billion dollars in 2024-25." in cleaned
    assert "likes" not in cleaned
    assert "View all" not in cleaned


# ── Exports ──────────────────────────────────────────────────────────────────

def _sample_state():
    from langchain_core.documents import Document
    from serve.schemas import EvidenceAssessment, VerificationReport

    return {
        "claim": "India's software exports reached 222 billion dollars",
        "language": "en",
        "outcome": "verified",
        "evidence": [Document(page_content="…", metadata={
            "url": "https://economictimes.indiatimes.com/x",
            "domain": "economictimes.indiatimes.com",
            "credibility": 0.80, "credibility_tier": "MEDIUM",
            "relevance_score": 0.99,
        })],
        "report": VerificationReport(
            claim="India's software exports reached 222 billion dollars",
            claim_type="statistical",
            evidence_analysis=[EvidenceAssessment(
                evidence_index=1, source="economictimes.indiatimes.com",
                stance="SUPPORTS", directly_relevant=True, reasoning="Confirms.")],
            contradictions=[], verdict="TRUE", confidence=80,
            key_findings=[], limitations=["Single source."],
            conclusion="Supported by the evidence.",
        ),
        "diagnostics": {"llm_calls": 1},
    }


def test_html_export_is_self_contained(tmp_path, monkeypatch):
    """No external requests: the CSP-free case still shouldn't need a CDN."""
    from core.config import cfg
    from serve.nodes.export import to_html

    monkeypatch.setattr(cfg, "output_dir", tmp_path)
    html = to_html(_sample_state()).read_text(encoding="utf-8")

    assert "<!doctype html>" in html.lower()
    assert "TRUE" in html
    assert "http://" not in html.replace("https://economictimes", "")
    assert "<script" not in html.lower()


def test_docx_export_writes_a_file(tmp_path, monkeypatch):
    from core.config import cfg
    from serve.nodes.export import to_docx

    monkeypatch.setattr(cfg, "output_dir", tmp_path)
    path = to_docx(_sample_state())
    assert path.exists() and path.stat().st_size > 0


def test_export_skips_formats_that_fail(tmp_path, monkeypatch):
    """One broken format must not lose the others."""
    from core.config import cfg
    from serve.nodes.export import export

    monkeypatch.setattr(cfg, "output_dir", tmp_path)
    artifacts = export(_sample_state(), ["html", "nonsense"])
    assert "html" in artifacts and "nonsense" not in artifacts
