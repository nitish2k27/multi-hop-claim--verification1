"""
Language detection and translation.

Two facts drive the whole design:

1. **The index is English.** A Hindi claim cannot retrieve anything from it, so
   the claim must be translated to English *for retrieval*.
2. **The report must come back in the user's language.** Not translated
   afterwards — *generated* in it. `core/prompts.py` already carries the
   `LANG_INSTRUCTIONS` table and the system prompt takes a language instruction;
   this node just supplies the language code.

So `VerifyState` carries both strings, which is why it has had `claim`,
`claim_english` and `language` since M2:

    claim          what the user said, in their language   -> report is in this
    claim_english  translated                              -> used for retrieval

Generating rather than back-translating matters: a translated report drifts from
the one the API returned, and the verdict label can shift in the round trip. One
generation, one verdict, expressed once.

WHY THE SEED IS SET
-------------------
`langdetect` is a port of a Java library that seeds its own RNG from the clock.
Without `DetectorFactory.seed = 0` the *same string can return different
languages on different runs* — a genuinely maddening source of flaky behaviour,
and one that would make the eval harness non-reproducible.
"""

from __future__ import annotations

import logging
import re

from langchain_core.runnables import RunnableLambda

from core.config import cfg
from core.prompts import LANG_INSTRUCTIONS
from serve.schemas import VerifyState

logger = logging.getLogger(__name__)

# Below this, detection is guesswork. "Ok" is valid in a dozen languages.
MIN_DETECT_CHARS = 20

# Scripts that are decisive on sight. langdetect is trained on prose and gets
# short strings wrong, but a string of Devanagari is not Portuguese — checking
# the script first is both faster and more reliable for exactly the short,
# mixed-script inputs a fact-check claim tends to be.
_SCRIPTS = [
    ("hi", re.compile(r"[ऀ-ॿ]")),   # Devanagari (Hindi/Marathi)
    ("bn", re.compile(r"[ঀ-৿]")),   # Bengali
    ("pa", re.compile(r"[਀-੿]")),   # Gurmukhi
    ("gu", re.compile(r"[઀-૿]")),   # Gujarati
    ("ta", re.compile(r"[஀-௿]")),   # Tamil
    ("te", re.compile(r"[ఀ-౿]")),   # Telugu
    ("kn", re.compile(r"[ಀ-೿]")),   # Kannada
    ("ml", re.compile(r"[ഀ-ൿ]")),   # Malayalam
    ("ar", re.compile(r"[؀-ۿ]")),   # Arabic
    ("ur", re.compile(r"[؀-ۿ]\s*[ٹپچڈ]")),  # Urdu
    ("ru", re.compile(r"[Ѐ-ӿ]")),   # Cyrillic
    ("ja", re.compile(r"[぀-ゟ゠-ヿ]")),  # kana
    ("ko", re.compile(r"[가-힯]")),   # Hangul
    ("zh", re.compile(r"[一-鿿]")),   # Han
]


def detect_language(text: str) -> str:
    """
    ISO 639-1 code for `text`, defaulting to English.

    Script detection first (decisive and cheap), then `langdetect` for the
    Latin-script languages where it actually performs well.
    """
    stripped = text.strip()
    if not stripped:
        return "en"

    for code, pattern in _SCRIPTS:
        if pattern.search(stripped):
            # Urdu and Arabic share a block; the Urdu-specific letters win.
            if code == "ar" and _SCRIPTS[9][1].search(stripped):
                return "ur"
            return code

    if len(stripped) < MIN_DETECT_CHARS:
        return "en"

    try:
        from langdetect import DetectorFactory, detect

        # Deterministic. Without this the same input can detect differently
        # between runs — see the module docstring.
        DetectorFactory.seed = 0
        code = detect(stripped)
    except Exception as exc:
        logger.debug("langdetect failed (%s) — defaulting to English", exc)
        return "en"

    # langdetect emits regional variants; the prompt table is keyed on the base.
    return code.split("-")[0]


def translate_to_english(text: str, source: str) -> str:
    """
    Translate a claim into English so it can hit the index.

    One Groq call, deliberately constrained: fact-checkable claims turn on
    specific numbers, names and dates, and a model that "improves" the phrasing
    while translating changes what is being checked.
    """
    from core.llm import get_llm

    prompt = (
        f"Translate this {LANG_NAMES.get(source, source)} statement into English.\n"
        f"Rules:\n"
        f"- Output ONLY the translation. No preamble, no quotes, no notes.\n"
        f"- Preserve every number, date, name and unit exactly.\n"
        f"- Do not soften, hedge, correct or complete the statement. If it is "
        f"false, translate it faithfully as a false statement.\n\n"
        f"{text}"
    )
    # Zero temperature: translation is not a place for creativity.
    result = get_llm(temperature=0.0, max_tokens=500).invoke(prompt)
    translated = (result.content or "").strip().strip('"')

    logger.info("Translated %s -> en: %r", source, translated[:70])
    return translated or text


LANG_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "es": "Spanish", "fr": "French",
    "de": "German", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ru": "Russian", "pt": "Portuguese", "en": "English",
}


def to_english(state: VerifyState) -> dict:
    """
    Detect the claim's language and produce an English version for retrieval.

    English input short-circuits entirely — no detection cost worth mentioning
    and no LLM call, which keeps the common path free.
    """
    claim = (state.get("claim") or "").strip()
    if not claim:
        return {"language": "en", "claim_english": ""}

    language = detect_language(claim)

    # An explicit REPORT_LANGUAGE overrides detection for the *output* only;
    # retrieval still needs English regardless.
    report_language = (
        language if cfg.report_language == "auto" else cfg.report_language
    )

    if language == "en":
        return {"language": report_language, "claim_english": claim}

    supported = language in LANG_INSTRUCTIONS
    if not supported:
        logger.warning("No prompt instruction for %r — report will be English",
                       language)

    english = translate_to_english(claim, language)

    diagnostics = dict(state.get("diagnostics", {}))
    diagnostics["llm_calls"] = diagnostics.get("llm_calls", 0) + 1
    diagnostics["translated_from"] = language

    return {
        "language": report_language if supported else "en",
        "claim_english": english,
        "diagnostics": diagnostics,
    }


language_node = RunnableLambda(to_english, name="to_english")
