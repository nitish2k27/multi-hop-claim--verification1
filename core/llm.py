"""
The single LLM seam.

Every Groq call in the project goes through get_llm(). That replaces the old
report_generator.py / report_generator_groq.py pair and the `if llm_mode ==
"groq" / elif "colab"` dispatch — swapping provider is now one env var, because
ChatGroq, ChatOllama and the rest are interchangeable behind the same interface.

The startup model check matters more than it looks: Groq deprecates and renames
models on a rolling basis, and a stale id would otherwise surface as a 404 in the
middle of a user's verification. verify_models() turns that into a boot error.
"""

from __future__ import annotations

import functools
import logging

import requests

from core.config import cfg

logger = logging.getLogger(__name__)

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


@functools.lru_cache(maxsize=1)
def available_models() -> frozenset[str]:
    """
    Model ids this key can actually use.

    Cached for the process lifetime — the list does not change under a running
    app, and we do not want a network round trip per verification.
    """
    resp = requests.get(
        GROQ_MODELS_URL,
        headers={"Authorization": f"Bearer {cfg.require_groq()}"},
        timeout=15,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "Groq rejected the API key (401). Rotate it at "
            "https://console.groq.com/keys and update .env"
        )
    resp.raise_for_status()
    return frozenset(m["id"] for m in resp.json().get("data", []))


def verify_models(*, strict: bool = True) -> dict[str, bool]:
    """
    Check every configured model id against the live catalogue.

    Call once at startup (CLI entry, FastAPI lifespan). Blank ids are skipped —
    GROQ_VISION_MODEL is legitimately unset until M5.

    Raises RuntimeError listing the bad ids when strict=True.
    """
    configured = {
        "GROQ_MODEL": cfg.groq_model,
        "GROQ_WHISPER_MODEL": cfg.groq_whisper_model,
        "GROQ_VISION_MODEL": cfg.groq_vision_model,
    }
    catalogue = available_models()
    result = {
        var: (model in catalogue)
        for var, model in configured.items()
        if model  # unset is not a failure
    }

    missing = [
        f"{var}={configured[var]!r}" for var, ok in result.items() if not ok
    ]
    if missing and strict:
        sample = sorted(m for m in catalogue if "llama" in m or "whisper" in m)[:8]
        raise RuntimeError(
            "Configured Groq model(s) not available to this key:\n"
            + "".join(f"  - {m}\n" for m in missing)
            + "Groq renames models regularly. Pick a current id from:\n"
            + "".join(f"  - {m}\n" for m in sample)
            + "then update .env"
        )
    return result


@functools.lru_cache(maxsize=4)
def get_llm(temperature: float = 0.2, max_tokens: int = 2000):
    """
    The chat model used for verdict generation.

    temperature is low by default: this model's job is to reason over supplied
    evidence and emit a schema-locked verdict, not to write creatively.

    Cached per (temperature, max_tokens) so nodes can request their own settings
    without re-instantiating a client each call.
    """
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=cfg.groq_model,
        api_key=cfg.require_groq(),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=2,
    )


def health() -> dict:
    """Human-readable status, for the CLI banner and GET /health."""
    try:
        checks = verify_models(strict=False)
        return {
            "groq_key": "present",
            "models": checks,
            "ok": all(checks.values()),
        }
    except Exception as exc:  # network down, bad key, etc.
        return {"groq_key": "unusable", "error": str(exc), "ok": False}


if __name__ == "__main__":
    # `python -m core.llm`         -> M0 acceptance check
    # `python -m core.llm --list`  -> every model id this key can use
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if "--list" in sys.argv:
        for model in sorted(available_models()):
            print(model)
        sys.exit(0)

    verify_models()
    print(f"[ok] model available: {cfg.groq_model}")
    print(get_llm().invoke("Reply with exactly: OK").content)
