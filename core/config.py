"""
Single source of truth for configuration.

Everything reads settings from here — no module reads os.environ directly, and
nothing reads a secret out of a text file. Validation happens at import time, so
a missing key or a malformed path is a startup error with a clear message rather
than an AttributeError somewhere inside a request.

Usage:
    from core.config import cfg
    print(cfg.groq_model)
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory containing this package's parent.
# Resolved from __file__ rather than cwd so that `python -m ingest.run` behaves
# the same whether it is launched from the repo root or anywhere else.
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings, loaded from .env with environment overrides."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",             # tolerate unrelated vars in a shared .env
        case_sensitive=False,
    )

    # ── Credentials ─────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key")
    tavily_api_key: str = Field(default="", description="Web search, M4")
    langsmith_api_key: str = Field(default="")
    langsmith_tracing: bool = Field(default=False)
    langsmith_project: str = Field(default="verifai")

    # ── Groq models ─────────────────────────────────────────────────────────
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_whisper_model: str = Field(default="whisper-large-v3")
    groq_vision_model: str = Field(default="")

    # ── Index / retrieval ───────────────────────────────────────────────────
    index_path: Path = Field(default=Path("index/current"))
    embed_model: str = Field(default="all-MiniLM-L6-v2")
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    chunk_size: int = Field(default=700, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    candidate_k: int = Field(default=20, gt=0)
    top_k: int = Field(default=8, gt=0)
    relevance_floor: float = Field(default=0.15, ge=0.0, le=1.0)

    # P(is_claim) below which the gate rejects. The trained detector separates
    # claims from questions very sharply (0.999 vs 0.0002 on probes), so 0.5
    # sits in a wide empty band rather than on a decision boundary.
    claim_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # A stance prediction below this is passed to the LLM marked "LOW - be
    # sceptical". The stance model is 73.6% accurate overall and weakest on
    # REFUTES, so its confidence is worth surfacing rather than hiding.
    stance_low_confidence: float = Field(default=0.60, ge=0.0, le=1.0)

    # ── Web search fallback (M4) ────────────────────────────────────────────
    # Fires only when the index returns nothing above the relevance floor, so a
    # claim the corpus can answer never costs a search. Requires TAVILY_API_KEY;
    # with no key the node logs once and routes to abstain, which is why the
    # default can safely be on.
    web_search_enabled: bool = Field(default=True)
    web_search_k: int = Field(default=6, gt=0)

    # ── Input adaptation (M5) ───────────────────────────────────────────────
    # Path to the Tesseract binary when it is installed somewhere off PATH.
    # pytesseract is only a wrapper; the engine is a separate system install.
    tesseract_cmd: str = Field(default="")

    # Hard ceiling on the credibility of a user-supplied document.
    #
    # A user can upload a file asserting anything. Without this cap the system
    # would confirm any claim its own attachment made — the most obvious way to
    # fool a fact-checker. Uploads are evidence, but never authoritative
    # evidence, and the report says so.
    upload_credibility_cap: float = Field(default=0.6, ge=0.0, le=1.0)

    # Report language. `auto` generates the report in whatever language the
    # claim was written or spoken in; an ISO code forces one.
    report_language: str = Field(default="auto")

    # Speak the conclusion back when the input was voice (gTTS).
    voice_output: bool = Field(default=True)

    # Where rendered html/docx/mp3 exports are written.
    output_dir: Path = Field(default=Path("data/outputs"))
    # "advanced" costs 2 Tavily credits per call instead of 1 and returns
    # longer extracts. Worth it here: this only runs when the corpus has already
    # failed, so it is the last chance to find evidence.
    web_search_depth: str = Field(default="advanced")

    # ── Trained models (local, CPU) ─────────────────────────────────────────
    claim_detector_path: Path = Field(default=Path("models/claim_detector/final"))
    stance_detector_path: Path = Field(default=Path("models/stance_detector/final"))
    device: str = Field(default="cpu")

    # Load HuggingFace models from the local cache only, never phoning home.
    #
    # This is not just a speed optimisation. sentence-transformers checks the
    # Hub for updated files on every load, and that httpx call was crashing this
    # machine with a hard access violation (exit 0xC0000005, no traceback) —
    # intermittently, depending on network conditions. Offline loading removes
    # the network from the critical path entirely, which is what Tier 1 wants
    # anyway: an index build should not depend on huggingface.co being reachable.
    #
    # Set HF_OFFLINE=false for the first run on a machine with a cold cache, so
    # the models can download; then leave it true.
    hf_offline: bool = Field(default=True)

    # ── Data locations ──────────────────────────────────────────────────────
    source_csv: Path = Field(default=Path("data/processed/news_articles_rag.csv"))
    index_root: Path = Field(default=Path("index"))
    embed_cache: Path = Field(default=Path("data/.embed_cache"))

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator(
        "groq_api_key", "tavily_api_key", "langsmith_api_key",
        "groq_model", "groq_whisper_model", "groq_vision_model",
        mode="before",
    )
    @classmethod
    def _strip_inline_comment(cls, v):
        """
        Treat a value that is only a dotenv comment as unset.

        dotenv strips a trailing `# ...` when a value is present, but on an EMPTY
        value (`KEY=            # note`) it parses the comment text as the value.
        That surfaced as a baffling "model not available: '# M5 - image OCR'"
        startup error. The template now keeps comments on their own lines; this
        guard means a stray inline comment can never cause that class of bug again.
        """
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("#"):
                return ""
            # also drop a trailing comment dotenv may have left attached
            if "#" in v and " #" in v:
                v = v.split(" #", 1)[0].strip()
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info) -> int:
        """An overlap >= chunk_size makes the splitter loop forever."""
        size = info.data.get("chunk_size", 700)
        if v >= size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({size})")
        return v

    @field_validator(
        "index_path", "claim_detector_path", "stance_detector_path",
        "source_csv", "index_root", "embed_cache", "output_dir",
        mode="after",
    )
    @classmethod
    def _absolutise(cls, v: Path) -> Path:
        """
        Resolve relative paths against the project root, not the working
        directory. The old code checked `Path("data")` relative to cwd, which was
        only correct when the server happened to be launched from the repo root.
        """
        return v if v.is_absolute() else (ROOT / v)

    # ── The index pointer ───────────────────────────────────────────────────
    #
    # Tier 1 builds into index/v{n}/ and, once the build validates, writes the
    # version name into index/CURRENT. Tier 2 reads CURRENT at startup.
    #
    # A text file rather than a symlink: symlinks need Developer Mode or admin
    # rights on Windows, and a half-created junction is a worse failure than a
    # one-line file. Swapping versions is an atomic single-file write either way.

    # ClassVar, not a field — pydantic treats any bare class attribute as a
    # settings field and rejects it without an annotation.
    POINTER: ClassVar[str] = "CURRENT"

    @property
    def active_index(self) -> Path:
        """Directory of the index Tier 2 should serve."""
        pointer = self.index_root / self.POINTER
        if pointer.exists():
            version = pointer.read_text(encoding="utf-8").strip()
            if version:
                return self.index_root / version
        # No pointer yet (fresh clone, or a build that never validated) —
        # fall back to the literal configured path so errors name a real target.
        return self.index_path

    def next_version(self) -> str:
        """Next unused `v{n}` under index/. Never overwrites a live build."""
        existing = [
            int(p.name[1:])
            for p in self.index_root.glob("v*")
            if p.is_dir() and p.name[1:].isdigit()
        ]
        return f"v{max(existing, default=0) + 1}"

    # ── Artifact paths within an index build ────────────────────────────────
    # Take a build dir explicitly so ingest can address the version it is
    # writing, while serve addresses whatever CURRENT points at.

    @staticmethod
    def manifest_path(build: Path) -> Path:
        return build / "manifest.json"

    @staticmethod
    def chroma_path(build: Path) -> Path:
        return build / "chroma"

    @staticmethod
    def parents_path(build: Path) -> Path:
        return build / "parents"

    @staticmethod
    def bm25_path(build: Path) -> Path:
        return build / "bm25.pkl"

    def require_groq(self) -> str:
        """
        Fetch the Groq key or fail with an actionable message.

        Deliberately not a validator: `ingest` never calls Groq, so requiring the
        key at import time would stop you building an index offline.
        """
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n"
                f"  1. cp {ROOT / '.env.example'} {ROOT / '.env'}\n"
                "  2. paste a key from https://console.groq.com/keys"
            )
        return self.groq_api_key


# Module-level singleton. Importing core.config validates the settings, so a bad
# .env fails at the first import rather than at first use.
cfg = Settings()


# Applied at import, before anything can pull in huggingface_hub — these are read
# once at *its* import time, so setting them later has no effect. Every module
# that touches a model imports cfg first, which is what makes this reliable.
if cfg.hf_offline:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# LangSmith tracing. LangChain reads these names from the environment directly,
# so the settings object has to publish them — this is the one place allowed to
# do that, for the same reason as the HF flags above.
#
# Requires BOTH the flag and a key: turning tracing on without a key makes every
# LLM call attempt a background upload that quietly fails, which looks like
# latency with no explanation. Off by default; free to enable at
# https://smith.langchain.com.
if cfg.langsmith_tracing and cfg.langsmith_api_key:
    import os

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", cfg.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", cfg.langsmith_project)
    # Hosts contacted by the tracer must be pre-resolved, or the first upload
    # crashes the process — see core.init_native_libs.
    os.environ.setdefault(
        "VERIFAI_DNS_PREWARM",
        "api.groq.com,api.tavily.com,api.smith.langchain.com",
    )
