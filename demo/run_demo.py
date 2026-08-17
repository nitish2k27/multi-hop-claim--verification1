"""
VerifAI demo — one command, about 90 seconds.

    python -m demo.run_demo              # all available scenarios
    python -m demo.run_demo --fast       # skip the LLM-backed one
    python -m demo.run_demo --list       # what would run, and what is skipped

Built as a screen-recording target: each scenario prints a heading, what it is
meant to show, a live trace of the graph, and the outcome — so a 90-second
capture reads as a narrative rather than a wall of logs.

SCENARIOS DEGRADE, THEY DO NOT FAIL
-----------------------------------
Two of the four scenarios depend on things that may not be configured: the web
fallback needs a Tavily key, and voice input needs M5. Rather than crash or be
silently dropped, an unavailable scenario prints WHY it is skipped and what
would enable it. That keeps the demo honest about the system's current reach
and means this file needs no edits when those land.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Progress bars would corrupt a screen recording.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from core.config import cfg

RULE = "─" * 74


class Scenario:
    """One demo case: what it shows, and whether it can run right now."""

    def __init__(self, title: str, claim: str, shows: str,
                 expect: str, requires=None, no_web: bool = False):
        self.title = title
        self.claim = claim
        self.shows = shows
        self.expect = expect
        self.requires = requires or (lambda: None)
        self.no_web = no_web

    def unavailable(self) -> str | None:
        return self.requires()


def _needs_tavily() -> str | None:
    if not cfg.tavily_api_key:
        return ("needs TAVILY_API_KEY — free tier at https://app.tavily.com, "
                "then add it to .env")
    return None


HINDI_CLIP = ROOT_FIXTURES = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "claim_hi.mp3"
)


def _needs_voice_clip() -> str | None:
    if not cfg.groq_api_key:
        return "needs GROQ_API_KEY for Whisper transcription"
    if not HINDI_CLIP.exists():
        return (f"no audio fixture at {HINDI_CLIP.name} — generate with "
                f"python -m tests.make_fixtures")
    return None


def _needs_ocr() -> str | None:
    from serve.nodes.adapt import ocr_unavailable

    reason = ocr_unavailable()
    return reason.split("\n")[0] if reason else None


# Ordered as a narrative: it works → it does not over-reach → it stops → it does
# not even start. Each scenario removes a different way the system could have
# been fooling you.
SCENARIOS = [
    Scenario(
        title="1. A claim the corpus can actually verify",
        claim="India's software exports reached 222 billion dollars in 2024-25",
        shows="the full path — hybrid dense+BM25 retrieval, cross-encoder "
              "reranking, per-source credibility, stance detection, and a "
              "schema-constrained verdict where every citation resolves",
        expect="TRUE, with evidence",
        # Forced off so this scenario always demonstrates the *index* path.
        no_web=True,
    ),
    Scenario(
        title="2. Related material, but not the actual fact",
        claim="India's GDP grew 8% in 2024",
        shows="the hardest case — the corpus is full of Indian economy "
              "articles, so retrieval returns high-scoring, genuinely on-topic "
              "evidence. A relevance threshold alone cannot catch this.",
        expect="UNVERIFIABLE — it reads the evidence and declines, instead of "
               "assembling a confident answer out of adjacent material",
        no_web=True,
    ),
    Scenario(
        title="3. Nothing relevant exists  ← the one that matters",
        claim="the moon is square in shape",
        shows="abstention: nothing clears the relevance floor, so the graph "
              "stops before reaching the LLM at all",
        expect="ABSTAINED, zero LLM calls — the previous version of this system "
               "answered this with a 700-word analysis of celebrity news",
        no_web=True,
    ),
    Scenario(
        title="4. Not a claim at all",
        claim="what time is it",
        shows="the claim gate — a fine-tuned BERT rejects non-assertions before "
              "any retrieval happens",
        expect="REJECTED in a fraction of a second, zero cost",
    ),
    Scenario(
        title="5. Outside the corpus, found on the web",
        claim="Who won the most recent Formula 1 world championship?",
        shows="the M4 fallback — the index abstains, live web search runs, and "
              "results are held to the same relevance floor",
        expect="a verdict sourced from the web, labelled as such",
        requires=_needs_tavily,
    ),
    Scenario(
        title="6. The same claim, spoken in Hindi",
        claim=str(HINDI_CLIP),
        shows="the full multimodal round trip — Whisper transcribes the audio, "
              "the language is detected, the claim is translated to English to "
              "search an English index, and the report is GENERATED in Hindi "
              "rather than translated back",
        expect="TRUE, a Hindi report, and a spoken Hindi answer as an mp3",
        requires=_needs_voice_clip,
        no_web=True,
    ),
    Scenario(
        title="7. A claim pasted as a screenshot",
        claim=str(Path(__file__).resolve().parent.parent
                  / "tests" / "fixtures" / "insta_post.png"),
        shows="OCR on a social-media screenshot, then the claim gate answering "
              "the question the user actually has — is this even checkable?",
        expect="the sentence lifted out of the image, then verified",
        requires=_needs_ocr,
        no_web=True,
    ),
]


def _print_header() -> None:
    print()
    print(RULE)
    print("  VerifAI — two-tier RAG fact verification")
    print(RULE)
    manifest_path = cfg.manifest_path(cfg.active_index)
    if manifest_path.exists():
        import json

        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"  index      {m['version']} · {m['documents']:,} documents · "
              f"{m['chunks']:,} chunks")
        print(f"  embedder   {m['embed_model']} (dim {m['embed_dim']})")
    print(f"  reranker   {cfg.rerank_model}")
    print(f"  floor      {cfg.relevance_floor}  (calibrated at M3, not guessed)")
    print(f"  LLM        {cfg.groq_model} via Groq")
    print(RULE)


def _run_one(scenario: Scenario, index: int, total: int) -> str:
    print()
    print(RULE)
    print(f"  {scenario.title}")
    print(RULE)
    print(f"  claim    \"{scenario.claim}\"")
    print(f"  shows    {scenario.shows}")
    print(f"  expect   {scenario.expect}")

    reason = scenario.unavailable()
    if reason:
        print(f"\n  ⏭  SKIPPED — {reason}")
        return "skipped"

    original = cfg.web_search_enabled
    if scenario.no_web:
        cfg.web_search_enabled = False

    print()
    try:
        from serve.graph import get_app
        from serve.schemas import new_state

        started = time.monotonic()
        final: dict = {}
        for chunk in get_app().stream(new_state(scenario.claim),
                                      stream_mode="updates"):
            for node, update in chunk.items():
                print(f"    → {node:<9} {_summarise(node, update)}")
                final.update(update)
        elapsed = time.monotonic() - started

    except Exception as exc:
        message = str(exc)
        if "rate_limit" in message or "429" in message:
            print("    ⏭  SKIPPED — Groq quota reached; retrieval ran fine, "
                  "only generation was refused")
            return "skipped"
        raise
    finally:
        cfg.web_search_enabled = original

    outcome = final.get("outcome", "?")
    calls = final.get("diagnostics", {}).get("llm_calls", 0)
    report = final.get("report")

    print()
    print(f"  ✓ {outcome.upper()}   {elapsed:.1f}s   {calls} LLM call(s)")
    if report:
        print(f"    verdict: {report.verdict} at {report.confidence}% confidence")
        if report.conclusion:
            print(f"    {_wrap(report.conclusion, 68)}")
    return outcome


def _summarise(node: str, update: dict) -> str:
    if node == "gate":
        return ("claim" if update.get("is_claim") else "NOT a claim") + \
               f"  (p={update.get('claim_confidence', 0.0):.3f})"
    if node == "retrieve":
        stats = update.get("retrieval_stats", {})
        return (f"{stats.get('candidates', 0)} candidates → "
                f"{len(update.get('evidence') or [])} above floor "
                f"{stats.get('floor', 0):.2f}")
    if node == "web":
        stats = update.get("retrieval_stats", {})
        if stats.get("web_skipped"):
            return f"skipped ({stats['web_skipped']})"
        return (f"{stats.get('web_results', 0)} web results → "
                f"{stats.get('web_after_floor', 0)} above floor")
    if node == "stance":
        counts = update.get("diagnostics", {}).get("stance_counts", {})
        return "  ".join(f"{k}={v}" for k, v in counts.items()) or "-"
    if node == "generate":
        report = update.get("report")
        return f"{report.verdict} @ {report.confidence}%" if report else "-"
    if node in ("abstain", "reject"):
        return f"→ {update.get('outcome', node)}, no LLM call"
    if node == "render":
        return f"{len(update.get('rendered', ''))} chars of markdown"
    return ""


def _wrap(text: str, width: int) -> str:
    import textwrap

    return textwrap.fill(text, width, subsequent_indent="    ")


def main() -> int:
    parser = argparse.ArgumentParser(prog="demo.run_demo", description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="skip scenarios that call the LLM")
    parser.add_argument("--list", action="store_true",
                        help="show what would run, without running it")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.ERROR)
    for noisy in ("httpx", "chromadb", "transformers", "sentence_transformers",
                  "serve", "core", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.list:
        print()
        for scenario in SCENARIOS:
            reason = scenario.unavailable()
            mark = "skip" if reason else "run "
            print(f"  [{mark}] {scenario.title}")
            if reason:
                print(f"          {reason}")
        return 0

    _print_header()

    scenarios = SCENARIOS
    if args.fast:
        scenarios = [s for s in SCENARIOS if s.no_web or s.unavailable()]

    started = time.monotonic()
    results = [
        _run_one(s, i, len(scenarios)) for i, s in enumerate(scenarios, start=1)
    ]

    print()
    print(RULE)
    ran = [r for r in results if r != "skipped"]
    print(f"  {len(ran)} scenario(s) in {time.monotonic() - started:.0f}s"
          f"   ·   {len(results) - len(ran)} skipped")
    print(RULE)
    print()
    print("  The abstention case is the point. Retrieval always returns")
    print("  something; similarity has no notion of \"nothing here matches\".")
    print("  The relevance floor supplies one, and it is calibrated against a")
    print("  hand-built adversarial set rather than guessed.")
    print()
    print("  Full numbers:  eval_harness/results/RESULTS.md")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
