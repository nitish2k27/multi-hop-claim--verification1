"""
Tier 2 entry point.

    python -m serve.cli "India's GDP grew 8% in 2024"
    python -m serve.cli "the moon is square in shape"      # abstains, 0 LLM calls
    python -m serve.cli "what time is it"                  # rejected at the gate

    python -m serve.cli --json "claim"        # the structured report
    python -m serve.cli --trace "claim"       # per-node progress
    python -m serve.cli --graph               # mermaid diagram, for the README

Exists before the API on purpose: it exercises the whole graph with no server,
no async and no SSE in the way, so M3's eval harness can drive the same code
path directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

from serve.graph import get_app, mermaid
from serve.retriever import IndexIncompatible, IndexUnavailable
from serve.schemas import new_state


def _force_utf8_stdio() -> None:
    """
    Make stdout/stderr UTF-8 regardless of the console's code page.

    A Windows console is cp1252 by default, which cannot encode most of what
    this system legitimately produces: a Hindi or Tamil report, a curly quote
    lifted from an article, or the narrow no-break space (U+202F) that some
    models put between a number and its unit. Without this the run completes,
    the verdict is correct, and then the final `print` raises
    UnicodeEncodeError — the work is done and thrown away at the last step.

    `errors="replace"` rather than "strict" so an unmappable glyph degrades to
    a question mark instead of losing the whole report.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _log_setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Chatty at INFO and they drown the run.
    for noisy in ("httpx", "chromadb", "sentence_transformers", "transformers",
                  "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # transformers draws a "Loading weights" bar per model — four of them, on
    # every run, interleaved with the trace output. Set before the first model
    # import so it takes effect.
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")


def _summarise(node: str, update: dict) -> str:
    """One line per node for --trace. Same shape the API's SSE emits."""
    if node == "adapt":
        if update.get("input_error"):
            return f"could not read input — {update['input_error'].splitlines()[0]}"
        kind = update.get("input_kind", "text")
        if kind == "text":
            return "text input"
        extracted = update.get("extracted_text") or ""
        return (f"{kind} → {len(extracted)} chars extracted"
                f"  ({update.get('source_file', '')})")
    if node == "translate":
        language = update.get("language", "en")
        english = update.get("claim_english", "")
        if language == "en":
            return "English — no translation needed"
        return f"detected {language} → EN: {english[:52]!r}"
    if node == "gate":
        return ("claim" if update.get("is_claim") else "NOT a claim") + \
               f" (p={update.get('claim_confidence', 0.0):.3f})"
    if node == "retrieve":
        stats = update.get("retrieval_stats", {})
        return (f"{stats.get('candidates', 0)} candidates -> "
                f"{len(update.get('evidence') or [])} above floor "
                f"{stats.get('floor', 0):.2f}")
    if node == "web":
        stats = update.get("retrieval_stats", {})
        if stats.get("web_skipped"):
            return f"skipped — {stats['web_skipped']}"
        if stats.get("web_error"):
            return f"FAILED — {stats['web_error']}"
        return (f"{stats.get('web_results', 0)} web results -> "
                f"{stats.get('web_after_floor', 0)} above floor "
                f"(top {stats.get('web_top_relevance', 0.0):.3f})")
    if node == "stance":
        counts = update.get("diagnostics", {}).get("stance_counts", {})
        return ", ".join(f"{k}={v}" for k, v in counts.items()) or "none"
    if node == "generate":
        report = update.get("report")
        return f"{report.verdict} @ {report.confidence}%" if report else "-"
    if node in ("abstain", "reject"):
        return update.get("outcome", node)
    if node == "render":
        return f"{len(update.get('rendered', ''))} chars"
    return ""


def run(claim: str, *, trace: bool, document: str | None = None) -> dict:
    """Invoke the graph, optionally printing each node as it completes."""
    app = get_app()
    # With a document, the *document* is the input to adapt and the claim is
    # what it should be checked against — so they swap roles here.
    state = (new_state(document, claim=claim) if document
             else new_state(claim))

    if not trace:
        return app.invoke(state)

    final: dict = dict(state)
    for chunk in app.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"  [{node:9s}] {_summarise(node, update)}", file=sys.stderr)
            final.update(update)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(prog="serve.cli", description=__doc__)
    parser.add_argument("claim", nargs="?", help="the claim to verify")
    parser.add_argument("--json", action="store_true",
                        help="print the structured report instead of markdown")
    parser.add_argument("--trace", action="store_true",
                        help="print each graph node as it completes")
    parser.add_argument("--graph", action="store_true",
                        help="print the mermaid diagram and exit")
    parser.add_argument("--no-web", action="store_true",
                        help="disable the web search fallback — index only. "
                             "Use this to see the pure abstention behaviour.")
    parser.add_argument("--document", metavar="PATH",
                        help="attach a PDF/DOCX as evidence for the claim. Its "
                             "credibility is capped — an uploaded file cannot "
                             "make its own assertion true.")
    parser.add_argument("--export", metavar="FMT",
                        help="also write these formats: html,docx,mp3")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Before any logging handler is created — logging binds the stream it finds.
    _force_utf8_stdio()
    _log_setup(args.verbose)

    # Mutating cfg rather than the environment: the settings object is the
    # single source of truth (see the invariants in CLAUDE.md), and the graph
    # reads this flag at routing time, so a per-run override works.
    if args.no_web:
        from core.config import cfg

        cfg.web_search_enabled = False

    if args.graph:
        print(mermaid())
        return 0

    if not args.claim:
        parser.print_help()
        return 1

    started = time.monotonic()
    try:
        final = run(args.claim, trace=args.trace, document=args.document)
    except (IndexUnavailable, IndexIncompatible) as exc:
        # A misconfigured or missing index is a setup problem with a known fix,
        # not a bug. The exception messages carry the command that resolves it,
        # so print those rather than a traceback the user has to read past.
        print(f"\n{exc}\n", file=sys.stderr)
        return 4
    except Exception as exc:
        # Groq's free tier is 100,000 tokens per DAY. Running out is expected
        # operationally, not a defect, and a 40-line traceback ending in a JSON
        # blob buries the one useful fact: when it resets. Everything up to the
        # LLM call already succeeded, so say that too.
        message = str(exc)
        if "rate_limit" in message or "429" in message:
            wait = re.search(r"try again in\s+((?:\d+h)?(?:\d+m)?[\d.]+s)", message)
            per_day = "per day" in message or "TPD" in message
            print(
                f"\nGroq {'daily quota' if per_day else 'rate limit'} reached."
                + (f" Resets in {wait.group(1)}." if wait else "")
                + "\n  Free tier is 100,000 tokens/day. Retrieval and stance "
                  "detection completed normally — only the report generation "
                  "call was refused."
                  "\n  Options: wait for the reset, or upgrade at "
                  "https://console.groq.com/settings/billing\n",
                file=sys.stderr,
            )
            return 5
        raise
    elapsed = time.monotonic() - started

    report = final.get("report")
    outcome = final.get("outcome", "unknown")
    diagnostics = final.get("diagnostics", {})

    if args.export:
        from serve.nodes.export import export

        produced = export(final, [f.strip() for f in args.export.split(",")])
        final["artifacts"] = {**final.get("artifacts", {}), **produced}

    if args.json:
        print(json.dumps({
            "claim": args.claim,
            "outcome": outcome,
            "report": report.model_dump() if report else None,
            "evidence": [
                {
                    "index": i,
                    "url": d.metadata.get("url"),
                    "domain": d.metadata.get("domain"),
                    "credibility": d.metadata.get("credibility"),
                    "relevance_score": d.metadata.get("relevance_score"),
                }
                for i, d in enumerate(final.get("evidence") or [], start=1)
            ],
            "stances": final.get("stances") or [],
            "retrieval_stats": final.get("retrieval_stats", {}),
            "diagnostics": diagnostics,
            "elapsed_seconds": round(elapsed, 2),
        }, indent=2, ensure_ascii=False))
    else:
        print()
        print(final.get("rendered", "(nothing rendered)"))

    for name, path in (final.get("artifacts") or {}).items():
        print(f"  {name:<5} → {path}", file=sys.stderr)

    print(
        f"\n[{outcome}]  {elapsed:.1f}s  ·  "
        f"{diagnostics.get('llm_calls', 0)} LLM call(s)",
        file=sys.stderr,
    )

    # Exit codes let the demo script and the eval harness branch without
    # parsing stdout. Non-zero here means "did not produce a verdict", not
    # "crashed" — both are legitimate outcomes.
    return {"verified": 0, "abstained": 2, "rejected": 3}.get(outcome, 1)


if __name__ == "__main__":
    raise SystemExit(main())
