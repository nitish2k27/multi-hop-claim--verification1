"""
Run the evaluations.

    python -m eval_harness.run_eval --retrieval     # A/B, no LLM, ~5 min
    python -m eval_harness.run_eval --sweep         # calibrate the floor, no LLM
    python -m eval_harness.run_eval --system        # full pipeline, uses Groq
    python -m eval_harness.run_eval --all

Results land in `eval_harness/results/` as JSON plus a markdown table ready to
paste into the README.

THREE RUNNERS, DELIBERATELY SEPARATE
------------------------------------
`--retrieval` and `--sweep` make **no LLM calls at all**. They are the ones that
produce the resume numbers and the calibrated floor, they are free, and they
finish in minutes — so they can be re-run after every retrieval change without
thinking about cost. `--system` is the slow, metered one.

That split is deliberate: the most valuable measurement in this project should
not be the one you are reluctant to run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from core.config import ROOT, cfg
from eval_harness.metrics import (
    AbstainScore,
    RetrievalScore,
    VerdictScore,
)
from eval_harness.prepare_datasets import DATA_DIR, read_jsonl

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Floors to sweep. Spans the placeholder (0.15) generously in both directions.
SWEEP_FLOORS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25,
                0.30, 0.35, 0.40, 0.50, 0.60]


def _log_setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "chromadb", "sentence_transformers", "transformers",
                  "urllib3", "httpcore", "serve", "core"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")


def _latest(pattern: str) -> Path | None:
    hits = sorted(DATA_DIR.glob(pattern))
    return hits[-1] if hits else None


def _save(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  results -> {path.relative_to(ROOT)}")
    return path


def _progress(i: int, total: int, started: float, label: str = "") -> None:
    if i % 10 and i != total:
        return
    elapsed = time.monotonic() - started
    rate = i / elapsed if elapsed else 0
    remaining = (total - i) / rate if rate else 0
    print(f"    {i:>4}/{total}  {elapsed:5.0f}s elapsed, "
          f"~{remaining:4.0f}s left  {label}", file=sys.stderr)


# ── Retrieval A/B ────────────────────────────────────────────────────────────

def run_retrieval(skip_baseline: bool = False) -> dict:
    """
    Old index vs new index on the known-item set.

    Three systems are measured, and the ordering of the columns is the argument:

      old_dense    the pre-rebuild index. Dense only, whole articles embedded.
      new_dense    the rebuilt index, dense only. **Only the chunking changed.**
      new_full     the shipped pipeline: dense + BM25 fusion + cross-encoder.

    `old_dense -> new_dense` is the truncation fix, isolated. `new_dense ->
    new_full` is everything added on top. Reporting only the end-to-end delta
    would credit the chunking fix for the reranker's contribution too.
    """
    from serve.retriever import get_retriever

    path = _latest("known_item_*.jsonl")
    if path is None:
        print("No known-item dataset. Run: python -m eval_harness.prepare_datasets")
        return {}

    queries = read_jsonl(path)
    print(f"Retrieval A/B over {len(queries)} queries from {path.name}")

    retriever = get_retriever()
    baseline = None
    if not skip_baseline:
        from eval_harness.baseline import BaselineUnavailable, get_baseline

        try:
            baseline = get_baseline()
            print(f"  baseline: {baseline.count():,} articles in data/chroma_db")
        except BaselineUnavailable as exc:
            print(f"  {exc}")
            return {}

    systems = ["new_dense", "new_full"] + (["old_dense"] if baseline else [])
    slices = sorted({q["slice"] for q in queries})
    scores = {
        (system, slice_name): RetrievalScore(system, slice_name)
        for system in systems for slice_name in slices + ["all"]
    }

    K = 10
    started = time.monotonic()
    for i, item in enumerate(queries, start=1):
        query, gold, slice_name = item["query"], item["gold_url"], item["slice"]

        if baseline:
            urls = [h["url"] for h in baseline.search(query, K)]
            scores[("old_dense", slice_name)].add(urls, gold)
            scores[("old_dense", "all")].add(urls, gold)

        # Dense-only on the new index. Chunks are deduplicated to article level
        # before scoring, so a k of 10 means ten distinct articles for every
        # system — otherwise the chunked index would get ten shots at the same
        # document and the comparison would be meaningless.
        dense = retriever.dense_only(query, k=K * 4)
        scores[("new_dense", slice_name)].add(_unique_urls(dense, K), gold)
        scores[("new_dense", "all")].add(_unique_urls(dense, K), gold)

        full = retriever.rank_chunks(query, k=K * 4)
        scores[("new_full", slice_name)].add(_unique_urls(full, K), gold)
        scores[("new_full", "all")].add(_unique_urls(full, K), gold)

        _progress(i, len(queries), started)

    rows = [scores[key].as_dict() for key in scores if scores[key].n]
    payload = {
        "dataset": path.name,
        "n_queries": len(queries),
        "k": K,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "rows": rows,
        # Measured, not assumed: running this twice moved the old index's head
        # recall@1 by 0.010 and its MRR by 0.005, while the new index reproduced
        # exactly. Chroma's HNSW search is approximate, so single-run figures
        # carry roughly +/-0.01 of jitter. Quote two decimals, not three.
        "run_to_run_variance": (
            "~0.01 on the old index (Chroma HNSW is an approximate index). "
            "The new index reproduced identically across runs. This is far "
            "below the effect being measured (0.47 -> 0.88 on tail recall@1), "
            "but it means the third decimal place is not meaningful."
        ),
        "note": (
            "The head slice is the control: both indexes embedded that text, "
            "and they score within 0.01 of each other there. The tail slice is "
            "the truncated region. Truncation did NOT make tail content "
            "unfindable — news articles are topically coherent, so the head "
            "vector still represents the whole article well enough to retrieve "
            "it by subject. What it destroyed was ranking precision: on tail "
            "queries the old index put the correct article first only 47% of "
            "the time, against 88% for the same embedder over chunks."
        ),
    }
    _print_retrieval_table(rows, slices)
    _save("retrieval_ab", payload)
    return payload


def _unique_urls(documents, k: int) -> list[str]:
    """Collapse chunks to distinct article URLs, preserving rank order."""
    seen: list[str] = []
    for doc in documents:
        url = doc.metadata.get("url")
        if url and url not in seen:
            seen.append(url)
        if len(seen) >= k:
            break
    return seen


def _print_retrieval_table(rows: list[dict], slices: list[str]) -> None:
    order = {"old_dense": 0, "new_dense": 1, "new_full": 2}
    print()
    for slice_name in slices + ["all"]:
        subset = sorted(
            (r for r in rows if r["slice"] == slice_name),
            key=lambda r: order.get(r["system"], 9),
        )
        if not subset:
            continue
        print(f"  slice: {slice_name}   (n={subset[0]['n']})")
        print(f"    {'system':<12}{'recall@1':>10}{'recall@5':>10}"
              f"{'recall@10':>11}{'MRR':>8}")
        for row in subset:
            print(f"    {row['system']:<12}{row['recall@1']:>10.3f}"
                  f"{row['recall@5']:>10.3f}{row['recall@10']:>11.3f}"
                  f"{row['mrr']:>8.3f}")
        print()


# ── Floor sweep ──────────────────────────────────────────────────────────────

def run_sweep() -> dict:
    """
    Calibrate `RELEVANCE_FLOOR` — no LLM calls.

    Scores every query in both sets **once**, caching the top reranked score,
    then evaluates each candidate floor against those cached scores. Sweeping
    thirteen floors therefore costs one pass, not thirteen.

    The trade-off being measured is direct: a higher floor abstains more often,
    which is good on the adversarial set and bad on the known-item set. The
    recommendation is the highest floor that keeps known-item retention at or
    above `--min-retention`.
    """
    from serve.retriever import get_retriever

    known_path = _latest("known_item_*.jsonl")
    adversarial_path = DATA_DIR / "adversarial_20.jsonl"
    if known_path is None or not adversarial_path.exists():
        print("Datasets missing. Run: python -m eval_harness.prepare_datasets")
        return {}

    retriever = get_retriever()
    known = read_jsonl(known_path)
    adversarial = read_jsonl(adversarial_path)

    print(f"Floor sweep: scoring {len(known)} known-item + "
          f"{len(adversarial)} adversarial queries once")

    started = time.monotonic()

    def top_score(query: str) -> float:
        ranked = retriever.rank_chunks(query, k=1)
        return ranked[0].metadata.get("relevance_score", 0.0) if ranked else 0.0

    # For known-item, what matters is the score of the GOLD document, not the
    # top score — a floor that keeps some other article is not "retention".
    known_gold_scores: list[float] = []
    for i, item in enumerate(known, start=1):
        ranked = retriever.rank_chunks(item["query"], k=20)
        gold = 0.0
        for doc in ranked:
            if doc.metadata.get("url") == item["gold_url"]:
                gold = doc.metadata.get("relevance_score", 0.0)
                break
        known_gold_scores.append(gold)
        _progress(i, len(known), started, "known-item")

    adversarial_top: list[float] = []
    adversarial_detail: list[dict] = []
    for i, item in enumerate(adversarial, start=1):
        score = top_score(item["claim"])
        adversarial_top.append(score)
        adversarial_detail.append({
            "id": item["id"],
            "category": item.get("category"),
            "claim": item["claim"],
            "top_relevance": round(score, 4),
        })
        _progress(i, len(adversarial), started, "adversarial")

    # Which claims resist every floor, and what category are they? This is the
    # diagnostic that says whether the floor is the right tool at all.
    adversarial_detail.sort(key=lambda d: -d["top_relevance"])
    print("\n    adversarial claims by top relevance score:")
    for detail in adversarial_detail:
        print(f"      {detail['top_relevance']:.4f}  [{detail['category']:<13}] "
              f"{detail['claim'][:58]}")

    rows = []
    for floor in SWEEP_FLOORS:
        retention = sum(1 for s in known_gold_scores if s >= floor) / max(len(known_gold_scores), 1)
        abstain = sum(1 for s in adversarial_top if s < floor) / max(len(adversarial_top), 1)
        rows.append({
            "floor": floor,
            "known_item_retention": round(retention, 4),
            "adversarial_abstain_precision": round(abstain, 4),
        })

    print(f"\n    {'floor':>7}{'gold retained':>16}{'adversarial abstain':>22}")
    for row in rows:
        print(f"    {row['floor']:>7.2f}{row['known_item_retention']:>16.3f}"
              f"{row['adversarial_abstain_precision']:>22.3f}")

    ordered = sorted(known_gold_scores)
    percentiles = {
        f"p{p}": round(ordered[min(int(len(ordered) * p / 100), len(ordered) - 1)], 4)
        for p in (1, 5, 10, 25, 50)
    } if ordered else {}

    print(f"\n    gold-document score percentiles: {percentiles}")
    print(f"    adversarial top scores, highest 5: "
          f"{[round(s, 4) for s in sorted(adversarial_top, reverse=True)[:5]]}")

    payload = {
        "n_known": len(known),
        "n_adversarial": len(adversarial),
        "current_floor": cfg.relevance_floor,
        "rows": rows,
        "gold_score_percentiles": percentiles,
        "adversarial_detail": adversarial_detail,
        "adversarial_top_scores": sorted(adversarial_top, reverse=True),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "caveat": (
            "Known-item queries are verbatim passages from the corpus, so their "
            "gold documents score higher than a paraphrased real-world claim "
            "would. The retention side of this trade-off is therefore "
            "optimistic, and the recommended floor should be read as an upper "
            "bound. The adversarial side uses real claim text and is realistic."
        ),
    }
    _recommend(rows, payload)
    _save("floor_sweep", payload)
    return payload


def _recommend(rows: list[dict], payload: dict, min_retention: float = 0.99) -> None:
    """
    Recommend the **lowest** floor that captures most of the available benefit.

    Not the highest-scoring floor. Two reasons, both from the measured data:

    * Retention here is optimistic (verbatim queries), so a floor that looks
      free may not be. When one side of a trade-off is measured optimistically,
      err toward the other side.
    * Abstention barely moves across the whole range. Pushing the floor up buys
      a few percent on the adversarial set while costing real recall on
      paraphrased claims that the known-item set cannot model.

    So: find the best abstention any floor achieves, then take the lowest floor
    that gets within `tolerance` of it.
    """
    viable = [r for r in rows if r["known_item_retention"] >= min_retention]
    if not viable:
        print(f"\n  No floor retains {min_retention:.0%} of gold documents.")
        return

    best_abstain = max(r["adversarial_abstain_precision"] for r in viable)
    tolerance = 0.05
    cheapest = min(
        (r for r in viable
         if r["adversarial_abstain_precision"] >= best_abstain - tolerance),
        key=lambda r: r["floor"],
    )

    payload["recommended_floor"] = cheapest["floor"]
    payload["best_achievable_abstain"] = best_abstain

    print(f"\n  Recommended RELEVANCE_FLOOR = {cheapest['floor']}")
    print(f"    retains  {cheapest['known_item_retention']:.1%} of gold documents")
    print(f"    abstains {cheapest['adversarial_abstain_precision']:.1%} "
          f"of adversarial claims "
          f"(best any floor achieves: {best_abstain:.1%})")
    print(f"    chosen as the lowest floor within {tolerance:.0%} of the best, "
          f"because retention is measured optimistically")
    if cheapest["floor"] != cfg.relevance_floor:
        print(f"    currently {cfg.relevance_floor} — set "
              f"RELEVANCE_FLOOR={cheapest['floor']} in .env")
    else:
        print(f"    this matches the current setting")

    resistant = 1.0 - best_abstain
    if resistant > 0.01:
        print(f"\n  {resistant:.0%} of adversarial claims abstain at NO floor. "
              f"See the per-claim scores above —")
        print(f"  if those are 'near_miss', the floor is not the mechanism that "
              f"can catch them.")


# ── Full-system eval ─────────────────────────────────────────────────────────

class QuotaExhausted(RuntimeError):
    """The API's daily budget is gone. Retrying will not help."""


def _retry_after_seconds(message: str) -> float | None:
    """Pull Groq's 'try again in 43m37.056s' out of the error text."""
    match = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", message)
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    return minutes * 60 + float(match.group(2))


def _verify(claim: str) -> dict:
    """
    One run through the compiled graph, with retry on *genuinely* transient errors.

    A per-minute rate limit clears in seconds and is worth retrying. A **daily**
    token cap does not — Groq's free tier is 100k tokens/day, and when it is gone
    the error says "try again in 43m". Retrying that on a 5-second backoff burns
    two more failed calls and then raises anyway, which is how the first version
    of this function lost a whole eval run. Daily exhaustion now fails fast with
    a message that says when to come back.
    """
    from serve.graph import get_app
    from serve.schemas import new_state

    for attempt in range(3):
        try:
            return get_app().invoke(new_state(claim))
        except Exception as exc:
            message = str(exc).lower()

            if "per day" in message or "tpd" in message or "rpd" in message:
                wait = _retry_after_seconds(message)
                raise QuotaExhausted(
                    "Groq daily quota exhausted"
                    + (f" — resets in about {wait / 60:.0f} minutes."
                       if wait else ".")
                    + "\n  Free tier is 100,000 tokens/day. Options: wait, use "
                      "--limit to shrink the run, or upgrade at "
                      "console.groq.com/settings/billing"
                ) from exc

            transient = any(
                marker in message
                for marker in ("rate limit", "429", "timeout", "503",
                               "overloaded", "connection")
            )
            if not transient or attempt == 2:
                raise

            # Honour the server's own retry hint when it gives one.
            wait = _retry_after_seconds(message) or 5 * (attempt + 1)
            wait = min(wait, 60)
            print(f"      transient error, retrying in {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run_system(limit: int | None, threshold: int,
               skip_fever: bool = False, web: bool = False) -> dict:
    """
    Drive the full graph over FEVER and the adversarial set. **Uses Groq.**

    Abstentions cost nothing, so the real API spend is only the claims that
    retrieve evidence — on this corpus that is a small minority of FEVER.

    WEB SEARCH IS OFF BY DEFAULT HERE, AND THAT IS A MEASUREMENT DECISION
    --------------------------------------------------------------------
    M4 added a live web fallback. Leaving it on would change what
    `abstain_precision` means: the adversarial set was written as claims *this
    corpus* cannot answer, and most of them the open web answers easily. With
    web search on, correctly answering "Napoleon was born in Corsica" counts as
    a failure against a set whose expected outcome is abstention — the metric
    would report a regression for behaviour that actually improved.

    So the default measures the property the architecture claims: given a fixed,
    inspectable corpus, does the system decline when it should? Pass `--web` to
    measure end-to-end product behaviour instead, and read the abstention
    numbers as "how often did BOTH sources come up empty".

    Practical reason too: the adversarial + FEVER sets abstain on ~211 of 220
    claims, so a web-enabled run spends ~211 Tavily searches — a fifth of the
    free tier's monthly quota, per run.
    """
    fever_path = _latest("fever_dev_*.jsonl")
    adversarial_path = DATA_DIR / "adversarial_20.jsonl"

    cfg.web_search_enabled = web
    payload: dict = {
        "confident_threshold": threshold,
        "web_search_enabled": web,
        "measures": (
            "end-to-end product behaviour: abstention means BOTH the indexed "
            "corpus and a live web search came up empty"
            if web else
            "corpus-only behaviour: abstention means the indexed corpus could "
            "not answer. This is the number the two-tier architecture claims."
        ),
    }
    print(f"  web search fallback: {'ENABLED' if web else 'disabled'}")
    if web:
        print("  NOTE: ~211 of 220 claims abstain on the corpus, so this run "
              "will spend ~211 Tavily searches.")
    started = time.monotonic()

    # -- adversarial first: it is small, and if abstention is broken there is
    #    no point spending 200 FEVER calls to find out.
    if adversarial_path.exists():
        items = read_jsonl(adversarial_path)
        print(f"\nAdversarial set ({len(items)} claims — all should abstain)")
        score = AbstainScore()
        try:
            for i, item in enumerate(items, start=1):
                final = _verify(item["claim"])
                report = final.get("report")
                score.add(
                    item=item,
                    outcome=final.get("outcome", "?"),
                    verdict=report.verdict if report else None,
                    confidence=report.confidence if report else 0,
                    evidence_count=len(final.get("evidence") or []),
                )
                _progress(i, len(items), started, "adversarial")
        except QuotaExhausted as exc:
            # Keep what was scored. A partial result is still information, and
            # losing 19 completed verifications because the 20th hit a daily cap
            # is pure waste — that happened once already.
            print(f"\n  {exc}", file=sys.stderr)
            print(f"  Keeping {score.n} completed results.", file=sys.stderr)
            payload["adversarial_incomplete"] = True
        payload["adversarial"] = score.as_dict()
        if score.n:
            _print_abstain(payload["adversarial"])

    # -- FEVER. Re-running this costs ~17 minutes and real API calls, so it can
    #    be skipped when only the (small, fast) adversarial set has changed.
    if fever_path and not skip_fever:
        items = read_jsonl(fever_path)
        if limit:
            items = items[:limit]
        print(f"\nFEVER ({len(items)} claims from {fever_path.name})")
        print("  Gold evidence is Wikipedia, which this corpus does not contain,")
        print("  so a high abstain rate here is CORRECT behaviour, not a failure.")
        score = VerdictScore(confident_threshold=threshold)
        details = []
        try:
            for i, item in enumerate(items, start=1):
                final = _verify(item["claim"])
                report = final.get("report")
                outcome = final.get("outcome", "?")
                score.add(
                    outcome=outcome,
                    verdict=report.verdict if report else None,
                    confidence=report.confidence if report else 0,
                    gold_label=item.get("gold_label"),
                )
                details.append({
                    "id": item["id"],
                    "claim": item["claim"],
                    "gold": item.get("gold_label"),
                    "outcome": outcome,
                    "verdict": report.verdict if report else None,
                    "confidence": report.confidence if report else 0,
                    "evidence": len(final.get("evidence") or []),
                })
                _progress(i, len(items), started, "fever")
        except QuotaExhausted as exc:
            print(f"\n  {exc}", file=sys.stderr)
            print(f"  Keeping {score.n} of {len(items)} results.", file=sys.stderr)
            payload["fever_incomplete"] = True
            payload["fever_completed"] = score.n
        payload["fever"] = score.as_dict()
        payload["fever_details"] = details
        if score.n:
            _print_verdicts(payload["fever"])

    payload["elapsed_seconds"] = round(time.monotonic() - started, 1)

    # Separate files per mode. A web-enabled run measures something different
    # from a corpus-only run, so overwriting one with the other would silently
    # replace a number with a differently-defined number under the same name.
    name = "system_eval_web" if web else "system_eval"

    # Merge rather than overwrite, so an adversarial-only re-run does not
    # silently delete the FEVER results from the last full pass.
    if skip_fever:
        previous = RESULTS_DIR / f"{name}.json"
        if previous.exists():
            old = json.loads(previous.read_text(encoding="utf-8"))
            for key in ("fever", "fever_details"):
                if key in old and key not in payload:
                    payload[key] = old[key]
            payload["fever_from_previous_run"] = True

    _save(name, payload)
    return payload


def _print_abstain(result: dict) -> None:
    print(f"\n    abstain_precision   {result['abstain_precision']:.3f}  "
          f"({result['abstained']}/{result['n']})")
    print(f"      via relevance floor (no LLM call)  "
          f"{result['abstained_by_relevance_floor']}")
    print(f"      via LLM judgement  (evidence read) "
          f"{result['abstained_by_llm_judgement']}")
    print(f"    rejected at gate    {result['rejected_at_gate']}")
    print(f"    wrongly answered    {result['wrongly_answered']}")
    print("    by category:")
    for category, stats in result["by_category"].items():
        print(f"      {category:<14} {stats['abstain_precision']:.3f}  "
              f"(n={stats['n']})")
    for failure in result["failures"]:
        print(f"      FAILED  [{failure['category']}] {failure['verdict']} "
              f"@{failure['confidence']}%  {failure['claim'][:60]}")


def _print_verdicts(result: dict) -> None:
    print(f"\n    n                   {result['n']}")
    print(f"    answered            {result['answered']}")
    print(f"    abstain_rate        {result['abstain_rate']:.3f}")
    accuracy = result["verdict_accuracy"]
    print(f"    verdict_accuracy    "
          f"{f'{accuracy:.3f}' if accuracy is not None else 'n/a (nothing answered)'}")
    print(f"    false_confidence    {result['false_confidence']}   "
          f"(wrong verdicts at >= {70}% confidence — target 0)")
    if result["mean_confidence_when_correct"] is not None:
        print(f"    mean confidence when correct  "
              f"{result['mean_confidence_when_correct']}")
    if result["mean_confidence_when_wrong"] is not None:
        print(f"    mean confidence when wrong    "
              f"{result['mean_confidence_when_wrong']}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(prog="eval_harness.run_eval",
                                     description=__doc__)
    parser.add_argument("--retrieval", action="store_true",
                        help="old vs new index recall A/B (no LLM)")
    parser.add_argument("--sweep", action="store_true",
                        help="calibrate RELEVANCE_FLOOR (no LLM)")
    parser.add_argument("--system", action="store_true",
                        help="full graph over FEVER + adversarial (uses Groq)")
    parser.add_argument("--all", action="store_true", help="all three")
    parser.add_argument("--limit", type=int,
                        help="cap FEVER claims, for a quick pass")
    parser.add_argument("--skip-fever", action="store_true",
                        help="run only the adversarial set; keeps the previous "
                             "FEVER results in the saved output")
    parser.add_argument("--web", action="store_true",
                        help="enable the M4 web fallback during --system. Off by "
                             "default: the adversarial set measures corpus-only "
                             "abstention, and a web-enabled run spends ~211 "
                             "Tavily searches. Results save separately.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="run without data/chroma_db")
    parser.add_argument("--confident-threshold", type=int, default=70,
                        help="confidence at or above which a wrong verdict "
                             "counts as false confidence")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _log_setup(args.verbose)

    if not any((args.retrieval, args.sweep, args.system, args.all)):
        parser.print_help()
        return 1

    if args.retrieval or args.all:
        run_retrieval(skip_baseline=args.skip_baseline)
    if args.sweep or args.all:
        run_sweep()
    if args.system or args.all:
        run_system(limit=args.limit, threshold=args.confident_threshold,
                   skip_fever=args.skip_fever, web=args.web)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
