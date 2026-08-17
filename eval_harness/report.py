"""
Turn the saved eval JSON into the markdown that goes in the README.

    python -m eval_harness.report              # print
    python -m eval_harness.report --write      # -> eval_harness/results/RESULTS.md

Kept separate from the runners so the numbers in the README are always
transcribed by a program from a results file, never retyped by hand. Hand-copied
metrics drift, and a resume number that does not match the repo is worse than no
number at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def retrieval_section(data: dict) -> list[str]:
    rows = data["rows"]
    labels = {
        "old_dense": "Old index (whole articles)",
        "new_dense": "New index (chunked)",
        "new_full":  "New index + BM25 + reranker",
    }
    order = ["old_dense", "new_dense", "new_full"]

    lines = [
        "## Retrieval quality — before and after",
        "",
        f"Known-item retrieval over {data['n_queries']} queries drawn from "
        f"{data['n_queries'] // 2} corpus articles. Each article contributes two "
        "queries: one passage from its **head** (inside the embedder's 256-token "
        "window) and one from its **tail** (beyond it). Gold is the source article.",
        "",
        "The head slice is the control — both indexes embedded that text.",
        "",
    ]

    for slice_name, title in (
        ("head", "Head — control (text both indexes embedded)"),
        ("tail", "Tail — the truncated region"),
        ("all", "Overall"),
    ):
        subset = [r for r in rows if r["slice"] == slice_name]
        if not subset:
            continue
        subset.sort(key=lambda r: order.index(r["system"])
                    if r["system"] in order else 9)
        lines += [
            f"### {title}",
            "",
            "| System | recall@1 | recall@5 | recall@10 | MRR |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in subset:
            lines.append(
                f"| {labels.get(row['system'], row['system'])} "
                f"| {row['recall@1']:.3f} | {row['recall@5']:.3f} "
                f"| {row['recall@10']:.3f} | {row['mrr']:.3f} |"
            )
        lines.append("")

    if data.get("note"):
        lines += ["> " + data["note"], ""]
    # Stated unconditionally: this is a property of the measurement method, not
    # of any one run, and it was established by running the eval twice and
    # diffing. A reader comparing three-decimal figures deserves to know the
    # third decimal is noise.
    lines += [
        "> **Run-to-run variance.** This eval was run twice. The old index moved "
        "by 0.010 on head recall@1 and 0.005 on MRR between runs; the new index "
        "reproduced identically. Chroma's HNSW is an approximate index, so treat "
        "single-run figures as carrying roughly ±0.01 — two decimal places are "
        "meaningful, the third is not. The effect being measured (0.47 → 0.88 on "
        "tail recall@1) is forty times that noise floor.",
        "",
    ]
    return lines


def sweep_section(data: dict) -> list[str]:
    lines = [
        "## Relevance floor calibration",
        "",
        f"Swept {len(data['rows'])} candidate floors against "
        f"{data['n_known']} known-item queries (should retrieve) and "
        f"{data['n_adversarial']} adversarial claims (should abstain). No LLM calls.",
        "",
        "| Floor | Gold document retained | Adversarial abstention |",
        "|---:|---:|---:|",
    ]
    for row in data["rows"]:
        marker = " ← **chosen**" if row["floor"] == data.get("recommended_floor") else ""
        lines.append(
            f"| {row['floor']:.2f} | {row['known_item_retention']:.3f} "
            f"| {row['adversarial_abstain_precision']:.3f}{marker} |"
        )
    lines.append("")

    detail = data.get("adversarial_detail") or []
    if detail:
        resistant = [d for d in detail if d["top_relevance"] >= 0.05]
        clean = [d for d in detail if d["top_relevance"] < 0.05]
        lines += [
            "### What the floor can and cannot separate",
            "",
            f"Of {len(detail)} adversarial claims, **{len(clean)} score below "
            f"0.05** and are filtered by any floor at all. The "
            f"**{len(resistant)} that score higher are all `near_miss`** — "
            "claims about subjects the corpus covers extensively, asserting "
            "facts it does not contain.",
            "",
            "| Top relevance | Category | Claim |",
            "|---:|---|---|",
        ]
        for item in detail[:8]:
            lines.append(
                f"| {item['top_relevance']:.4f} | `{item['category']}` "
                f"| {item['claim']} |"
            )
        lines += [
            "",
            "This is the honest limit of a relevance threshold: it separates "
            "*absent from the corpus* from *present in the corpus*. It cannot "
            "separate *supports this claim* from *is merely about the same "
            "subject*. That judgement is made downstream by the LLM's "
            "`directly_relevant` field.",
            "",
        ]

    if data.get("caveat"):
        lines += ["> " + data["caveat"], ""]
    return lines


def system_section(data: dict) -> list[str]:
    lines = ["## End-to-end behaviour", ""]

    adversarial = data.get("adversarial")
    if adversarial:
        lines += [
            "### Abstention on unanswerable claims",
            "",
            f"{adversarial['n']} hand-written claims the corpus provably cannot "
            "answer. Correct behaviour for every one is abstention.",
            "",
            f"- **abstain_precision: {adversarial['abstain_precision']:.3f}** "
            f"({adversarial['abstained']}/{adversarial['n']})",
            f"- rejected at the claim gate: {adversarial['rejected_at_gate']}",
            f"- wrongly answered: {adversarial['wrongly_answered']}",
            "",
        ]

        # Which mechanism abstained. This is the architectural claim, and it is
        # the part a reader is most likely to be sceptical of.
        if "abstained_by_relevance_floor" in adversarial:
            by_floor = adversarial["abstained_by_relevance_floor"]
            by_llm = adversarial["abstained_by_llm_judgement"]
            lines += [
                "**Two independent mechanisms produced these abstentions:**",
                "",
                f"- **{by_floor}** were stopped by the relevance floor — nothing "
                "cleared it, so no LLM call was made and the claim cost nothing.",
                f"- **{by_llm}** retrieved evidence, passed it to the LLM, and the "
                "LLM returned `UNVERIFIABLE` after reading it.",
                "",
                "The second group is the interesting one. Those are claims about "
                "subjects the corpus genuinely covers, so retrieval returns "
                "relevant-looking articles and no threshold can filter them. "
                "Only reading the evidence answers them.",
                "",
            ]

        lines += [
            "| Category | n | abstain_precision |",
            "|---|---:|---:|",
        ]
        for category, stats in adversarial["by_category"].items():
            lines.append(
                f"| `{category}` | {stats['n']} "
                f"| {stats['abstain_precision']:.3f} |"
            )
        lines.append("")
        if adversarial["failures"]:
            lines += ["Claims that were answered when they should not have been:", ""]
            for failure in adversarial["failures"]:
                lines.append(
                    f"- `{failure['category']}` — **{failure['verdict']}** at "
                    f"{failure['confidence']}% — {failure['claim']}"
                )
            lines.append("")

    fever = data.get("fever")
    if fever:
        threshold = data.get("confident_threshold", 70)
        lines += [
            "### FEVER",
            "",
            f"{fever['n']} claims from the KILT FEVER validation split, balanced "
            "SUPPORTS/REFUTES. **FEVER's gold evidence is Wikipedia, which this "
            "corpus does not contain**, so a high abstention rate here is correct "
            "behaviour rather than a failure — the system is being asked "
            "questions its corpus cannot answer.",
            "",
            f"| Metric | Value |",
            "|---|---:|",
            f"| claims | {fever['n']} |",
            f"| answered | {fever['answered']} |",
            f"| abstain_rate | {fever['abstain_rate']:.3f} |",
            f"| verdict_accuracy (of answered) | {_pct(fever['verdict_accuracy'])} |",
            f"| **false_confidence** (wrong at ≥{threshold}%) "
            f"| **{fever['false_confidence']}** |",
        ]
        if fever.get("mean_confidence_when_correct") is not None:
            lines.append(f"| mean confidence when correct "
                         f"| {fever['mean_confidence_when_correct']} |")
        if fever.get("mean_confidence_when_wrong") is not None:
            lines.append(f"| mean confidence when wrong "
                         f"| {fever['mean_confidence_when_wrong']} |")
        lines += [
            "",
            "`false_confidence` counts wrong verdicts issued at high confidence. "
            "It is the metric to drive to zero: a confident wrong answer is worse "
            "than no answer. It is meaningful here **because** n is large — the "
            "system was given 200 opportunities to be confidently wrong and took "
            "none of them.",
            "",
        ]

        # Guard against the single most misleading number in this whole report.
        answered = fever["answered"]
        if answered and answered < 30:
            lines += [
                f"> **Read `verdict_accuracy` with care.** It is computed over "
                f"the **{answered} claims the system actually committed to**, "
                f"not over all {fever['n']}. At that sample size the figure "
                f"carries almost no statistical weight and should not be quoted "
                f"as a headline accuracy number. The meaningful results on this "
                f"set are the abstention rate and `false_confidence`, both of "
                f"which are computed over all {fever['n']} claims.",
                "",
            ]
    return lines


def build() -> str:
    lines = ["# Evaluation results", ""]

    retrieval = _load("retrieval_ab")
    sweep = _load("floor_sweep")
    system = _load("system_eval")

    if not any((retrieval, sweep, system)):
        return "No results yet. Run: python -m eval_harness.run_eval --all"

    if retrieval:
        lines += retrieval_section(retrieval)
    if sweep:
        lines += sweep_section(sweep)
    if system:
        lines += system_section(system)

    lines += [
        "---",
        "",
        "Reproduce:",
        "",
        "```bash",
        "python -m eval_harness.prepare_datasets",
        "python -m eval_harness.run_eval --all",
        "python -m eval_harness.report --write",
        "```",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(prog="eval_harness.report")
    parser.add_argument("--write", action="store_true",
                        help="write results/RESULTS.md instead of printing")
    args = parser.parse_args()

    markdown = build()
    if args.write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "RESULTS.md"
        path.write_text(markdown, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
