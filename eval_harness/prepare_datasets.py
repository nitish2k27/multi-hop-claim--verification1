"""
Build the three evaluation datasets.

    python -m eval_harness.prepare_datasets            # all three
    python -m eval_harness.prepare_datasets --articles 150

WHY THERE ARE THREE, AND WHY recall@5 IS NOT MEASURED ON FEVER
---------------------------------------------------------------
BUILD_PLAN §M3 defines `recall@5` as "fraction of FEVER claims where gold
evidence is in the top 5". That cannot be measured here. FEVER's gold evidence
is Wikipedia sentences, keyed by `fever_page_id`; this corpus is 1,687 news
articles containing no Wikipedia at all. The metric would return 0.00 for the
old index and 0.00 for the new one — a before/after table of two zeros, which
proves nothing about the change that was made.

So retrieval quality is measured on a **known-item** set derived from the corpus
itself, and FEVER is kept for the questions it *can* answer:

  known_item_*.jsonl   query = a passage from a real article
                       gold  = that article's URL
                       -> recall@k, MRR, and the old-vs-new A/B

  fever_dev_*.jsonl    200 real checkable claims whose evidence is NOT in the
                       corpus. Correct behaviour is overwhelmingly abstention.
                       -> abstain rate, and for anything it does answer,
                          verdict accuracy and false-confidence against
                          FEVER's SUPPORTS/REFUTES gold

  adversarial_20.jsonl 20 hand-written claims the corpus genuinely cannot
                       answer -> abstain_precision

THE HEAD/TAIL SPLIT IS THE EXPERIMENT
-------------------------------------
`all-MiniLM-L6-v2` truncates at 256 tokens (~1,000 characters). The old index
embedded whole articles, so everything past that point was never in a vector —
median article is 2,996 chars, so roughly two thirds of it was invisible.

Each sampled article therefore yields two queries:

    head  passage starting before char 900   <- the old index DID embed this
    tail  passage starting after  char 1500  <- the old index did NOT

`head` is the control. If the old index scores well on `head` and badly on
`tail`, the gap is the truncation bug and nothing else — same articles, same
embedder, same queries. Without the control, a `tail` result alone would just be
two systems differing for unknown reasons.

Tail recall for the old index is expected to be near zero *by construction*.
That is the finding being quantified, not a rigged comparison — and the eval
report says so out loud.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

import pandas as pd

from core.config import ROOT, cfg

DATA_DIR = Path(__file__).resolve().parent / "datasets"

# MiniLM's ~256-token window is about 1,000 characters. The bands are set either
# side of it with a margin so a passage cannot straddle the boundary.
HEAD_MAX_START = 900
TAIL_MIN_START = 1500

# A query has to be specific enough to identify one article out of 1,687, and
# short enough to look like something a person would type.
MIN_QUERY_CHARS = 110
MAX_QUERY_CHARS = 340
MIN_CONTENT_WORDS = 10

# Navigation furniture and syndication boilerplate. These appear in many
# articles, so a query drawn from one identifies nothing.
_BOILERPLATE = re.compile(
    r"(subscribe|newsletter|sign up|click here|read more|follow us|all rights "
    r"reserved|copyright|advertisement|cookie|privacy policy|terms of service|"
    r"share this|photo:|image:|getty images|reuters/|©)",
    re.IGNORECASE,
)

# Deliberately simple: no nltk, so no punkt download and no network at prep time.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _sentences_with_offsets(text: str) -> list[tuple[int, str]]:
    """Split into sentences, keeping each one's character offset in the article."""
    out: list[tuple[int, str]] = []
    cursor = 0
    for piece in _SENTENCE_END.split(text):
        start = text.find(piece, cursor)
        if start < 0:
            start = cursor
        out.append((start, piece.strip()))
        cursor = start + len(piece)
    return out


def _usable(sentence: str) -> bool:
    """Is this sentence specific enough to serve as a known-item query?"""
    if not (MIN_QUERY_CHARS <= len(sentence) <= MAX_QUERY_CHARS):
        return False
    if _BOILERPLATE.search(sentence):
        return False
    return len(_WORD.findall(sentence)) >= MIN_CONTENT_WORDS


def _pick_passage(
    sentences: list[tuple[int, str]], *, lo: int, hi: int | None
) -> tuple[int, str] | None:
    """
    First usable sentence whose start offset falls in [lo, hi).

    Adjacent sentences are joined when one alone is too short, which keeps
    otherwise-good articles in the sample instead of discarding them for having
    terse prose.
    """
    for i, (start, sentence) in enumerate(sentences):
        if start < lo or (hi is not None and start >= hi):
            continue
        if _usable(sentence):
            return start, sentence
        # Try joining with the next sentence.
        if i + 1 < len(sentences):
            joined = f"{sentence} {sentences[i + 1][1]}"
            if _usable(joined):
                return start, joined
    return None


# ── Known-item set ───────────────────────────────────────────────────────────

def build_known_item(n_articles: int, seed: int = 20260817) -> list[dict]:
    """Sample articles and emit a head query and a tail query for each."""
    frame = pd.read_csv(cfg.source_csv)
    frame = frame.dropna(subset=["text", "url"])

    # Only articles long enough to *have* a tail. Selecting on length is
    # required by the design, not a convenience — an article shorter than the
    # embedder's window has no truncated region to test.
    long_enough = frame[frame["text"].str.len() >= TAIL_MIN_START + 600]
    print(f"  {len(long_enough):,} of {len(frame):,} articles are long enough "
          f"to have a truncated tail")

    rows = long_enough.sample(
        n=min(n_articles * 3, len(long_enough)), random_state=seed
    ).to_dict("records")

    records: list[dict] = []
    kept = 0
    for row in rows:
        if kept >= n_articles:
            break
        sentences = _sentences_with_offsets(str(row["text"]))

        head = _pick_passage(sentences, lo=0, hi=HEAD_MAX_START)
        tail = _pick_passage(sentences, lo=TAIL_MIN_START, hi=None)
        if not head or not tail:
            continue  # need both, or the article cannot serve as its own control

        kept += 1
        for slice_name, (offset, passage) in (("head", head), ("tail", tail)):
            records.append({
                "id": f"ki-{kept:04d}-{slice_name}",
                "slice": slice_name,
                "query": passage,
                "gold_url": row["url"],
                "char_offset": offset,
                "article_chars": len(str(row["text"])),
                "domain": row.get("domain"),
            })

    print(f"  built {len(records)} queries from {kept} articles "
          f"({kept} head + {kept} tail)")
    return records


# ── FEVER ────────────────────────────────────────────────────────────────────

def _find_kilt_fever() -> Path | None:
    """
    Locate the cached KILT FEVER validation parquet.

    KILT is used rather than raw FEVER because the `fever` dataset in the cache
    is a loading *script* with no data — it would need the network. KILT ships
    parquet, so preparation stays offline like everything else in this project.
    """
    patterns = [
        r"C:\Users\**\.cache\huggingface\hub\datasets--kilt_tasks\snapshots\*\fever\validation-*.parquet",
        str(Path.home() / ".cache/huggingface/hub/datasets--kilt_tasks/snapshots/*/fever/validation-*.parquet"),
        "D:/hf-cache/hub/datasets--kilt_tasks/snapshots/*/fever/validation-*.parquet",
    ]
    for pattern in patterns:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return Path(hits[0])
    return None


def build_fever(n: int, seed: int = 20260817) -> list[dict]:
    """
    Sample FEVER claims with their SUPPORTS/REFUTES gold labels.

    KILT's FEVER split drops NOT ENOUGH INFO, so the gold set is two-class. That
    is worth stating plainly: this set measures whether the system gets a
    *decidable* claim right, or correctly declines. It cannot measure NEI
    detection, and the eval report does not pretend otherwise.
    """
    path = _find_kilt_fever()
    if path is None:
        print("  KILT FEVER parquet not found in the HuggingFace cache.")
        print("  Skipping — retrieval and adversarial evals do not need it.")
        return []

    print(f"  reading {path.name}")
    frame = pd.read_parquet(path)

    records: list[dict] = []
    for row in frame.to_dict("records"):
        output = row.get("output")
        if output is None or len(output) == 0:
            continue
        answer = output[0].get("answer")
        if answer not in ("SUPPORTS", "REFUTES"):
            continue
        claim = str(row["input"]).strip()
        if len(claim) < 15:
            continue

        page = ""
        try:
            page = output[0]["provenance"][0]["meta"]["fever_page_id"]
        except (KeyError, IndexError, TypeError):
            pass

        records.append({
            "id": f"fever-{row['id']}",
            "claim": claim,
            "gold_label": answer,
            # Recorded for provenance only. It names a Wikipedia page, which is
            # exactly why recall cannot be scored against this corpus.
            "gold_wikipedia_page": page,
        })

    random.Random(seed).shuffle(records)
    balanced = _balance(records, n)
    counts: dict[str, int] = {}
    for record in balanced:
        counts[record["gold_label"]] = counts.get(record["gold_label"], 0) + 1
    print(f"  sampled {len(balanced)} claims: {counts}")
    return balanced


def _balance(records: list[dict], n: int) -> list[dict]:
    """Equal SUPPORTS/REFUTES, so accuracy cannot be gamed by a constant answer."""
    supports = [r for r in records if r["gold_label"] == "SUPPORTS"]
    refutes = [r for r in records if r["gold_label"] == "REFUTES"]
    half = n // 2
    return supports[:half] + refutes[: n - half]


# ── Adversarial ──────────────────────────────────────────────────────────────
#
# Hand-written, and every one is a claim the corpus genuinely cannot speak to.
# The correct behaviour for all twenty is abstention.
#
# Graded on purpose. The absurd ones are easy — nothing in a news corpus is
# topically near "the moon is square". The `near_miss` ones are the real test:
# they are *about* subjects the corpus covers extensively (India's economy,
# Apple, Netflix, TechCrunch) but assert specific facts it does not contain.
# Those are where a relevance floor alone fails, because retrieval will return
# genuinely on-topic articles with high scores — exactly what was observed at M2
# with the factcheck.org economy articles.

ADVERSARIAL: list[dict] = [
    # -- physically absurd; retrieval should find nothing topical at all
    {"claim": "The moon is square in shape.",
     "category": "absurd"},
    {"claim": "Water boils at 40 degrees Celsius at sea level.",
     "category": "absurd"},
    {"claim": "The Pacific Ocean is smaller than Lake Superior.",
     "category": "absurd"},
    {"claim": "Adult humans have three hearts and two spines.",
     "category": "absurd"},

    # -- true, checkable, and simply not in a 2024-2026 news corpus
    {"claim": "The Treaty of Westphalia was signed in 1648.",
     "category": "absent_true"},
    {"claim": "Napoleon Bonaparte was born on the island of Corsica.",
     "category": "absent_true"},
    {"claim": "Insulin was first isolated at the University of Toronto in 1921.",
     "category": "absent_true"},
    {"claim": "The human genome contains roughly 20,000 protein-coding genes.",
     "category": "absent_true"},

    # -- false, checkable, and equally absent
    {"claim": "The Great Wall of China is visible from the Moon with the naked eye.",
     "category": "absent_false"},
    {"claim": "Vitamin C reliably cures the common cold within 24 hours.",
     "category": "absent_false"},
    {"claim": "Sweden adopted the euro as its official currency in 2023.",
     "category": "absent_false"},
    {"claim": "The population of Reykjavik exceeded two million people in 2024.",
     "category": "absent_false"},

    # -- NEAR MISS: on-topic for this corpus, but the specific fact is absent.
    #    These are the ones that matter. Retrieval will score highly here.
    {"claim": "India's GDP contracted by 12 percent during the 2019 fiscal year.",
     "category": "near_miss"},
    {"claim": "Apple sold exactly 41 million iPhones in the first quarter of 2019.",
     "category": "near_miss"},
    {"claim": "TechCrunch was founded in 1998 by Michael Arrington.",
     "category": "near_miss"},
    {"claim": "Netflix reported 500 million paying subscribers in 2023.",
     "category": "near_miss"},
    {"claim": "OpenAI was founded in 2010 as a for-profit hardware company.",
     "category": "near_miss"},
    {"claim": "Bitcoin traded above 500,000 US dollars in March 2024.",
     "category": "near_miss"},
    {"claim": "The Guardian newspaper published its first edition in 1921.",
     "category": "near_miss"},
    {"claim": "Infosys relocated its global headquarters from Bengaluru to Chennai in 2022.",
     "category": "near_miss"},
]


def build_adversarial() -> list[dict]:
    return [
        {
            "id": f"adv-{i:02d}",
            "claim": item["claim"],
            "category": item["category"],
            # The whole set shares one expectation, which is what makes
            # abstain_precision a single clean number.
            "expected": "abstained",
        }
        for i, item in enumerate(ADVERSARIAL, start=1)
    ]


# ── IO ───────────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):,} -> {path.relative_to(ROOT)}")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(prog="eval_harness.prepare_datasets")
    parser.add_argument("--articles", type=int, default=100,
                        help="articles to sample for the known-item set "
                             "(each yields a head and a tail query)")
    parser.add_argument("--fever", type=int, default=200,
                        help="FEVER claims to sample")
    args = parser.parse_args()

    print("Known-item retrieval set")
    known = build_known_item(args.articles)
    write_jsonl(known, DATA_DIR / f"known_item_{len(known)}.jsonl")

    print("\nFEVER")
    fever = build_fever(args.fever)
    if fever:
        write_jsonl(fever, DATA_DIR / f"fever_dev_{len(fever)}.jsonl")

    print("\nAdversarial")
    adversarial = build_adversarial()
    write_jsonl(adversarial, DATA_DIR / "adversarial_20.jsonl")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
