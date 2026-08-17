"""
Publish the two trained BERTs to the HuggingFace Hub.

    python scripts/push_to_hub.py --user YOUR_HF_USERNAME --dry-run
    python scripts/push_to_hub.py --user YOUR_HF_USERNAME

Why a script rather than `hf upload` twice:

* **The model card is generated from `results.json` and `config.json`**, so the
  numbers on the Hub cannot drift from the numbers the training run actually
  produced. A hand-written card is a second copy of the truth, and second
  copies rot.
* **`training_args.bin` is excluded.** It is a pickle of the TrainingArguments
  object — useless to anyone downloading the model, and pickles on a public
  repo are a trust smell worth not having.
* **The caveats travel with the weights.** The claim detector reports 1.00
  test accuracy, which is a red flag rather than a result; anyone who finds
  that model on the Hub without the explanation will draw the wrong conclusion.

Authenticate first (a token with *write* scope, from
https://huggingface.co/settings/tokens):

    hf auth login

Re-running is safe: it overwrites the files in place and leaves history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files never uploaded, whatever is in the directory.
SKIP = {"training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"}


# ── Model cards ──────────────────────────────────────────────────────────────
# The honest limitations are not optional garnish. Both are stated in
# core/models.py and the project README; a card that omitted them would be
# advertising a claim detector with perfect accuracy, which is not what this is.

CARDS = {
    "claim_detector": {
        "repo": "verifai-claim-detector",
        "base": "bert-base-uncased",
        "examples": "280,713",
        "task": "Binary claim detection — is this text a verifiable factual assertion?",
        "labels": "`LABEL_0` = not a claim · `LABEL_1` = a claim",
        "caveat": """## Read this before trusting the accuracy

This model reports **1.00 test accuracy**, and that is a red flag rather than a
result.

Probed by hand, it separates *declarative sentences* from *questions and
opinions* — which is what its training negatives actually were. It does **not**
judge verifiability: the string `"asdkjh askjdh"` scores 0.997 as a claim.

In VerifAI it is used for exactly one thing: a cheap filter that keeps
questions and conversational text out of the retrieval path, before any
expensive work happens. Do not use it as a check on whether something is
*checkable*.""",
        "usage": '''from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()

text = "India's software exports reached 222 billion dollars in 2024-25"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=1)

print(f"P(is a claim) = {{probs[0][1].item():.4f}}")   # ~0.9999''',
    },
    "stance_detector": {
        "repo": "verifai-stance-detector",
        "base": "bert-base-cased",
        "examples": "208,346",
        "task": "Claim–evidence stance classification (FEVER-NLI).",
        "labels": "`SUPPORTS` · `REFUTES` · `NOT ENOUGH INFO`",
        "caveat": """## Read this before trusting the labels

73.6% accuracy overall, and **REFUTES is its weakest class** — probed by hand
it called a directly contradicting sentence `SUPPORTS` at 0.52 confidence.

In VerifAI its output is never the verdict. Each label travels to the reasoning
LLM *with its confidence attached*, and the prompt instructs the model to
follow the evidence text wherever the two disagree. On a typical run it is
overridden on roughly a third of items.

Treat it as a signal, not an answer.""",
        "usage": '''from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()

claim = "India's software exports reached 222 billion dollars in 2024-25"
evidence = "Exports of computer software and services reached 222 billion in 2024-25."

inputs = tokenizer(claim, evidence, return_tensors="pt",
                   truncation=True, max_length=256)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=1)[0]

for i, label in model.config.id2label.items():
    print(f"{{label:16s}} {{probs[int(i)].item():.4f}}")''',
    },
}


def build_card(key: str, repo_id: str, results: dict, config: dict) -> str:
    """Render the model card. Every metric comes from results.json."""
    spec = CARDS[key]

    metrics = "\n".join(
        f"| {name} | {results[field]:.4f} |"
        for name, field in (
            ("Accuracy", "test_accuracy"),
            ("Precision", "test_precision"),
            ("Recall", "test_recall"),
            ("F1", "test_f1"),
        )
        if field in results
    )

    per_class = ""
    if "per_class_f1" in results:
        rows = "\n".join(f"| `{k}` | {v:.4f} |"
                         for k, v in results["per_class_f1"].items())
        per_class = f"\n\n**Per-class F1**\n\n| Label | F1 |\n|---|---|\n{rows}"

    id2label = config.get("id2label") or {}
    label_block = ""
    if id2label:
        rows = "\n".join(f"| `{i}` | `{lab}` |" for i, lab in sorted(id2label.items()))
        label_block = f"\n\n| Index | Label |\n|---|---|\n{rows}"

    return f"""---
license: mit
language:
  - en
base_model: {spec["base"]}
pipeline_tag: text-classification
tags:
  - fact-checking
  - fever
  - claim-verification
  - bert
datasets:
  - fever
metrics:
  - accuracy
  - f1
library_name: transformers
---

# {repo_id.split("/")[-1]}

{spec["task"]}

Fine-tuned from `{spec["base"]}` on {spec["examples"]} FEVER examples over 3
epochs. Runs on CPU. Part of [VerifAI](https://github.com/nitish2k27/multi-hop-claim--verification1),
a two-tier fact-verification RAG system built around knowing when to abstain.

**Labels** — {spec["labels"]}{label_block}

## Results

| Metric | Score |
|---|---|
{metrics}{per_class}

{spec["caveat"]}

## Usage

```python
{spec["usage"].format(repo_id=repo_id)}
```

## How it is used in VerifAI

This model is one stage of a graph, not a system on its own. The full pipeline
runs a claim gate, hybrid dense+BM25 retrieval, cross-encoder reranking, a
calibrated relevance floor, per-source credibility scoring, this stance model,
and only then an LLM constrained to a Pydantic schema with every citation
validated against the documents actually retrieved.

Two of its three exits never reach an LLM at all — the design goal is a system
that declines rather than guesses. On a 20-claim adversarial set it abstained
on 20/20, with zero false-confident answers.

Details, measured results and an end-to-end execution trace are in the
[repository](https://github.com/nitish2k27/multi-hop-claim--verification1).
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="your HuggingFace username")
    parser.add_argument("--private", action="store_true",
                        help="create the repos private (default is public)")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the cards and list files, upload nothing")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed:  pip install huggingface_hub",
              file=sys.stderr)
        return 1

    api = HfApi()

    if not args.dry_run:
        try:
            who = api.whoami()
            print(f"Authenticated as {who['name']}\n")
        except Exception:
            print("Not logged in. Run:  hf auth login\n"
                  "  (needs a token with WRITE scope from "
                  "https://huggingface.co/settings/tokens)", file=sys.stderr)
            return 1

    for key, spec in CARDS.items():
        local = ROOT / "models" / key / "final"
        repo_id = f"{args.user}/{spec['repo']}"
        print(f"── {repo_id} " + "─" * max(0, 50 - len(repo_id)))

        if not (local / "model.safetensors").exists():
            print(f"   SKIP — no weights at {local}\n")
            continue

        results = json.loads((local / "results.json").read_text(encoding="utf-8"))
        config = json.loads((local / "config.json").read_text(encoding="utf-8"))

        card = build_card(key, repo_id, results, config)
        (local / "README.md").write_text(card, encoding="utf-8")
        print(f"   model card written ({len(card):,} chars)")

        uploads = [p for p in sorted(local.iterdir())
                   if p.is_file() and p.name not in SKIP]
        total_mb = sum(p.stat().st_size for p in uploads) / 1024 / 1024
        for p in uploads:
            print(f"     + {p.name:24s} {p.stat().st_size / 1024 / 1024:8.2f} MB")
        for name in sorted(SKIP):
            if (local / name).exists():
                print(f"     - {name:24s} (excluded)")
        print(f"   {len(uploads)} files, {total_mb:.1f} MB")

        if args.dry_run:
            print("   dry run — nothing uploaded\n")
            continue

        api.create_repo(repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(local),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=list(SKIP),
            commit_message="Publish VerifAI fine-tuned classifier",
        )
        print(f"   uploaded -> https://huggingface.co/{repo_id}\n")

    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to upload.")
    else:
        print("Done. Update the model paths in .env or point "
              "CLAIM_DETECTOR_PATH / STANCE_DETECTOR_PATH at the Hub IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
