"""
prepare_fever_data.py
─────────────────────
⚠️  DEPRECATED: This script has issues with HuggingFace dataset changes.
    Use prepare_fever_data_fixed.py instead.

The FEVER dataset on HuggingFace no longer supports the old loading method.
Run: python scripts/prepare_fever_data_fixed.py
"""

import sys

def main():
    print("⚠️  DEPRECATED SCRIPT")
    print("=" * 50)
    print("This script is deprecated due to HuggingFace dataset changes.")
    print("The FEVER dataset no longer supports trust_remote_code.")
    print("")
    print("✅ SOLUTION:")
    print("   Run: python scripts/prepare_fever_data_fixed.py")
    print("")
    print("The fixed version uses the KILT FEVER dataset which works")
    print("with current HuggingFace datasets library.")
    print("=" * 50)
    
    sys.exit(1)

if __name__ == "__main__":
    main()


# ── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("data/training")
OUTPUT_FILE = OUTPUT_DIR / "fever_converted.jsonl"

# How many FEVER rows to convert.
# 800 is enough — quality matters more than quantity for LoRA.
MAX_ROWS = 800

# FEVER label → your pipeline's verdict
LABEL_MAP = {
    "SUPPORTS":         "TRUE",
    "REFUTES":          "FALSE",
    "NOT ENOUGH INFO":  "UNVERIFIABLE",
}

# FEVER label → stance string used in your evidence block
STANCE_MAP = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES":  "REFUTES",
    "NOT ENOUGH INFO": "NEUTRAL",
}

# Fake credibility scores — realistic spread matching your real outputs
CREDIBILITY_BY_LABEL = {
    "SUPPORTS": (0.55, 0.85),
    "REFUTES":  (0.55, 0.85),
    "NOT ENOUGH INFO": (0.40, 0.65),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _credibility(label: str) -> float:
    lo, hi = CREDIBILITY_BY_LABEL.get(label, (0.4, 0.7))
    return round(random.uniform(lo, hi), 2)


def _support_score(label: str) -> float:
    if label == "SUPPORTS":
        return round(random.uniform(70, 95), 1)
    if label == "REFUTES":
        return round(random.uniform(0, 15), 1)
    return round(random.uniform(30, 55), 1)


def _refute_score(label: str) -> float:
    if label == "REFUTES":
        return round(random.uniform(70, 95), 1)
    if label == "SUPPORTS":
        return round(random.uniform(0, 10), 1)
    return round(random.uniform(15, 40), 1)


def build_llm_context(claim: str, evidence_text: str, label: str) -> str:
    """
    Build the exact same llm_context string your NLP pipeline produces.
    This must match the format in STEP 3 of your pipeline output.
    """

    verdict      = LABEL_MAP.get(label, "UNVERIFIABLE")
    stance       = STANCE_MAP.get(label, "NEUTRAL")
    cred         = _credibility(label)
    sup_score    = _support_score(label)
    ref_score    = _refute_score(label)
    neu_score    = round(100 - sup_score - ref_score, 1)
    neu_score    = max(0.0, neu_score)

    # Preliminary verdict mirrors your pipeline
    prelim = verdict

    context = f"""=== FACT VERIFICATION CONTEXT ===

CLAIM:
  {claim}

CLAIM ANALYSIS:
  Is verifiable claim: True
  Claim confidence:    {round(random.uniform(0.75, 1.0), 3)}

NAMED ENTITIES:
  - Extracted from claim text

TEMPORAL EXPRESSIONS:
  - None explicitly detected

KNOWLEDGE BASE LINKS:
  None found

RETRIEVED EVIDENCE:
  [1] [{stance}] (source: wikipedia, credibility: {cred})
      {evidence_text[:300]}

EVIDENCE AGGREGATION:
  - Preliminary verdict:  {prelim}
  - Support score:        {sup_score}%
  - Refute score:         {ref_score}%
  - Neutral score:        {neu_score}%
  - Evidence pieces:      1
  - Supports:             {1 if stance == 'SUPPORTS' else 0}
  - Refutes:              {1 if stance == 'REFUTES' else 0}

TASK:
Based on the claim, entities, temporal context, and retrieved evidence above,
provide a detailed fact-verification analysis with:
1. Final verdict (TRUE / FALSE / MOSTLY TRUE / MOSTLY FALSE / UNVERIFIABLE)
2. Confidence score (0-100%)
3. Key evidence supporting your verdict
4. Any conflicting evidence
5. Important caveats or limitations
=== END CONTEXT ==="""

    return context


def convert_fever(max_rows: int = MAX_ROWS) -> list[dict]:
    """
    Load FEVER from HuggingFace and convert to llm_context format.
    Returns list of dicts ready to write as JSONL.
    """

    print("\n[1/3] Loading FEVER dataset from HuggingFace...")
    print("      (first run downloads ~1 GB — subsequent runs use cache)\n")

    # Try different approaches to load FEVER dataset
    dataset = None
    
    try:
        # Try the new approach without trust_remote_code
        print("      Trying new dataset format...")
        dataset = load_dataset("fever", "v1.0", split="train")
    except Exception as e1:
        print(f"      New format failed: {e1}")
        
        try:
            # Try alternative dataset name
            print("      Trying alternative dataset...")
            dataset = load_dataset("fever", split="train")
        except Exception as e2:
            print(f"      Alternative failed: {e2}")
            
            try:
                # Try the simplified fever dataset
                print("      Trying simplified fever dataset...")
                dataset = load_dataset("fever", "v2.0", split="train")
            except Exception as e3:
                print(f"      Simplified failed: {e3}")
                
                # Final fallback - use a different fever dataset
                print("      Using fallback dataset...")
                dataset = load_dataset("kilt_tasks", "fever", split="train")

    if dataset is None:
        raise RuntimeError("Could not load FEVER dataset with any method")

    print(f"      Loaded {len(dataset):,} rows total")
    print(f"      Sampling {max_rows} rows...\n")

    # Sample evenly across labels so we get balanced training data
    by_label: dict[str, list] = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}
    
    for row in dataset:
        # Handle different dataset formats
        if 'label' in row:
            lbl = row["label"]
        elif 'output' in row and isinstance(row['output'], list) and len(row['output']) > 0:
            # KILT format
            lbl = row['output'][0].get('answer', 'NOT ENOUGH INFO')
        else:
            lbl = 'NOT ENOUGH INFO'
            
        if lbl in by_label and len(by_label[lbl]) < max_rows:
            by_label[lbl].append(row)

    per_class = max_rows // 3
    selected  = []
    for lbl, rows in by_label.items():
        selected.extend(random.sample(rows, min(per_class, len(rows))))

    random.shuffle(selected)
    print(f"      Selected: {len(selected)} rows")
    print(f"      SUPPORTS: {sum(1 for r in selected if r['label']=='SUPPORTS')}")
    print(f"      REFUTES:  {sum(1 for r in selected if r['label']=='REFUTES')}")
    print(f"      NEI:      {sum(1 for r in selected if r['label']=='NOT ENOUGH INFO')}\n")

    return selected


def process_row(row: dict) -> dict | None:
    """Convert one FEVER row → training dict."""

    # Handle different dataset formats
    if 'claim' in row:
        claim = row.get("claim", "").strip()
        label = row.get("label", "NOT ENOUGH INFO")
        evidence_groups = row.get("evidence", [])
    elif 'input' in row:
        # KILT format
        claim = row.get("input", "").strip()
        if 'output' in row and isinstance(row['output'], list) and len(row['output']) > 0:
            label = row['output'][0].get('answer', 'NOT ENOUGH INFO')
        else:
            label = 'NOT ENOUGH INFO'
        evidence_groups = []
    else:
        return None

    # FEVER evidence is a list of annotation groups
    # Each group is a list of [annotation_id, wikipedia_url, sent_id, sentence]
    # We extract the first available evidence sentence
    evidence_text = ""
    
    for group in evidence_groups:
        for annotation in group:
            # annotation = [ann_id, wiki_url, sent_id, sentence_text]
            if isinstance(annotation, list) and len(annotation) >= 4:
                text = annotation[3]
                if text and isinstance(text, str) and len(text) > 20:
                    evidence_text = text.strip()
                    break
        if evidence_text:
            break

    # Skip rows with no usable evidence (common for NEI)
    if not evidence_text:
        evidence_text = "No direct evidence sentence available from source."

    if not claim:
        return None

    llm_context = build_llm_context(claim, evidence_text, label)

    return {
        "llm_context": llm_context,
        "report": None,          # filled in by Kaggle notebook 1
        "metadata": {
            "source":  "fever",
            "label":   label,
            "verdict": LABEL_MAP.get(label, "UNVERIFIABLE"),
            "claim":   claim,
        }
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows     = convert_fever(MAX_ROWS)
    examples = []

    print("[2/3] Converting rows to llm_context format...")
    skipped = 0
    for row in rows:
        result = process_row(row)
        if result is None:
            skipped += 1
            continue
        examples.append(result)

    print(f"      Converted: {len(examples)}  |  Skipped: {skipped}\n")

    print(f"[3/3] Writing to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[OK] Done. {len(examples)} examples saved to:")
    print(f"     {OUTPUT_FILE.resolve()}\n")
    print("Next step:")
    print("  1. Run your NLP pipeline on real claims →")
    print("     python scripts/export_pipeline_outputs.py")
    print("  2. Upload both .jsonl files to Kaggle as a dataset")
    print("  3. Run Kaggle notebook 1 to generate reports\n")


if __name__ == "__main__":
    random.seed(42)
    main()
