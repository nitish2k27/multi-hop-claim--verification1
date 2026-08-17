"""
prepare_fever_data_fixed.py
───────────────────────────
Fixed version that works with current HuggingFace datasets.
Uses a more reliable approach to load FEVER-like data.

Run this LOCALLY (no GPU needed) before uploading to Kaggle.

Output: data/training/fever_converted.jsonl
        Each line: {"llm_context": "...", "report": null}

Install:
    pip install datasets
"""

import json
import random
from pathlib import Path

# ── Try importing datasets ──────────────────────────────────────────────────
try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit(
        "\n[ERROR] Run:  pip install datasets\n"
        "Then re-run this script."
    )


# ── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("data/training")
OUTPUT_FILE = OUTPUT_DIR / "fever_converted.jsonl"

# How many FEVER rows to convert.
MAX_ROWS = 800

# FEVER label → your pipeline's verdict
LABEL_MAP = {
    "SUPPORTS": "TRUE",
    "REFUTES": "FALSE", 
    "NOT ENOUGH INFO": "UNVERIFIABLE",
    "SUPPORTED": "TRUE",
    "REFUTED": "FALSE",
    "NEI": "UNVERIFIABLE"
}

# FEVER label → stance string used in your evidence block
STANCE_MAP = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NEUTRAL",
    "SUPPORTED": "SUPPORTS", 
    "REFUTED": "REFUTES",
    "NEI": "NEUTRAL"
}

# Fake credibility scores — realistic spread matching your real outputs
CREDIBILITY_BY_LABEL = {
    "SUPPORTS": (0.55, 0.85),
    "REFUTES": (0.55, 0.85),
    "NOT ENOUGH INFO": (0.40, 0.65),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _credibility(label: str) -> float:
    lo, hi = CREDIBILITY_BY_LABEL.get(label, (0.4, 0.7))
    return round(random.uniform(lo, hi), 2)


def _support_score(label: str) -> float:
    if label in ["SUPPORTS", "SUPPORTED"]:
        return round(random.uniform(70, 95), 1)
    if label in ["REFUTES", "REFUTED"]:
        return round(random.uniform(0, 15), 1)
    return round(random.uniform(30, 55), 1)


def _refute_score(label: str) -> float:
    if label in ["REFUTES", "REFUTED"]:
        return round(random.uniform(70, 95), 1)
    if label in ["SUPPORTS", "SUPPORTED"]:
        return round(random.uniform(0, 10), 1)
    return round(random.uniform(15, 40), 1)


def build_llm_context(claim: str, evidence_text: str, label: str) -> str:
    """
    Build the exact same llm_context string your NLP pipeline produces.
    """

    verdict = LABEL_MAP.get(label, "UNVERIFIABLE")
    stance = STANCE_MAP.get(label, "NEUTRAL")
    cred = _credibility(label)
    sup_score = _support_score(label)
    ref_score = _refute_score(label)
    neu_score = round(100 - sup_score - ref_score, 1)
    neu_score = max(0.0, neu_score)

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
  - Preliminary verdict:  {verdict}
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


def load_fever_alternative():
    """Try multiple approaches to load FEVER-like data"""
    
    print("\n[1/3] Loading fact-checking dataset...")
    
    # Try multiple dataset sources
    datasets_to_try = [
        ("fever", "v1.0", "labelled_dev"),
        ("fever", "v2.0", "train"), 
        ("fever", None, "train"),
        ("kilt_tasks", "fever", "train"),
        ("climate_fever", None, "test"),
        ("scifact", None, "train")
    ]
    
    for dataset_name, config, split in datasets_to_try:
        try:
            print(f"      Trying {dataset_name}...")
            if config:
                dataset = load_dataset(dataset_name, config, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            print(f"      ✓ Successfully loaded {dataset_name}")
            return dataset, dataset_name
            
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            continue
    
    # If all fail, create synthetic data
    print("      Creating synthetic data as fallback...")
    return create_synthetic_data(), "synthetic"


def create_synthetic_data():
    """Create synthetic fact-checking data as fallback"""
    
    synthetic_claims = [
        {
            "claim": "The Earth orbits around the Sun",
            "label": "SUPPORTS",
            "evidence": "The Earth orbits the Sun in an elliptical path, completing one orbit approximately every 365.25 days."
        },
        {
            "claim": "Water boils at 100 degrees Celsius at sea level",
            "label": "SUPPORTS", 
            "evidence": "At standard atmospheric pressure (1 atmosphere), pure water boils at exactly 100 degrees Celsius or 212 degrees Fahrenheit."
        },
        {
            "claim": "The Great Wall of China is visible from space",
            "label": "REFUTES",
            "evidence": "Contrary to popular belief, the Great Wall of China is not visible from space with the naked eye, as confirmed by astronauts."
        },
        {
            "claim": "Humans use only 10% of their brain",
            "label": "REFUTES",
            "evidence": "Neuroimaging studies show that humans use virtually all of their brain, even during simple tasks. The 10% myth has been thoroughly debunked."
        },
        {
            "claim": "Lightning never strikes the same place twice",
            "label": "REFUTES", 
            "evidence": "Lightning frequently strikes the same location multiple times, especially tall structures like the Empire State Building."
        }
    ]
    
    # Expand the dataset by creating variations
    expanded_data = []
    for _ in range(MAX_ROWS):
        base_claim = random.choice(synthetic_claims)
        expanded_data.append({
            "claim": base_claim["claim"],
            "label": base_claim["label"],
            "evidence": base_claim["evidence"]
        })
    
    return expanded_data


def process_row(row: dict, dataset_name: str) -> dict | None:
    """Convert one row to training format"""
    
    # Extract claim and label based on dataset format
    if dataset_name == "synthetic":
        claim = row.get("claim", "")
        label = row.get("label", "NOT ENOUGH INFO")
        evidence_text = row.get("evidence", "")
    elif "kilt" in dataset_name:
        claim = row.get("input", "")
        if 'output' in row and row['output']:
            label = row['output'][0].get('answer', 'NOT ENOUGH INFO')
        else:
            label = 'NOT ENOUGH INFO'
        evidence_text = "Evidence from knowledge base."
    else:
        # Standard FEVER format
        claim = row.get("claim", "")
        label = row.get("label", "NOT ENOUGH INFO")
        
        # Try to extract evidence
        evidence_text = ""
        if 'evidence' in row:
            evidence = row['evidence']
            if isinstance(evidence, list) and evidence:
                # Handle nested evidence structure
                for group in evidence:
                    if isinstance(group, list):
                        for item in group:
                            if isinstance(item, list) and len(item) >= 4:
                                evidence_text = item[3]
                                break
                    elif isinstance(group, str):
                        evidence_text = group
                        break
                    if evidence_text:
                        break
        
        if not evidence_text:
            evidence_text = "Supporting evidence from reliable sources."
    
    if not claim or not claim.strip():
        return None
    
    # Normalize labels
    label = label.upper()
    if label not in LABEL_MAP:
        label = "NOT ENOUGH INFO"
    
    llm_context = build_llm_context(claim.strip(), evidence_text, label)
    
    return {
        "llm_context": llm_context,
        "report": None,
        "metadata": {
            "source": dataset_name,
            "label": label,
            "verdict": LABEL_MAP.get(label, "UNVERIFIABLE"),
            "claim": claim.strip(),
        }
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset, dataset_name = load_fever_alternative()
    
    print(f"      Dataset: {dataset_name}")
    print(f"      Size: {len(dataset):,} rows")
    print(f"      Sampling up to {MAX_ROWS} rows...\n")
    
    # Sample data
    if len(dataset) > MAX_ROWS:
        indices = random.sample(range(len(dataset)), MAX_ROWS)
        sampled_data = [dataset[i] for i in indices]
    else:
        sampled_data = list(dataset)
    
    print("[2/3] Converting rows to llm_context format...")
    examples = []
    skipped = 0
    
    for row in sampled_data:
        result = process_row(row, dataset_name)
        if result is None:
            skipped += 1
            continue
        examples.append(result)
    
    print(f"      Converted: {len(examples)}  |  Skipped: {skipped}\n")
    
    # Write output
    print(f"[3/3] Writing to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"\n[OK] Done. {len(examples)} examples saved to:")
    print(f"     {OUTPUT_FILE.resolve()}\n")
    
    # Show label distribution
    label_counts = {}
    for ex in examples:
        label = ex['metadata']['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    print("\nNext steps:")
    print("  1. Run your NLP pipeline on real claims →")
    print("     python scripts/export_pipeline_outputs.py")
    print("  2. Upload both .jsonl files to Kaggle as a dataset")
    print("  3. Run Kaggle notebook 1 to generate reports\n")


if __name__ == "__main__":
    random.seed(42)
    main()