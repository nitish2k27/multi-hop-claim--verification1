import json
import pandas as pd
from datasets import load_dataset

print("Collecting all processed datasets...")


# ════════════════════════════════════════
# claim_train.csv - FROM LOCAL FILES
# ════════════════════════════════════════
print("\n1. Processing FEVER → claim_train.csv")

rows = []
try:
    with open("data/raw/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "claim": item.get("claim", ""),
                "label": item.get("label", ""),
                "evidence_wiki_url": str(
                    item.get("evidence_wiki_url", "")
                ),
                "evidence_sentence": str(
                    item.get("evidence_sentence", "")
                ),
                "split": "train"
            })
    print(f"  ✓ Loaded train.jsonl")
except FileNotFoundError:
    print("  ⚠ train.jsonl not found in data/raw/")

pd.DataFrame(rows).to_csv(
    "data/processed/claim_train.csv", index=False
)
print(f"DONE → claim_train.csv ({len(rows)} records)")


# ════════════════════════════════════════
# stance_train.csv - FROM LOCAL FILES
# ════════════════════════════════════════
print("\n2. Processing FEVER → stance_train.csv")

stance_map = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NEUTRAL"
}
rows = []
try:
    with open("data/raw/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "claim": item.get("claim", ""),
                "evidence": str(
                    item.get("evidence_sentence", "")
                ),
                "stance": stance_map.get(
                    item.get("label", "").upper(), "NEUTRAL"
                ),
                "split": "train"
            })
    print(f"  ✓ Loaded train.jsonl")
except FileNotFoundError:
    print("  ⚠ train.jsonl not found in data/raw/")

pd.DataFrame(rows).to_csv(
    "data/processed/stance_train.csv", index=False
)
print(f"DONE → stance_train.csv ({len(rows)} records)")


# ════════════════════════════════════════
# ner_train.csv - CoNLL-2003
# ════════════════════════════════════════
print("\n3. Downloading CoNLL-2003 → ner_train.csv")

tag_map = {
    0:"O", 1:"B-PER", 2:"I-PER",
    3:"B-ORG", 4:"I-ORG",
    5:"B-LOC", 6:"I-LOC",
    7:"B-MISC", 8:"I-MISC"
}
rows = []
try:
    # NEW
    # NEW
    conll = load_dataset("BramVanroy/conll2003")
    for split in ["train", "validation", "test"]:
        for item in conll[split]:
            rows.append({
                "tokens": " ".join(item["tokens"]),
                "ner_tags": " ".join([
                    tag_map.get(t, "O")
                    for t in item["ner_tags"]
                ]),
                "split": split
            })
    pd.DataFrame(rows).to_csv(
        "data/processed/ner_train.csv", index=False
    )
    print(f"DONE → ner_train.csv ({len(rows)} records)")
except Exception as e:
    print(f"  ⚠ CoNLL error: {e}")


# ════════════════════════════════════════
# ner_multilingual_train.csv - WikiANN
# ════════════════════════════════════════
print("\n4. Downloading WikiANN → ner_multilingual_train.csv")

languages = {
    "en": "English", "hi": "Hindi",
    "es": "Spanish", "ar": "Arabic",
    "zh": "Chinese", "fr": "French",
    "de": "German",  "ta": "Tamil"
}
tag_map2 = {
    0:"O", 1:"B-PER", 2:"I-PER",
    3:"B-ORG", 4:"I-ORG",
    5:"B-LOC", 6:"I-LOC"
}
rows = []
for code, name in languages.items():
    try:
        data = load_dataset("wikiann", code)
        for item in data["train"]:
            rows.append({
                "language": code,
                "language_name": name,
                "tokens": " ".join(item["tokens"]),
                "ner_tags": " ".join([
                    tag_map2.get(t, "O")
                    for t in item["ner_tags"]
                ])
            })
        print(f"  ✓ {name} done")
    except Exception as e:
        print(f"  ⚠ Skipped {name}: {e}")

pd.DataFrame(rows).to_csv(
    "data/processed/ner_multilingual_train.csv", index=False
)
print(f"DONE → ner_multilingual_train.csv ({len(rows)} records)")


# ════════════════════════════════════════
# llm_train.json - FEVER + NEWS
# ════════════════════════════════════════
print("\n5. Building LLM fine-tuning → llm_train.json")

llm_data = []
explanations = {
    "SUPPORTS":        "The claim is SUPPORTED by the evidence.",
    "REFUTES":         "The claim is REFUTED by the evidence.",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO to verify this claim."
}

# 5k from FEVER
count = 0
try:
    with open("data/raw/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if count >= 5000:
                break
            item = json.loads(line)
            llm_data.append({
                "instruction": (
                    "You are a fact-checking assistant. "
                    "Verify the claim using the evidence."
                ),
                "input": (
                    f"Claim: {item.get('claim','')}\n\n"
                    f"Evidence: {str(item.get('evidence_sentence',''))}\n\n"
                    f"Verdict: {item.get('label','')}"
                ),
                "output": explanations.get(
                    item.get("label", ""), ""
                ),
                "label": item.get("label", ""),
                "source": "FEVER"
            })
            count += 1
    print(f"  ✓ FEVER: {count} samples")
except FileNotFoundError:
    print("  ⚠ train.jsonl not found")

# 5k from news articles
try:
    news_df = pd.read_csv("data/raw/news_articles.csv")
    for _, row in news_df.head(5000).iterrows():
        llm_data.append({
            "instruction": (
                "You are a fact-checking assistant. "
                "Analyze this news article."
            ),
            "input": (
                f"Title: {str(row.get('title',''))}\n\n"
                f"Text: {str(row.get('text',''))[:500]}\n\n"
                f"Source: {str(row.get('source',''))}"
            ),
            "output": (
                f"Article from {row.get('source','')} "
                f"requires fact verification."
            ),
            "label": "CUSTOM",
            "source": str(row.get("source", ""))
        })
    print(f"  ✓ News: {len(news_df.head(5000))} samples")
except FileNotFoundError:
    print("  ⚠ news_articles.csv not found")

with open("data/processed/llm_train.json", "w",
          encoding="utf-8") as f:
    json.dump(llm_data, f, indent=2, ensure_ascii=False)
print(f"DONE → llm_train.json ({len(llm_data)} records)")


