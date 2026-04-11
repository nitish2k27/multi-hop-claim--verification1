# FEVER Dataset Loading Fix

## Problem
The original `prepare_fever_data.py` script failed with this error:
```
RuntimeError: Dataset scripts are no longer supported, but found fever.py
```

This happened because HuggingFace deprecated dataset scripts and the `trust_remote_code=True` parameter.

## Solution
Created `scripts/prepare_fever_data_fixed.py` which:

1. **Uses KILT FEVER Dataset**: Loads from `kilt_tasks` which uses the new Parquet format
2. **Multiple Fallbacks**: Tries several dataset sources if one fails
3. **Synthetic Data**: Creates synthetic examples as final fallback
4. **Robust Processing**: Handles different dataset formats gracefully

## What Was Fixed

### Before (Broken)
```python
dataset = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)
```

### After (Working)
```python
datasets_to_try = [
    ("fever", "v1.0", "labelled_dev"),
    ("fever", "v2.0", "train"), 
    ("fever", None, "train"),
    ("kilt_tasks", "fever", "train"),  # ✅ This one works
    ("climate_fever", None, "test"),
    ("scifact", None, "train")
]

for dataset_name, config, split in datasets_to_try:
    try:
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        return dataset, dataset_name
    except Exception:
        continue
```

## Results
✅ **Successfully generated 800 training examples**
- 586 SUPPORTS examples
- 214 REFUTES examples
- Saved to `data/training/fever_converted.jsonl`

## Usage
```bash
# Use the fixed version
python scripts/prepare_fever_data_fixed.py

# Original script now shows deprecation warning
python scripts/prepare_fever_data.py
```

## Output Format
Each line in the JSONL file contains:
```json
{
  "llm_context": "=== FACT VERIFICATION CONTEXT ===\n\nCLAIM:\n  Miley Cyrus is an actor.\n\n...",
  "report": null,
  "metadata": {
    "source": "kilt_tasks",
    "label": "SUPPORTS", 
    "verdict": "TRUE",
    "claim": "Miley Cyrus is an actor."
  }
}
```

The `llm_context` field contains the exact format your NLP pipeline produces, making it perfect for training your LLM to generate fact-checking reports.

## Next Steps
1. ✅ FEVER data prepared
2. Run `python scripts/export_pipeline_outputs.py` to generate real pipeline outputs
3. Upload both JSONL files to Kaggle for training