# NLP Pipeline Integration Guide

## Quick Start
```python
from src.nlp.nlp_pipeline import NLPPipeline

# Initialize
pipeline = NLPPipeline()

# Analyze text
result = pipeline.analyze("India's GDP grew 8% in 2024")

# Access results
print(result['analysis']['claim_detection'])
print(result['analysis']['entities'])
print(result['analysis']['temporal'])
```

## Plugging in Trained Models

### Step 1: Train Your Model

Train your claim detection model and save to:
```
models/claim_detector/final/
```

### Step 2: Update Config

Edit `configs/nlp_config.yaml`:
```yaml
models:
  claim_detector:
    use_trained: true  # Change to true
```

### Step 3: Restart Pipeline
```python
pipeline = NLPPipeline()  # Will now load trained model
```

That's it! No code changes needed.

## Model Directory Structure
```
models/
├── claim_detector/
│   └── final/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer_config.json
├── ner_model/
│   └── final/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer_config.json
└── stance_detector/
    └── final/
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer_config.json
```

## Testing Your Integration
```bash
cd tests
python test_nlp_pipeline.py
```

All tests should pass with both placeholder and trained models.