# Models Directory

This directory contains trained models for the fact verification system.

## Model Structure
```
models/
├── claim_detector/final/       # BERT claim detection model
├── stance_detector/final/      # BERT stance detection model  
├── language_detection/         # FastText language detection
├── ner_model/                  # Named Entity Recognition
└── llm_finetuned/             # Fine-tuned LLM models
```

## Download Instructions

The trained models are large files (>100MB each) and are not included in this repository.

### Option 1: Download from Release
Download the models from the GitHub releases page when available.

### Option 2: Train Your Own
Use the training scripts in `src/data_collection/` to download datasets and train models:

```bash
# Download training data
python src/data_collection/download_claim_datasets.py
python src/data_collection/download_stance_dataset.py

# Train models (instructions coming soon)
```

### Option 3: Use Placeholder Models
The system will automatically fall back to pre-trained HuggingFace models if trained models are not available.

## Model Sizes
- `claim_detector/final/model.safetensors`: ~418MB
- `stance_detector/final/model.safetensors`: ~413MB  
- `language_detection/lid.176.bin`: ~125MB

Total: ~956MB