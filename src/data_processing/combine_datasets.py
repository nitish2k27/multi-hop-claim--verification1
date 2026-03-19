#!/bin/bash

echo "========================================"
echo "COMPLETE DATA GENERATION PIPELINE"
echo "========================================"

# Get API key
read -p "Enter your Gemini API key: " API_KEY

# Step 1: Generate synthetic data
echo ""
echo "Step 1: Generating multilingual synthetic data..."
echo "This will take ~3-4 hours for 6,000 examples"
python src/data_collection/generate_multilingual_synthetic.py "$API_KEY"

# Step 2: Download FEVER
echo ""
echo "Step 2: Downloading FEVER dataset..."
python -c "from datasets import load_dataset; load_dataset('fever', 'v1.0').save_to_disk('data/raw/fever')"

# Step 3: Combine datasets
echo ""
echo "Step 3: Combining synthetic + real data..."
python src/data_processing/combine_datasets.py

# Done
echo ""
echo "========================================"
echo "✓ DATA GENERATION COMPLETE"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - data/synthetic/raw/fact_check_*.json (6 languages)"
echo "  - data/processed/llm_training_combined.json"
echo "  - data/processed/claim_detection_train.csv"
echo "  - data/processed/stance_detection_train.csv"