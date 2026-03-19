# Fact Verification System

A comprehensive, multi-modal fact verification system that can verify claims using 8 different input types with user-provided context documents and multi-language support.

## 🚀 Features

### Multi-Modal Input Processing
- **8 Input Types**: Text, voice/audio, PDF, DOCX, XLSX, CSV, images (OCR), URLs
- **Language Detection**: FastText with 99.1% accuracy across 176 languages
- **Context Documents**: Users can provide supporting documents with priority-based ranking
- **Batch Processing**: Handle multiple inputs simultaneously

### Advanced NLP Pipeline
- **Claim Detection**: BERT model trained on FEVER dataset (85.7% accuracy)
- **Named Entity Recognition**: Extracts people, organizations, locations, misc entities
- **Entity Linking**: Links entities to Wikidata knowledge base
- **Temporal Extraction**: Extracts dates, quarters, durations, time ranges
- **Stance Detection**: BERT model trained on FEVER-NLI (SUPPORTS/REFUTES/NEUTRAL)

### RAG (Retrieval-Augmented Generation)
- **Hybrid Search**: Combines dense vector search + sparse BM25 keyword search
- **Priority System**: User documents (1.0) → Vector DB (0.7) → Web search (0.5)
- **Re-ranking**: Cross-encoder for better result quality
- **Credibility Scoring**: Multi-factor assessment (domain authority, recency, source type)

### Data Collection
- **News Scrapers**: Reuters, BBC, Guardian, AP, CNN, Indian sources
- **Regional Coverage**: 25 Indian news sources across 10 languages
- **Quality Control**: Deduplication, validation, rate limiting

## 📋 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/fact-verification-system.git
cd fact-verification-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup NLTK Data
```bash
python scripts/setup_nltk.py
```

### 4. Check System Status
```bash
python scripts/check_system_status.py
```

### 5. Download Models & Data (Optional)
```bash
# Download training datasets
python src/data_collection/download_claim_datasets.py
python src/data_collection/download_stance_dataset.py

# Scrape news data
python src/data_collection/reliable_news_scraper.py

# Create vector database
python scripts/ingest_to_rag.py
```

### 6. Test System (Works with Placeholder Models)
```bash
# Test NLP pipeline (uses HuggingFace models as fallback)
python tests/test_complete_nlp_pipeline.py

# Test end-to-end system
python tests/test_fact_verification_e2e.py
```

## 🎯 For Team Members

### Immediate Access (No Large Files)
The repository contains all source code and can run immediately with placeholder models:
- **Claim Detection**: Uses zero-shot classification
- **Stance Detection**: Uses NLI model as fallback  
- **NER**: Uses pre-trained BERT-NER from HuggingFace
- **Language Detection**: Downloads FastText model automatically (125MB)

### Full System (With Trained Models)
For production use, download the trained models:
1. Check the `models/README.md` for download instructions
2. Check the `data/README.md` for dataset instructions
3. Models will be provided via releases or cloud storage

## 🏗️ Architecture

```
INPUT TEXT/DOCUMENT
    ↓
┌─────────────────────────────────┐
│      INPUT PROCESSING           │
├─────────────────────────────────┤
│ • Multi-format support          │
│ • Language detection            │
│ • Priority-based handling       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│      NLP PIPELINE               │
├─────────────────────────────────┤
│ • Claim Detection               │
│ • Named Entity Recognition      │
│ • Entity Linking                │
│ • Temporal Extraction           │
│ • Stance Detection              │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│      RAG PIPELINE               │
├─────────────────────────────────┤
│ • Evidence Retrieval            │
│ • Re-ranking                    │
│ • Credibility Scoring           │
│ • Stance Classification         │
└─────────────────────────────────┘
    ↓
VERIFICATION RESULT
```

## 📁 Project Structure

```
fact-verification-system/
├── src/
│   ├── main_pipeline.py           # Main verification pipeline
│   ├── preprocessing/             # Input processing
│   ├── nlp/                       # NLP components
│   ├── rag/                       # RAG pipeline
│   ├── data_collection/           # Data scrapers
│   └── data_processing/           # Data cleaning
├── models/                        # Trained models (Git LFS)
├── data/                          # Training/test data (Git LFS)
├── tests/                         # Test suite
├── docs/                          # Documentation
├── configs/                       # Configuration files
└── scripts/                       # Utility scripts
```

## 🧪 Testing

Run the complete test suite:
```bash
# Test individual components
python tests/test_trained_claim_detector.py
python tests/test_stance_detector.py
python tests/test_rag_pipeline.py

# Test complete pipeline
python tests/test_complete_nlp_pipeline.py
python tests/test_fact_verification_e2e.py
```

## 📊 Model Performance

- **Claim Detection**: 85.7% accuracy on test set
- **Stance Detection**: Trained on 208k FEVER-NLI examples
- **Language Detection**: 99.1% accuracy across 176 languages
- **Entity Linking**: Wikidata integration with disambiguation

## 🔧 Configuration

Edit `configs/nlp_config.yaml` to:
- Enable/disable trained models
- Adjust confidence thresholds
- Configure batch sizes
- Set device (CPU/GPU)

## 📚 Documentation

- [Input Processor Capabilities](docs/INPUT_PROCESSOR_CAPABILITIES.md)
- [NLP Pipeline Guide](docs/NLP_PIPELINE_GUIDE.md)
- [RAG Pipeline Guide](docs/RAG_PIPELINE_GUIDE.md)
- [Language Detection](docs/LANGUAGE_DETECTION.md)
- [Regional Scraper Guide](docs/REGIONAL_INDIAN_SCRAPER_GUIDE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FEVER dataset for training data
- HuggingFace for model hosting
- Wikidata for entity linking
- FastText for language detection