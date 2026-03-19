# Data Directory

This directory contains training data, processed datasets, and the vector database.

## Directory Structure
```
data/
├── raw/                       # Raw scraped data
│   ├── fever_official/        # FEVER dataset files
│   ├── fever_simplified/      # Simplified FEVER dataset
│   └── scraped_news.csv       # News articles from scrapers
├── processed/                 # Processed training data
│   ├── claim_detection_*.csv  # Claim detection datasets
│   ├── stance_detection_*.csv # Stance detection datasets
│   └── news_articles_rag.csv  # RAG-ready news articles
└── chroma_db/                 # Vector database files
    └── chroma.sqlite3         # ChromaDB database
```

## Download Instructions

The data files are large (>5MB each) and are not included in this repository.

### Option 1: Download Training Data
```bash
# Download FEVER datasets for training
python src/data_collection/download_claim_datasets.py
python src/data_collection/download_stance_dataset.py
```

### Option 2: Scrape Fresh Data
```bash
# Scrape news articles
python src/data_collection/reliable_news_scraper.py
python src/data_collection/regional_indian_scraper.py

# Process scraped data
python src/data_processing/clean_news_data.py
```

### Option 3: Ingest to RAG Database
```bash
# Create vector database from news articles
python scripts/ingest_to_rag.py
```

## Data Sizes
- Training datasets: ~120MB total
- News articles: ~15MB
- Vector database: ~51MB
- Raw FEVER data: ~31MB

Total: ~217MB