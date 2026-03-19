# RAG Pipeline Documentation

## Overview

The RAG (Retrieval-Augmented Generation) Pipeline retrieves relevant evidence from a knowledge base to verify claims. It combines vector search, keyword search, re-ranking, and credibility scoring.

## Architecture

```
Claim Input
    ↓
[Vector Search] ← Dense embeddings (semantic)
    ↓
[Sparse Search] ← BM25 (keyword matching)
    ↓
[Hybrid Retrieval] ← Combines both
    ↓
[Re-ranking] ← Cross-encoder scoring
    ↓
[Credibility Scoring] ← Source assessment
    ↓
[Stance Detection] ← SUPPORTS/REFUTES/NEUTRAL
    ↓
Ranked Evidence
```

## Components

### 1. Vector Database (`src/rag/vector_database.py`)

**Purpose:** Stores and retrieves documents using semantic embeddings.

**Features:**
- ✅ ChromaDB integration
- ✅ Sentence-transformers embeddings
- ✅ Metadata filtering
- ✅ Batch ingestion
- ✅ Collection management

**Usage:**
```python
from src.rag.vector_database import VectorDatabase

# Initialize
db = VectorDatabase(collection_name="fact_check_kb")

# Add documents
db.add_documents([
    {
        'text': "India's GDP grew 8% in 2024",
        'metadata': {
            'source': 'Economic Times',
            'date': '2024-01-15',
            'url': 'https://...'
        }
    }
])

# Search
results = db.search(
    query="GDP growth India",
    top_k=5,
    filter_metadata={'date': {'$gte': '2024-01-01'}}
)
```

**Methods:**
- `add_documents(documents)` - Add documents to database
- `search(query, top_k, filter_metadata)` - Semantic search
- `delete_collection()` - Remove collection
- `get_collection_stats()` - Get statistics

### 2. Sparse Retrieval (`src/rag/sparse_retrieval.py`)

**Purpose:** Keyword-based search using BM25 algorithm.

**Features:**
- ✅ BM25 ranking
- ✅ Fast keyword matching
- ✅ Complements vector search
- ✅ Good for exact matches

**Usage:**
```python
from src.rag.sparse_retrieval import SparseRetriever

# Initialize
retriever = SparseRetriever()

# Index documents
retriever.index_documents([
    {'id': '1', 'text': 'India GDP growth...'},
    {'id': '2', 'text': 'Economic report...'}
])

# Search
results = retriever.search("GDP growth", top_k=5)
```

**When to use:**
- Exact keyword matches needed
- Named entities (company names, locations)
- Technical terms
- Acronyms

### 3. Hybrid Retrieval (`src/rag/hybrid_retrieval.py`)

**Purpose:** Combines vector and sparse search for best results.

**Features:**
- ✅ Weighted combination (vector + BM25)
- ✅ Reciprocal Rank Fusion (RRF)
- ✅ Configurable weights
- ✅ Deduplication

**Usage:**
```python
from src.rag.hybrid_retrieval import HybridRetriever

# Initialize
retriever = HybridRetriever(
    vector_db=vector_db,
    sparse_retriever=sparse_retriever,
    vector_weight=0.7,  # 70% vector, 30% sparse
    sparse_weight=0.3
)

# Search
results = retriever.search(
    query="India GDP growth 2024",
    top_k=10
)
```

**Fusion Methods:**
- **Weighted Sum:** `score = α * vector_score + β * sparse_score`
- **RRF:** `score = Σ(1 / (k + rank_i))` for each retriever

### 4. Re-ranker (`src/rag/reranker.py`)

**Purpose:** Re-ranks retrieved documents using cross-encoder for better relevance.

**Features:**
- ✅ Cross-encoder scoring
- ✅ Query-document relevance
- ✅ More accurate than bi-encoder
- ✅ Configurable threshold

**Usage:**
```python
from src.rag.reranker import Reranker

# Initialize
reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# Re-rank
reranked = reranker.rerank(
    query="India GDP growth",
    documents=retrieved_docs,
    top_k=5
)
```

**How it works:**
1. Initial retrieval gets ~50-100 candidates
2. Cross-encoder scores each query-document pair
3. Re-sorts by cross-encoder score
4. Returns top-k most relevant

### 5. Credibility Scorer (`src/rag/credibility_scorer.py`)

**Purpose:** Assesses source credibility and document quality.

**Features:**
- ✅ Domain authority scoring
- ✅ Recency scoring
- ✅ Source type classification
- ✅ Configurable weights

**Usage:**
```python
from src.rag.credibility_scorer import CredibilityScorer

# Initialize
scorer = CredibilityScorer()

# Score document
score = scorer.score_document({
    'text': '...',
    'metadata': {
        'source': 'Reuters',
        'date': '2024-01-15',
        'url': 'https://reuters.com/...'
    }
})

print(score)
# {
#     'overall_score': 0.85,
#     'domain_score': 0.9,
#     'recency_score': 1.0,
#     'source_type': 'news'
# }
```

**Scoring Factors:**
- **Domain Authority:** Trusted sources get higher scores
- **Recency:** Recent documents score higher
- **Source Type:** Academic > News > Blog > Social Media
- **URL Structure:** Official domains score higher

### 6. Complete RAG Pipeline (`src/rag/rag_pipeline.py`)

**Purpose:** Integrates all components into end-to-end pipeline.

**Features:**
- ✅ Hybrid retrieval
- ✅ Re-ranking
- ✅ Credibility scoring
- ✅ Stance detection
- ✅ Evidence aggregation
- ✅ User context prioritization

**Usage:**
```python
from src.rag.rag_pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline(
    collection_name="fact_check_kb",
    use_reranker=True,
    use_credibility=True
)

# Verify claim
result = pipeline.verify_claim(
    claim="India's GDP grew 8% in 2024",
    user_context_docs=None,  # Optional user-provided docs
    top_k=5
)

print(result)
# {
#     'claim': '...',
#     'evidence': [
#         {
#             'text': '...',
#             'relevance_score': 0.92,
#             'credibility_score': 0.85,
#             'stance': 'SUPPORTS',
#             'source': 'Reuters'
#         }
#     ],
#     'aggregated_stance': {
#         'overall': 'SUPPORTS',
#         'confidence': 0.87
#     }
# }
```

## Configuration

### Vector Database Config

```python
# In your code
db = VectorDatabase(
    collection_name="fact_check_kb",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    persist_directory="data/chroma_db"
)
```

### Hybrid Retrieval Config

```python
retriever = HybridRetriever(
    vector_weight=0.7,  # Adjust based on your use case
    sparse_weight=0.3,
    fusion_method="rrf"  # or "weighted_sum"
)
```

### Re-ranker Config

```python
reranker = Reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size=32
)
```

### Credibility Scorer Config

```python
scorer = CredibilityScorer(
    domain_weight=0.4,
    recency_weight=0.3,
    source_type_weight=0.3
)
```

## Data Ingestion

### Step 1: Prepare Documents

```python
documents = [
    {
        'text': "India's GDP grew 8.2% in fiscal year 2024...",
        'metadata': {
            'source': 'Economic Times',
            'date': '2024-01-15',
            'url': 'https://economictimes.com/...',
            'author': 'John Doe',
            'category': 'economics'
        }
    },
    # More documents...
]
```

### Step 2: Ingest into Vector DB

```python
from src.rag.vector_database import VectorDatabase

db = VectorDatabase(collection_name="fact_check_kb")

# Batch ingestion
db.add_documents(documents)

print(f"Added {len(documents)} documents")
```

### Step 3: Index for Sparse Retrieval

```python
from src.rag.sparse_retrieval import SparseRetriever

sparse = SparseRetriever()
sparse.index_documents(documents)

print("Sparse index created")
```

### Step 4: Test Retrieval

```python
from src.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

result = pipeline.verify_claim("India GDP growth 2024")
print(f"Found {len(result['evidence'])} evidence pieces")
```

## User Context Documents

### Priority System

When users provide context documents:

1. **User Documents:** Priority 1.0 (HIGHEST)
2. **Knowledge Base:** Priority 0.7 (MEDIUM)
3. **Web Search:** Priority 0.5 (LOWEST)

### Usage

```python
# User provides claim + supporting documents
result = pipeline.verify_claim(
    claim="Our revenue grew 50% in Q3",
    user_context_docs=[
        {
            'text': "Q3 Financial Report...",
            'metadata': {
                'document_name': 'Q3 Report',
                'is_user_provided': True
            }
        }
    ]
)

# User docs appear FIRST in evidence
for evidence in result['evidence']:
    if evidence.get('is_user_provided'):
        print(f"User doc: {evidence['text'][:100]}")
```

## Performance Optimization

### 1. Batch Processing

```python
# Process multiple claims at once
claims = ["claim1", "claim2", "claim3"]

results = []
for claim in claims:
    result = pipeline.verify_claim(claim)
    results.append(result)
```

### 2. Caching

```python
# Cache embeddings
db = VectorDatabase(
    collection_name="fact_check_kb",
    cache_embeddings=True
)
```

### 3. GPU Acceleration

```python
# Use GPU for embeddings
db = VectorDatabase(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # or "cpu"
)
```

### 4. Limit Results

```python
# Retrieve fewer documents for faster processing
result = pipeline.verify_claim(
    claim="...",
    top_k=5  # Instead of 10 or 20
)
```

## Testing

### Test Vector Database

```bash
cd src/rag
python vector_database.py
```

**Expected Output:**
- Created collection
- Added sample documents
- Search results

### Test Hybrid Retrieval

```bash
python hybrid_retrieval.py
```

**Expected Output:**
- Vector search results
- Sparse search results
- Hybrid results (combined)

### Test Complete Pipeline

```bash
python rag_pipeline.py
```

**Expected Output:**
- Retrieved evidence
- Relevance scores
- Credibility scores
- Stance detection
- Aggregated results

## Integration with Main Pipeline

```python
from src.main_pipeline import FactVerificationPipeline

# Initialize (RAG is automatically included)
pipeline = FactVerificationPipeline()

# Verify claim
result = pipeline.verify_claim_with_context(
    claim_input="India's GDP grew 8% in Q3",
    context_documents=[...]  # Optional
)

# RAG results are in result['evidence']
for evidence in result['evidence']:
    print(f"Source: {evidence['source']}")
    print(f"Relevance: {evidence['relevance_score']:.2f}")
    print(f"Credibility: {evidence['credibility_score']:.2f}")
    print(f"Stance: {evidence['stance']}")
```

## Troubleshooting

### Issue: No results found

**Causes:**
- Empty database
- Query too specific
- Embedding mismatch

**Solutions:**
1. Check if data is ingested: `db.get_collection_stats()`
2. Try broader query
3. Lower similarity threshold

### Issue: Slow retrieval

**Solutions:**
1. Reduce `top_k`
2. Use GPU for embeddings
3. Enable caching
4. Use smaller embedding model

### Issue: Low relevance scores

**Solutions:**
1. Enable re-ranker
2. Adjust hybrid weights
3. Use better embedding model
4. Add more diverse data

## Best Practices

1. **Data Quality:** Ingest high-quality, diverse documents
2. **Metadata:** Include rich metadata (source, date, author)
3. **Hybrid Search:** Use both vector and sparse for best results
4. **Re-ranking:** Always use re-ranker for top results
5. **Credibility:** Weight trusted sources higher
6. **User Context:** Prioritize user-provided documents
7. **Monitoring:** Track retrieval quality metrics

## Summary

✅ **Complete RAG Pipeline** with 6 components

✅ **Hybrid Retrieval** (vector + sparse)

✅ **Re-ranking** with cross-encoder

✅ **Credibility Scoring** for source assessment

✅ **User Context Priority** for user-provided docs

✅ **Production-Ready** with optimization options

✅ **Well-Tested** with comprehensive examples
