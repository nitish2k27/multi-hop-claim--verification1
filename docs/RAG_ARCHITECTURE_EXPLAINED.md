# RAG Architecture Explained - Complete System Structure

## 🏗️ **RAG System Overview**

Your fact verification system has **TWO RAG ARCHITECTURES**:

1. **Original RAG** (`src/rag/rag_pipeline.py`) - Single collection, basic retrieval
2. **Enhanced RAG** (`src/rag/enhanced_rag_pipeline.py`) - Multi-collection, smart prioritization

---

## 📁 **File Structure & Paths**

```
src/rag/
├── vector_database.py              # ChromaDB wrapper & data ingestion
├── rag_pipeline.py                 # Original RAG (single collection)
├── enhanced_rag_pipeline.py        # Enhanced RAG (multi-collection)
├── retrieval.py                    # Basic retrieval with user context priority
├── hybrid_retrieval.py             # Vector + BM25 search
├── reranker.py                     # Cross-encoder reranking
├── credibility_scorer.py           # Source credibility assessment
└── sparse_retrieval.py             # BM25 keyword search

data/
├── chroma_db/                      # ChromaDB storage (UUID-based)
│   ├── chroma.sqlite3              # ChromaDB metadata database
│   └── {collection_uuid}/          # Collection data (e.g., 6776f2bc-a214-4d6a-a19e-92a1560e610f)
│       ├── data_level0.bin         # Vector embeddings
│       ├── header.bin              # Collection header
│       ├── index_metadata.pickle   # Index metadata
│       ├── length.bin              # Data lengths
│       └── link_lists.bin          # HNSW index links
├── uploads/                        # Original uploaded files
│   └── {user_id}/                  # User-specific folder (e.g., test_user)
│       └── {date}/                 # Date folder (e.g., 20260403)
│           └── {file_hash}.{ext}   # Hashed filename (e.g., 6f50c243...e2.txt)
├── processed/                      # Processed datasets
├── raw/                           # Raw scraped data
└── README.md
```

**Important Note**: ChromaDB uses UUID-based storage internally. Collection names like `news_articles` and `uploaded_documents` are mapped to UUIDs in `chroma.sqlite3`, and the actual data is stored in UUID-named folders.

---

## 🔄 **Enhanced RAG Architecture (Multi-Collection)**

### **System Flow Diagram**
```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
│              "Did our company revenue grow 20%?"               │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                   CONTEXT-AWARE ROUTING                        │
├─────────────────────────────────────────────────────────────────┤
│ Keywords Analysis:                                              │
│ • "our company" → PRIVATE context → Prioritize user docs       │
│ • "GDP", "India" → PUBLIC context → Prioritize news articles   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-COLLECTION SEARCH                       │
├─────────────────────────────────────────────────────────────────┤
│ Collection 1: news_articles        Collection 2: uploaded_docs │
│ ├─ Weight: 0.6                     ├─ Weight: 1.0              │
│ ├─ Credibility: 0.85               ├─ Credibility: 0.70        │
│ ├─ Search: "revenue grow 20%"      ├─ Search: "revenue grow 20%" │
│ └─ Results: 3 articles             └─ Results: 5 documents     │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MERGE & RERANK                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. Apply collection weights to scores                          │
│ 2. Remove duplicates (same content)                            │
│ 3. Cross-encoder reranking                                     │
│ 4. Credibility scoring per source type                        │
│ 5. Final score = rerank_score * 0.7 + credibility * 0.3      │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FINAL RESULTS                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. User Doc: "Q4 revenue increased 22%" (score: 0.95)         │
│ 2. User Doc: "Annual growth target met" (score: 0.88)         │
│ 3. News: "Tech sector growth trends" (score: 0.72)            │
│ 4. User Doc: "Financial summary 2024" (score: 0.70)           │
│ 5. News: "Market analysis report" (score: 0.65)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 **User Document Scoring System**

### **1. Document Upload Process**
**File**: `src/document_processing/document_handler.py`

```python
# Step 1: Upload & Extract
document_data = document_handler.process_upload("company_report.pdf", user_id="user123")

# Step 2: Chunk Document
chunks = [
    {
        'chunk_id': 0,
        'text': 'Our Q4 revenue grew by 22% compared to last year...',
        'start_word': 0,
        'end_word': 500,
        'word_count': 500
    },
    # ... more chunks
]

# Step 3: Add to RAG Database
document_handler.add_to_rag(
    document_data,
    vector_db,
    collection_name='uploaded_documents'
)
```

**Storage Structure**:
```
# Original uploaded file
data/uploads/test_user/20260403/6f50c243df3802d93b9b1a6c881b8e906b6d233dd628b1bec071ee4c52c462e2.txt

# ChromaDB collections (UUID-based storage)
data/chroma_db/
├── chroma.sqlite3                                    # Metadata database
└── 6776f2bc-a214-4d6a-a19e-92a1560e610f/           # Collection UUID folder
    ├── data_level0.bin                              # Vector embeddings
    ├── header.bin                                   # Collection header
    ├── index_metadata.pickle                        # Index metadata
    ├── length.bin                                   # Data lengths
    └── link_lists.bin                               # HNSW index links

# Note: ChromaDB uses UUIDs for collection storage, not collection names
# Collection names (news_articles, uploaded_documents) are mapped to UUIDs internally
```

### **2. Scoring Hierarchy**
**File**: `src/rag/enhanced_rag_pipeline.py`

```python
# Collection Configuration
collection_config = {
    'news_articles': {
        'credibility_base': 0.85,    # High credibility (verified sources)
        'weight': 1.0,              # Standard weight
        'description': 'Scraped news articles from reliable sources'
    },
    'uploaded_documents': {
        'credibility_base': 0.70,    # Medium credibility (unverified)
        'weight': 0.8,              # Slightly lower weight
        'description': 'User-uploaded documents'
    }
}
```

### **3. Context-Aware Prioritization**
**File**: `src/rag/enhanced_rag_pipeline.py` → `_context_aware_weights()`

```python
def _context_aware_weights(self, claim: str) -> Dict[str, float]:
    """Smart weighting based on query content"""
    
    claim_lower = claim.lower()
    
    # Private/Company Keywords
    private_keywords = [
        'our', 'we', 'company', 'revenue', 'profit', 'quarter',
        'fiscal', 'internal', 'department', 'team', 'project'
    ]
    
    # Public/News Keywords  
    public_keywords = [
        'gdp', 'economy', 'government', 'country', 'nation',
        'global', 'world', 'international', 'market', 'election'
    ]
    
    private_score = sum(1 for keyword in private_keywords if keyword in claim_lower)
    public_score = sum(1 for keyword in public_keywords if keyword in claim_lower)
    
    if private_score > public_score:
        # Query about company/private matters → Prioritize user uploads
        return {
            'news_articles': 0.6,
            'uploaded_documents': 1.0  # Higher priority
        }
    elif public_score > private_score:
        # Query about public matters → Prioritize news
        return {
            'news_articles': 1.0,      # Higher priority
            'uploaded_documents': 0.7
        }
    else:
        # Neutral → Equal weight
        return {
            'news_articles': 1.0,
            'uploaded_documents': 0.9
        }
```

### **4. Final Scoring Formula**
**File**: `src/rag/enhanced_rag_pipeline.py` → `_merge_and_rerank()`

```python
# For each search result:
for result in all_results:
    # Step 1: Apply collection weight
    original_score = result.get('score', 0.0)           # Vector similarity
    collection_weight = result.get('collection_weight', 1.0)  # Context-aware weight
    weighted_score = original_score * collection_weight
    
    # Step 2: Cross-encoder reranking
    rerank_score = reranker.rerank(query, result['text'])
    
    # Step 3: Credibility scoring
    credibility = credibility_scorer.score_document(
        result['text'],
        result.get('metadata', {})
    )
    
    # Step 4: Final score combination
    final_score = (rerank_score * 0.7) + (credibility * 0.3)
    
    result['final_score'] = final_score
```

---

## 🔍 **Normal Retrieval RAG (Original)**

### **Single Collection Architecture**
**File**: `src/rag/rag_pipeline.py`

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                   SINGLE COLLECTION SEARCH                     │
├─────────────────────────────────────────────────────────────────┤
│ Collection: news_articles (default)                            │
│ ├─ Hybrid Retrieval (Vector + BM25)                           │
│ ├─ Get top 20 candidates                                       │
│ └─ No collection weighting                                     │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RERANK & SCORE                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Cross-encoder reranking                                     │
│ 2. Credibility scoring                                         │
│ 3. Stance detection                                            │
│ 4. Return top 5 results                                        │
└─────────────────────────────────────────────────────────────────┘
```

### **Usage**:
```python
# Original RAG - Single collection
from src.rag.rag_pipeline import RAGPipeline

rag = RAGPipeline(vector_db, collection_name='news_articles')
result = rag.verify_claim("India GDP grew 8%", top_k=5)

# Only searches news_articles collection
# No multi-collection support
# No context-aware prioritization
```

---

## 🆚 **Comparison: Original vs Enhanced RAG**

| Feature | Original RAG | Enhanced RAG |
|---------|-------------|--------------|
| **Collections** | Single (`news_articles`) | Multiple (`news_articles` + `uploaded_documents`) |
| **Prioritization** | None | Context-aware smart weighting |
| **User Documents** | ❌ Not supported | ✅ Full support with priority |
| **Credibility** | Basic scoring | Per-collection credibility tiers |
| **Search Strategy** | Fixed | Configurable (`context_aware`, `equal_weight`, `prioritize_uploads`) |
| **File Path** | `src/rag/rag_pipeline.py` | `src/rag/enhanced_rag_pipeline.py` |
| **Use Case** | Simple fact-checking | Production system with user uploads |

---

## 📈 **Scoring Examples**

### **Example 1: Company Query**
**Query**: "Did our Q4 revenue grow 20%?"

```python
# Context Analysis
private_keywords_found = ['our', 'revenue']  # 2 matches
public_keywords_found = []                   # 0 matches

# Collection Weights
weights = {
    'news_articles': 0.6,        # Lower priority
    'uploaded_documents': 1.0    # Higher priority
}

# Search Results
news_result = {
    'text': 'Tech sector shows growth trends...',
    'original_score': 0.75,
    'collection_weight': 0.6,
    'weighted_score': 0.75 * 0.6 = 0.45,
    'credibility': 0.85,
    'final_score': (0.45 * 0.7) + (0.85 * 0.3) = 0.57
}

user_doc_result = {
    'text': 'Our Q4 revenue increased by 22%...',
    'original_score': 0.80,
    'collection_weight': 1.0,
    'weighted_score': 0.80 * 1.0 = 0.80,
    'credibility': 0.70,
    'final_score': (0.80 * 0.7) + (0.70 * 0.3) = 0.77
}

# Final Ranking: User doc (0.77) > News (0.57)
```

### **Example 2: Public Query**
**Query**: "India's GDP grew 8% in 2024"

```python
# Context Analysis
private_keywords_found = []                    # 0 matches
public_keywords_found = ['gdp', 'india']      # 2 matches

# Collection Weights
weights = {
    'news_articles': 1.0,        # Higher priority
    'uploaded_documents': 0.7    # Lower priority
}

# Final Ranking: News articles prioritized over user docs
```

---

## 🔧 **Integration Points**

### **1. Document Upload Integration**
**File**: `src/enhanced_main_pipeline.py`

```python
# When user uploads document
def upload_document(self, file_path: str, user_id: str = "default"):
    # 1. Process document
    document_data = self.document_handler.process_upload(file_path, user_id)
    
    # 2. Add to RAG database
    self.document_handler.add_to_rag(
        document_data,
        self.vector_db,
        collection_name='uploaded_documents'  # Separate collection
    )
    
    # 3. Now available for search in enhanced RAG
```

### **2. Query Processing Integration**
**File**: `src/enhanced_main_pipeline.py`

```python
def verify_claim(self, claim: str):
    # Uses Enhanced RAG automatically
    result = self.multilingual_pipeline.verify_claim(claim)
    
    # Enhanced RAG searches both:
    # - news_articles (scraped news)
    # - uploaded_documents (user files)
```

---

## 🎯 **Key Benefits of Enhanced RAG**

1. **Smart Prioritization**: Automatically prioritizes relevant collections based on query context
2. **User Document Support**: Seamlessly integrates user uploads with existing knowledge base
3. **Credibility Tiers**: Different credibility scores for different source types
4. **Deduplication**: Removes duplicate content across collections
5. **Configurable Strategy**: Can switch between different search strategies
6. **Scalable**: Easy to add new collections (research papers, Wikipedia, etc.)

Your enhanced RAG system provides a production-ready, intelligent fact verification pipeline that adapts to user context and maintains high accuracy across diverse information sources! 🚀