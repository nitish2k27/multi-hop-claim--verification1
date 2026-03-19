"""
BM25 Sparse Retrieval
Keyword-based document retrieval
"""

import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class BM25Retriever:
    """
    BM25-based sparse retrieval
    Complements dense vector search with keyword matching
    """
    
    def __init__(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Initialize BM25 retriever
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]
        
        logger.info(f"Initializing BM25 with {len(documents)} documents")
        
        # Tokenize documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        logger.info("✓ BM25 index built")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        
        - Lowercase
        - Remove stopwords
        - Word tokenization
        """
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
        
        return tokens
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using BM25
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of results with documents, scores, metadata
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(scores[idx]),
                'metadata': self.metadatas[idx],
                'index': int(idx)
            })
        
        return results


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample documents
    docs = [
        "India's GDP grew 8% in 2024",
        "Economic growth driven by services",
        "Manufacturing sector improvement",
        "Stock market reaches new highs",
        "Inflation remains stable"
    ]
    
    metadata = [
        {'source': 'reuters'},
        {'source': 'bbc'},
        {'source': 'times'},
        {'source': 'bloomberg'},
        {'source': 'guardian'}
    ]
    
    # Initialize
    bm25 = BM25Retriever(docs, metadata)
    
    # Search
    query = "GDP growth India"
    results = bm25.search(query, top_k=3)
    
    print(f"\nQuery: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   Doc: {result['document']}")
        print(f"   Source: {result['metadata']['source']}")
        print()