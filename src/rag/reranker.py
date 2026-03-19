"""
Re-Ranking Module
Uses cross-encoder to re-rank retrieved documents
"""

import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-rank retrieved documents using cross-encoder
    
    Cross-encoders are more accurate than bi-encoders for ranking
    but slower, so we use them only on top candidates
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker
        
        Args:
            model_name: Cross-encoder model name
        """
        logger.info(f"Loading cross-encoder: {model_name}")
        
        self.model = CrossEncoder(model_name)
        
        logger.info("✓ Reranker initialized")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents
        
        Args:
            query: Search query
            documents: List of candidate documents
            top_k: Number of top results to return
            return_scores: Whether to include scores
        
        Returns:
            Re-ranked documents with scores
        """
        if not documents:
            return []
        
        logger.debug(f"Re-ranking {len(documents)} documents...")
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked_indices = scores.argsort()[::-1][:top_k]
        
        # Build results
        results = []
        for idx in ranked_indices:
            result = {
                'document': documents[idx],
                'score': float(scores[idx]),
                'original_index': int(idx)
            }
            results.append(result)
        
        logger.debug(f"Re-ranked to top {len(results)} documents")
        
        return results
    
    def rerank_with_metadata(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        document_key: str = 'document'
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results that include metadata
        
        Args:
            query: Search query
            results: List of result dicts with documents and metadata
            top_k: Number of top results
            document_key: Key in dict containing document text
        
        Returns:
            Re-ranked results with original metadata preserved
        """
        if not results:
            return []
        
        # Extract documents
        documents = [r[document_key] for r in results]
        
        # Create pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked_indices = scores.argsort()[::-1][:top_k]
        
        # Build results with metadata
        reranked = []
        for idx in ranked_indices:
            result = results[idx].copy()
            result['rerank_score'] = float(scores[idx])
            result['original_rank'] = int(idx)
            reranked.append(result)
        
        return reranked


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    reranker = Reranker()
    
    query = "GDP growth in India"
    
    docs = [
        "Stock market reaches new highs",  # Less relevant
        "India's GDP grew 8% in 2024",  # Most relevant
        "Inflation remains stable",  # Less relevant
        "Economic growth driven by services",  # Relevant
        "Manufacturing sector improvement"  # Somewhat relevant
    ]
    
    print("\n" + "="*60)
    print("RE-RANKING TEST")
    print("="*60)
    print(f"\nQuery: {query}\n")
    
    print("Original order:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc}")
    
    # Re-rank
    results = reranker.rerank(query, docs, top_k=5)
    
    print("\nRe-ranked order:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   {result['document']}")
        print()