"""
Hybrid Retrieval
Combines dense (vector) and sparse (BM25) search using Reciprocal Rank Fusion
"""

import logging
from typing import List, Dict, Any
from src.rag.vector_database import VectorDatabase
from src.rag.sparse_retrieval import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse search
    
    Uses Reciprocal Rank Fusion (RRF) to combine rankings
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        collection_name: str,
        k: int = 60  # RRF parameter
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vector_db: VectorDatabase instance
            collection_name: Collection to search
            k: RRF parameter (default: 60)
        """
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.k = k
        
        # Get collection for BM25
        collection = vector_db.get_collection(collection_name)
        
        # Get all documents for BM25
        logger.info("Loading documents for BM25 indexing...")
        all_docs = collection.get()
        
        self.documents = all_docs['documents']
        self.metadatas = all_docs['metadatas']
        self.ids = all_docs['ids']
        
        # Initialize BM25
        self.bm25_retriever = BM25Retriever(
            documents=self.documents,
            metadatas=self.metadatas
        )
        
        logger.info(f"✓ HybridRetriever initialized with {len(self.documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search
        
        Args:
            query: Search query
            top_k: Number of final results
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
        
        Returns:
            Fused results from both retrievers
        """
        # Get more results from each retriever for fusion
        retrieve_k = top_k * 3
        
        # Dense retrieval (vector search)
        logger.debug("Running dense retrieval...")
        dense_results = self.vector_db.search(
            collection_name=self.collection_name,
            query=query,
            top_k=retrieve_k
        )
        
        # Sparse retrieval (BM25)
        logger.debug("Running sparse retrieval...")
        sparse_results = self.bm25_retriever.search(query, top_k=retrieve_k)
        
        # Reciprocal Rank Fusion
        logger.debug("Fusing results with RRF...")
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight,
            sparse_weight
        )
        
        # Return top-k
        return fused_results[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: Dict,
        sparse_results: List[Dict],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine rankings using Reciprocal Rank Fusion
        
        RRF formula: score(d) = Σ weight / (k + rank(d))
        """
        scores = {}
        
        # Process dense results
        for rank, doc_id in enumerate(dense_results['ids'][0]):
            if doc_id not in scores:
                scores[doc_id] = {
                    'score': 0,
                    'document': dense_results['documents'][0][rank],
                    'metadata': dense_results['metadatas'][0][rank],
                    'id': doc_id
                }
            
            # RRF score
            scores[doc_id]['score'] += dense_weight / (self.k + rank + 1)
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = self.ids[result['index']]
            
            if doc_id not in scores:
                scores[doc_id] = {
                    'score': 0,
                    'document': result['document'],
                    'metadata': result['metadata'],
                    'id': doc_id
                }
            
            scores[doc_id]['score'] += sparse_weight / (self.k + rank + 1)
        
        # Sort by score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # This requires a populated vector database
    # See vector_database.py demo for setup
    
    print("HybridRetriever requires a populated VectorDatabase")
    print("Run vector_database.py demo first to create sample collection")