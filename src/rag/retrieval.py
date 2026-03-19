"""
RAG Pipeline with User Context Document Priority
Handles evidence retrieval with priority for user-provided documents
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline that prioritizes user-provided context documents
    
    Priority order:
    1. User-provided context documents (if any) - HIGHEST PRIORITY
    2. Vector DB search results
    3. Web search (if needed)
    """
    
    def __init__(self, vector_db=None, web_search=None):
        """
        Initialize RAG pipeline
        
        Args:
            vector_db: Vector database instance (ChromaDB, FAISS, etc.)
            web_search: Web search instance (optional)
        """
        self.vector_db = vector_db
        self.web_search = web_search
        logger.info("RAG Pipeline initialized")
    
    def retrieve_evidence(
        self,
        claim: str,
        user_context_docs: List[Dict] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve evidence for claim
        
        Priority order:
        1. User-provided context documents (if any) - HIGHEST PRIORITY
        2. Vector DB search results
        3. Web search (if needed)
        
        Args:
            claim: The claim to verify
            user_context_docs: List of user-provided context documents
                (from InputProcessor.process_with_context)
            top_k: Number of results from vector DB
        
        Returns:
            List of evidence with priority scores
        """
        all_evidence = []
        
        # ==========================================
        # PRIORITY 1: User-provided context docs
        # ==========================================
        
        if user_context_docs:
            logger.info(f"Adding {len(user_context_docs)} user-provided context docs")
            
            for doc in user_context_docs:
                # Split into chunks (for better matching)
                chunks = self._chunk_document(doc['text'])
                
                for chunk in chunks:
                    all_evidence.append({
                        'text': chunk,
                        'source': doc['metadata'].get('document_name', 'User Document'),
                        'source_type': 'user_provided',
                        'priority': 1.0,  # HIGHEST PRIORITY
                        'credibility': 1.0,  # User-provided = trusted
                        'language': doc['language'],
                        'metadata': doc['metadata']
                    })
            
            logger.info(f"✓ Added {len(all_evidence)} chunks from user documents")
        
        # ==========================================
        # PRIORITY 2: Vector DB search
        # ==========================================
        
        if self.vector_db:
            logger.info("Searching vector database...")
            
            try:
                db_results = self.vector_db.search(claim, top_k=top_k)
                
                for result in db_results:
                    all_evidence.append({
                        'text': result['text'],
                        'source': result['source'],
                        'source_type': 'knowledge_base',
                        'priority': 0.7,  # Lower than user docs
                        'credibility': result.get('credibility', 0.8),
                        'language': result.get('language', 'en'),
                        'metadata': result.get('metadata', {})
                    })
                
                logger.info(f"✓ Added {len(db_results)} results from vector DB")
            except Exception as e:
                logger.error(f"Vector DB search failed: {str(e)}")
        
        # ==========================================
        # PRIORITY 3: Web search (optional)
        # ==========================================
        
        # Only if not enough evidence from user docs + DB
        if len(all_evidence) < top_k and self.web_search:
            logger.info("Insufficient evidence, performing web search...")
            
            try:
                web_results = self.web_search.search(claim, top_k=top_k - len(all_evidence))
                
                for result in web_results:
                    all_evidence.append({
                        'text': result['text'],
                        'source': result['url'],
                        'source_type': 'web',
                        'priority': 0.5,  # Lowest priority
                        'credibility': result.get('credibility', 0.6),
                        'language': result.get('language', 'en'),
                        'metadata': result.get('metadata', {})
                    })
                
                logger.info(f"✓ Added {len(web_results)} results from web search")
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}")
        
        # ==========================================
        # Sort by priority
        # ==========================================
        
        all_evidence.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"✓ Total evidence retrieved: {len(all_evidence)}")
        
        return all_evidence
    
    def _chunk_document(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split document into chunks for better matching
        
        Args:
            text: Document text
            chunk_size: Number of words per chunk
        
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    def index_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        priority: str = 'normal'
    ):
        """
        Index a document in the vector database
        
        Args:
            text: Document text
            metadata: Document metadata
            priority: Priority level ('high', 'normal', 'low')
        """
        if not self.vector_db:
            logger.warning("No vector database configured")
            return
        
        try:
            # Chunk document
            chunks = self._chunk_document(text)
            
            # Index each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunk_metadata['priority'] = priority
                
                self.vector_db.add(
                    text=chunk,
                    metadata=chunk_metadata
                )
            
            logger.info(f"✓ Indexed {len(chunks)} chunks with priority: {priority}")
        except Exception as e:
            logger.error(f"Failed to index document: {str(e)}")
