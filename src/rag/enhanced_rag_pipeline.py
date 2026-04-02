"""
Enhanced RAG Pipeline with Multi-Collection Support
Handles both news articles and uploaded documents with smart prioritization
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from src.rag.vector_database import VectorDatabase
from src.rag.retrieval import Retrieval
from src.rag.reranker import Reranker
from src.rag.credibility_scorer import CredibilityScorer

logger = logging.getLogger(__name__)


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with multi-collection support
    
    Features:
    - Search multiple collections (news_articles, uploaded_documents)
    - Context-aware prioritization
    - Credibility scoring per source type
    - Smart result merging and reranking
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        model_manager,
        collections: List[str] = None,
        search_strategy: str = 'context_aware'
    ):
        """
        Initialize enhanced RAG pipeline
        
        Args:
            vector_db: Vector database instance
            model_manager: Model manager for embeddings
            collections: List of collections to search
            search_strategy: 'context_aware', 'equal_weight', 'prioritize_uploads'
        """
        self.vector_db = vector_db
        self.model_manager = model_manager
        self.collections = collections or ['news_articles', 'uploaded_documents']
        self.search_strategy = search_strategy
        
        # Initialize components
        self.retrieval = Retrieval(vector_db, model_manager)
        self.reranker = Reranker(model_manager)
        self.credibility_scorer = CredibilityScorer()
        
        # Collection-specific settings
        self.collection_config = {
            'news_articles': {
                'credibility_base': 0.85,
                'weight': 1.0,
                'description': 'Scraped news articles from reliable sources'
            },
            'uploaded_documents': {
                'credibility_base': 0.70,
                'weight': 0.8,
                'description': 'User-uploaded documents'
            }
        }
        
        logger.info(f"✓ EnhancedRAGPipeline initialized")
        logger.info(f"  Collections: {self.collections}")
        logger.info(f"  Strategy: {search_strategy}")
    
    def verify_claim(
        self,
        claim: str,
        top_k: int = 5,
        collection_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Verify claim against multiple collections
        
        Args:
            claim: Claim to verify
            top_k: Number of top results to return
            collection_weights: Custom weights per collection
        
        Returns:
            Verification result with evidence from multiple sources
        """
        logger.info(f"Verifying claim across {len(self.collections)} collections")
        logger.info(f"Claim: {claim[:100]}...")
        
        # Step 1: Determine search strategy
        search_weights = self._determine_search_weights(
            claim, collection_weights
        )
        
        # Step 2: Search each collection
        all_results = []
        collection_results = {}
        
        for collection in self.collections:
            if collection not in search_weights:
                continue
                
            try:
                # Search collection
                results = self._search_collection(
                    collection, claim, top_k * 2  # Get more for reranking
                )
                
                # Add collection metadata
                for result in results:
                    result['collection'] = collection
                    result['collection_weight'] = search_weights[collection]
                
                collection_results[collection] = results
                all_results.extend(results)
                
                logger.info(f"  {collection}: {len(results)} results")
                
            except Exception as e:
                logger.error(f"Search failed for {collection}: {e}")
                collection_results[collection] = []
        
        if not all_results:
            return self._no_evidence_response(claim)
        
        # Step 3: Merge and rerank results
        final_results = self._merge_and_rerank(
            all_results, claim, top_k
        )
        
        # Step 4: Generate verification verdict
        verdict_result = self._generate_verdict(
            claim, final_results, collection_results
        )
        
        return verdict_result
    
    def _determine_search_weights(
        self,
        claim: str,
        custom_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Determine search weights based on strategy and claim content"""
        
        if custom_weights:
            return custom_weights
        
        if self.search_strategy == 'equal_weight':
            return {col: 1.0 for col in self.collections}
        
        elif self.search_strategy == 'prioritize_uploads':
            return {
                'news_articles': 0.7,
                'uploaded_documents': 1.0
            }
        
        elif self.search_strategy == 'context_aware':
            return self._context_aware_weights(claim)
        
        else:
            return {col: 1.0 for col in self.collections}
    
    def _context_aware_weights(self, claim: str) -> Dict[str, float]:
        """Determine weights based on claim content"""
        
        claim_lower = claim.lower()
        
        # Keywords that suggest public/news content
        public_keywords = [
            'gdp', 'economy', 'government', 'country', 'nation', 'global',
            'world', 'international', 'market', 'stock', 'election',
            'president', 'minister', 'policy', 'law', 'regulation'
        ]
        
        # Keywords that suggest private/company content
        private_keywords = [
            'our', 'we', 'company', 'revenue', 'profit', 'quarter',
            'fiscal', 'internal', 'department', 'team', 'project',
            'budget', 'expenses', 'sales', 'customers'
        ]
        
        public_score = sum(1 for keyword in public_keywords if keyword in claim_lower)
        private_score = sum(1 for keyword in private_keywords if keyword in claim_lower)
        
        if private_score > public_score:
            # Prioritize uploaded documents
            return {
                'news_articles': 0.6,
                'uploaded_documents': 1.0
            }
        elif public_score > private_score:
            # Prioritize news articles
            return {
                'news_articles': 1.0,
                'uploaded_documents': 0.7
            }
        else:
            # Equal weight
            return {
                'news_articles': 1.0,
                'uploaded_documents': 0.9
            }
    
    def _search_collection(
        self,
        collection: str,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search a specific collection"""
        
        try:
            # Use existing retrieval component
            results = self.retrieval.retrieve(
                collection_name=collection,
                query=query,
                top_k=top_k
            )
            
            # Add collection-specific metadata
            config = self.collection_config.get(collection, {})
            
            for result in results:
                result['source_type'] = collection
                result['base_credibility'] = config.get('credibility_base', 0.75)
                result['source_description'] = config.get('description', 'Unknown source')
            
            return results
            
        except Exception as e:
            logger.error(f"Collection search failed: {e}")
            return []
    
    def _merge_and_rerank(
        self,
        all_results: List[Dict[str, Any]],
        claim: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Merge results from all collections and rerank"""
        
        logger.info(f"Merging and reranking {len(all_results)} results")
        
        # Step 1: Apply collection weights to scores
        for result in all_results:
            original_score = result.get('score', 0.0)
            collection_weight = result.get('collection_weight', 1.0)
            result['weighted_score'] = original_score * collection_weight
        
        # Step 2: Remove duplicates (same content from different collections)
        unique_results = self._deduplicate_results(all_results)
        
        # Step 3: Rerank using cross-encoder
        try:
            reranked_results = self.reranker.rerank(
                query=claim,
                documents=[r['text'] for r in unique_results],
                metadata=unique_results
            )
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            # Fallback to weighted score sorting
            reranked_results = sorted(
                unique_results,
                key=lambda x: x['weighted_score'],
                reverse=True
            )
        
        # Step 4: Apply credibility scoring
        for result in reranked_results:
            credibility = self.credibility_scorer.score_document(
                result['text'],
                result.get('metadata', {})
            )
            result['credibility_score'] = credibility
            
            # Combine rerank score with credibility
            rerank_score = result.get('rerank_score', result['weighted_score'])
            result['final_score'] = (rerank_score * 0.7) + (credibility * 0.3)
        
        # Step 5: Final sorting and top-k selection
        final_results = sorted(
            reranked_results,
            key=lambda x: x['final_score'],
            reverse=True
        )[:top_k]
        
        logger.info(f"✓ Final results: {len(final_results)}")
        return final_results
    
    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate content from different collections"""
        
        seen_texts = set()
        unique_results = []
        
        for result in results:
            text = result['text']
            
            # Simple deduplication by text similarity
            is_duplicate = False
            for seen_text in seen_texts:
                if self._text_similarity(text, seen_text) > 0.9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(text)
                unique_results.append(result)
        
        logger.info(f"Deduplication: {len(results)} → {len(unique_results)}")
        return unique_results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple Jaccard similarity)"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_verdict(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
        collection_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Generate final verification verdict"""
        
        if not evidence:
            return self._no_evidence_response(claim)
        
        # Analyze evidence for stance
        support_count = 0
        refute_count = 0
        total_confidence = 0.0
        
        evidence_summary = []
        
        for result in evidence:
            # Simple stance detection (you can use trained model here)
            stance = self._detect_stance(claim, result['text'])
            
            result['stance'] = stance['label']
            result['stance_confidence'] = stance['confidence']
            
            if stance['label'] == 'SUPPORTS':
                support_count += stance['confidence']
            elif stance['label'] == 'REFUTES':
                refute_count += stance['confidence']
            
            total_confidence += result['final_score']
            
            evidence_summary.append({
                'text': result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                'source': result['collection'],
                'stance': stance['label'],
                'confidence': result['final_score'],
                'credibility': result['credibility_score']
            })
        
        # Determine overall verdict
        if support_count > refute_count and support_count > 0.5:
            verdict = 'SUPPORTED'
            confidence = min(support_count / len(evidence), 0.95)
        elif refute_count > support_count and refute_count > 0.5:
            verdict = 'REFUTED'
            confidence = min(refute_count / len(evidence), 0.95)
        else:
            verdict = 'INSUFFICIENT'
            confidence = 0.3
        
        # Generate explanation
        explanation = self._generate_explanation(
            verdict, evidence_summary, collection_results
        )
        
        return {
            'verdict': verdict,
            'confidence': confidence * 100,  # Convert to percentage
            'explanation': explanation,
            'evidence': evidence_summary,
            'evidence_sources': {
                collection: len(results) 
                for collection, results in collection_results.items()
            },
            'search_metadata': {
                'total_results_found': sum(len(results) for results in collection_results.values()),
                'collections_searched': list(collection_results.keys()),
                'search_strategy': self.search_strategy
            }
        }
    
    def _detect_stance(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Simple stance detection (replace with trained model)"""
        
        # Simple keyword-based stance detection
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Extract key terms from claim
        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        
        # Calculate overlap
        overlap = len(claim_words.intersection(evidence_words))
        total_words = len(claim_words)
        
        overlap_ratio = overlap / total_words if total_words > 0 else 0
        
        # Simple heuristic
        if overlap_ratio > 0.3:
            # Check for negation words
            negation_words = ['not', 'no', 'never', 'false', 'incorrect', 'wrong']
            has_negation = any(word in evidence_lower for word in negation_words)
            
            if has_negation:
                return {'label': 'REFUTES', 'confidence': 0.6}
            else:
                return {'label': 'SUPPORTS', 'confidence': 0.7}
        else:
            return {'label': 'NOT_ENOUGH_INFO', 'confidence': 0.4}
    
    def _generate_explanation(
        self,
        verdict: str,
        evidence: List[Dict[str, Any]],
        collection_results: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate human-readable explanation"""
        
        source_counts = {
            'news_articles': len(collection_results.get('news_articles', [])),
            'uploaded_documents': len(collection_results.get('uploaded_documents', []))
        }
        
        explanation_parts = []
        
        # Verdict explanation
        if verdict == 'SUPPORTED':
            explanation_parts.append("The claim is supported by available evidence.")
        elif verdict == 'REFUTED':
            explanation_parts.append("The claim is contradicted by available evidence.")
        else:
            explanation_parts.append("There is insufficient evidence to verify this claim.")
        
        # Source information
        source_info = []
        if source_counts['news_articles'] > 0:
            source_info.append(f"{source_counts['news_articles']} news articles")
        if source_counts['uploaded_documents'] > 0:
            source_info.append(f"{source_counts['uploaded_documents']} uploaded documents")
        
        if source_info:
            explanation_parts.append(f"Evidence found in: {', '.join(source_info)}.")
        
        # Top evidence
        if evidence:
            top_evidence = evidence[0]
            explanation_parts.append(
                f"Key evidence: {top_evidence['text'][:150]}..."
            )
        
        return " ".join(explanation_parts)
    
    def _no_evidence_response(self, claim: str) -> Dict[str, Any]:
        """Response when no evidence is found"""
        
        return {
            'verdict': 'INSUFFICIENT',
            'confidence': 0.0,
            'explanation': "No relevant evidence found in the available knowledge base.",
            'evidence': [],
            'evidence_sources': {collection: 0 for collection in self.collections},
            'search_metadata': {
                'total_results_found': 0,
                'collections_searched': self.collections,
                'search_strategy': self.search_strategy
            }
        }
    
    def add_collection(
        self,
        collection_name: str,
        config: Dict[str, Any] = None
    ):
        """Add a new collection to search"""
        
        if collection_name not in self.collections:
            self.collections.append(collection_name)
        
        if config:
            self.collection_config[collection_name] = config
        
        logger.info(f"Added collection: {collection_name}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        
        stats = {}
        
        for collection in self.collections:
            try:
                collection_obj = self.vector_db.get_collection(collection)
                count = len(collection_obj.get()['ids'])
                stats[collection] = {
                    'document_count': count,
                    'status': 'available'
                }
            except Exception as e:
                stats[collection] = {
                    'document_count': 0,
                    'status': f'error: {str(e)}'
                }
        
        return stats


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("ENHANCED RAG PIPELINE TEST")
    print("="*80)
    
    # This would require initialized components
    print("\nTo test:")
    print("1. Initialize: rag = EnhancedRAGPipeline(vector_db, model_manager)")
    print("2. Verify: result = rag.verify_claim('India GDP grew 8%')")
    print("3. Check stats: stats = rag.get_collection_stats()")