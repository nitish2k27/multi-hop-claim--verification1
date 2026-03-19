"""
Complete RAG Pipeline
Integrates retrieval, re-ranking, credibility, stance detection
"""

import logging
from typing import List, Dict, Any, Optional
from src.rag.vector_database import VectorDatabase
from src.rag.hybrid_retrieval import HybridRetriever
from src.rag.reranker import Reranker
from src.rag.credibility_scorer import CredibilityScorer
from src.nlp.stance_detection import StanceDetector
from src.nlp.model_manager import ModelManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG Pipeline
    
    Flow:
    1. Hybrid retrieval (vector + BM25)
    2. Re-ranking
    3. Credibility scoring
    4. Stance detection
    5. Evidence aggregation
    6. Context preparation for LLM
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        collection_name: str = "news_articles",
        nlp_model_manager: Optional[ModelManager] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            vector_db: VectorDatabase instance
            collection_name: Default collection to search
            nlp_model_manager: ModelManager for NLP components
        """
        logger.info("Initializing RAG Pipeline...")
        
        self.vector_db = vector_db
        self.collection_name = collection_name
        
        # Initialize components
        logger.info("Loading RAG components...")
        
        # Check if collection exists and has documents
        try:
            collection = vector_db.get_collection(collection_name)
            doc_count = collection.count()
            
            if doc_count == 0:
                logger.warning(f"Collection '{collection_name}' is empty!")
                logger.warning("Add documents using DataIngestionHelper before searching")
                self.has_data = False
            else:
                logger.info(f"Collection '{collection_name}' has {doc_count} documents")
                self.has_data = True
                
                # Initialize hybrid retriever (needs documents)
                self.hybrid_retriever = HybridRetriever(
                    vector_db=vector_db,
                    collection_name=collection_name
                )
        except Exception as e:
            logger.warning(f"Collection '{collection_name}' not found: {str(e)}")
            logger.warning("Create collection and add documents first")
            self.has_data = False
            self.hybrid_retriever = None
        
        # Initialize other components (don't require collection)
        self.reranker = Reranker()
        self.credibility_scorer = CredibilityScorer()
        
        # Initialize stance detector
        if nlp_model_manager is None:
            nlp_model_manager = ModelManager()
        
        self.stance_detector = StanceDetector(nlp_model_manager)
        
        logger.info("✓ RAG Pipeline initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        user_context_docs: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Complete retrieval pipeline
        
        Args:
            query: Search query
            top_k: Number of final results
            filters: Metadata filters for search
            user_context_docs: User-provided context documents (high priority)
        
        Returns:
            Retrieved and processed evidence
        """
        logger.info(f"Running RAG pipeline for query: '{query[:100]}...'")
        
        all_evidence = []
        
        # ==========================================
        # PRIORITY 1: User-provided context
        # ==========================================
        
        if user_context_docs:
            logger.info(f"Adding {len(user_context_docs)} user-provided documents")
            
            for doc in user_context_docs:
                all_evidence.append({
                    'document': doc['text'],
                    'source': doc.get('metadata', {}).get('document_name', 'User Document'),
                    'source_type': 'user_provided',
                    'priority': 1.0,  # HIGHEST
                    'credibility': {
                        'total_score': 1.0,
                        'tier': 'USER_PROVIDED'
                    },
                    'metadata': doc.get('metadata', {})
                })
        
        # ==========================================
        # PRIORITY 2: Database search
        # ==========================================
        
        if self.has_data and self.hybrid_retriever:
            logger.info("→ Step 1: Hybrid retrieval (vector + BM25)")
            
            # Retrieve more candidates for re-ranking
            candidates = self.hybrid_retriever.search(
                query=query,
                top_k=top_k * 4  # Get 4x for re-ranking
            )
            
            logger.info(f"  ✓ Retrieved {len(candidates)} candidates")
            
            # ==========================================
            # Step 2: Re-ranking
            # ==========================================
            
            logger.info("→ Step 2: Re-ranking with cross-encoder")
            
            reranked = self.reranker.rerank_with_metadata(
                query=query,
                results=candidates,
                top_k=top_k,
                document_key='document'
            )
            
            logger.info(f"  ✓ Re-ranked to top {len(reranked)}")
            
            # ==========================================
            # Step 3: Credibility scoring
            # ==========================================
            
            logger.info("→ Step 3: Credibility scoring")
            
            for result in reranked:
                metadata = result.get('metadata', {})
                
                credibility = self.credibility_scorer.score(
                    url=metadata.get('url'),
                    source=metadata.get('source'),
                    publish_date=metadata.get('publish_date'),
                    source_type=metadata.get('source_type')
                )
                
                result['credibility'] = credibility
                result['source'] = metadata.get('source', 'unknown')
                result['source_type'] = 'knowledge_base'
                result['priority'] = 0.7  # Lower than user docs
            
            logger.info(f"  ✓ Scored {len(reranked)} sources")
            
            all_evidence.extend(reranked)
        
        else:
            logger.warning("No database to search! Add documents first.")
        
        # ==========================================
        # Sort by priority (user docs first)
        # ==========================================
        
        all_evidence.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"✓ Total evidence collected: {len(all_evidence)}")
        
        return all_evidence
    
    def detect_stances(
        self,
        claim: str,
        evidence_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect stance for each evidence piece
        
        Args:
            claim: The claim being verified
            evidence_list: List of evidence dicts
        
        Returns:
            Evidence with stance information added
        """
        logger.info(f"→ Step 4: Stance detection for {len(evidence_list)} evidence pieces")
        
        for evidence in evidence_list:
            # Detect stance
            stance_result = self.stance_detector.detect(
                claim=claim,
                evidence=evidence['document']
            )
            
            # Add to evidence
            evidence['stance'] = stance_result['stance']
            evidence['stance_confidence'] = stance_result['confidence']
        
        # Count stances
        supports = sum(1 for e in evidence_list if e['stance'] == 'SUPPORTS')
        refutes = sum(1 for e in evidence_list if e['stance'] == 'REFUTES')
        neutral = sum(1 for e in evidence_list if e['stance'] == 'NEUTRAL')
        
        logger.info(f"  ✓ Stances: {supports} SUPPORTS, {refutes} REFUTES, {neutral} NEUTRAL")
        
        return evidence_list
    
    def aggregate_evidence(
        self,
        claim: str,
        evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate evidence and calculate verdict
        
        Args:
            claim: The claim
            evidence_list: Evidence with stances
        
        Returns:
            Aggregated analysis
        """
        logger.info("→ Step 5: Evidence aggregation")
        
        # Calculate weighted scores
        support_score = 0.0
        refute_score = 0.0
        neutral_score = 0.0
        
        for evidence in evidence_list:
            # Weight = stance_confidence * credibility * priority
            weight = (
                evidence['stance_confidence'] *
                evidence['credibility']['total_score'] *
                evidence['priority']
            )
            
            if evidence['stance'] == 'SUPPORTS':
                support_score += weight
            elif evidence['stance'] == 'REFUTES':
                refute_score += weight
            else:
                neutral_score += weight
        
        # Normalize to percentages
        total = support_score + refute_score + neutral_score
        
        if total > 0:
            support_pct = (support_score / total) * 100
            refute_pct = (refute_score / total) * 100
            neutral_pct = (neutral_score / total) * 100
        else:
            support_pct = refute_pct = neutral_pct = 0
        
        # Determine verdict
        if support_pct > 70:
            verdict = 'TRUE'
        elif support_pct > 50:
            verdict = 'MOSTLY TRUE'
        elif refute_pct > 70:
            verdict = 'FALSE'
        elif refute_pct > 50:
            verdict = 'MOSTLY FALSE'
        elif abs(support_pct - refute_pct) < 10:
            verdict = 'CONFLICTING'
        else:
            verdict = 'UNVERIFIABLE'
        
        # Confidence
        confidence = abs(support_pct - refute_pct)
        
        aggregation = {
            'verdict': verdict,
            'confidence': confidence,
            'support_percentage': support_pct,
            'refute_percentage': refute_pct,
            'neutral_percentage': neutral_pct,
            'support_score': support_score,
            'refute_score': refute_score,
            'num_evidence': len(evidence_list),
            'num_supports': sum(1 for e in evidence_list if e['stance'] == 'SUPPORTS'),
            'num_refutes': sum(1 for e in evidence_list if e['stance'] == 'REFUTES'),
            'num_neutral': sum(1 for e in evidence_list if e['stance'] == 'NEUTRAL')
        }
        
        logger.info(f"  ✓ Verdict: {verdict} (confidence: {confidence:.1f}%)")
        logger.info(f"    Support: {support_pct:.1f}%, Refute: {refute_pct:.1f}%")
        
        return aggregation
    
    def prepare_context_for_llm(
        self,
        claim: str,
        evidence_list: List[Dict[str, Any]],
        aggregation: Dict[str, Any],
        max_context_length: int = 2000
    ) -> str:
        """
        Prepare context for LLM generation
        
        Args:
            claim: The claim
            evidence_list: Evidence with stances
            aggregation: Aggregated analysis
            max_context_length: Maximum context length (chars)
        
        Returns:
            Formatted context string for LLM
        """
        logger.info("→ Step 6: Preparing context for LLM")
        
        context = f"""CLAIM TO VERIFY:
{claim}

EVIDENCE ANALYSIS:
"""
        
        # Add evidence pieces
        for i, evidence in enumerate(evidence_list[:10], 1):  # Max 10 pieces
            context += f"""
{i}. [{evidence['stance']}] (Credibility: {evidence['credibility']['total_score']:.2f}, Source: {evidence['source']})
   {evidence['document'][:200]}...
"""
            
            # Check length
            if len(context) > max_context_length:
                context += "\n[Additional evidence truncated...]"
                break
        
        # Add aggregation summary
        context += f"""

VERDICT CALCULATION:
- Support: {aggregation['support_percentage']:.1f}% ({aggregation['num_supports']} pieces)
- Refute: {aggregation['refute_percentage']:.1f}% ({aggregation['num_refutes']} pieces)
- Neutral: {aggregation['neutral_percentage']:.1f}% ({aggregation['num_neutral']} pieces)
- Preliminary Verdict: {aggregation['verdict']}
- Confidence: {aggregation['confidence']:.1f}%
"""
        
        logger.info(f"  ✓ Context prepared ({len(context)} chars)")
        
        return context
    
    def verify_claim(
        self,
        claim: str,
        top_k: int = 5,
        user_context_docs: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Complete claim verification pipeline
        
        MAIN METHOD - Use this for end-to-end verification
        
        Args:
            claim: Claim to verify
            top_k: Number of evidence pieces
            user_context_docs: Optional user-provided context
        
        Returns:
            Complete verification result
        """
        logger.info("\n" + "="*80)
        logger.info("CLAIM VERIFICATION PIPELINE")
        logger.info("="*80)
        logger.info(f"Claim: {claim}")
        logger.info("="*80 + "\n")
        
        # Step 1-3: Retrieve evidence
        evidence = self.retrieve(
            query=claim,
            top_k=top_k,
            user_context_docs=user_context_docs
        )
        
        if not evidence:
            logger.warning("No evidence found!")
            return {
                'claim': claim,
                'verdict': 'UNVERIFIABLE',
                'confidence': 0,
                'evidence': [],
                'error': 'No evidence found'
            }
        
        # Step 4: Detect stances
        evidence = self.detect_stances(claim, evidence)
        
        # Step 5: Aggregate
        aggregation = self.aggregate_evidence(claim, evidence)
        
        # Step 6: Prepare context for LLM
        llm_context = self.prepare_context_for_llm(claim, evidence, aggregation)
        
        logger.info("\n" + "="*80)
        logger.info("VERIFICATION COMPLETE")
        logger.info("="*80 + "\n")
        
        return {
            'claim': claim,
            'verdict': aggregation['verdict'],
            'confidence': aggregation['confidence'],
            'evidence': evidence,
            'aggregation': aggregation,
            'llm_context': llm_context,
            'ready_for_llm': True
        }


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("RAG PIPELINE DEMO")
    print("="*80)
    
    # Initialize (will warn if no data)
    vdb = VectorDatabase()
    
    # Check if demo collection exists
    try:
        stats = vdb.get_collection_stats('demo_collection')
        
        if stats['exists'] and stats['count'] > 0:
            # Initialize pipeline
            rag = RAGPipeline(vdb, collection_name='demo_collection')
            
            # Test claim
            claim = "India's GDP grew 8% in 2024"
            
            print(f"\nVerifying claim: {claim}\n")
            
            # Run verification
            result = rag.verify_claim(claim, top_k=3)
            
            # Display results
            print("\n" + "="*80)
            print("RESULTS")
            print("="*80)
            print(f"\nVerdict: {result['verdict']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"\nEvidence count: {len(result['evidence'])}")
            
            print("\nEvidence summary:")
            for i, ev in enumerate(result['evidence'][:3], 1):
                print(f"\n{i}. {ev['document'][:100]}...")
                print(f"   Stance: {ev['stance']}")
                print(f"   Source: {ev['source']}")
                print(f"   Credibility: {ev['credibility']['total_score']:.2f}")
        
        else:
            print("\nNo data in collection!")
            print("Run vector_database.py demo first to create sample data")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTo use RAG pipeline:")
        print("1. Create vector database")
        print("2. Add documents using DataIngestionHelper")
        print("3. Then run RAG pipeline")