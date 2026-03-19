"""
Main verification pipeline with context document support
Complete workflow from input processing to verification
"""

import logging
from typing import List, Dict, Any, Optional

from src.preprocessing.input_processor import InputProcessor
from src.rag.retrieval import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactVerificationPipeline:
    """
    Complete fact verification pipeline
    
    Workflow:
    1. Input Processing (claim + optional context docs)
    2. NLP Analysis (claim decomposition, entity extraction)
    3. Evidence Retrieval (prioritizing user-provided docs)
    4. Verification (stance detection, credibility scoring)
    5. Explanation Generation
    """
    
    def __init__(
        self,
        vector_db=None,
        web_search=None,
        nlp_pipeline=None,
        verifier=None
    ):
        """
        Initialize verification pipeline
        
        Args:
            vector_db: Vector database instance
            web_search: Web search instance
            nlp_pipeline: NLP pipeline instance
            verifier: Verification model instance
        """
        self.input_processor = InputProcessor()
        self.rag_pipeline = RAGPipeline(vector_db=vector_db, web_search=web_search)
        self.nlp_pipeline = nlp_pipeline
        self.verifier = verifier
        
        logger.info("✓ Fact Verification Pipeline initialized")
    
    def verify_claim_with_context(
        self,
        claim_input: str,
        context_documents: List[Dict] = None,
        claim_type: str = 'text'
    ) -> Dict[str, Any]:
        """
        Complete verification pipeline with optional context docs
        
        Args:
            claim_input: The claim to verify
            context_documents: Optional user-provided documents
                Each document: {'data': path/url/text, 'type': 'pdf'/'docx'/etc}
            claim_type: Type of claim input (if claim_input is str)
        
        Returns:
            Complete verification result with evidence and verdict
        """
        logger.info("="*60)
        logger.info("STARTING FACT VERIFICATION PIPELINE")
        logger.info("="*60)
        
        # ==========================================
        # STEP 1: Input Processing
        # ==========================================
        
        logger.info("\n[STEP 1] Input Processing...")
        
        if context_documents:
            # Process claim + context
            processed = self.input_processor.process_with_context(
                claim_input=claim_input,
                claim_type=claim_type,
                context_documents=context_documents
            )
            
            claim_text = processed['claim']['text']
            claim_language = processed['claim']['language']
            user_docs = processed['context_documents']
            
            logger.info(f"✓ Processed claim with {len(user_docs)} context documents")
        else:
            # Process claim only
            processed = self.input_processor.process(claim_input, claim_type)
            claim_text = processed['text']
            claim_language = processed['language']
            user_docs = None
            
            logger.info("✓ Processed claim (no context documents)")
        
        logger.info(f"Claim: {claim_text}")
        logger.info(f"Language: {claim_language}")
        
        # ==========================================
        # STEP 2: NLP Analysis
        # ==========================================
        
        logger.info("\n[STEP 2] NLP Analysis...")
        
        nlp_results = {}
        if self.nlp_pipeline:
            try:
                nlp_results = self.nlp_pipeline.analyze(claim_text)
                logger.info("✓ NLP analysis complete")
            except Exception as e:
                logger.error(f"NLP analysis failed: {str(e)}")
        else:
            logger.warning("No NLP pipeline configured, skipping analysis")
        
        # ==========================================
        # STEP 3: Evidence Retrieval (with context priority)
        # ==========================================
        
        logger.info("\n[STEP 3] Evidence Retrieval...")
        
        evidence = self.rag_pipeline.retrieve_evidence(
            claim=claim_text,
            user_context_docs=user_docs,  # HIGH PRIORITY
            top_k=5
        )
        
        # User-provided docs appear FIRST in evidence list
        logger.info(f"✓ Retrieved {len(evidence)} evidence items")
        
        # Log evidence sources
        user_provided_count = sum(1 for e in evidence if e['source_type'] == 'user_provided')
        kb_count = sum(1 for e in evidence if e['source_type'] == 'knowledge_base')
        web_count = sum(1 for e in evidence if e['source_type'] == 'web')
        
        logger.info(f"  - User-provided: {user_provided_count}")
        logger.info(f"  - Knowledge base: {kb_count}")
        logger.info(f"  - Web search: {web_count}")
        
        # ==========================================
        # STEP 4: Verification
        # ==========================================
        
        logger.info("\n[STEP 4] Verification...")
        
        verification_result = {}
        if self.verifier:
            try:
                verification_result = self.verifier.verify(
                    claim=claim_text,
                    evidence=evidence,
                    nlp_results=nlp_results
                )
                logger.info("✓ Verification complete")
            except Exception as e:
                logger.error(f"Verification failed: {str(e)}")
        else:
            logger.warning("No verifier configured, skipping verification")
            verification_result = {
                'verdict': 'UNKNOWN',
                'confidence': 0.0,
                'explanation': 'No verifier configured'
            }
        
        # ==========================================
        # STEP 5: Compile Results
        # ==========================================
        
        logger.info("\n[STEP 5] Compiling results...")
        
        result = {
            'claim': claim_text,
            'language': claim_language,
            'evidence': evidence,
            'nlp_analysis': nlp_results,
            'verification': verification_result,
            'metadata': {
                'has_user_context': context_documents is not None,
                'num_user_docs': len(user_docs) if user_docs else 0,
                'num_evidence': len(evidence),
                'evidence_sources': {
                    'user_provided': user_provided_count,
                    'knowledge_base': kb_count,
                    'web': web_count
                }
            }
        }
        
        logger.info("="*60)
        logger.info("VERIFICATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Verdict: {verification_result.get('verdict', 'UNKNOWN')}")
        logger.info(f"Confidence: {verification_result.get('confidence', 0.0):.2f}")
        
        return result
    
    def verify_claim(self, claim_input: str, claim_type: str = 'text') -> Dict[str, Any]:
        """
        Verify claim without context documents (original workflow)
        
        Args:
            claim_input: The claim to verify
            claim_type: Type of claim input
        
        Returns:
            Verification result
        """
        return self.verify_claim_with_context(
            claim_input=claim_input,
            context_documents=None,
            claim_type=claim_type
        )


# ==========================================
# USAGE EXAMPLE
# ==========================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = FactVerificationPipeline()
    
    # Example: Verify with context documents
    result = pipeline.verify_claim_with_context(
        claim_input="Our revenue grew 50% in Q3",
        context_documents=[
            {
                'data': 'data/user_uploads/financial_report.pdf',
                'type': 'pdf',
                'name': 'Q3 Financial Report'
            }
        ]
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Claim: {result['claim']}")
    print(f"\nVerdict: {result['verification'].get('verdict', 'UNKNOWN')}")
    print(f"Confidence: {result['verification'].get('confidence', 0.0):.2f}")
    
    print("\nEVIDENCE USED:")
    for i, ev in enumerate(result['evidence'][:3], 1):  # Show top 3
        print(f"{i}. [{ev['source_type']}] {ev['text'][:100]}...")
        print(f"   Priority: {ev['priority']}, Source: {ev['source']}")
