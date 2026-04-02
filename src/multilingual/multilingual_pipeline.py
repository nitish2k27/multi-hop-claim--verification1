"""
Multilingual Fact Verification Pipeline
Handles queries in any language using translation
"""

import logging
from typing import Dict, Any
from src.multilingual.translator import Translator
from src.nlp.nlp_pipeline import NLPPipeline
from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class MultilingualVerificationPipeline:
    """
    Multilingual fact verification
    
    Workflow:
    1. Detect user's language
    2. Translate claim to English
    3. Process in English (NLP + RAG)
    4. Translate response back to user's language
    """
    
    def __init__(
        self,
        nlp_pipeline: NLPPipeline,
        rag_pipeline: RAGPipeline,
        translator: Translator = None
    ):
        """
        Initialize multilingual pipeline
        
        Args:
            nlp_pipeline: NLP pipeline instance
            rag_pipeline: RAG pipeline instance
            translator: Translator instance (optional)
        """
        self.nlp = nlp_pipeline
        self.rag = rag_pipeline
        self.translator = translator or Translator(backend='google')
        
        logger.info("✓ MultilingualVerificationPipeline initialized")
    
    def verify_claim(
        self,
        claim: str,
        user_language: str = None
    ) -> Dict[str, Any]:
        """
        Verify claim in any language
        
        Args:
            claim: Claim text in any language
            user_language: User's language code (auto-detected if None)
        
        Returns:
            Verification result in user's language
        """
        logger.info(f"\n→ Processing claim: {claim[:50]}...")
        
        # Step 1: Detect language if not provided
        if not user_language:
            logger.info("→ Detecting language...")
            nlp_result = self.nlp.analyze(claim)
            user_language = nlp_result.get('language', 'en')
            logger.info(f"  Detected: {user_language}")
        
        # Step 2: Translate to English if needed
        if user_language != 'en':
            logger.info(f"→ Translating from {user_language} to English...")
            claim_en = self.translator.to_english(claim, user_language)
            logger.info(f"  Translated: {claim_en[:50]}...")
        else:
            claim_en = claim
        
        # Step 3: Process in English
        logger.info("→ Processing in English...")
        
        # NLP Analysis
        nlp_result = self.nlp.analyze(claim_en)
        
        # If not a claim, return early
        if not nlp_result['is_claim']:
            return self._format_response(
                verdict='NOT_A_CLAIM',
                confidence=0.0,
                explanation="This does not appear to be a verifiable claim.",
                user_language=user_language,
                original_claim=claim,
                english_claim=claim_en
            )
        
        # RAG Verification
        rag_result = self.rag.verify_claim(claim_en, top_k=3)
        
        # Step 4: Translate response back to user's language
        logger.info(f"→ Translating response to {user_language}...")
        
        response = self._format_response(
            verdict=rag_result['verdict'],
            confidence=rag_result['confidence'],
            explanation=rag_result.get('explanation', ''),
            evidence=rag_result.get('evidence', []),
            user_language=user_language,
            original_claim=claim,
            english_claim=claim_en
        )
        
        return response
    
    def _format_response(
        self,
        verdict: str,
        confidence: float,
        explanation: str,
        user_language: str,
        original_claim: str,
        english_claim: str,
        evidence: list = None
    ) -> Dict[str, Any]:
        """Format and translate response"""
        
        # Translate explanation if needed
        if user_language != 'en' and explanation:
            explanation_translated = self.translator.from_english(
                explanation,
                user_language
            )
        else:
            explanation_translated = explanation
        
        # Translate verdict
        verdict_translations = {
            'en': {
                'SUPPORTED': 'SUPPORTED',
                'REFUTED': 'REFUTED',
                'INSUFFICIENT': 'INSUFFICIENT EVIDENCE',
                'NOT_A_CLAIM': 'NOT A CLAIM'
            },
            'hi': {
                'SUPPORTED': 'समर्थित',
                'REFUTED': 'खंडित',
                'INSUFFICIENT': 'अपर्याप्त साक्ष्य',
                'NOT_A_CLAIM': 'दावा नहीं'
            },
            'es': {
                'SUPPORTED': 'APOYADO',
                'REFUTED': 'REFUTADO',
                'INSUFFICIENT': 'EVIDENCIA INSUFICIENTE',
                'NOT_A_CLAIM': 'NO ES UNA AFIRMACIÓN'
            },
            # Add more languages as needed
        }
        
        verdict_translated = verdict_translations.get(
            user_language, {}
        ).get(verdict, verdict)
        
        return {
            'verdict': verdict,
            'verdict_translated': verdict_translated,
            'confidence': confidence,
            'explanation': explanation_translated,
            'evidence': evidence or [],
            'metadata': {
                'user_language': user_language,
                'original_claim': original_claim,
                'english_claim': english_claim,
                'processing_language': 'en'
            }
        }


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from src.nlp.model_manager import ModelManager
    from src.rag.vector_database import VectorDatabase
    
    print("\n" + "="*80)
    print("MULTILINGUAL VERIFICATION TEST")
    print("="*80)
    
    # Initialize components
    nlp = NLPPipeline()
    vdb = VectorDatabase()
    model_manager = ModelManager()
    rag = RAGPipeline(vdb, 'news_articles', model_manager)
    
    # Create multilingual pipeline
    ml_pipeline = MultilingualVerificationPipeline(nlp, rag)
    
    # Test claims in different languages
    test_claims = [
        ("India's GDP grew 8% in 2024", 'en'),
        ("भारत की जीडीपी 2024 में 8% बढ़ी", 'hi'),
        ("El PIB de India creció un 8% en 2024", 'es'),
    ]
    
    for claim, lang in test_claims:
        print(f"\n{'='*80}")
        print(f"Testing ({lang}): {claim}")
        print('='*80)
        
        result = ml_pipeline.verify_claim(claim, user_language=lang)
        
        print(f"\nVerdict: {result['verdict_translated']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Explanation: {result['explanation']}")