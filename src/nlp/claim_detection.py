"""
Claim Detection Module
Detects if text contains verifiable claims
"""

import logging
from typing import Dict, Any, List
from src.nlp.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ClaimDetector:
    """
    Detects verifiable claims in text
    
    Supports both trained and placeholder models
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize claim detector
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.model_info = model_manager.load_claim_detector()
        self.model = self.model_info['model']
        self.model_type = self.model_info['type']
        
        logger.info(f"ClaimDetector initialized (type: {self.model_type})")
    
    def detect(self, text: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Detect if text is a verifiable claim
        
        Args:
            text: Input text
            threshold: Confidence threshold for classification
        
        Returns:
            {
                'is_claim': bool,
                'confidence': float,
                'label': str
            }
        """
        if not text or len(text) < 5:
            return {
                'is_claim': False,
                'confidence': 0.0,
                'label': 'too_short'
            }
        
        if self.model_type == 'trained':
            return self._detect_with_trained_model(text, threshold)
        else:
            return self._detect_with_placeholder(text, threshold)
    
    def _detect_with_trained_model(self, text: str, threshold: float) -> Dict[str, Any]:
        """
        Use trained binary classifier
        
        Expected labels: LABEL_1 (is claim), LABEL_0 (not claim)
        """
        result = self.model(text)[0]
        
        is_claim = result['label'] == 'LABEL_1'
        confidence = result['score']
        
        # Apply threshold
        if confidence < threshold:
            is_claim = False
        
        return {
            'is_claim': is_claim,
            'confidence': confidence,
            'label': 'claim' if is_claim else 'not_claim'
        }
    
    def _detect_with_placeholder(self, text: str, threshold: float) -> Dict[str, Any]:
        """
        Use zero-shot classifier as placeholder
        
        Classifies into: factual claim, opinion, question, statement
        """
        candidate_labels = self.model_info['labels']
        
        result = self.model(text, candidate_labels)
        
        # Check if top prediction is "factual claim"
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        is_claim = (top_label == 'factual claim' and top_score >= threshold)
        
        return {
            'is_claim': is_claim,
            'confidence': top_score,
            'label': top_label,
            'all_scores': dict(zip(result['labels'], result['scores']))
        }
    
    def extract_claims_from_text(self, text: str, threshold: float = 0.7) -> List[Dict]:
        """
        Extract all claims from a longer text
        
        Splits text into sentences and checks each
        
        Args:
            text: Input text (can be multiple sentences)
            threshold: Confidence threshold
        
        Returns:
            List of claim dictionaries
        """
        # Split into sentences
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            # Download punkt if not available
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(text)
        
        claims = []
        
        for i, sentence in enumerate(sentences):
            result = self.detect(sentence, threshold)
            
            if result['is_claim']:
                claims.append({
                    'text': sentence,
                    'sentence_index': i,
                    'confidence': result['confidence'],
                    'label': result['label']
                })
        
        logger.info(f"Extracted {len(claims)} claims from {len(sentences)} sentences")
        
        return claims


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = ModelManager()
    detector = ClaimDetector(manager)
    
    # Test cases
    test_texts = [
        "India's GDP grew 8% in 2024",  # Claim
        "I think the economy is doing well",  # Opinion
        "What is the GDP growth rate?",  # Question
        "The report was published yesterday",  # Statement
    ]
    
    print("\n" + "="*60)
    print("CLAIM DETECTION TESTS")
    print("="*60)
    
    for text in test_texts:
        result = detector.detect(text)
        
        print(f"\nText: {text}")
        print(f"Is Claim: {result['is_claim']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Label: {result['label']}")
        
        if 'all_scores' in result:
            print("All scores:", result['all_scores'])
    
    # Test claim extraction
    print("\n" + "="*60)
    print("CLAIM EXTRACTION TEST")
    print("="*60)
    
    long_text = """
    India's GDP grew 8% in 2024 according to official data. 
    This is a significant achievement. 
    The growth was driven by strong services sector performance.
    Experts believe this trend will continue.
    """
    
    claims = detector.extract_claims_from_text(long_text)
    
    print(f"\nFound {len(claims)} claims:")
    for claim in claims:
        print(f"- {claim['text']} (confidence: {claim['confidence']:.2f})")