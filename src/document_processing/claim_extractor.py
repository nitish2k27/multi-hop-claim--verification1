"""
Claim Extraction from Documents
Extracts verifiable claims from uploaded documents (PDF, DOCX, etc.)
"""

import logging
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """
    Extract claims from documents
    
    Features:
    - Sentence-level claim detection
    - Batch processing for efficiency
    - Confidence scoring
    - Context preservation
    """
    
    def __init__(
        self,
        claim_detector,
        confidence_threshold: float = 0.8,
        batch_size: int = 32
    ):
        """
        Initialize claim extractor
        
        Args:
            claim_detector: Trained claim detection model
            confidence_threshold: Minimum confidence for claims
            batch_size: Batch size for processing
        """
        self.claim_detector = claim_detector
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        logger.info(f"✓ ClaimExtractor initialized")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
    
    def extract_claims_from_text(
        self,
        text: str,
        preserve_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract claims from text
        
        Args:
            text: Input text
            preserve_context: Include surrounding sentences as context
        
        Returns:
            List of claims with metadata
        """
        logger.info(f"Extracting claims from text ({len(text)} chars)")
        
        # Step 1: Split into sentences
        sentences = self._split_sentences(text)
        logger.info(f"Split into {len(sentences)} sentences")
        
        # Step 2: Pre-filter sentences
        filtered_sentences = self._prefilter_sentences(sentences)
        logger.info(f"After pre-filtering: {len(filtered_sentences)} sentences")
        
        # Step 3: Batch claim detection
        claims = self._batch_claim_detection(filtered_sentences)
        
        # Step 4: Add context if requested
        if preserve_context:
            claims = self._add_context(claims, sentences)
        
        logger.info(f"✓ Extracted {len(claims)} claims")
        return claims
    
    def extract_claims_from_document(
        self,
        document_data: Dict[str, Any],
        mode: str = 'full'
    ) -> List[Dict[str, Any]]:
        """
        Extract claims from processed document
        
        Args:
            document_data: Output from DocumentHandler.process_upload()
            mode: 'full' (all text) or 'summary' (summarize first)
        
        Returns:
            List of claims with document metadata
        """
        logger.info(f"Extracting claims from document: {document_data['metadata']['filename']}")
        
        text = document_data['text']
        metadata = document_data['metadata']
        
        # Choose processing mode
        if mode == 'summary' and len(text.split()) > 5000:
            # Summarize large documents first
            text = self._summarize_document(text)
            logger.info(f"Summarized document to {len(text)} chars")
        
        # Extract claims
        claims = self.extract_claims_from_text(text)
        
        # Add document metadata to each claim
        for claim in claims:
            claim['document_metadata'] = metadata
            claim['extraction_mode'] = mode
        
        return claims
    
    def _split_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with metadata"""
        
        # Simple sentence splitting (you can use spaCy for better results)
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        result = []
        char_offset = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                result.append({
                    'id': i,
                    'text': sentence,
                    'char_start': char_offset,
                    'char_end': char_offset + len(sentence),
                    'word_count': len(sentence.split())
                })
                char_offset += len(sentence) + 1  # +1 for the delimiter
        
        return result
    
    def _prefilter_sentences(
        self,
        sentences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Pre-filter sentences to reduce processing load"""
        
        filtered = []
        
        for sentence in sentences:
            text = sentence['text']
            
            # Skip if too short
            if sentence['word_count'] < 5:
                continue
            
            # Skip questions
            if text.strip().endswith('?'):
                continue
            
            # Skip obvious non-claims
            opinion_words = ['i think', 'i believe', 'maybe', 'perhaps', 'possibly']
            if any(word in text.lower() for word in opinion_words):
                continue
            
            # Skip if mostly numbers/symbols
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.5:
                continue
            
            filtered.append(sentence)
        
        return filtered
    
    def _batch_claim_detection(
        self,
        sentences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run claim detection in batches"""
        
        claims = []
        
        # Process in batches
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_texts = [s['text'] for s in batch]
            
            # Run claim detection
            try:
                predictions = self.claim_detector.predict(batch_texts)
                
                # Process results
                for sentence, prediction in zip(batch, predictions):
                    # Assuming prediction is {'is_claim': bool, 'confidence': float}
                    if prediction['is_claim'] and prediction['confidence'] >= self.confidence_threshold:
                        claims.append({
                            'text': sentence['text'],
                            'confidence': prediction['confidence'],
                            'sentence_id': sentence['id'],
                            'char_start': sentence['char_start'],
                            'char_end': sentence['char_end'],
                            'word_count': sentence['word_count']
                        })
                        
            except Exception as e:
                logger.error(f"Batch claim detection failed: {e}")
                # Fallback to individual processing
                for sentence in batch:
                    try:
                        prediction = self.claim_detector.predict([sentence['text']])[0]
                        if prediction['is_claim'] and prediction['confidence'] >= self.confidence_threshold:
                            claims.append({
                                'text': sentence['text'],
                                'confidence': prediction['confidence'],
                                'sentence_id': sentence['id'],
                                'char_start': sentence['char_start'],
                                'char_end': sentence['char_end'],
                                'word_count': sentence['word_count']
                            })
                    except Exception as e2:
                        logger.error(f"Individual claim detection failed: {e2}")
                        continue
        
        return claims
    
    def _add_context(
        self,
        claims: List[Dict[str, Any]],
        all_sentences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add surrounding sentences as context"""
        
        # Create sentence lookup
        sentence_lookup = {s['id']: s for s in all_sentences}
        
        for claim in claims:
            sentence_id = claim['sentence_id']
            
            # Get previous and next sentences
            prev_sentence = sentence_lookup.get(sentence_id - 1)
            next_sentence = sentence_lookup.get(sentence_id + 1)
            
            context_before = prev_sentence['text'] if prev_sentence else ""
            context_after = next_sentence['text'] if next_sentence else ""
            
            claim['context'] = {
                'before': context_before,
                'after': context_after,
                'full': f"{context_before} {claim['text']} {context_after}".strip()
            }
        
        return claims
    
    def _summarize_document(self, text: str) -> str:
        """Summarize document to extract key points"""
        
        # Simple extractive summarization
        # You can replace this with a proper summarization model
        
        sentences = self._split_sentences(text)
        
        # Score sentences by keyword frequency
        word_freq = {}
        for sentence in sentences:
            words = sentence['text'].lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        for sentence in sentences:
            score = 0
            words = sentence['text'].lower().split()
            for word in words:
                score += word_freq.get(word, 0)
            sentence['score'] = score / len(words) if words else 0
        
        # Select top sentences (top 20%)
        sentences.sort(key=lambda x: x['score'], reverse=True)
        top_sentences = sentences[:max(1, len(sentences) // 5)]
        
        # Sort by original order
        top_sentences.sort(key=lambda x: x['id'])
        
        # Combine into summary
        summary = ' '.join(s['text'] for s in top_sentences)
        return summary
    
    def analyze_document_claims(
        self,
        document_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive claim analysis of document
        
        Args:
            document_data: Processed document data
        
        Returns:
            Analysis report with claims, statistics, and insights
        """
        logger.info("Performing comprehensive claim analysis")
        
        # Extract claims
        claims = self.extract_claims_from_document(document_data)
        
        # Calculate statistics
        total_sentences = len(self._split_sentences(document_data['text']))
        claim_density = len(claims) / total_sentences if total_sentences > 0 else 0
        
        # Categorize claims by confidence
        high_confidence = [c for c in claims if c['confidence'] >= 0.9]
        medium_confidence = [c for c in claims if 0.8 <= c['confidence'] < 0.9]
        low_confidence = [c for c in claims if c['confidence'] < 0.8]
        
        # Extract key topics (simple keyword extraction)
        all_claim_text = ' '.join(c['text'] for c in claims)
        keywords = self._extract_keywords(all_claim_text)
        
        return {
            'document_info': {
                'filename': document_data['metadata']['filename'],
                'total_words': document_data['metadata']['word_count'],
                'total_sentences': total_sentences
            },
            'claim_statistics': {
                'total_claims': len(claims),
                'claim_density': claim_density,
                'high_confidence_claims': len(high_confidence),
                'medium_confidence_claims': len(medium_confidence),
                'low_confidence_claims': len(low_confidence)
            },
            'claims': {
                'all': claims,
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence,
                'low_confidence': low_confidence
            },
            'key_topics': keywords,
            'analysis_metadata': {
                'confidence_threshold': self.confidence_threshold,
                'extraction_timestamp': document_data['metadata']['upload_date']
            }
        }
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key topics from claims"""
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:top_k]]


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("CLAIM EXTRACTION TEST")
    print("="*80)
    
    # Mock claim detector for testing
    class MockClaimDetector:
        def predict(self, texts):
            # Simple mock: sentences with numbers are likely claims
            results = []
            for text in texts:
                has_number = any(c.isdigit() for c in text)
                confidence = 0.9 if has_number else 0.3
                results.append({
                    'is_claim': has_number,
                    'confidence': confidence
                })
            return results
    
    # Test
    extractor = ClaimExtractor(MockClaimDetector())
    
    test_text = """
    India's GDP grew by 8% in 2024. This is a significant achievement.
    The unemployment rate fell to 5.2%. Many experts believe this is positive.
    What caused this growth? The government implemented new policies.
    Revenue increased by 20% last quarter.
    """
    
    claims = extractor.extract_claims_from_text(test_text)
    
    print(f"\nExtracted {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim['text']} (confidence: {claim['confidence']:.2f})")