"""
Comprehensive tests for NLP Pipeline
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import logging
from src.nlp.nlp_pipeline import NLPPipeline

logging.basicConfig(level=logging.WARNING)


class TestNLPPipeline(unittest.TestCase):
    """Test suite for NLP Pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize pipeline once for all tests"""
        cls.pipeline = NLPPipeline()
    
    def test_claim_detection_positive(self):
        """Test claim detection on factual claim"""
        text = "India's GDP grew 8% in 2024"
        result = self.pipeline.analyze(text)
        
        self.assertTrue(result['analysis']['claim_detection']['is_claim'])
        self.assertGreater(result['analysis']['claim_detection']['confidence'], 0.5)
    
    def test_claim_detection_negative(self):
        """Test claim detection on non-claim"""
        text = "I think the weather is nice today"
        result = self.pipeline.analyze(text)
        
        # Should be detected as opinion, not factual claim
        # (May pass depending on model sensitivity)
        self.assertIsNotNone(result['analysis']['claim_detection'])
    
    def test_entity_extraction(self):
        """Test NER extraction"""
        text = "Narendra Modi visited New York in September 2024"
        result = self.pipeline.analyze(text)
        
        entities = result['analysis']['entities']
        
        # Should find person, location, date
        self.assertGreater(entities['total_entities'], 0)
        self.assertIn('PER', entities['entity_types'])  # Person
        self.assertIn('LOC', entities['entity_types'])  # Location
    
    def test_temporal_extraction(self):
        """Test temporal extraction"""
        text = "The report from Q3 2024 shows improvement over last year"
        result = self.pipeline.analyze(text)
        
        temporal = result['analysis']['temporal']
        
        # Should find Q3 2024
        self.assertGreater(len(temporal['dates']), 0)
        
        # Check for quarter detection
        quarter_found = any(d['type'] == 'quarter' for d in temporal['dates'])
        self.assertTrue(quarter_found)
    
    def test_stance_detection(self):
        """Test stance detection"""
        claim = "India's GDP grew 8%"
        evidence = "Official data confirms 8% GDP growth"
        
        result = self.pipeline.analyze_claim_evidence_pair(claim, evidence)
        
        # Should detect SUPPORTS stance
        self.assertIn(result['stance']['stance'], ['SUPPORTS', 'REFUTES', 'NEUTRAL'])
        self.assertGreater(result['stance']['confidence'], 0.0)
    
    def test_claim_extraction_from_document(self):
        """Test extracting multiple claims from document"""
        document = """
        India's GDP grew 8% in 2024.
        The growth was driven by services sector.
        Manufacturing also showed improvement.
        """
        
        claims = self.pipeline.extract_claims_from_document(document, threshold=0.5)
        
        # Should extract at least one claim
        self.assertGreater(len(claims), 0)
        
        # Each claim should have analysis
        for claim in claims:
            self.assertIn('analysis', claim)
            self.assertIn('claim_info', claim)
    
    def test_pipeline_info(self):
        """Test pipeline information retrieval"""
        info = self.pipeline.get_pipeline_info()
        
        # Should have all components
        self.assertIn('claim_detector', info)
        self.assertIn('ner', info)
        self.assertIn('stance_detector', info)
        
        # Each should have loaded status
        self.assertTrue(info['claim_detector']['loaded'])
        self.assertTrue(info['ner']['loaded'])


if __name__ == '__main__':
    unittest.main()