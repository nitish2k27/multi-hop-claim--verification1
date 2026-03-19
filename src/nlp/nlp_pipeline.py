"""
Complete NLP Analysis Pipeline
Integrates all NLP components
"""

import logging
from typing import Dict, Any, Optional, List
from src.nlp.model_manager import ModelManager
from src.nlp.claim_detection import ClaimDetector
from src.nlp.entity_extraction import EntityExtractor
from src.nlp.entity_linking import WikidataEntityLinker
from src.nlp.temporal_extraction import TemporalExtractor
from src.nlp.stance_detection import StanceDetector

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Complete NLP analysis pipeline

    Processes text through all NLP stages:
    1. Claim Detection
    2. Named Entity Recognition
    3. Entity Linking
    4. Temporal Extraction
    5. (Stance Detection - used later with evidence)
    """

    def __init__(self, config_path: str = "configs/nlp_config.yaml"):
        logger.info("Initializing NLP Pipeline...")

        self.model_manager     = ModelManager(config_path)
        self.claim_detector    = ClaimDetector(self.model_manager)
        self.entity_extractor  = EntityExtractor(self.model_manager)
        self.entity_linker     = WikidataEntityLinker()
        self.temporal_extractor = TemporalExtractor()
        self.stance_detector   = StanceDetector(self.model_manager)

        logger.info("✓ NLP Pipeline initialized successfully")

    def analyze(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """
        Run complete NLP analysis on text

        Args:
            text: Input text to analyze
            language: Language code

        Returns:
            Complete NLP analysis results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running NLP Analysis on: {text[:80]}...")
        logger.info(f"{'='*60}\n")

        results = {
            'text':     text,
            'language': language,
            'analysis': {}
        }

        # ── Step 1: Claim Detection ──────────────────────────────────────────
        logger.info("→ Step 1: Claim Detection")
        claim_result = self.claim_detector.detect(text)
        results['analysis']['claim_detection'] = claim_result
        logger.info(f"  ✓ Is claim: {claim_result['is_claim']} ({claim_result['confidence']:.3f})")

        # ── Step 2: Named Entity Recognition ────────────────────────────────
        logger.info("→ Step 2: Named Entity Recognition")
        entity_summary = self.entity_extractor.get_entity_summary(text)
        results['analysis']['entities'] = entity_summary
        logger.info(f"  ✓ Found {entity_summary['total_entities']} entities: {entity_summary['counts']}")

        # ── Step 3: Entity Linking ───────────────────────────────────────────
        logger.info("→ Step 3: Entity Linking")

        # Flatten grouped NER output into list of dicts for entity linker
        # entity_summary['entities'] = {'PER': [...], 'ORG': [...], ...}
        # entity_linker.link_entities() expects: [{'word': ..., 'entity_group': ...}, ...]
        flat_entities = []
        for entity_type, entity_list in entity_summary['entities'].items():
            for ent in entity_list:
                flat_entities.append({
                    'word':         ent['text'],
                    'entity_group': entity_type,
                    'score':        ent.get('score', 1.0),
                    'start':        ent.get('start', 0),
                    'end':          ent.get('end', 0)
                })

        linked_entities = self.entity_linker.link_entities(flat_entities, language)
        results['analysis']['linked_entities'] = linked_entities
        logger.info(f"  ✓ Linked {len(linked_entities)} entities to knowledge base")

        # ── Step 4: Temporal Extraction ──────────────────────────────────────
        logger.info("→ Step 4: Temporal Extraction")

        # temporal_extractor.extract() returns a list — wrap it properly
        temporal_list = self.temporal_extractor.extract(text)
        temporal_info = {
            'dates':       temporal_list,
            'total_count': len(temporal_list)
        }
        results['analysis']['temporal'] = temporal_info

        logger.info(f"  ✓ Extracted {len(temporal_list)} temporal expressions")
        for item in temporal_list:
            original   = item.get('text',       item.get('original', ''))
            normalized = item.get('normalized', '')
            expr_type  = item.get('type',       '')
            logger.info(f"    - {original} → {normalized} ({expr_type})")

        logger.info(f"\n{'='*60}")
        logger.info("NLP Analysis Complete")
        logger.info(f"{'='*60}\n")

        return results

    def extract_claims_from_document(self, text: str, threshold: float = 0.7) -> List[Dict]:
        """
        Extract all verifiable claims from a document

        Args:
            text: Document text
            threshold: Confidence threshold for claim detection

        Returns:
            List of claims with full NLP analysis
        """
        logger.info("Extracting claims from document...")

        claims = self.claim_detector.extract_claims_from_text(text, threshold)

        analyzed_claims = []
        for claim in claims:
            claim_analysis = self.analyze(claim['text'])
            claim_analysis['claim_info'] = claim
            analyzed_claims.append(claim_analysis)

        logger.info(f"✓ Extracted and analyzed {len(analyzed_claims)} claims")
        return analyzed_claims

    def analyze_claim_evidence_pair(
        self,
        claim: str,
        evidence: str
    ) -> Dict[str, Any]:
        """
        Analyze a claim-evidence pair for stance detection

        Args:
            claim:    The claim being verified
            evidence: Evidence text

        Returns:
            Analysis including stance detection
        """
        claim_analysis    = self.analyze(claim)
        evidence_analysis = self.analyze(evidence)
        stance_result     = self.stance_detector.detect(claim, evidence)

        return {
            'claim_analysis':    claim_analysis,
            'evidence_analysis': evidence_analysis,
            'stance':            stance_result
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about loaded models in pipeline"""
        return {
            'claim_detector':     self.model_manager.get_model_info('claim_detector'),
            'ner':                self.model_manager.get_model_info('ner'),
            'stance_detector':    self.model_manager.get_model_info('stance_detector'),
            'entity_linker':      {'type': 'wikidata_api'},
            'temporal_extractor': {'type': 'rule_based'}
        }


# Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pipeline = NLPPipeline()

    # Test 1: Single claim
    print("\n" + "="*80)
    print("TEST 1: SINGLE CLAIM ANALYSIS")
    print("="*80)

    test_claim = "India's GDP grew 8% in Q3 2024 according to official government data"
    result = pipeline.analyze(test_claim)

    print("\nRESULTS:")
    print(f"  Claim Detection: {result['analysis']['claim_detection']}")
    print(f"  Entities:        {result['analysis']['entities']['counts']}")
    print(f"  Temporal items:  {result['analysis']['temporal']['total_count']}")

    # Test 2: Document claim extraction
    print("\n" + "="*80)
    print("TEST 2: CLAIM EXTRACTION FROM DOCUMENT")
    print("="*80)

    document = """
    India's economy has shown remarkable growth in recent years.
    The GDP expanded by 8% in 2024, driven by strong performance in the services sector.
    Manufacturing output also increased significantly.
    Experts believe this growth trend will continue into 2025.
    However, some economists question the sustainability of this growth rate.
    """

    claims = pipeline.extract_claims_from_document(document, threshold=0.6)

    print(f"\nExtracted {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"\n  {i}. {claim['claim_info']['text']}")
        print(f"     Confidence: {claim['claim_info']['confidence']:.2f}")
        print(f"     Entities:   {claim['analysis']['entities']['counts']}")

    # Test 3: Claim-Evidence stance
    print("\n" + "="*80)
    print("TEST 3: CLAIM-EVIDENCE STANCE DETECTION")
    print("="*80)

    claim    = "India's GDP grew 8% in 2024"
    evidence = "Official statistics show India's economy expanded by 8.2% in fiscal year 2024"

    pair = pipeline.analyze_claim_evidence_pair(claim, evidence)

    print(f"\n  Claim:      {claim}")
    print(f"  Evidence:   {evidence}")
    print(f"  Stance:     {pair['stance']['stance']}")
    print(f"  Confidence: {pair['stance']['confidence']:.3f}")

    # Pipeline info
    print("\n" + "="*80)
    print("PIPELINE INFORMATION")
    print("="*80)

    import json
    print(json.dumps(pipeline.get_pipeline_info(), indent=2))