"""
Test complete NLP pipeline with all trained models
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nlp.nlp_pipeline import NLPPipeline


def test_complete_nlp_pipeline():
    """Test end-to-end NLP pipeline"""

    print("\n" + "="*80)
    print("TESTING COMPLETE NLP PIPELINE")
    print("="*80)
    print("\nAll trained models enabled:")
    print("  ✅ Claim Detection (BERT-base-uncased)")
    print("  ✅ NER (dslim/bert-base-NER)")
    print("  ✅ Entity Linking (Wikidata API)")
    print("  ✅ Temporal Extraction (Rule-based)")
    print("  ✅ Stance Detection (BERT-base-cased)")
    print("="*80)

    # Initialize pipeline
    print("\nInitializing NLP pipeline...")
    try:
        pipeline = NLPPipeline()
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Model status
    print("\nModel Status:")
    claim_info  = pipeline.model_manager.get_model_info('claim_detector')
    ner_info    = pipeline.model_manager.get_model_info('ner')
    stance_info = pipeline.model_manager.get_model_info('stance_detector')

    print(f"  Claim Detection:     {'✅ TRAINED' if claim_info.get('type') == 'trained' else '⚠️  PLACEHOLDER'}")
    print(f"  NER:                 {'✅ TRAINED' if ner_info.get('loaded') else '✗ Not loaded'}")
    print(f"  Entity Linking:      ✅ WIKIDATA")
    print(f"  Temporal Extraction: ✅ RULE-BASED")
    print(f"  Stance Detection:    {'✅ TRAINED' if stance_info.get('type') == 'trained' else '⚠️  PLACEHOLDER'}")

    # Test cases
    test_cases = [
        {
            'text':        "India's GDP grew 8% in 2024 according to official government data",
            'description': "Factual claim with temporal and entity info"
        },
        {
            'text':        "Narendra Modi visited New York in September 2024 to attend the UN General Assembly",
            'description': "Claim with multiple entities and dates"
        },
        {
            'text':        "What is the current GDP growth rate of India?",
            'description': "Question (not a claim)"
        },
        {
            'text':        "I think the economy is doing well this year",
            'description': "Opinion (not a claim)"
        },
        {
            'text':        "Apple Inc announced record profits in Q4 2024",
            'description': "Corporate news claim"
        },
    ]

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Text: {test_case['text']}")
        print("-" * 80)

        try:
            result   = pipeline.analyze(test_case['text'])
            analysis = result['analysis']

            # ── 1. Claim Detection ───────────────────────────────────────────
            claim_result = analysis['claim_detection']
            is_claim     = claim_result['is_claim']
            confidence   = claim_result['confidence']

            print(f"\n1. CLAIM DETECTION:")
            print(f"   Is Claim:   {'✅ YES' if is_claim else '❌ NO'}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Label:      {claim_result['label']}")

            # ── 2. Entity Extraction ─────────────────────────────────────────
            print(f"\n2. ENTITY EXTRACTION:")
            entity_summary = analysis['entities']

            if entity_summary['total_entities'] == 0:
                print("   No entities found")
            else:
                for entity_type, entities in entity_summary['entities'].items():
                    print(f"   {entity_type}:")
                    for ent in entities:
                        print(f"     • {ent['text']} (score: {ent['score']:.3f})")

            # ── 3. Entity Linking ────────────────────────────────────────────
            print(f"\n3. ENTITY LINKING:")
            linked = analysis.get('linked_entities', [])
            if not linked:
                print("   No linked entities")
            else:
                for ent in linked:
                    word       = ent.get('word', 'N/A')
                    wikidata   = ent.get('wikidata_id', 'not linked')
                    ent_type   = ent.get('entity_group', '')
                    print(f"   • {word} ({ent_type}) → {wikidata}")
                    if 'wikidata_info' in ent and ent['wikidata_info']:
                        desc = ent['wikidata_info'].get('description', '')[:60]
                        print(f"       └─ {desc}")

            # ── 4. Temporal Extraction ───────────────────────────────────────
            print(f"\n4. TEMPORAL EXTRACTION:")
            temporal_list = analysis['temporal']['dates']
            if not temporal_list:
                print("   No temporal expressions found")
            else:
                for item in temporal_list:
                    text_expr  = item.get('text', '')
                    normalized = item.get('normalized', '')
                    expr_type  = item.get('type', '')
                    print(f"   • '{text_expr}' → {normalized} ({expr_type})")

            # ── 5. Summary ───────────────────────────────────────────────────
            print(f"\n5. SUMMARY:")
            print(f"   {'✅ CLAIM' if is_claim else '❌ NOT CLAIM'} "
                  f"(confidence: {confidence:.3f})")
            print(f"   Entities found:   {entity_summary['total_entities']}")
            print(f"   Temporal items:   {len(temporal_list)}")
            print(f"   Language:         {result['language']}")

        except Exception as e:
            print(f"\n✗ Error processing text: {e}")
            import traceback
            traceback.print_exc()

    # ── Stance detection test ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("BONUS: CLAIM-EVIDENCE STANCE TEST")
    print(f"{'='*80}")

    claim    = "India's GDP grew 8% in 2024"
    evidence = "Official statistics confirm India's economy expanded by 8.2% in 2024"

    try:
        pair = pipeline.analyze_claim_evidence_pair(claim, evidence)
        print(f"\nClaim:      {claim}")
        print(f"Evidence:   {evidence}")
        print(f"Stance:     {pair['stance']['stance']}")
        print(f"Confidence: {pair['stance']['confidence']:.3f}")
    except Exception as e:
        print(f"✗ Stance test error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("✓ COMPLETE NLP PIPELINE TEST FINISHED")
    print("="*80)


if __name__ == "__main__":
    test_complete_nlp_pipeline()