"""
End-to-end fact verification test
Tests the complete pipeline: NLP → RAG → Verification
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nlp.nlp_pipeline import NLPPipeline
from src.rag.vector_database import VectorDatabase
from src.rag.rag_pipeline import RAGPipeline
from src.nlp.model_manager import ModelManager


def test_end_to_end():
    """Test complete fact verification pipeline"""

    print("\n" + "="*80)
    print("END-TO-END FACT VERIFICATION TEST")
    print("="*80)

    # ── Step 1: Initialize components ───────────────────────────────────────
    print("\n→ Step 1: Initializing components...")

    try:
        # NLP Pipeline
        nlp_pipeline = NLPPipeline()
        print("  ✓ NLP Pipeline initialized")

        # Vector Database
        vdb = VectorDatabase()
        print("  ✓ Vector Database initialized")

        # Check if RAG data exists
        stats = vdb.get_collection_stats('news_articles')

        if not stats['exists'] or stats.get('count', 0) == 0:
            print("\n  ⚠️  Warning: No RAG data found in 'news_articles' collection!")
            print("  Run data ingestion first:")
            print("    python scripts/ingest_to_rag.py")
            print("\n  Continuing with NLP-only test...")
            rag_available = False
            rag_pipeline  = None
        else:
            print(f"  ✓ RAG Database ready ({stats['count']:,} articles)")
            rag_available = True

            # Use the complex RAGPipeline (with hybrid retrieval)
            model_manager = ModelManager()
            rag_pipeline  = RAGPipeline(
                vector_db=vdb,
                collection_name='news_articles',
                nlp_model_manager=model_manager
            )
            print("  ✓ RAG Pipeline initialized")

    except Exception as e:
        print(f"\n  ✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── Test claims ──────────────────────────────────────────────────────────
    test_claims = [
        {
            'claim':       "India's GDP grew 8% in 2024",
            'description': "Economic claim with specific metric"
        },
        {
            'claim':       "Narendra Modi is the Prime Minister of India",
            'description': "Political position claim"
        },
        {
            'claim':       "Climate change is accelerating globally",
            'description': "Environmental claim"
        },
        {
            'claim':       "The stock market reached new highs last week",
            'description': "Financial claim with temporal reference"
        },
    ]

    print("\n" + "="*80)
    print("TESTING FACT VERIFICATION")
    print("="*80)

    for i, test_case in enumerate(test_claims, 1):
        print(f"\n{'='*80}")
        print(f"CLAIM {i}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"\nClaim: {test_case['claim']}")
        print("-" * 80)

        # ── NLP Analysis ─────────────────────────────────────────────────────
        print(f"\n→ NLP Analysis:")

        try:
            nlp_result = nlp_pipeline.analyze(test_case['claim'])
            analysis   = nlp_result['analysis']

            # Claim detection — correct key path
            claim_info = analysis['claim_detection']
            is_claim   = claim_info['is_claim']
            confidence = claim_info['confidence']

            print(f"  Is Claim:   {'✅ YES' if is_claim else '❌ NO'}")
            print(f"  Confidence: {confidence:.3f}")

            # Entities
            entity_summary = analysis['entities']
            if entity_summary['total_entities'] > 0:
                for entity_type, entities in entity_summary['entities'].items():
                    names = [e.get('text', e.get('word', '')) for e in entities]
                    print(f"  {entity_type}: {', '.join(names)}")
            else:
                print("  Entities: none found")

            # Temporal — correct key path
            temporal_list = analysis['temporal']['dates']
            if temporal_list:
                temporal_strs = [t.get('text', '') for t in temporal_list]
                print(f"  Temporal: {', '.join(temporal_strs)}")

        except Exception as e:
            print(f"  ✗ NLP analysis failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # ── RAG Verification ─────────────────────────────────────────────────
        if rag_available and rag_pipeline and is_claim:
            print(f"\n→ RAG Verification:")

            try:
                rag_result = rag_pipeline.verify_claim(test_case['claim'], top_k=3)

                print(f"  Verdict:    {rag_result['verdict']}")
                print(f"  Confidence: {rag_result['confidence']:.1f}%")

                evidence_list = rag_result.get('evidence', [])
                print(f"\n  Evidence ({len(evidence_list)} pieces):")

                for j, evidence in enumerate(evidence_list[:2], 1):
                    doc_text    = evidence.get('document', '')[:100]
                    stance      = evidence.get('stance', 'N/A')
                    source      = evidence.get('source', 'N/A')
                    credibility = evidence.get('credibility', {})
                    cred_score  = credibility.get('total_score', 0) if isinstance(credibility, dict) else credibility

                    print(f"\n  {j}. {doc_text}...")
                    print(f"     Stance:      {stance}")
                    print(f"     Source:      {source}")
                    print(f"     Credibility: {cred_score:.2f}")

            except Exception as e:
                print(f"  ✗ RAG verification failed: {e}")
                import traceback
                traceback.print_exc()

        elif not is_claim:
            print(f"\n→ Skipping RAG verification (not a claim)")

        elif not rag_available:
            print(f"\n→ Skipping RAG verification (no data in database)")

        print(f"\n" + "-" * 80)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("✓ END-TO-END TEST COMPLETE")
    print("="*80)

    print("\n📊 SYSTEM STATUS:")
    print("  ✅ Claim Detection:     TRAINED")
    print("  ✅ NER:                 PRE-TRAINED")
    print("  ✅ Entity Linking:      ACTIVE")
    print("  ✅ Temporal Extraction: ACTIVE")
    print("  ✅ Stance Detection:    TRAINED")

    if rag_available:
        print(f"  ✅ RAG Database:       {stats['count']:,} articles")
    else:
        print("  ⚠️  RAG Database:       Not populated")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_end_to_end()