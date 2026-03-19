"""
Test complete RAG pipeline with real data
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.vector_database import VectorDatabase
from src.rag.rag_pipeline import RAGPipeline
from src.nlp.model_manager import ModelManager

def test_rag():
    """Test RAG with real data"""
    
    print("\n" + "="*80)
    print("TESTING RAG PIPELINE")
    print("="*80)
    
    # Initialize
    vdb = VectorDatabase()
    
    # Check collection
    stats = vdb.get_collection_stats('news_articles')
    
    if not stats['exists']:
        print("\n✗ Collection 'news_articles' not found!")
        print("Run ingestion first:")
        print("  python scripts/ingest_to_rag.py")
        return
    
    print(f"\n✓ Collection found: {stats['count']:,} articles")
    
    # Initialize RAG pipeline
    model_manager = ModelManager()
    rag = RAGPipeline(vdb, collection_name='news_articles', nlp_model_manager=model_manager)
    
    # Test claims
    test_claims = [
        "India's GDP grew 8% in 2024",
        "Climate change is accelerating",
        "The stock market reached new highs",
    ]
    
    for claim in test_claims:
        print("\n" + "="*80)
        print(f"TESTING: {claim}")
        print("="*80)
        
        result = rag.verify_claim(claim, top_k=3)
        
        print(f"\nVerdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        
        print(f"\nEvidence ({len(result['evidence'])} pieces):")
        for i, ev in enumerate(result['evidence'][:3], 1):
            print(f"\n{i}. {ev['document'][:150]}...")
            print(f"   Stance: {ev['stance']}")
            print(f"   Source: {ev['source']}")
            print(f"   Credibility: {ev['credibility']['total_score']:.2f}")
    
    print("\n" + "="*80)
    print("✓ RAG TESTING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    test_rag()
