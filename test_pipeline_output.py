#!/usr/bin/env python3
"""
Test script to show actual NLP pipeline output structure
"""

import sys
import json
sys.path.append('src')

def test_nlp_pipeline():
    """Test NLP pipeline with the requested claim"""
    
    claim = "The Indian government launched a new scheme in 2024"
    
    print("="*60)
    print(f"TESTING CLAIM: {claim}")
    print("="*60)
    
    # Test NLP Pipeline
    try:
        print("\n=== NLP PIPELINE OUTPUT ===")
        from src.nlp.nlp_pipeline import NLPPipeline
        nlp = NLPPipeline()
        nlp_result = nlp.analyze(claim)
        
        print(json.dumps(nlp_result, indent=2, default=str))
        
    except Exception as e:
        print(f"NLP Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Test RAG Pipeline if possible
    try:
        print("\n=== RAG PIPELINE OUTPUT ===")
        from src.rag.vector_database import VectorDatabase
        from src.rag.rag_pipeline import RAGPipeline
        from src.nlp.model_manager import ModelManager
        
        vdb = VectorDatabase()
        stats = vdb.get_collection_stats('news_articles')
        
        if stats.get('exists') and stats.get('count', 0) > 0:
            model_manager = ModelManager()
            rag = RAGPipeline(vdb, 'news_articles', model_manager)
            rag_result = rag.verify_claim(claim, top_k=3)
            
            print(json.dumps(rag_result, indent=2, default=str))
        else:
            print("RAG database is empty - no articles to search")
            
    except Exception as e:
        print(f"RAG Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Enhanced Main Pipeline
    try:
        print("\n=== ENHANCED MAIN PIPELINE OUTPUT ===")
        from src.enhanced_main_pipeline import EnhancedFactVerificationPipeline
        
        pipeline = EnhancedFactVerificationPipeline()
        result = pipeline.verify_claim(claim, input_type='text')
        
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"Enhanced Pipeline Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nlp_pipeline()