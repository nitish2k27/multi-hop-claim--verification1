#!/usr/bin/env python3
"""
Test script to run a single claim through the pipeline
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_single_claim():
    """Test a single claim through the enhanced pipeline"""
    
    claim = "O governo indiano lançou o programa PM Kisan em 2024."
    
    print("="*60)
    print(f"Testing claim: {claim}")
    print("="*60)
    
    try:
        print("Loading enhanced pipeline...")
        from src.enhanced_main_pipeline import EnhancedFactVerificationPipeline
        
        pipeline = EnhancedFactVerificationPipeline()
        print("✓ Pipeline loaded successfully")
        
        print(f"\nProcessing claim: {claim}")
        result = pipeline.verify_claim(claim, input_type='text')
        
        print("\n" + "="*60)
        print("PIPELINE OUTPUT:")
        print("="*60)
        print(json.dumps(result, indent=2, default=str))
        
        # Extract key information
        verdict = result.get('verdict', 'UNKNOWN')
        confidence = result.get('confidence', 0)
        evidence_count = len(result.get('evidence', []))
        
        print(f"\n📊 SUMMARY:")
        print(f"   Verdict: {verdict}")
        print(f"   Confidence: {confidence}%")
        print(f"   Evidence pieces: {evidence_count}")
        
        # Check if we can build llm_context
        llm_context = result.get('llm_context', '')
        if not llm_context:
            print("\n🔧 Building LLM context...")
            llm_context = f"""=== FACT VERIFICATION CONTEXT ===

CLAIM:
  {claim}

VERDICT: {verdict}
CONFIDENCE: {confidence}%

EXPLANATION:
{result.get('explanation', 'No explanation available')}

EVIDENCE: {evidence_count} pieces found

=== END CONTEXT ==="""
        
        print(f"\n📝 LLM CONTEXT ({len(llm_context)} chars):")
        print("-" * 40)
        print(llm_context[:500] + "..." if len(llm_context) > 500 else llm_context)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_claim()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)