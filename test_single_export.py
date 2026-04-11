#!/usr/bin/env python3
"""
Test single claim export to verify fixes
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_single_claim_export():
    """Test exporting a single claim"""
    print("="*60)
    print("TESTING SINGLE CLAIM EXPORT")
    print("="*60)
    
    try:
        # Import after adding to path
        from src.enhanced_main_pipeline import EnhancedFactVerificationPipeline
        
        # Initialize pipeline
        print("→ Loading pipeline...")
        pipeline = EnhancedFactVerificationPipeline()
        print("✓ Pipeline loaded successfully")
        
        # Test claim
        test_claim = "India is a member of the G20"
        print(f"\n→ Processing claim: {test_claim}")
        
        # Process claim
        result = pipeline.verify_claim(test_claim, input_type='text')
        
        print(f"\n✓ Processing complete!")
        print(f"  Verdict: {result.get('verdict', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
            return False
        
        # Format for LLM training
        llm_context = {
            "claim": test_claim,
            "verdict": result.get('verdict', 'UNKNOWN'),
            "confidence": result.get('confidence', 0.0),
            "explanation": result.get('explanation', ''),
            "evidence": result.get('evidence', []),
            "metadata": result.get('metadata', {})
        }
        
        print(f"\n→ LLM Context Generated:")
        print(json.dumps(llm_context, indent=2))
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_claim_export()
    print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}")
    sys.exit(0 if success else 1)