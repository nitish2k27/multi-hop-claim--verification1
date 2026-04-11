#!/usr/bin/env python3
"""
Test script to verify the pipeline fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.enhanced_main_pipeline import EnhancedFactVerificationPipeline

def test_single_claim():
    """Test a single claim to verify the fix"""
    print("="*60)
    print("TESTING PIPELINE FIX")
    print("="*60)
    
    try:
        # Initialize pipeline
        print("→ Loading pipeline...")
        pipeline = EnhancedFactVerificationPipeline()
        print("✓ Pipeline loaded successfully")
        
        # Test claim
        test_claim = "India is a member of the G20"
        print(f"\n→ Testing claim: {test_claim}")
        
        # Process claim
        result = pipeline.verify_claim(test_claim, input_type='text')
        
        print(f"\n✓ Processing complete!")
        print(f"  Verdict: {result.get('verdict', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Error: {result.get('error', 'None')}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_claim()
    sys.exit(0 if success else 1)