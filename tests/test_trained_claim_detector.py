"""
Test trained claim detection model
"""

from transformers import pipeline

def test_claim_detector():
    """Test the trained model"""
    
    print("\n" + "="*60)
    print("TESTING TRAINED CLAIM DETECTOR")
    print("="*60)
    
    # Load model
    classifier = pipeline(
        "text-classification",
        model="models/claim_detector/final"
    )
    
    # Test cases
    test_cases = [
        ("India's GDP grew 8% in 2024", True),  # Claim
        ("What is the GDP growth rate?", False),  # Question
        ("I think the economy is good", False),  # Opinion
        ("The unemployment rate decreased to 5%", True),  # Claim
        ("Please read the document", False),  # Instruction
        ("Climate change is accelerating", True),  # Claim
        ("That seems interesting", False),  # Opinion
    ]
    
    print("\nTest Results:\n")
    
    correct = 0
    total = len(test_cases)
    
    for text, expected_is_claim in test_cases:
        result = classifier(text)[0]
        
        # LABEL_1 = claim, LABEL_0 = not claim
        predicted_is_claim = result['label'] == 'LABEL_1'
        is_correct = predicted_is_claim == expected_is_claim
        
        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} Text: {text}")
        print(f"  Expected: {'Claim' if expected_is_claim else 'Not Claim'}")
        print(f"  Predicted: {result['label']} (score: {result['score']:.3f})")
        print()
    
    accuracy = correct / total * 100
    print("="*60)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    test_claim_detector()