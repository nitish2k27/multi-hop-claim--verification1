"""
Test trained stance detection model
"""

from transformers import pipeline
import os
from pathlib import Path
import json

def test_stance_detector():
    """Test the trained stance detection model"""
    
    print("\n" + "="*80)
    print("TESTING TRAINED STANCE DETECTOR")
    print("="*80)
    
    # Get absolute path
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "stance_detector" / "final"
    
    # Check if model exists
    if not model_path.exists():
        print("\n✗ Model not found at:", model_path)
        print("\nThe model hasn't been trained yet!")
        print("\nOptions:")
        print("  1. Train the model in Kaggle (see training guide)")
        print("  2. Use placeholder model by setting use_trained: false in config")
        return
    
    # Check required files
    required_files = ['config.json']
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    
    if missing_files:
        print(f"\n✗ Missing required files: {missing_files}")
        print(f"Model path: {model_path}")
        return
    
    print(f"\n✓ Model found at: {model_path}")
    
    # Load label mapping
    label_file = model_path / "labels.json"
    if label_file.exists():
        with open(label_file, 'r') as f:
            label_info = json.load(f)
        id2label = {int(k): v for k, v in label_info['id2label'].items()}
        print(f"\n✓ Labels: {label_info['labels']}")
    else:
        # Default mapping
        id2label = {
            0: 'SUPPORTS',
            1: 'REFUTES',
            2: 'NOT ENOUGH INFO'
        }
        print(f"\n⚠ Using default label mapping")
    
    # Load model
    print("\nLoading model...")
    classifier = pipeline(
        "text-classification",
        model=str(model_path),
        tokenizer=str(model_path)
    )
    print("✓ Model loaded")
    
    # Test cases: (claim, evidence, expected_label)
    test_cases = [
        # SUPPORTS examples
        (
            "Narendra Modi is the Prime Minister of India",
            "Narendra Modi became India's Prime Minister in 2014 and continues to serve",
            "SUPPORTS"
        ),
        (
            "India's GDP grew 8% in 2024",
            "Official government data shows India's economy expanded by 8% in 2024",
            "SUPPORTS"
        ),
        (
            "The Eiffel Tower is in Paris",
            "The Eiffel Tower is located on the Champ de Mars in Paris, France",
            "SUPPORTS"
        ),
        
        # REFUTES examples
        (
            "The sun revolves around the Earth",
            "The Earth revolves around the sun in our solar system",
            "REFUTES"
        ),
        (
            "India's GDP grew 8% in 2024",
            "India's GDP contracted by 2% in 2024 due to economic downturn",
            "REFUTES"
        ),
        (
            "Nikolaj Coster-Waldau worked with Fox Broadcasting",
            "He is best known for his role in the HBO series Game of Thrones",
            "REFUTES"
        ),
        
        # NOT ENOUGH INFO examples
        (
            "India's GDP grew 8% in 2024",
            "The weather was sunny in Delhi today",
            "NOT ENOUGH INFO"
        ),
        (
            "Tesla was founded in 2003",
            "Electric vehicles are becoming increasingly popular",
            "NOT ENOUGH INFO"
        ),
        (
            "The stock market reached new highs",
            "Many investors are interested in technology stocks",
            "NOT ENOUGH INFO"
        ),
    ]
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    correct = 0
    total = len(test_cases)
    
    for i, (claim, evidence, expected_label) in enumerate(test_cases, 1):
        # Combine claim + evidence for the model
        # Format: claim [SEP] evidence
        text = f"{claim} [SEP] {evidence}"
        
        # Get prediction
        result = classifier(text, truncation=True, max_length=256)[0]
        
        # Extract predicted label
        # Model outputs LABEL_0, LABEL_1, LABEL_2
        predicted_id = int(result['label'].split('_')[1])
        predicted_label = id2label[predicted_id]
        
        # Check if correct
        is_correct = predicted_label == expected_label
        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"\n{status} Test {i}/{total}:")
        print(f"  Claim: {claim}")
        print(f"  Evidence: {evidence[:60]}...")
        print(f"  Expected: {expected_label}")
        print(f"  Predicted: {predicted_label} (confidence: {result['score']:.3f})")
    
    # Summary
    accuracy = correct / total * 100
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        print("\n✓ Model is performing well!")
    elif accuracy >= 60:
        print("\n⚠ Model needs improvement")
    else:
        print("\n✗ Model performance is poor - consider retraining")
    
    print("="*80)


if __name__ == "__main__":
    test_stance_detector()