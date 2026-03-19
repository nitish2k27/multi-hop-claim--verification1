"""
Test stance detector with real FEVER examples
Standalone test — no ModelManager needed
"""
 
import torch
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
 
def test_with_real_data():
    print("\n" + "="*80)
    print("TESTING STANCE DETECTOR WITH REAL FEVER DATA")
    print("="*80)
 
    # Paths
    project_root = Path(__file__).parent.parent
    model_path   = project_root / "models" / "stance_detector" / "final"
    test_data_path = project_root / "data" / "processed" / "stance_detection_test.csv"
 
    # Checks
    if not model_path.exists():
        print(f"\n✗ Model not found at: {model_path}")
        return
 
    if not test_data_path.exists():
        print(f"\n✗ Test data not found at: {test_data_path}")
        print("Run first: python src/data_collection/download_stance_dataset.py")
        return
 
    print(f"\n✓ Model:     {model_path}")
    print(f"✓ Test data: {test_data_path}")
 
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
 
    # Load label mapping
    label_file = model_path / "labels.json"
    with open(label_file, 'r') as f:
        label_info = json.load(f)
    id2label = {int(k): v for k, v in label_info['id2label'].items()}
 
    print(f"✓ Labels: {id2label}")
 
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(test_data_path)
    sample_df = test_df.sample(20, random_state=42)
    sample_df = sample_df.dropna(subset=['claim', 'evidence', 'label'])
    sample_df = sample_df.reset_index(drop=True)
    print(f"Testing on {len(sample_df)} random examples...")
 
    # Predict function
    def predict(claim, evidence):
        inputs = tokenizer(
            claim,
            evidence,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
 
        probs = torch.softmax(logits, dim=1)[0]
        predicted_id = probs.argmax().item()
 
        # Map NOT ENOUGH INFO → NEUTRAL for display
        raw_label = id2label[predicted_id]
        label_map = {
            'SUPPORTS': 'SUPPORTS',
            'REFUTES': 'REFUTES',
            'NOT ENOUGH INFO': 'NEUTRAL',
            'NEUTRAL': 'NEUTRAL'
        }
        return label_map.get(raw_label, 'NEUTRAL'), probs[predicted_id].item(), raw_label
 
    # Run tests
    correct = 0
    results = []
 
    for idx, row in sample_df.iterrows():
        claim      = row['claim']
        evidence   = row['evidence']
        true_label = row['label']
 
        # Normalize true label for comparison
        true_normalized = 'NEUTRAL' if true_label == 'NOT ENOUGH INFO' else true_label
 
        predicted_label, confidence, raw_label = predict(str(claim), str(evidence))
 
        is_correct = predicted_label == true_normalized
        if is_correct:
            correct += 1
 
        results.append({
            'claim':      claim[:60],
            'true':       true_normalized,
            'predicted':  predicted_label,
            'raw_label':  raw_label,
            'confidence': confidence,
            'correct':    is_correct
        })
 
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
 
    for i, r in enumerate(results, 1):
        status = "✓" if r['correct'] else "✗"
        print(f"\n{status} {i}. {r['claim']}...")
        print(f"   True: {r['true']:20} | Predicted: {r['predicted']:20} ({r['confidence']:.3f})")
 
    # Summary
    accuracy = correct / len(sample_df) * 100
 
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Accuracy: {correct}/{len(sample_df)} ({accuracy:.1f}%)")
 
    print("\nPer-class accuracy:")
    for label in ['SUPPORTS', 'REFUTES', 'NEUTRAL']:
        label_results = [r for r in results if r['true'] == label]
        if label_results:
            label_correct = sum(1 for r in label_results if r['correct'])
            label_acc = label_correct / len(label_results) * 100
            print(f"  {label}: {label_correct}/{len(label_results)} ({label_acc:.1f}%)")
 
    print("="*80)
 
 
def test_with_custom_examples():
    """Test with hand-crafted examples to verify model logic"""
 
    print("\n" + "="*80)
    print("TESTING WITH CUSTOM EXAMPLES")
    print("="*80)
 
    project_root = Path(__file__).parent.parent
    model_path   = project_root / "models" / "stance_detector" / "final"
 
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
 
    label_file = model_path / "labels.json"
    with open(label_file, 'r') as f:
        label_info = json.load(f)
    id2label = {int(k): v for k, v in label_info['id2label'].items()}
 
    test_cases = [
        {
            'claim':    "India's GDP grew 8% in 2024",
            'evidence': "Official government data shows India's economy expanded by 8% in 2024",
            'expected': "SUPPORTS"
        },
        {
            'claim':    "India's GDP grew 8% in 2024",
            'evidence': "India's GDP actually fell by 2% in 2024 according to official data",
            'expected': "REFUTES"
        },
        {
            'claim':    "India's GDP grew 8% in 2024",
            'evidence': "The weather in New Delhi was sunny today",
            'expected': "NEUTRAL"
        },
        {
            'claim':    "The Eiffel Tower is located in London",
            'evidence': "The Eiffel Tower is a wrought-iron lattice tower in Paris, France",
            'expected': "REFUTES"
        },
        {
            'claim':    "Berlin is the capital of Germany",
            'evidence': "Berlin is the capital and largest city of Germany",
            'expected': "SUPPORTS"
        }
    ]
 
    correct = 0
    print()
 
    for i, case in enumerate(test_cases, 1):
        inputs = tokenizer(
            case['claim'],
            case['evidence'],
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
 
        with torch.no_grad():
            logits = model(**inputs).logits
 
        probs        = torch.softmax(logits, dim=1)[0]
        predicted_id = probs.argmax().item()
        raw_label    = id2label[predicted_id]
        confidence   = probs[predicted_id].item()
 
        label_map = {
            'SUPPORTS': 'SUPPORTS',
            'REFUTES': 'REFUTES',
            'NOT ENOUGH INFO': 'NEUTRAL',
            'NEUTRAL': 'NEUTRAL'
        }
        predicted = label_map.get(raw_label, 'NEUTRAL')
 
        is_correct = predicted == case['expected']
        if is_correct:
            correct += 1
 
        status = "✓" if is_correct else "✗"
        print(f"{status} Test {i}:")
        print(f"  Claim:     {case['claim']}")
        print(f"  Evidence:  {case['evidence'][:70]}...")
        print(f"  Expected:  {case['expected']}")
        print(f"  Predicted: {predicted} ({confidence:.3f})")
        print()
 
    print(f"Custom test accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
    print("="*80)
 
 
if __name__ == "__main__":
    test_with_custom_examples()   # quick sanity check first
    test_with_real_data()         # then full FEVER tes