"""
Test FastText Language Detection
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.language_detector import LanguageDetector
from src.preprocessing.input_processor import InputProcessor


def test_language_detector():
    """Test FastText language detector directly"""
    print("\n" + "="*60)
    print("TEST: FastText Language Detector")
    print("="*60)
    
    detector = LanguageDetector()
    
    # Test cases
    test_cases = [
        ("Hello, how are you?", "en", "English"),
        ("नमस्ते, आप कैसे हैं?", "hi", "Hindi"),
        ("Hola, ¿cómo estás?", "es", "Spanish"),
        ("مرحبا كيف حالك؟", "ar", "Arabic"),
        ("Bonjour, comment allez-vous?", "fr", "French"),
        ("你好，你好吗？", "zh", "Chinese"),
        ("Привет, как дела?", "ru", "Russian"),
        ("こんにちは、お元気ですか？", "ja", "Japanese"),
        ("안녕하세요, 어떻게 지내세요?", "ko", "Korean"),
        ("Olá, como você está?", "pt", "Portuguese"),
    ]
    
    print("\nSingle Language Detection:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for text, expected_lang, expected_name in test_cases:
        lang, conf = detector.detect(text)
        lang_name = detector.get_language_name(lang)
        
        status = "✓" if lang == expected_lang else "✗"
        if lang == expected_lang:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} Text: {text[:50]}...")
        print(f"  Expected: {expected_lang} ({expected_name})")
        print(f"  Detected: {lang} ({lang_name})")
        print(f"  Confidence: {conf:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Accuracy: {(passed / len(test_cases)) * 100:.1f}%")


def test_short_text():
    """Test detection on short text"""
    print("\n" + "="*60)
    print("TEST: Short Text Detection")
    print("="*60)
    
    detector = LanguageDetector()
    
    short_texts = [
        ("Hello", "en"),
        ("नमस्ते", "hi"),
        ("Hola", "es"),
        ("مرحبا", "ar"),
        ("你好", "zh"),
    ]
    
    for text, expected in short_texts:
        lang, conf = detector.detect(text)
        status = "✓" if lang == expected else "✗"
        
        print(f"\n{status} Text: '{text}'")
        print(f"  Expected: {expected}")
        print(f"  Detected: {lang} (confidence: {conf:.4f})")


def test_code_mixed():
    """Test detection on code-mixed text"""
    print("\n" + "="*60)
    print("TEST: Code-Mixed Text Detection")
    print("="*60)
    
    detector = LanguageDetector()
    
    mixed_texts = [
        "This is English और यह हिंदी है",
        "Je parle français and English",
        "Hablo español and English",
        "This is English mixed with français and немного русского"
    ]
    
    for text in mixed_texts:
        # Get top 3 predictions
        predictions = detector.detect_multiple(text, top_k=3)
        
        print(f"\nText: {text}")
        print("Top 3 predictions:")
        for lang, conf in predictions:
            lang_name = detector.get_language_name(lang)
            print(f"  {lang} ({lang_name}): {conf:.4f}")


def test_input_processor_integration():
    """Test language detection in InputProcessor"""
    print("\n" + "="*60)
    print("TEST: InputProcessor Integration")
    print("="*60)
    
    processor = InputProcessor()
    
    test_texts = [
        ("India's GDP grew 8% in 2024", "en"),
        ("भारत की जीडीपी 2024 में 8% बढ़ी", "hi"),
        ("El PIB de India creció un 8% en 2024", "es"),
        ("نما الناتج المحلي الإجمالي للهند بنسبة 8٪ في عام 2024", "ar"),
    ]
    
    for text, expected_lang in test_texts:
        result = processor.process(text, 'text')
        
        status = "✓" if result['language'] == expected_lang else "✗"
        
        print(f"\n{status} Text: {text[:50]}...")
        print(f"  Expected: {expected_lang}")
        print(f"  Detected: {result['language']}")
        print(f"  Text length: {len(result['text'])} chars")


def test_detailed_detection():
    """Test detailed language detection with alternatives"""
    print("\n" + "="*60)
    print("TEST: Detailed Language Detection")
    print("="*60)
    
    processor = InputProcessor()
    
    text = "India's GDP यानी जीडीपी grew 8% in 2024"
    
    details = processor._detect_language_with_details(text)
    
    print(f"\nText: {text}")
    print(f"\nPrimary Language: {details['language']} ({details['language_name']})")
    print(f"Confidence: {details['confidence']:.4f}")
    print("\nAlternative Predictions:")
    for lang, conf in details['alternatives']:
        lang_name = processor.language_detector.get_language_name(lang)
        print(f"  {lang} ({lang_name}): {conf:.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FASTTEXT LANGUAGE DETECTION - TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Basic language detection
        test_language_detector()
        
        # Test 2: Short text
        test_short_text()
        
        # Test 3: Code-mixed text
        test_code_mixed()
        
        # Test 4: InputProcessor integration
        test_input_processor_integration()
        
        # Test 5: Detailed detection
        test_detailed_detection()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
