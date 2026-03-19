"""
Language Detection using FastText
Fast, accurate, offline-capable language detection
"""

import os
import logging
from typing import Tuple, Optional, List
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import fasttext, fall back to langdetect if not available
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    logger.warning("FastText not available, falling back to langdetect")
    try:
        from langdetect import detect, LangDetectException
        LANGDETECT_AVAILABLE = True
    except ImportError:
        LANGDETECT_AVAILABLE = False
        logger.error("Neither FastText nor langdetect available!")


class LanguageDetector:
    """
    Language detector using FastText's pre-trained model
    
    Supports 176 languages with 99.1% accuracy
    """
    
    MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    MODEL_FILENAME = "lid.176.bin"
    
    def __init__(self, model_dir: str = "models/language_detection"):
        """
        Initialize language detector
        
        Args:
            model_dir: Directory to store/load the FastText model
        """
        if not FASTTEXT_AVAILABLE:
            logger.warning("FastText not available. Using fallback detection method.")
            self.model = None
            self.use_fasttext = False
            return
        
        self.use_fasttext = True
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.model_dir / self.MODEL_FILENAME
        
        # Download model if not exists
        if not self.model_path.exists():
            logger.info("FastText language model not found. Downloading...")
            self._download_model()
        
        # Load model
        logger.info(f"Loading FastText language model from {self.model_path}")
        
        # Suppress FastText warnings
        fasttext.FastText.eprint = lambda x: None
        
        self.model = fasttext.load_model(str(self.model_path))
        
        logger.info("✓ Language detector initialized")
    
    def _download_model(self):
        """Download FastText language identification model"""
        logger.info(f"Downloading from {self.MODEL_URL}")
        logger.info("This is a one-time download (126 MB)...")
        
        try:
            response = requests.get(self.MODEL_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (5 * 1024 * 1024) == 0:  # Every 5MB
                                logger.info(f"Downloaded {percent:.1f}%")
            
            logger.info("✓ Model downloaded successfully")
            
        except Exception as e:
            if self.model_path.exists():
                self.model_path.unlink()  # Delete partial download
            raise Exception(f"Failed to download model: {str(e)}")
    
    def detect(
        self,
        text: str,
        k: int = 1,
        threshold: float = 0.0
    ) -> Tuple[str, float]:
        """
        Detect language of text
        
        Args:
            text: Input text
            k: Number of top predictions to return
            threshold: Minimum confidence threshold (0.0 to 1.0)
        
        Returns:
            Tuple of (language_code, confidence)
            Example: ('en', 0.9876)
        
        Language codes are ISO 639-1/639-3:
            - 'en' = English
            - 'hi' = Hindi
            - 'es' = Spanish
            - 'ar' = Arabic
            - 'zh' = Chinese
            - 'fr' = French
            - etc.
        """
        # Clean text
        text = self._preprocess_text(text)
        
        # Check if text is too short
        if len(text) < 10:
            logger.warning(f"Text too short for reliable detection ({len(text)} chars)")
            # Fallback to script detection
            return self._detect_by_script(text), 0.5
        
        # Use FastText if available
        if self.use_fasttext and self.model:
            # Predict
            predictions = self.model.predict(text, k=k, threshold=threshold)
            
            # Extract results
            # predictions format: (('__label__en', '__label__hi'), (0.98, 0.02))
            labels, confidences = predictions
            
            if len(labels) == 0:
                # No prediction above threshold
                logger.warning("No language detected above threshold")
                return 'unknown', 0.0
            
            # Get top prediction
            top_label = labels[0].replace('__label__', '')
            top_confidence = float(confidences[0])
            
            logger.debug(f"Detected language: {top_label} (confidence: {top_confidence:.4f})")
            
            return top_label, top_confidence
        
        # Fallback to langdetect or script detection
        else:
            if LANGDETECT_AVAILABLE:
                try:
                    lang = detect(text)
                    return lang, 0.8  # Assume 80% confidence for langdetect
                except:
                    pass
            
            # Final fallback: script detection
            return self._detect_by_script(text), 0.5
    
    def detect_multiple(
        self,
        text: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Detect top-k possible languages
        
        Useful for code-mixed text or uncertain cases
        
        Args:
            text: Input text
            top_k: Number of language predictions
        
        Returns:
            List of (language, confidence) tuples
            Example: [('en', 0.85), ('es', 0.10), ('fr', 0.05)]
        """
        text = self._preprocess_text(text)
        
        if len(text) < 10:
            return [(self._detect_by_script(text), 0.5)]
        
        predictions = self.model.predict(text, k=top_k)
        labels, confidences = predictions
        
        results = []
        for label, conf in zip(labels, confidences):
            lang_code = label.replace('__label__', '')
            results.append((lang_code, float(conf)))
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better detection"""
        # Remove URLs
        import re
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # FastText expects newlines to be replaced
        text = text.replace('\n', ' ')
        
        return text.strip()
    
    def _detect_by_script(self, text: str) -> str:
        """
        Fallback: Detect language by Unicode script
        
        Used when text is too short for FastText
        """
        # Count characters in different scripts
        scripts = {
            'devanagari': 0,  # Hindi, Marathi, Nepali
            'arabic': 0,
            'chinese': 0,
            'cyrillic': 0,  # Russian, Ukrainian
            'latin': 0  # English, Spanish, French, etc.
        }
        
        for char in text:
            code_point = ord(char)
            
            # Devanagari (Hindi)
            if 0x0900 <= code_point <= 0x097F:
                scripts['devanagari'] += 1
            
            # Arabic
            elif 0x0600 <= code_point <= 0x06FF:
                scripts['arabic'] += 1
            
            # Chinese (CJK)
            elif 0x4E00 <= code_point <= 0x9FFF:
                scripts['chinese'] += 1
            
            # Cyrillic (Russian)
            elif 0x0400 <= code_point <= 0x04FF:
                scripts['cyrillic'] += 1
            
            # Latin (English, etc.)
            elif (0x0041 <= code_point <= 0x005A) or (0x0061 <= code_point <= 0x007A):
                scripts['latin'] += 1
        
        # Determine dominant script
        dominant = max(scripts, key=scripts.get)
        
        # Map script to language
        script_to_lang = {
            'devanagari': 'hi',  # Hindi
            'arabic': 'ar',
            'chinese': 'zh',
            'cyrillic': 'ru',
            'latin': 'en'  # Default to English for Latin
        }
        
        return script_to_lang.get(dominant, 'unknown')
    
    def is_supported(self, language_code: str) -> bool:
        """
        Check if a language is supported
        
        Args:
            language_code: ISO language code (e.g., 'en', 'hi')
        
        Returns:
            True if supported
        """
        # FastText supports 176 languages
        # Common ones:
        supported_langs = [
            'en', 'hi', 'es', 'ar', 'fr', 'de', 'it', 'pt', 'ru', 'zh',
            'ja', 'ko', 'tr', 'pl', 'nl', 'sv', 'da', 'fi', 'no', 'cs',
            'el', 'he', 'th', 'vi', 'id', 'ms', 'bn', 'ta', 'te', 'mr',
            'ur', 'fa', 'sw', 'am', 'km', 'my', 'ne', 'si', 'pa'
            # ... and 137 more
        ]
        
        return language_code in supported_langs
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get full language name from code
        
        Args:
            language_code: ISO code (e.g., 'en')
        
        Returns:
            Language name (e.g., 'English')
        """
        # Common language names
        lang_names = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            'ar': 'Arabic',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'tr': 'Turkish',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'ur': 'Urdu',
            'th': 'Thai',
            'vi': 'Vietnamese'
        }
        
        return lang_names.get(language_code, language_code.upper())


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    # Initialize detector
    detector = LanguageDetector()
    
    # Test cases
    test_texts = [
        ("Hello, how are you?", "en"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("Hola, ¿cómo estás?", "es"),
        ("مرحبا كيف حالك؟", "ar"),
        ("Bonjour, comment allez-vous?", "fr"),
        ("你好，你好吗？", "zh"),
        ("This is a mix of English और हिंदी", "en"),  # Code-mixed
        ("短い", "ja"),  # Short text (Japanese)
    ]
    
    print("="*60)
    print("LANGUAGE DETECTION TESTS")
    print("="*60)
    
    for text, expected in test_texts:
        lang, conf = detector.detect(text)
        lang_name = detector.get_language_name(lang)
        
        status = "✓" if lang == expected else "✗"
        
        print(f"\n{status} Text: {text}")
        print(f"  Detected: {lang} ({lang_name}) - Confidence: {conf:.4f}")
        print(f"  Expected: {expected}")
    
    # Test multiple predictions
    print("\n" + "="*60)
    print("TOP-3 LANGUAGE PREDICTIONS")
    print("="*60)
    
    mixed_text = "This is English mixed with français and немного русского"
    predictions = detector.detect_multiple(mixed_text, top_k=3)
    
    print(f"\nText: {mixed_text}")
    print("Predictions:")
    for lang, conf in predictions:
        print(f"  {lang} ({detector.get_language_name(lang)}): {conf:.4f}")
