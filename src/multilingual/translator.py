"""
Translation Service for Multilingual Support
Handles translation to/from English for RAG processing
"""

import logging
from typing import Dict, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class Translator:
    """
    Translation service for multilingual support
    
    Supports multiple translation backends:
    - Helsinki-NLP (free, good quality)
    - Google Translate (unofficial, free)
    - Azure Translator (paid, best quality)
    """
    
    def __init__(self, backend: str = 'google'):
        """
        Initialize translator
        
        Args:
            backend: Translation backend ('helsinki', 'google', 'azure')
        """
        self.backend = backend
        self.translator = None
        
        self._initialize_backend()
        
        logger.info(f"✓ Translator initialized (backend: {backend})")
    
    def _initialize_backend(self):
        """Initialize translation backend"""
        
        if self.backend == 'helsinki':
            self._initialize_helsinki()
        elif self.backend == 'google':
            self._initialize_google()
        elif self.backend == 'azure':
            self._initialize_azure()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _initialize_helsinki(self):
        """Initialize Helsinki-NLP models (free, offline)"""
        try:
            from transformers import pipeline
            
            # We'll load models on-demand to save memory
            self.translator = 'helsinki'
            logger.info("Using Helsinki-NLP translation (free, good quality)")
            
        except ImportError:
            logger.error("transformers not installed")
            raise
    
    def _initialize_google(self):
        """Initialize Google Translate using deep-translator (no httpx conflict)"""
        try:
            from deep_translator import GoogleTranslator
            
            self.translator = GoogleTranslator()
            logger.info("Using deep-translator GoogleTranslator (free, requires internet)")
            
        except ImportError:
            logger.error("deep-translator not installed")
            logger.error("Install with: pip install deep-translator")
            raise
    
    def _initialize_azure(self):
        """Initialize Azure Translator (paid, best quality)"""
        try:
            import os
            
            # Azure requires API key
            api_key = os.getenv('AZURE_TRANSLATOR_KEY')
            if not api_key:
                raise ValueError("AZURE_TRANSLATOR_KEY environment variable not set")
            
            self.translator = {
                'key': api_key,
                'endpoint': os.getenv('AZURE_TRANSLATOR_ENDPOINT', 'https://api.cognitive.microsofttranslator.com'),
                'region': os.getenv('AZURE_TRANSLATOR_REGION', 'global')
            }
            
            logger.info("Using Azure Translator (paid, best quality)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Translator: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'hi', 'es')
            target_lang: Target language code (e.g., 'en')
        
        Returns:
            Translated text
        """
        # No translation needed
        if source_lang == target_lang:
            return text
        
        # Translate based on backend
        if self.backend == 'helsinki':
            return self._translate_helsinki(text, source_lang, target_lang)
        elif self.backend == 'google':
            return self._translate_google(text, source_lang, target_lang)
        elif self.backend == 'azure':
            return self._translate_azure(text, source_lang, target_lang)
    
    def _translate_helsinki(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate using Helsinki-NLP"""
        from transformers import pipeline
        
        # Map language codes to Helsinki model names
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        try:
            translator = pipeline("translation", model=model_name)
            result = translator(text, max_length=512)[0]['translation_text']
            return result
            
        except Exception as e:
            logger.warning(f"Helsinki translation failed: {e}")
            logger.warning(f"Model not found: {model_name}")
            logger.warning("Falling back to Google Translate")
            
            # Fallback to Google
            return self._translate_google(text, source_lang, target_lang)
    
    def _translate_google(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate using deep-translator GoogleTranslator"""
        try:
            from deep_translator import GoogleTranslator
            
            # Create translator with source and target languages
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            result = translator.translate(text)
            return result
            
        except Exception as e:
            logger.error(f"Google translation failed: {e}")
            return text  # Return original on failure
    
    def _translate_azure(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate using Azure Translator"""
        import requests
        
        endpoint = f"{self.translator['endpoint']}/translate"
        params = {
            'api-version': '3.0',
            'from': source_lang,
            'to': target_lang
        }
        headers = {
            'Ocp-Apim-Subscription-Key': self.translator['key'],
            'Ocp-Apim-Subscription-Region': self.translator['region'],
            'Content-type': 'application/json'
        }
        body = [{'text': text}]
        
        try:
            response = requests.post(endpoint, params=params, headers=headers, json=body)
            response.raise_for_status()
            
            result = response.json()
            return result[0]['translations'][0]['text']
            
        except Exception as e:
            logger.error(f"Azure translation failed: {e}")
            return text
    
    def to_english(self, text: str, source_lang: str) -> str:
        """Translate to English"""
        return self.translate(text, source_lang, 'en')
    
    def from_english(self, text: str, target_lang: str) -> str:
        """Translate from English"""
        return self.translate(text, 'en', target_lang)


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("TRANSLATION TESTS")
    print("="*80)
    
    # Use Google Translate (easiest for testing)
    translator = Translator(backend='google')
    
    test_cases = [
        ("India's GDP grew 8% in 2024", 'en', 'hi'),
        ("भारत की जीडीपी 2024 में 8% बढ़ी", 'hi', 'en'),
        ("El PIB de India creció un 8% en 2024", 'es', 'en'),
    ]
    
    for text, src, tgt in test_cases:
        translated = translator.translate(text, src, tgt)
        print(f"\n{src} → {tgt}:")
        print(f"  Original: {text}")
        print(f"  Translated: {translated}")