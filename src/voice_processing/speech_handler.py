"""
Voice Processing Handler
Handles speech-to-text and text-to-speech for multilingual support
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class SpeechHandler:
    """
    Handle voice input and output
    
    Features:
    - Speech-to-Text (STT) with language detection
    - Text-to-Speech (TTS) for responses
    - Support for multiple backends (OpenAI Whisper, Google Speech)
    """
    
    def __init__(
        self,
        stt_backend: str = 'whisper',
        tts_backend: str = 'gtts'
    ):
        """
        Initialize speech handler
        
        Args:
            stt_backend: STT backend ('whisper', 'google', 'azure')
            tts_backend: TTS backend ('gtts', 'azure', 'elevenlabs')
        """
        self.stt_backend = stt_backend
        self.tts_backend = tts_backend
        
        self.stt_model = None
        self.tts_engine = None
        
        self._initialize_stt()
        self._initialize_tts()
        
        logger.info(f"✓ SpeechHandler initialized")
        logger.info(f"  STT: {stt_backend}, TTS: {tts_backend}")
    
    def _initialize_stt(self):
        """Initialize Speech-to-Text backend"""
        
        if self.stt_backend == 'whisper':
            self._initialize_whisper()
        elif self.stt_backend == 'google':
            self._initialize_google_stt()
        elif self.stt_backend == 'azure':
            self._initialize_azure_stt()
        else:
            raise ValueError(f"Unknown STT backend: {self.stt_backend}")
    
    def _initialize_whisper(self):
        """Initialize OpenAI Whisper (recommended)"""
        try:
            import whisper
            
            # Load model (base is good balance of speed/accuracy)
            self.stt_model = whisper.load_model("base")
            logger.info("Using OpenAI Whisper (free, local, excellent quality)")
            
        except ImportError:
            logger.error("whisper not installed: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _initialize_google_stt(self):
        """Initialize Google Speech-to-Text"""
        try:
            import speech_recognition as sr
            
            self.stt_model = sr.Recognizer()
            logger.info("Using Google Speech Recognition (free, requires internet)")
            
        except ImportError:
            logger.error("SpeechRecognition not installed: pip install SpeechRecognition")
            raise
    
    def _initialize_azure_stt(self):
        """Initialize Azure Speech Services"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            import os
            
            # Azure requires API key
            speech_key = os.getenv('AZURE_SPEECH_KEY')
            service_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            
            if not speech_key:
                raise ValueError("AZURE_SPEECH_KEY environment variable not set")
            
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, 
                region=service_region
            )
            
            self.stt_model = speech_config
            logger.info("Using Azure Speech Services (paid, best quality)")
            
        except ImportError:
            logger.error("azure-cognitiveservices-speech not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Azure Speech: {e}")
            raise
    
    def _initialize_tts(self):
        """Initialize Text-to-Speech backend"""
        
        if self.tts_backend == 'gtts':
            self._initialize_gtts()
        elif self.tts_backend == 'azure':
            self._initialize_azure_tts()
        elif self.tts_backend == 'elevenlabs':
            self._initialize_elevenlabs()
        else:
            logger.warning(f"Unknown TTS backend: {self.tts_backend}")
            logger.warning("TTS will be disabled")
    
    def _initialize_gtts(self):
        """Initialize Google Text-to-Speech (gTTS)"""
        try:
            from gtts import gTTS
            
            self.tts_engine = 'gtts'
            logger.info("Using Google Text-to-Speech (free, requires internet)")
            
        except ImportError:
            logger.error("gtts not installed: pip install gtts")
            logger.warning("TTS will be disabled")
    
    def _initialize_azure_tts(self):
        """Initialize Azure Text-to-Speech"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            import os
            
            speech_key = os.getenv('AZURE_SPEECH_KEY')
            service_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            
            if not speech_key:
                raise ValueError("AZURE_SPEECH_KEY environment variable not set")
            
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, 
                region=service_region
            )
            
            self.tts_engine = speech_config
            logger.info("Using Azure Text-to-Speech (paid, best quality)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure TTS: {e}")
            logger.warning("TTS will be disabled")
    
    def _initialize_elevenlabs(self):
        """Initialize ElevenLabs TTS"""
        try:
            import elevenlabs
            import os
            
            api_key = os.getenv('ELEVENLABS_API_KEY')
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEY environment variable not set")
            
            elevenlabs.set_api_key(api_key)
            self.tts_engine = 'elevenlabs'
            logger.info("Using ElevenLabs TTS (paid, premium quality)")
            
        except ImportError:
            logger.error("elevenlabs not installed: pip install elevenlabs")
            logger.warning("TTS will be disabled")
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs: {e}")
            logger.warning("TTS will be disabled")
    
    def speech_to_text(
        self,
        audio_file: str,
        language: str = None
    ) -> Dict[str, Any]:
        """
        Convert speech to text with language detection
        
        Args:
            audio_file: Path to audio file (wav, mp3, etc.)
            language: Expected language (optional, auto-detected if None)
        
        Returns:
            Dict with 'text', 'language', 'confidence'
        """
        logger.info(f"Converting speech to text: {audio_file}")
        
        if self.stt_backend == 'whisper':
            return self._whisper_stt(audio_file, language)
        elif self.stt_backend == 'google':
            return self._google_stt(audio_file, language)
        elif self.stt_backend == 'azure':
            return self._azure_stt(audio_file, language)
        else:
            raise ValueError(f"STT backend not initialized: {self.stt_backend}")
    
    def _whisper_stt(
        self,
        audio_file: str,
        language: str = None
    ) -> Dict[str, Any]:
        """Speech-to-text using Whisper"""
        try:
            # Transcribe with language detection
            result = self.stt_model.transcribe(
                audio_file,
                language=language  # None = auto-detect
            )
            
            return {
                'text': result['text'].strip(),
                'language': result['language'],
                'confidence': 0.95,  # Whisper doesn't provide confidence
                'backend': 'whisper'
            }
            
        except Exception as e:
            logger.error(f"Whisper STT failed: {e}")
            raise
    
    def _google_stt(
        self,
        audio_file: str,
        language: str = None
    ) -> Dict[str, Any]:
        """Speech-to-text using Google Speech Recognition"""
        try:
            import speech_recognition as sr
            
            # Load audio file
            with sr.AudioFile(audio_file) as source:
                audio = self.stt_model.record(source)
            
            # Recognize speech
            if language:
                text = self.stt_model.recognize_google(audio, language=language)
            else:
                # Try multiple languages
                for lang in ['en-US', 'hi-IN', 'es-ES']:
                    try:
                        text = self.stt_model.recognize_google(audio, language=lang)
                        language = lang[:2]  # Extract language code
                        break
                    except sr.UnknownValueError:
                        continue
                else:
                    raise sr.UnknownValueError("Could not understand audio")
            
            return {
                'text': text,
                'language': language or 'en',
                'confidence': 0.85,
                'backend': 'google'
            }
            
        except Exception as e:
            logger.error(f"Google STT failed: {e}")
            raise
    
    def _azure_stt(
        self,
        audio_file: str,
        language: str = None
    ) -> Dict[str, Any]:
        """Speech-to-text using Azure Speech Services"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Set language
            if language:
                self.stt_model.speech_recognition_language = language
            else:
                # Auto-detect mode
                auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=["en-US", "hi-IN", "es-ES"]
                )
            
            # Create recognizer
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
            
            if language:
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.stt_model,
                    audio_config=audio_config
                )
            else:
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.stt_model,
                    auto_detect_source_language_config=auto_detect_source_language_config,
                    audio_config=audio_config
                )
            
            # Recognize
            result = speech_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                detected_language = result.properties.get(
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult,
                    language or 'en'
                )
                
                return {
                    'text': result.text,
                    'language': detected_language[:2],
                    'confidence': 0.90,
                    'backend': 'azure'
                }
            else:
                raise Exception(f"Speech recognition failed: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure STT failed: {e}")
            raise
    
    def text_to_speech(
        self,
        text: str,
        language: str = 'en',
        output_file: str = None
    ) -> Optional[str]:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            language: Language code (e.g., 'en', 'hi', 'es')
            output_file: Output audio file path (optional)
        
        Returns:
            Path to generated audio file (if output_file provided)
        """
        if not self.tts_engine:
            logger.warning("TTS not available")
            return None
        
        logger.info(f"Converting text to speech ({language}): {text[:50]}...")
        
        if self.tts_backend == 'gtts':
            return self._gtts_tts(text, language, output_file)
        elif self.tts_backend == 'azure':
            return self._azure_tts(text, language, output_file)
        elif self.tts_backend == 'elevenlabs':
            return self._elevenlabs_tts(text, language, output_file)
    
    def _gtts_tts(
        self,
        text: str,
        language: str,
        output_file: str = None
    ) -> Optional[str]:
        """Text-to-speech using gTTS"""
        try:
            from gtts import gTTS
            import pygame
            
            # Create TTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to file
            if not output_file:
                output_file = tempfile.mktemp(suffix='.mp3')
            
            tts.save(output_file)
            
            # Play audio (optional)
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(output_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
            except ImportError:
                logger.info("pygame not installed - audio saved but not played")
                logger.info("Install with: pip install pygame")
            
            return output_file
            
        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            return None
    
    def _azure_tts(
        self,
        text: str,
        language: str,
        output_file: str = None
    ) -> Optional[str]:
        """Text-to-speech using Azure"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Set voice based on language
            voice_map = {
                'en': 'en-US-JennyNeural',
                'hi': 'hi-IN-SwaraNeural',
                'es': 'es-ES-ElviraNeural'
            }
            
            voice = voice_map.get(language, 'en-US-JennyNeural')
            self.tts_engine.speech_synthesis_voice_name = voice
            
            # Create synthesizer
            if not output_file:
                output_file = tempfile.mktemp(suffix='.wav')
            
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.tts_engine,
                audio_config=audio_config
            )
            
            # Synthesize
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return output_file
            else:
                logger.error(f"Azure TTS failed: {result.reason}")
                return None
                
        except Exception as e:
            logger.error(f"Azure TTS failed: {e}")
            return None
    
    def _elevenlabs_tts(
        self,
        text: str,
        language: str,
        output_file: str = None
    ) -> Optional[str]:
        """Text-to-speech using ElevenLabs"""
        try:
            import elevenlabs
            
            # Generate audio
            audio = elevenlabs.generate(
                text=text,
                voice="Bella",  # You can customize this
                model="eleven_multilingual_v2"
            )
            
            # Save to file
            if not output_file:
                output_file = tempfile.mktemp(suffix='.mp3')
            
            with open(output_file, 'wb') as f:
                f.write(audio)
            
            return output_file
            
        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            return None


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("SPEECH PROCESSING TEST")
    print("="*80)
    
    # Initialize speech handler
    speech_handler = SpeechHandler(stt_backend='whisper', tts_backend='gtts')
    
    print("\nTo test:")
    print("1. Record audio file: test_audio.wav")
    print("2. Run: result = speech_handler.speech_to_text('test_audio.wav')")
    print("3. Run: speech_handler.text_to_speech('Hello world', 'en')")