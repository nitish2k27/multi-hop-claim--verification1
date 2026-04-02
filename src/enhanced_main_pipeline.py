"""
Enhanced Main Pipeline with Multilingual Support and Document Processing
Integrates all components for complete fact verification system
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import tempfile

# Import existing components
from src.preprocessing.input_processor import InputProcessor
from src.nlp.nlp_pipeline import NLPPipeline
from src.nlp.model_manager import ModelManager
from src.rag.rag_pipeline import RAGPipeline
from src.rag.vector_database import VectorDatabase

# Import new components
from src.multilingual.translator import Translator
from src.multilingual.multilingual_pipeline import MultilingualVerificationPipeline
from src.document_processing.document_handler import DocumentHandler
from src.document_processing.claim_extractor import ClaimExtractor
from src.voice_processing.speech_handler import SpeechHandler

logger = logging.getLogger(__name__)


class EnhancedFactVerificationPipeline:
    """
    Complete fact verification pipeline with multilingual and multimodal support
    
    Features:
    - Text input (any language)
    - Voice input (speech-to-text)
    - Document upload (PDF, DOCX, TXT)
    - Claim extraction from documents
    - Multilingual processing
    - Voice output (text-to-speech)
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None
    ):
        """
        Initialize enhanced pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Initialize core components
        self._initialize_components()
        
        logger.info("✓ EnhancedFactVerificationPipeline initialized")
        logger.info(f"  Supported inputs: text, voice, documents")
        logger.info(f"  Supported languages: multilingual")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'translation': {
                'backend': 'google',  # google, helsinki, azure
                'cache_size': 1000
            },
            'speech': {
                'stt_backend': 'whisper',  # whisper, google, azure
                'tts_backend': 'gtts',     # gtts, azure, elevenlabs
                'enable_tts': True
            },
            'documents': {
                'upload_dir': 'data/uploads',
                'chunk_size': 500,
                'chunk_overlap': 50,
                'supported_formats': ['.pdf', '.docx', '.txt']
            },
            'claim_extraction': {
                'confidence_threshold': 0.8,
                'batch_size': 32
            },
            'rag': {
                'collections': ['news_articles', 'uploaded_documents'],
                'search_strategy': 'context_aware',  # context_aware, equal_weight, prioritize_uploads
                'top_k': 5
            }
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        
        logger.info("Initializing pipeline components...")
        
        # Core NLP components
        self.input_processor = InputProcessor()
        self.model_manager = ModelManager()
        self.nlp_pipeline = NLPPipeline()
        
        # RAG components
        self.vector_db = VectorDatabase()
        self.rag_pipeline = RAGPipeline(
            self.vector_db,
            'news_articles',
            self.model_manager
        )
        
        # New components
        self.translator = Translator(
            backend=self.config['translation']['backend']
        )
        
        self.multilingual_pipeline = MultilingualVerificationPipeline(
            self.nlp_pipeline,
            self.rag_pipeline,
            self.translator
        )
        
        self.document_handler = DocumentHandler(
            upload_dir=self.config['documents']['upload_dir'],
            chunk_size=self.config['documents']['chunk_size'],
            chunk_overlap=self.config['documents']['chunk_overlap']
        )
        
        # Initialize claim extractor with trained model
        claim_detector = self.model_manager.get_model('claim_detector')
        self.claim_extractor = ClaimExtractor(
            claim_detector,
            confidence_threshold=self.config['claim_extraction']['confidence_threshold'],
            batch_size=self.config['claim_extraction']['batch_size']
        )
        
        # Speech components (optional)
        if self.config['speech']['stt_backend'] or self.config['speech']['tts_backend']:
            try:
                self.speech_handler = SpeechHandler(
                    stt_backend=self.config['speech']['stt_backend'],
                    tts_backend=self.config['speech']['tts_backend']
                )
            except Exception as e:
                logger.warning(f"Speech handler initialization failed: {e}")
                logger.warning("Voice processing will be disabled")
                self.speech_handler = None
        else:
            self.speech_handler = None
        
        logger.info("✓ All components initialized")
    
    def verify_claim(
        self,
        input_data: Union[str, Dict[str, Any]],
        input_type: str = 'text',
        user_language: str = None,
        enable_voice_output: bool = None
    ) -> Dict[str, Any]:
        """
        Verify claim from any input type
        
        Args:
            input_data: Input data (text, file path, or dict)
            input_type: 'text', 'voice', 'document'
            user_language: User's language (auto-detected if None)
            enable_voice_output: Enable TTS response
        
        Returns:
            Verification result with metadata
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"FACT VERIFICATION REQUEST")
        logger.info(f"Input type: {input_type}")
        logger.info(f"{'='*80}")
        
        # Set voice output default
        if enable_voice_output is None:
            enable_voice_output = self.config['speech']['enable_tts']
        
        try:
            # Step 1: Process input based on type
            if input_type == 'text':
                claim_text, detected_language = self._process_text_input(
                    input_data, user_language
                )
            elif input_type == 'voice':
                claim_text, detected_language = self._process_voice_input(
                    input_data, user_language
                )
            elif input_type == 'document':
                return self._process_document_input(input_data, user_language)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
            
            # Step 2: Verify claim using multilingual pipeline
            result = self.multilingual_pipeline.verify_claim(
                claim_text,
                user_language=detected_language or user_language
            )
            
            # Step 3: Add input metadata
            result['input_metadata'] = {
                'input_type': input_type,
                'original_input': str(input_data)[:100] + "..." if len(str(input_data)) > 100 else str(input_data),
                'detected_language': detected_language,
                'processing_timestamp': self._get_timestamp()
            }
            
            # Step 4: Generate voice output if requested
            if enable_voice_output and self.speech_handler:
                audio_file = self._generate_voice_response(
                    result['explanation'],
                    result['metadata']['user_language']
                )
                if audio_file:
                    result['audio_response'] = audio_file
            
            logger.info(f"✓ Verification complete: {result['verdict']}")
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                'verdict': 'ERROR',
                'confidence': 0.0,
                'explanation': f"Processing failed: {str(e)}",
                'error': str(e),
                'input_metadata': {
                    'input_type': input_type,
                    'error_timestamp': self._get_timestamp()
                }
            }
    
    def _process_text_input(
        self,
        text: str,
        user_language: str = None
    ) -> tuple[str, str]:
        """Process text input"""
        
        logger.info(f"Processing text input: {text[:50]}...")
        
        # Detect language if not provided
        if not user_language:
            result = self.input_processor.process(text)
            detected_language = result.get('language', 'en')
        else:
            detected_language = user_language
        
        return text, detected_language
    
    def _process_voice_input(
        self,
        audio_file: str,
        user_language: str = None
    ) -> tuple[str, str]:
        """Process voice input"""
        
        if not self.speech_handler:
            raise ValueError("Speech processing not available")
        
        logger.info(f"Processing voice input: {audio_file}")
        
        # Convert speech to text
        stt_result = self.speech_handler.speech_to_text(
            audio_file,
            language=user_language
        )
        
        claim_text = stt_result['text']
        detected_language = stt_result['language']
        
        logger.info(f"STT result: {claim_text}")
        logger.info(f"Detected language: {detected_language}")
        
        return claim_text, detected_language
    
    def _process_document_input(
        self,
        input_data: Union[str, Dict[str, Any]],
        user_language: str = None
    ) -> Dict[str, Any]:
        """Process document input"""
        
        logger.info("Processing document input")
        
        # Handle different input formats
        if isinstance(input_data, str):
            # File path
            file_path = input_data
            user_id = "default"
            mode = "verification"  # User will ask questions about the document
        elif isinstance(input_data, dict):
            file_path = input_data['file_path']
            user_id = input_data.get('user_id', 'default')
            mode = input_data.get('mode', 'verification')
        else:
            raise ValueError("Invalid document input format")
        
        # Process document
        document_data = self.document_handler.process_upload(file_path, user_id)
        
        # Add to RAG database
        self.document_handler.add_to_rag(
            document_data,
            self.vector_db,
            collection_name='uploaded_documents'
        )
        
        if mode == 'analysis':
            # Extract and analyze claims from document
            analysis = self.claim_extractor.analyze_document_claims(document_data)
            
            # Translate results if needed
            if user_language and user_language != 'en':
                analysis = self._translate_analysis_results(analysis, user_language)
            
            return {
                'mode': 'document_analysis',
                'document_info': analysis['document_info'],
                'claim_analysis': analysis,
                'message': f"Document processed and analyzed. Found {analysis['claim_statistics']['total_claims']} claims."
            }
        
        else:
            # Verification mode - document added to RAG for future queries
            return {
                'mode': 'document_upload',
                'document_info': {
                    'filename': document_data['metadata']['filename'],
                    'chunks_added': len(document_data['chunks']),
                    'word_count': document_data['metadata']['word_count']
                },
                'message': f"Document uploaded and processed. You can now ask questions about it."
            }
    
    def _translate_analysis_results(
        self,
        analysis: Dict[str, Any],
        target_language: str
    ) -> Dict[str, Any]:
        """Translate claim analysis results"""
        
        # Translate claim texts
        for claim_list in analysis['claims'].values():
            for claim in claim_list:
                if 'text' in claim:
                    claim['text_translated'] = self.translator.from_english(
                        claim['text'],
                        target_language
                    )
        
        return analysis
    
    def _generate_voice_response(
        self,
        text: str,
        language: str
    ) -> Optional[str]:
        """Generate voice response"""
        
        if not self.speech_handler:
            return None
        
        try:
            audio_file = self.speech_handler.text_to_speech(
                text,
                language=language
            )
            return audio_file
        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def upload_document(
        self,
        file_path: str,
        user_id: str = "default",
        analyze_claims: bool = False
    ) -> Dict[str, Any]:
        """
        Upload and process document
        
        Args:
            file_path: Path to document file
            user_id: User identifier
            analyze_claims: Whether to extract claims from document
        
        Returns:
            Processing result
        """
        mode = 'analysis' if analyze_claims else 'verification'
        
        return self._process_document_input({
            'file_path': file_path,
            'user_id': user_id,
            'mode': mode
        })
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities"""
        
        # Check component status
        components_status = {
            'input_processor': bool(self.input_processor),
            'nlp_pipeline': bool(self.nlp_pipeline),
            'rag_pipeline': bool(self.rag_pipeline),
            'translator': bool(self.translator),
            'document_handler': bool(self.document_handler),
            'claim_extractor': bool(self.claim_extractor),
            'speech_handler': bool(self.speech_handler)
        }
        
        # Check database collections
        try:
            collections = self.vector_db.list_collections()
            db_status = {
                'available_collections': collections,
                'news_articles_count': len(self.vector_db.get_collection('news_articles').get()['ids']) if 'news_articles' in collections else 0,
                'uploaded_documents_count': len(self.vector_db.get_collection('uploaded_documents').get()['ids']) if 'uploaded_documents' in collections else 0
            }
        except Exception as e:
            db_status = {'error': str(e)}
        
        return {
            'system_status': 'operational' if all(components_status.values()) else 'partial',
            'components': components_status,
            'database': db_status,
            'capabilities': {
                'text_input': True,
                'voice_input': bool(self.speech_handler),
                'document_upload': True,
                'multilingual': True,
                'voice_output': bool(self.speech_handler),
                'claim_extraction': True
            },
            'supported_languages': ['en', 'hi', 'es', 'fr', 'de', 'zh'],  # Add more as needed
            'supported_document_formats': self.config['documents']['supported_formats'],
            'configuration': self.config
        }


# ==========================================
# TESTING AND EXAMPLES
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("ENHANCED FACT VERIFICATION PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = EnhancedFactVerificationPipeline()
    
    # Check system status
    status = pipeline.get_system_status()
    print(f"\nSystem Status: {status['system_status']}")
    print(f"Available Collections: {status['database'].get('available_collections', [])}")
    
    # Example usage
    print("\n" + "="*50)
    print("EXAMPLE USAGE")
    print("="*50)
    
    print("\n1. Text Input:")
    print("result = pipeline.verify_claim('India GDP grew 8% in 2024')")
    
    print("\n2. Voice Input:")
    print("result = pipeline.verify_claim('audio.wav', input_type='voice')")
    
    print("\n3. Document Upload:")
    print("result = pipeline.upload_document('report.pdf', analyze_claims=True)")
    
    print("\n4. Multilingual:")
    print("result = pipeline.verify_claim('भारत की जीडीपी बढ़ी', user_language='hi')")
    
    # Test with sample text
    print("\n" + "="*50)
    print("SAMPLE TEST")
    print("="*50)
    
    try:
        result = pipeline.verify_claim(
            "India's GDP grew by 8% in 2024",
            input_type='text',
            user_language='en'
        )
        
        print(f"\nVerdict: {result.get('verdict', 'ERROR')}")
        print(f"Confidence: {result.get('confidence', 0):.1f}%")
        print(f"Explanation: {result.get('explanation', 'No explanation')[:100]}...")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if models are not trained yet.")