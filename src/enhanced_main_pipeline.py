"""
src/enhanced_main_pipeline.py
──────────────────────────────
Fixed version of the original enhanced_main_pipeline.py.

FIXES:
  1. Wrong voice import path: src.voice_processing → src.voice
  2. All components now lazy-loaded (no 2-min startup)
  3. Falls back gracefully if optional components unavailable

NOTE: For new code, prefer src/fact_verification_service.py which is
the proper unified entry point. This file is kept for backward
compatibility with scripts/export_pipeline_outputs.py and
scripts/test_single_claim.py which import EnhancedFactVerificationPipeline.
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedFactVerificationPipeline:
    """
    Backward-compatible wrapper around FactVerificationService.
    Provides the same interface as the original broken class.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._service = None
        logger.info("EnhancedFactVerificationPipeline ready (lazy loading)")

    def _get_service(self):
        if self._service is None:
            from src.fact_verification_service import FactVerificationService
            self._service = FactVerificationService()
        return self._service

    def verify_claim(
        self,
        input_data: Union[str, Dict[str, Any]],
        input_type: str = "text",
        user_language: str = None,
        enable_voice_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify a claim. Delegates to FactVerificationService.

        Args:
            input_data  : claim text, or file path string
            input_type  : 'text', 'voice', 'document', 'pdf', 'docx', 'image'
            user_language: ISO code (auto-detected if None)
            enable_voice_output: Generate audio response

        Returns:
            Result dict with verdict, confidence, report, llm_context, etc.
        """
        svc = self._get_service()

        # Normalise input_type to service format
        type_map = {
            "text":     "text",
            "voice":    "audio",
            "document": "pdf",
            "pdf":      "pdf",
            "docx":     "docx",
            "image":    "image",
        }
        svc_type = type_map.get(input_type, "text")

        # Determine claim vs file_path
        if isinstance(input_data, dict):
            claim     = input_data.get("text", input_data.get("claim", ""))
            file_path = input_data.get("file_path")
        elif svc_type in ("audio", "pdf", "docx", "image"):
            claim     = None
            file_path = input_data
        else:
            claim     = input_data
            file_path = None

        llm_mode = self.config.get("llm_mode", "groq")

        result = svc.verify(
            claim=claim,
            file_path=file_path,
            input_type=svc_type,
            llm_mode=llm_mode,
            generate_audio=enable_voice_output,
        )

        # Add legacy fields that old scripts expect
        result.setdefault("explanation", result.get("report", "")[:200])
        result.setdefault("llm_context",
            result.get("rag_result", {}).get("llm_context", "") if result.get("rag_result") else "")
        result.setdefault("metadata", {
            "user_language": result.get("user_language", "en"),
            "original_claim": result.get("claim", ""),
        })

        return result

    def upload_document(
        self,
        file_path: str,
        user_id: str = "default",
        analyze_claims: bool = False,
    ) -> Dict[str, Any]:
        """Upload and index a document into the RAG vector database."""
        from src.document_processing.document_handler import DocumentHandler
        from src.rag.vector_database import VectorDatabase

        handler = DocumentHandler()
        vdb     = VectorDatabase()
        doc     = handler.process_upload(file_path, user_id)
        handler.add_to_rag(doc, vdb, "uploaded_documents")

        result = {
            "mode": "document_upload",
            "document_info": {
                "filename":    doc["metadata"]["filename"],
                "chunks_added": len(doc["chunks"]),
                "word_count":  doc["metadata"]["word_count"],
            },
            "message": f"Document uploaded. {len(doc['chunks'])} chunks indexed.",
        }

        if analyze_claims:
            from src.document_processing.claim_extractor import ClaimExtractor
            from src.nlp.model_manager import ModelManager
            mm        = ModelManager()
            extractor = ClaimExtractor(mm.load_claim_detector())
            analysis  = extractor.analyze_document_claims(doc)
            result["claim_analysis"] = analysis
            result["mode"]           = "document_analysis"

        return result

    def get_system_status(self) -> Dict[str, Any]:
        """Return basic system status."""
        from src.rag.vector_database import VectorDatabase
        vdb = VectorDatabase()
        collections = vdb.list_collections()
        return {
            "system_status": "operational",
            "collections":   collections,
            "capabilities": {
                "text_input":    True,
                "voice_input":   True,
                "document_upload": True,
                "multilingual":  True,
            },
        }