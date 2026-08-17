"""
Document Upload and Processing
Handles PDF, DOCX, TXT files uploaded by users
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentHandler:
    """
    Handle uploaded documents
    
    Features:
    - Extract text from PDF, DOCX, TXT
    - Chunk documents for RAG
    - Store metadata
    - Add to vector database
    """
    
    def __init__(
        self,
        upload_dir: str = "data/uploads",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize document handler
        
        Args:
            upload_dir: Directory to store uploads
            chunk_size: Size of text chunks (words)
            chunk_overlap: Overlap between chunks
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"✓ DocumentHandler initialized")
        logger.info(f"  Upload dir: {self.upload_dir}")
    
    def process_upload(
        self,
        file_path: str,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process uploaded file
        
        Args:
            file_path: Path to uploaded file
            user_id: User ID for organization
        
        Returns:
            Processing result with extracted text and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing upload: {file_path.name}")
        
        # Extract text based on file type
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            text = self._extract_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            text = self._extract_docx(file_path)
        elif file_ext == '.txt':
            text = self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Generate file hash for deduplication
        file_hash = self._hash_file(file_path)
        
        # Create metadata
        metadata = {
            'filename': file_path.name,
            'file_type': file_ext[1:],
            'file_hash': file_hash,
            'user_id': user_id,
            'upload_date': datetime.now().isoformat(),
            'file_size': file_path.stat().st_size,
            'word_count': len(text.split())
        }
        
        # Chunk text
        chunks = self._chunk_text(text)
        
        logger.info(f"✓ Extracted {len(text)} chars, {len(chunks)} chunks")
        
        # Save to permanent storage
        stored_path = self._store_file(file_path, file_hash, user_id)
        metadata['stored_path'] = str(stored_path)
        
        return {
            'text': text,
            'chunks': chunks,
            'metadata': metadata
        }
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        try:
            import PyPDF2
            
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            
            return '\n\n'.join(text)
            
        except ImportError:
            logger.error("PyPDF2 not installed: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            import docx
            
            doc = docx.Document(file_path)
            text = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)
            
            return '\n\n'.join(text)
            
        except ImportError:
            logger.error("python-docx not installed: pip install python-docx")
            raise
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    def _extract_txt(self, file_path: Path) -> str:
        """Extract text from TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text for RAG
        
        Args:
            text: Full document text
        
        Returns:
            List of chunks with metadata
        """
        words = text.split()
        chunks = []
        
        i = 0
        chunk_id = 0
        
        while i < len(words):
            # Get chunk
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_word': i,
                'end_word': i + len(chunk_words),
                'word_count': len(chunk_words)
            })
            
            # Move to next chunk with overlap
            i += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        return chunks
    
    def _hash_file(self, file_path: Path) -> str:
        """Generate file hash for deduplication"""
        hasher = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _store_file(
        self,
        file_path: Path,
        file_hash: str,
        user_id: str
    ) -> Path:
        """Store file in permanent location"""
        
        # Organize by user and date
        date_folder = datetime.now().strftime('%Y%m%d')
        storage_dir = self.upload_dir / user_id / date_folder
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Use hash to prevent duplicates
        stored_path = storage_dir / f"{file_hash}{file_path.suffix}"
        
        if not stored_path.exists():
            import shutil
            shutil.copy2(file_path, stored_path)
        
        return stored_path
    
    def add_to_rag(
        self,
        document_data: Dict[str, Any],
        vector_db,
        collection_name: str = "uploaded_documents"
    ):
        """
        Add processed document to RAG
        
        Args:
            document_data: Output from process_upload()
            vector_db: VectorDatabase instance
            collection_name: Collection name
        """
        logger.info(f"Adding document to RAG collection: {collection_name}")
        
        # Create or get collection
        vector_db.create_collection(collection_name)
        
        # Add chunks
        chunks = document_data['chunks']
        metadata = document_data['metadata']
        
        for chunk in chunks:
            # Combine metadata
            chunk_metadata = {
                **metadata,
                'chunk_id': chunk['chunk_id'],
                'chunk_word_count': chunk['word_count']
            }
            
            # Add to vector DB
            vector_db.add_documents(
                collection_name=collection_name,
                documents=[chunk['text']],
                metadatas=[chunk_metadata],
                ids=[f"{metadata['file_hash']}_{chunk['chunk_id']}"]
            )
        
        logger.info(f"✓ Added {len(chunks)} chunks to RAG")


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    handler = DocumentHandler()
    
    # Test with sample file
    # You would replace this with actual uploaded file path
    print("\n" + "="*80)
    print("DOCUMENT PROCESSING TEST")
    print("="*80)
    print("\nCreate a test file and process it:")
    print("  result = handler.process_upload('path/to/file.pdf')")