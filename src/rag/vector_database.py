"""
Vector Database Manager
Handles ChromaDB operations with easy data ingestion
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector Database using ChromaDB
    
    Stores document embeddings for semantic search
    """
    
    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector database
        
        Args:
            persist_directory: Where to store the database
            embedding_model: SentenceTransformer model name
        """
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {self.persist_dir}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        logger.info(f"✓ Vector Database initialized (embedding dim: {self.embedding_dim})")
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict] = None
    ):
        """
        Create a new collection
        
        Args:
            collection_name: Name of collection
            metadata: Optional metadata for collection
        
        Returns:
            Collection object
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists (count: {collection.count()})")
            return collection
            
        except Exception:
            # Create new collection
            logger.info(f"Creating new collection: {collection_name}")
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            
            logger.info(f"✓ Collection '{collection_name}' created")
            return collection
    
    def get_collection(self, collection_name: str):
        """Get existing collection"""
        return self.client.get_collection(collection_name)
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"✓ Collection '{collection_name}' deleted")
        except Exception as e:
            logger.warning(f"Failed to delete collection: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        collections = self.client.list_collections()
        return [c.name for c in collections]
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ):
        """
        Add documents to collection
        
        Args:
            collection_name: Collection to add to
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
            batch_size: Batch size for processing
        """
        collection = self.get_collection(collection_name)
        
        # Generate IDs if not provided
        if ids is None:
            current_count = collection.count()
            ids = [f"doc_{current_count + i}" for i in range(len(documents))]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding"):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings)
        
        # Convert to list for ChromaDB
        embeddings = [emb.tolist() for emb in embeddings]
        
        # Add to collection in batches
        logger.info(f"Adding {len(documents)} documents to collection...")
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding to DB"):
            batch_end = min(i + batch_size, len(documents))
            
            collection.add(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end] if metadatas else None,
                ids=ids[i:batch_end]
            )
        
        logger.info(f"✓ Added {len(documents)} documents to '{collection_name}'")
        logger.info(f"  Total documents in collection: {collection.count()}")
    
    def add_from_dataframe(
        self,
        collection_name: str,
        df: pd.DataFrame,
        text_column: str,
        metadata_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None
    ):
        """
        Add documents from pandas DataFrame
        
        Args:
            collection_name: Collection to add to
            df: DataFrame with documents
            text_column: Column containing text
            metadata_columns: Columns to include as metadata
            id_column: Column to use as document ID
        """
        logger.info(f"Adding {len(df)} documents from DataFrame...")
        
        # Extract documents
        documents = df[text_column].astype(str).tolist()
        
        # Extract IDs
        if id_column:
            ids = df[id_column].astype(str).tolist()
        else:
            ids = None
        
        # Extract metadata
        if metadata_columns:
            metadatas = df[metadata_columns].to_dict('records')
            # Convert all values to strings
            metadatas = [
                {k: str(v) for k, v in meta.items()}
                for meta in metadatas
            ]
        else:
            metadatas = None
        
        # Add to collection
        self.add_documents(
            collection_name=collection_name,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def add_from_csv(
        self,
        collection_name: str,
        csv_path: str,
        text_column: str,
        metadata_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None
    ):
        """
        Add documents from CSV file
        
        Args:
            collection_name: Collection to add to
            csv_path: Path to CSV file
            text_column: Column containing text
            metadata_columns: Columns to include as metadata
            id_column: Column to use as document ID
        """
        logger.info(f"Loading documents from CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        self.add_from_dataframe(
            collection_name=collection_name,
            df=df,
            text_column=text_column,
            metadata_columns=metadata_columns,
            id_column=id_column
        )
    
    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents
        
        Args:
            collection_name: Collection to search
            query: Search query
            top_k: Number of results
            filters: Metadata filters (e.g., {'language': 'en'})
        
        Returns:
            Search results with documents, distances, metadatas
        """
        collection = self.get_collection(collection_name)
        
        # Embed query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        return results
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about collection"""
        try:
            collection = self.get_collection(collection_name)
            
            return {
                'name': collection_name,
                'count': collection.count(),
                'exists': True
            }
        except:
            return {
                'name': collection_name,
                'exists': False
            }


# ==========================================
# DATA INGESTION HELPER
# ==========================================

class DataIngestionHelper:
    """
    Helper class for easy data ingestion
    
    PLUGIN POINT: Use this to add your collected data later
    """
    
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
    
    def ingest_news_articles(
        self,
        csv_path: str,
        collection_name: str = "news_articles"
    ):
        """
        Ingest news articles from CSV
        
        PLUGIN POINT: Call this after you collect news data
        
        Expected CSV columns:
        - text: Article text
        - title: Article title
        - url: Source URL
        - source: Source name (e.g., 'reuters.com')
        - publish_date: Publication date
        - language: Language code
        """
        logger.info(f"Ingesting news articles from {csv_path}")
        
        # Create collection
        self.vector_db.create_collection(
            collection_name,
            metadata={'type': 'news_articles'}
        )
        
        # Add from CSV
        self.vector_db.add_from_csv(
            collection_name=collection_name,
            csv_path=csv_path,
            text_column='text',
            metadata_columns=['title', 'url', 'source', 'publish_date', 'language'],
            id_column='url'  # Use URL as unique ID
        )
        
        logger.info("✓ News articles ingested successfully")
    
    def ingest_user_documents(
        self,
        documents: List[Dict],
        collection_name: str = "user_documents"
    ):
        """
        Ingest user-provided documents
        
        PLUGIN POINT: Call this when user provides context docs
        
        Args:
            documents: List of dicts with 'text', 'source', 'metadata'
        """
        logger.info(f"Ingesting {len(documents)} user documents")
        
        # Create collection
        self.vector_db.create_collection(
            collection_name,
            metadata={'type': 'user_provided', 'priority': 'high'}
        )
        
        # Extract data
        texts = [doc['text'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        ids = [f"user_doc_{i}" for i in range(len(documents))]
        
        # Add to collection
        self.vector_db.add_documents(
            collection_name=collection_name,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info("✓ User documents ingested successfully")
    
    def ingest_wikipedia(
        self,
        texts: List[str],
        titles: List[str],
        collection_name: str = "wikipedia"
    ):
        """
        Ingest Wikipedia articles
        
        PLUGIN POINT: Call this with Wikipedia data
        """
        logger.info(f"Ingesting {len(texts)} Wikipedia articles")
        
        self.vector_db.create_collection(
            collection_name,
            metadata={'type': 'wikipedia'}
        )
        
        metadatas = [{'title': title, 'source': 'wikipedia'} for title in titles]
        
        self.vector_db.add_documents(
            collection_name=collection_name,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info("✓ Wikipedia articles ingested")


# ==========================================
# TESTING & DEMO
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    vdb = VectorDatabase()
    
    # Create sample collection
    print("\n" + "="*60)
    print("VECTOR DATABASE DEMO")
    print("="*60)
    
    collection_name = "demo_collection"
    
    # Create collection
    vdb.create_collection(collection_name)
    
    # Add sample documents
    sample_docs = [
        "India's GDP grew 8% in 2024 according to official data",
        "The economic growth was driven by services sector",
        "Manufacturing sector also showed improvement",
        "Inflation remained stable at 4.5%",
        "Government announced new economic policies"
    ]
    
    sample_metadata = [
        {'source': 'reuters', 'date': '2024-10-15'},
        {'source': 'bbc', 'date': '2024-10-16'},
        {'source': 'times', 'date': '2024-10-17'},
        {'source': 'bloomberg', 'date': '2024-10-18'},
        {'source': 'guardian', 'date': '2024-10-19'}
    ]
    
    vdb.add_documents(
        collection_name=collection_name,
        documents=sample_docs,
        metadatas=sample_metadata
    )
    
    # Search
    print("\n" + "="*60)
    print("SEARCH TEST")
    print("="*60)
    
    query = "GDP growth in India"
    results = vdb.search(collection_name, query, top_k=3)
    
    print(f"\nQuery: {query}\n")
    print("Top 3 results:")
    for i, doc in enumerate(results['documents'][0], 1):
        print(f"{i}. {doc}")
        print(f"   Distance: {results['distances'][0][i-1]:.4f}")
        print(f"   Metadata: {results['metadatas'][0][i-1]}")
        print()
    
    # Stats
    stats = vdb.get_collection_stats(collection_name)
    print(f"Collection stats: {stats}")