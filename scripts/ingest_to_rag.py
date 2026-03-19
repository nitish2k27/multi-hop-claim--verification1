"""
Ingest processed news data into RAG vector database
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging
from src.rag.vector_database import VectorDatabase, DataIngestionHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_news_to_rag():
    """Ingest processed news articles into RAG"""
    
    logger.info("\n" + "="*80)
    logger.info("INGESTING NEWS ARTICLES INTO RAG")
    logger.info("="*80)
    
    # ==========================================
    # STEP 1: Initialize Vector Database
    # ==========================================
    
    logger.info("\n→ Step 1: Initializing vector database")
    
    vdb = VectorDatabase(
        persist_directory="data/chroma_db",
        embedding_model="all-MiniLM-L6-v2"  # Fast, good quality
    )
    
    logger.info("✓ Vector database initialized")
    
    # ==========================================
    # STEP 2: Load processed data
    # ==========================================
    
    logger.info("\n→ Step 2: Loading processed data")
    
    processed_file = "data/processed/news_articles_rag.csv"
    
    if not Path(processed_file).exists():
        logger.error(f"✗ File not found: {processed_file}")
        logger.error("Run data processing first:")
        logger.error("  python src/data_collection/process_scraped_data.py")
        return
    
    df = pd.read_csv(processed_file)
    
    logger.info(f"✓ Loaded {len(df):,} articles")
    
    # ==========================================
    # STEP 3: Create/Clear collection
    # ==========================================
    
    logger.info("\n→ Step 3: Setting up collection")
    
    collection_name = "news_articles"
    
    # Check if collection exists
    existing_collections = vdb.list_collections()
    
    if collection_name in existing_collections:
        logger.warning(f"Collection '{collection_name}' already exists")
        
        response = input("Delete and recreate? (yes/no): ")
        
        if response.lower() == 'yes':
            vdb.delete_collection(collection_name)
            logger.info(f"✓ Deleted existing collection")
        else:
            logger.info("Appending to existing collection")
    
    # Create collection
    vdb.create_collection(
        collection_name,
        metadata={'type': 'news_articles', 'source': 'scraped'}
    )
    
    logger.info(f"✓ Collection '{collection_name}' ready")
    
    # ==========================================
    # STEP 4: Ingest data
    # ==========================================
    
    logger.info("\n→ Step 4: Ingesting articles")
    logger.info("This may take several minutes...")
    
    # Prepare metadata columns
    metadata_columns = ['title', 'source', 'publish_date', 'language', 'domain']
    
    # Keep only available columns
    available_metadata = [col for col in metadata_columns if col in df.columns]
    
    # Add to vector database
    vdb.add_from_dataframe(
        collection_name=collection_name,
        df=df,
        text_column='text',
        metadata_columns=available_metadata,
        id_column='url'
    )
    
    logger.info("✓ Ingestion complete!")
    
    # ==========================================
    # STEP 5: Verify
    # ==========================================
    
    logger.info("\n→ Step 5: Verifying ingestion")
    
    stats = vdb.get_collection_stats(collection_name)
    
    logger.info(f"\nCollection stats:")
    logger.info(f"  Name: {stats['name']}")
    logger.info(f"  Count: {stats['count']:,}")
    logger.info(f"  Exists: {stats['exists']}")
    
    # ==========================================
    # STEP 6: Test search
    # ==========================================
    
    logger.info("\n→ Step 6: Testing search")
    
    test_query = "GDP growth economy"
    
    results = vdb.search(
        collection_name=collection_name,
        query=test_query,
        top_k=3
    )
    
    logger.info(f"\nTest query: '{test_query}'")
    logger.info(f"Top 3 results:\n")
    
    for i, doc in enumerate(results['documents'][0], 1):
        metadata = results['metadatas'][0][i-1]
        distance = results['distances'][0][i-1]
        
        logger.info(f"{i}. {doc[:100]}...")
        logger.info(f"   Source: {metadata.get('source', 'N/A')}")
        logger.info(f"   Date: {metadata.get('publish_date', 'N/A')}")
        logger.info(f"   Distance: {distance:.4f}\n")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    
    logger.info("="*80)
    logger.info("✓ RAG INGESTION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nCollection: {collection_name}")
    logger.info(f"Articles: {stats['count']:,}")
    logger.info(f"Location: data/chroma_db/")
    logger.info("\nRAG system is ready to use!")
    logger.info("\nNext: Test the complete pipeline")
    logger.info("  python tests/test_rag_pipeline.py")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    ingest_news_to_rag()