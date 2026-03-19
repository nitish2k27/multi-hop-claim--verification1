"""
Process scraped news articles for RAG ingestion
Cleans, validates, and prepares data for vector database
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsDataProcessor:
    """Process scraped news data for RAG"""
    
    def __init__(
        self,
        input_csv: str,
        output_dir: str = "data/processed"
    ):
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Input: {self.input_csv}")
        logger.info(f"Output: {self.output_dir}")
    
    def load_data(self):
        """Load scraped data"""
        logger.info("\n" + "="*80)
        logger.info("LOADING SCRAPED DATA")
        logger.info("="*80)
        
        df = pd.read_csv(self.input_csv)
        
        logger.info(f"\n✓ Loaded {len(df):,} articles")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        logger.info("\n" + "="*80)
        logger.info("CLEANING DATA")
        logger.info("="*80)
        
        initial_count = len(df)
        
        # ==========================================
        # 1. Remove duplicates
        # ==========================================
        logger.info("\n→ Step 1: Removing duplicates")
        
        # Remove duplicate URLs
        df = df.drop_duplicates(subset=['url'], keep='first')
        logger.info(f"  Removed {initial_count - len(df):,} duplicate URLs")
        
        # Remove duplicate text
        df = df.drop_duplicates(subset=['text'], keep='first')
        logger.info(f"  Total after dedup: {len(df):,}")
        
        # ==========================================
        # 2. Remove null/empty values
        # ==========================================
        logger.info("\n→ Step 2: Removing null/empty values")
        
        before = len(df)
        
        # Required fields
        df = df[df['text'].notna() & (df['text'].str.strip() != '')]
        df = df[df['title'].notna() & (df['title'].str.strip() != '')]
        
        logger.info(f"  Removed {before - len(df):,} rows with null/empty text")
        
        # ==========================================
        # 3. Filter by text length
        # ==========================================
        logger.info("\n→ Step 3: Filtering by text length")
        
        # Calculate actual word count
        df['actual_word_count'] = df['text'].str.split().str.len()
        
        before = len(df)
        
        # Keep only articles with at least 100 words
        df = df[df['actual_word_count'] >= 100]
        
        logger.info(f"  Removed {before - len(df):,} short articles (< 100 words)")
        logger.info(f"  Remaining: {len(df):,}")
        
        # ==========================================
        # 4. Clean text content
        # ==========================================
        logger.info("\n→ Step 4: Cleaning text content")
        
        def clean_text(text):
            """Clean individual text"""
            if pd.isna(text):
                return ""
            
            # Convert to string
            text = str(text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
            
            # Remove URLs (if any left)
            text = re.sub(r'http\S+|www\.\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Strip
            text = text.strip()
            
            return text
        
        df['text'] = df['text'].apply(clean_text)
        df['title'] = df['title'].apply(clean_text)
        
        logger.info("  ✓ Text cleaned")
        
        # ==========================================
        # 5. Standardize dates
        # ==========================================
        logger.info("\n→ Step 5: Standardizing dates")
        
        def parse_date(date_str):
            """Parse various date formats"""
            if pd.isna(date_str):
                return None
            
            try:
                # Try parsing
                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
            except:
                return None
        
        df['publish_date'] = df['publish_date'].apply(parse_date)
        
        # Fill missing dates with today's date
        today = datetime.now().strftime('%Y-%m-%d')
        df['publish_date'] = df['publish_date'].fillna(today)
        
        logger.info("  ✓ Dates standardized")
        
        # ==========================================
        # 6. Standardize language codes
        # ==========================================
        logger.info("\n→ Step 6: Standardizing language codes")
        
        # Ensure lowercase
        df['language'] = df['language'].str.lower()
        
        # Fill missing languages with 'en'
        df['language'] = df['language'].fillna('en')
        
        logger.info(f"  Languages: {df['language'].value_counts().to_dict()}")
        
        # ==========================================
        # Summary
        # ==========================================
        logger.info("\n" + "="*80)
        logger.info("CLEANING SUMMARY")
        logger.info("="*80)
        logger.info(f"Initial articles:  {initial_count:,}")
        logger.info(f"Final articles:    {len(df):,}")
        logger.info(f"Removed:           {initial_count - len(df):,} ({(initial_count - len(df))/initial_count*100:.1f}%)")
        logger.info("="*80)
        
        return df
    
    def prepare_for_rag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for RAG ingestion"""
        logger.info("\n" + "="*80)
        logger.info("PREPARING FOR RAG")
        logger.info("="*80)
        
        # ==========================================
        # Create combined text field
        # ==========================================
        logger.info("\n→ Creating combined text field")
        
        # Combine title + text for better search
        df['combined_text'] = df['title'] + ". " + df['text']
        
        # ==========================================
        # Select columns for RAG
        # ==========================================
        logger.info("\n→ Selecting columns for RAG")
        
        rag_columns = [
            'url',           # Unique ID
            'combined_text', # Searchable text
            'title',         # For display
            'source',        # For credibility
            'publish_date',  # For credibility/recency
            'language',      # For filtering
            'domain',        # For credibility
            'category',      # For filtering
        ]
        
        # Keep only available columns
        available_cols = [col for col in rag_columns if col in df.columns]
        df_rag = df[available_cols].copy()
        
        # Rename for RAG
        df_rag = df_rag.rename(columns={'combined_text': 'text'})
        
        logger.info(f"  Selected columns: {list(df_rag.columns)}")
        logger.info(f"  Total articles: {len(df_rag):,}")
        
        # ==========================================
        # Statistics
        # ==========================================
        logger.info("\n" + "="*80)
        logger.info("DATASET STATISTICS")
        logger.info("="*80)
        
        logger.info(f"\nTotal articles: {len(df_rag):,}")
        
        logger.info(f"\nBy source:")
        for source, count in df_rag['source'].value_counts().head(10).items():
            logger.info(f"  {source}: {count:,}")
        
        logger.info(f"\nBy language:")
        for lang, count in df_rag['language'].value_counts().items():
            logger.info(f"  {lang}: {count:,}")
        
        if 'category' in df_rag.columns:
            logger.info(f"\nBy category:")
            for cat, count in df_rag['category'].value_counts().head(10).items():
                logger.info(f"  {cat}: {count:,}")
        
        # Text length stats
        text_lengths = df_rag['text'].str.split().str.len()
        logger.info(f"\nText length (words):")
        logger.info(f"  Min:    {text_lengths.min():,}")
        logger.info(f"  Mean:   {text_lengths.mean():.0f}")
        logger.info(f"  Median: {text_lengths.median():.0f}")
        logger.info(f"  Max:    {text_lengths.max():,}")
        
        logger.info("="*80)
        
        return df_rag
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data"""
        logger.info("\n→ Saving processed data")
        
        output_file = self.output_dir / "news_articles_rag.csv"
        
        df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Saved to: {output_file}")
        logger.info(f"  Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_file
    
    def run(self):
        """Run complete processing pipeline"""
        logger.info("\n" + "="*80)
        logger.info("NEWS DATA PROCESSING FOR RAG")
        logger.info("="*80)
        
        # Load
        df = self.load_data()
        
        # Clean
        df_clean = self.clean_data(df)
        
        # Prepare for RAG
        df_rag = self.prepare_for_rag(df_clean)
        
        # Save
        output_file = self.save_processed_data(df_rag)
        
        logger.info("\n" + "="*80)
        logger.info("✓ PROCESSING COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nProcessed file: {output_file}")
        logger.info(f"Articles ready for RAG: {len(df_rag):,}")
        logger.info("\nNext step: Ingest into vector database")
        logger.info("  python scripts/ingest_to_rag.py")
        logger.info("="*80 + "\n")
        
        return output_file


if __name__ == "__main__":
    # UPDATE THIS PATH to your scraped data
    processor = NewsDataProcessor(
        input_csv="data/raw/scraped_news.csv"  # ← Your CSV file
    )
    
    processor.run()