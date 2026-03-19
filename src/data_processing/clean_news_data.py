"""
News Data Cleaning and Quality Enhancement
Removes duplicates, fixes data quality issues, and standardizes the dataset
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
from pathlib import Path
from urllib.parse import urlparse
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataCleaner:
    def __init__(self):
        """Initialize the data cleaner"""
        
        # Quality thresholds
        self.min_title_length = 10
        self.max_title_length = 200
        self.min_text_length = 100
        self.max_text_length = 50000
        
        # Common spam/low-quality indicators
        self.spam_indicators = [
            'advertisement', 'sponsored', 'click here', 'buy now',
            'limited time', 'act now', 'free trial', 'subscribe now',
            'lorem ipsum', 'test article', 'placeholder'
        ]
        
        # Valid categories
        self.valid_categories = {
            'general', 'world', 'business', 'technology', 'science', 'health',
            'politics', 'entertainment', 'sports', 'india', 'geopolitics',
            'artificial_intelligence', 'startups', 'cybersecurity', 'fact-check',
            'bollywood', 'hollywood', 'music', 'celebrity', 'lifestyle'
        }
        
        # Source domain mapping for standardization
        self.source_standardization = {
            'bbc.co.uk': 'BBC',
            'guardian.com': 'The Guardian',
            'cnn.com': 'CNN',
            'reuters.com': 'Reuters',
            'timesofindia.indiatimes.com': 'Times of India',
            'thehindu.com': 'The Hindu',
            'indianexpress.com': 'Indian Express',
            'hindustantimes.com': 'Hindustan Times',
            'aljazeera.com': 'Al Jazeera',
            'techcrunch.com': 'TechCrunch',
            'wired.com': 'Wired'
        }
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load the news data"""
        logger.info(f"📂 Loading data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"   📊 Loaded {len(df):,} articles")
            logger.info(f"   📋 Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load data: {str(e)}")
            raise
    
    def analyze_data_quality(self, df: pd.DataFrame):
        """Analyze current data quality issues"""
        logger.info(f"\n🔍 DATA QUALITY ANALYSIS")
        logger.info(f"=" * 50)
        
        total_articles = len(df)
        
        # Missing data analysis
        logger.info(f"📊 Missing Data:")
        for col in df.columns:
            missing = df[col].isna().sum()
            empty_strings = (df[col] == '').sum() if df[col].dtype == 'object' else 0
            total_missing = missing + empty_strings
            percentage = (total_missing / total_articles) * 100
            logger.info(f"   {col}: {total_missing:,} ({percentage:.1f}%)")
        
        # Duplicate analysis
        url_duplicates = df.duplicated(subset=['url']).sum()
        title_duplicates = df.duplicated(subset=['title']).sum()
        content_duplicates = df.duplicated(subset=['text']).sum()
        
        logger.info(f"\n🔄 Duplicate Analysis:")
        logger.info(f"   URL duplicates: {url_duplicates:,}")
        logger.info(f"   Title duplicates: {title_duplicates:,}")
        logger.info(f"   Content duplicates: {content_duplicates:,}")
        
        # Content quality analysis
        if 'title' in df.columns:
            short_titles = (df['title'].str.len() < self.min_title_length).sum()
            long_titles = (df['title'].str.len() > self.max_title_length).sum()
            logger.info(f"\n📝 Title Quality:")
            logger.info(f"   Too short (<{self.min_title_length} chars): {short_titles:,}")
            logger.info(f"   Too long (>{self.max_title_length} chars): {long_titles:,}")
        
        if 'text' in df.columns:
            short_content = (df['text'].str.len() < self.min_text_length).sum()
            long_content = (df['text'].str.len() > self.max_text_length).sum()
            logger.info(f"\n📄 Content Quality:")
            logger.info(f"   Too short (<{self.min_text_length} chars): {short_content:,}")
            logger.info(f"   Too long (>{self.max_text_length} chars): {long_content:,}")
        
        # Date analysis
        if 'publish_date' in df.columns:
            valid_dates = pd.to_datetime(df['publish_date'], errors='coerce').notna().sum()
            invalid_dates = total_articles - valid_dates
            logger.info(f"\n📅 Date Quality:")
            logger.info(f"   Valid dates: {valid_dates:,}")
            logger.info(f"   Invalid/missing dates: {invalid_dates:,}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content if too long
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up again
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def standardize_source(self, source: str, url: str = '') -> str:
        """Standardize source names"""
        if pd.isna(source) or source == '':
            if url:
                domain = urlparse(url).netloc.lower()
                return self.source_standardization.get(domain, domain)
            return 'Unknown'
        
        source = str(source).strip()
        
        # Check if it's a domain
        if '.' in source and not ' ' in source:
            domain = source.lower()
            return self.source_standardization.get(domain, source)
        
        return source
    
    def standardize_category(self, category: str) -> str:
        """Standardize category names"""
        if pd.isna(category) or category == '':
            return 'general'
        
        category = str(category).lower().strip()
        
        # Direct match
        if category in self.valid_categories:
            return category
        
        # Fuzzy matching
        category_mapping = {
            'tech': 'technology',
            'ai': 'artificial_intelligence',
            'startup': 'startups',
            'cyber': 'cybersecurity',
            'entertainment': 'entertainment',
            'bollywood': 'bollywood',
            'hollywood': 'hollywood',
            'celebrity': 'celebrity',
            'music': 'music',
            'lifestyle': 'lifestyle',
            'national': 'india',
            'international': 'world',
            'global': 'world',
            'economy': 'business',
            'finance': 'business',
            'political': 'politics'
        }
        
        for key, value in category_mapping.items():
            if key in category:
                return value
        
        return 'general'
    
    def fix_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix and standardize publish dates"""
        logger.info(f"📅 Fixing publish dates...")
        
        if 'publish_date' not in df.columns:
            df['publish_date'] = ''
            return df
        
        # Convert to datetime
        df['publish_date_parsed'] = pd.to_datetime(df['publish_date'], errors='coerce')
        
        # For missing dates, try to extract from URL or use current date
        missing_dates = df['publish_date_parsed'].isna()
        
        # Extract year from URL if possible
        def extract_year_from_url(url):
            if pd.isna(url):
                return None
            match = re.search(r'/(\d{4})/', str(url))
            if match:
                year = int(match.group(1))
                if 2000 <= year <= datetime.now().year:
                    return f"{year}-01-01"
            return None
        
        # Try to extract dates from URLs
        for idx in df[missing_dates].index:
            url = df.loc[idx, 'url'] if 'url' in df.columns else ''
            extracted_date = extract_year_from_url(url)
            if extracted_date:
                df.loc[idx, 'publish_date_parsed'] = pd.to_datetime(extracted_date)
        
        # For remaining missing dates, use a default recent date
        still_missing = df['publish_date_parsed'].isna()
        df.loc[still_missing, 'publish_date_parsed'] = datetime(2024, 1, 1)
        
        # Convert back to string format
        df['publish_date'] = df['publish_date_parsed'].dt.strftime('%Y-%m-%d')
        df = df.drop('publish_date_parsed', axis=1)
        
        fixed_count = missing_dates.sum()
        logger.info(f"   ✅ Fixed {fixed_count:,} missing dates")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove various types of duplicates"""
        logger.info(f"🔄 Removing duplicates...")
        
        initial_count = len(df)
        
        # Remove exact URL duplicates
        df = df.drop_duplicates(subset=['url'], keep='first')
        url_removed = initial_count - len(df)
        logger.info(f"   🔗 Removed {url_removed:,} URL duplicates")
        
        # Remove title duplicates (case insensitive)
        if 'title' in df.columns:
            df['title_lower'] = df['title'].str.lower().str.strip()
            before_title = len(df)
            df = df.drop_duplicates(subset=['title_lower'], keep='first')
            df = df.drop('title_lower', axis=1)
            title_removed = before_title - len(df)
            logger.info(f"   📝 Removed {title_removed:,} title duplicates")
        
        # Remove near-duplicate content (using hash)
        if 'text' in df.columns:
            def content_hash(text):
                if pd.isna(text) or text == '':
                    return ''
                # Create hash of first 500 characters (normalized)
                normalized = re.sub(r'\s+', ' ', str(text)[:500].lower().strip())
                return hashlib.md5(normalized.encode()).hexdigest()
            
            df['content_hash'] = df['text'].apply(content_hash)
            before_content = len(df)
            df = df.drop_duplicates(subset=['content_hash'], keep='first')
            df = df.drop('content_hash', axis=1)
            content_removed = before_content - len(df)
            logger.info(f"   📄 Removed {content_removed:,} content duplicates")
        
        total_removed = initial_count - len(df)
        logger.info(f"   ✅ Total duplicates removed: {total_removed:,}")
        
        return df
    
    def filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out low-quality articles"""
        logger.info(f"🎯 Filtering low-quality articles...")
        
        initial_count = len(df)
        
        # Filter by title length
        if 'title' in df.columns:
            title_mask = (
                (df['title'].str.len() >= self.min_title_length) & 
                (df['title'].str.len() <= self.max_title_length)
            )
            df = df[title_mask]
            title_filtered = initial_count - len(df)
            logger.info(f"   📝 Filtered {title_filtered:,} articles by title length")
        
        # Filter by content length
        if 'text' in df.columns:
            before_content = len(df)
            content_mask = (
                (df['text'].str.len() >= self.min_text_length) & 
                (df['text'].str.len() <= self.max_text_length)
            )
            df = df[content_mask]
            content_filtered = before_content - len(df)
            logger.info(f"   📄 Filtered {content_filtered:,} articles by content length")
        
        # Filter spam content
        if 'text' in df.columns:
            before_spam = len(df)
            spam_pattern = '|'.join(self.spam_indicators)
            spam_mask = ~df['text'].str.contains(spam_pattern, case=False, na=False)
            df = df[spam_mask]
            spam_filtered = before_spam - len(df)
            logger.info(f"   🚫 Filtered {spam_filtered:,} spam articles")
        
        # Filter invalid URLs
        if 'url' in df.columns:
            before_url = len(df)
            valid_url_mask = df['url'].str.contains(r'^https?://', na=False)
            df = df[valid_url_mask]
            url_filtered = before_url - len(df)
            logger.info(f"   🔗 Filtered {url_filtered:,} invalid URLs")
        
        total_filtered = initial_count - len(df)
        logger.info(f"   ✅ Total low-quality articles filtered: {total_filtered:,}")
        
        return df
    
    def clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize all text fields"""
        logger.info(f"🧹 Cleaning and standardizing content...")
        
        # Clean titles
        if 'title' in df.columns:
            df['title'] = df['title'].apply(self.clean_text)
            logger.info(f"   📝 Cleaned titles")
        
        # Clean text content
        if 'text' in df.columns:
            df['text'] = df['text'].apply(self.clean_text)
            logger.info(f"   📄 Cleaned text content")
        
        # Standardize sources
        if 'source' in df.columns:
            if 'url' in df.columns:
                df['source'] = df.apply(lambda row: self.standardize_source(row['source'], row['url']), axis=1)
            else:
                df['source'] = df['source'].apply(lambda x: self.standardize_source(x))
            logger.info(f"   📰 Standardized sources")
        
        # Standardize categories
        if 'category' in df.columns:
            df['category'] = df['category'].apply(self.standardize_category)
            logger.info(f"   📂 Standardized categories")
        
        # Ensure language field
        if 'language' not in df.columns:
            df['language'] = 'en'
        else:
            df['language'] = df['language'].fillna('en')
        
        logger.info(f"   ✅ Content cleaning complete")
        
        return df
    
    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful metadata fields"""
        logger.info(f"📊 Adding metadata...")
        
        # Add word count
        if 'text' in df.columns:
            df['word_count'] = df['text'].str.split().str.len()
        
        # Add character count
        if 'text' in df.columns:
            df['char_count'] = df['text'].str.len()
        
        # Add domain from URL
        if 'url' in df.columns:
            df['domain'] = df['url'].apply(lambda x: urlparse(str(x)).netloc if pd.notna(x) else '')
        
        # Add cleaning timestamp
        df['cleaned_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"   ✅ Added metadata fields")
        
        return df
    
    def clean_dataset(self, input_path: str, output_path: str = None):
        """Complete data cleaning pipeline"""
        
        if output_path is None:
            output_path = input_path.replace('.csv', '_cleaned.csv')
        
        logger.info(f"🚀 STARTING DATA CLEANING PIPELINE")
        logger.info(f"=" * 60)
        logger.info(f"📂 Input: {input_path}")
        logger.info(f"📁 Output: {output_path}")
        
        # Load data
        df = self.load_data(input_path)
        
        # Analyze current quality
        self.analyze_data_quality(df)
        
        # Clean and process
        df = self.remove_duplicates(df)
        df = self.filter_quality(df)
        df = self.clean_and_standardize(df)
        df = self.fix_dates(df)
        df = self.add_metadata(df)
        
        # Final quality check
        logger.info(f"\n📊 FINAL DATASET STATISTICS")
        logger.info(f"=" * 40)
        logger.info(f"   Total articles: {len(df):,}")
        logger.info(f"   Unique sources: {df['source'].nunique()}")
        logger.info(f"   Categories: {sorted(df['category'].unique())}")
        logger.info(f"   Languages: {sorted(df['language'].unique())}")
        logger.info(f"   Date range: {df['publish_date'].min()} to {df['publish_date'].max()}")
        
        # Show top sources
        logger.info(f"\n📰 Top 10 Sources:")
        top_sources = df['source'].value_counts().head(10)
        for source, count in top_sources.items():
            logger.info(f"   {source}: {count:,} articles")
        
        # Show category distribution
        logger.info(f"\n📂 Category Distribution:")
        categories = df['category'].value_counts()
        for category, count in categories.items():
            percentage = (count / len(df)) * 100
            logger.info(f"   {category}: {count:,} ({percentage:.1f}%)")
        
        # Save cleaned data
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"\n✅ CLEANING COMPLETE!")
        logger.info(f"💾 Cleaned dataset saved to: {output_path}")
        
        return df

def main():
    """Run the data cleaning pipeline"""
    cleaner = NewsDataCleaner()
    
    # Clean the existing dataset
    cleaned_df = cleaner.clean_dataset(
        input_path="data/raw/news_articles.csv",
        output_path="data/processed/news_articles_cleaned.csv"
    )
    
    logger.info(f"🎉 Data cleaning pipeline completed successfully!")

if __name__ == "__main__":
    main()