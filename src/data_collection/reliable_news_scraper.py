"""
Reliable News Scraper - High Quality Data Collection
Focuses on working RSS feeds with robust error handling and quality validation
"""

import logging
import time
import requests
import feedparser
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from newspaper import Article
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReliableNewsCollector:
    def __init__(self):
        """Initialize with only verified working RSS feeds"""
        
        # VERIFIED WORKING FEEDS (tested and confirmed)
        self.reliable_feeds = {
            # Indian English Sources (High Success Rate)
            'times_of_india': [
                'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
                'https://timesofindia.indiatimes.com/rssfeeds/1898055.cms',  # India news
                'https://timesofindia.indiatimes.com/rssfeeds/1898024.cms',  # Business
            ],
            'the_hindu': [
                'https://www.thehindu.com/news/national/feeder/default.rss',
                'https://www.thehindu.com/business/feeder/default.rss',
                'https://www.thehindu.com/sci-tech/feeder/default.rss',
            ],
            'indian_express': [
                'https://indianexpress.com/section/india/feed/',
                'https://indianexpress.com/section/business/feed/',
                'https://indianexpress.com/section/technology/feed/',
            ],
            'hindustan_times': [
                'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
                'https://www.hindustantimes.com/feeds/rss/business-news/rssfeed.xml',
                'https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml',
            ],
            
            # International Sources (Verified Working)
            'guardian': [
                'https://www.theguardian.com/world/rss',
                'https://www.theguardian.com/business/rss',
                'https://www.theguardian.com/technology/rss',
                'https://www.theguardian.com/politics/rss',
            ],
            'al_jazeera': [
                'https://www.aljazeera.com/xml/rss/all.xml',
                'https://www.aljazeera.com/xml/rss/business.xml',
            ],
            'france24': [
                'https://www.france24.com/en/rss',
                'https://www.france24.com/en/business/rss',
            ],
            'dw_english': [
                'https://rss.dw.com/rdf/rss-en-all',
                'https://rss.dw.com/rdf/rss-en-bus',
            ],
            
            # Tech Sources (Usually Reliable)
            'techcrunch': [
                'https://techcrunch.com/feed/',
                'https://techcrunch.com/category/startups/feed/',
            ],
            'wired': [
                'https://www.wired.com/feed/rss',
            ],
            
            # News Aggregators (High Volume)
            'google_news_india': [
                'https://news.google.com/rss/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNRE55YXpBU0JXVnVMVWRDS0FBUAE?hl=en-IN&gl=IN&ceid=IN:en',
            ],
            'google_news_world': [
                'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pKVGlnQVAB?hl=en-IN&gl=IN&ceid=IN:en',
            ],
        }
        
        # Language mapping
        self.source_languages = {
            'times_of_india': 'en', 'the_hindu': 'en', 'indian_express': 'en', 'hindustan_times': 'en',
            'guardian': 'en', 'al_jazeera': 'en', 'france24': 'en', 'dw_english': 'en',
            'techcrunch': 'en', 'wired': 'en', 'google_news_india': 'en', 'google_news_world': 'en'
        }
        
        # Enhanced category mapping
        self.category_mapping = {
            'topstories': 'general', 'top': 'general', 'all': 'general',
            'world': 'world', 'international': 'world', 'global': 'world',
            'business': 'business', 'economy': 'business', 'finance': 'business',
            'technology': 'technology', 'tech': 'technology', 'sci-tech': 'technology',
            'science': 'science', 'health': 'science',
            'india': 'india', 'national': 'india',
            'politics': 'politics', 'political': 'politics',
            'startups': 'startups', 'startup': 'startups'
        }
        
        # Quality thresholds
        self.min_article_length = 200
        self.min_title_length = 10
        self.max_retries = 3
        self.request_timeout = 10
        
        # Rate limiting
        self.delay = 2
        
        # Request headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    def validate_feed(self, url: str) -> bool:
        """Validate if RSS feed is accessible and has content"""
        try:
            response = requests.get(url, timeout=self.request_timeout, headers=self.headers)
            if response.status_code != 200:
                return False
            
            feed = feedparser.parse(url)
            return len(feed.entries) > 0 and not feed.bozo
            
        except Exception:
            return False
    
    def extract_category_from_url(self, url: str, source: str) -> str:
        """Extract category from RSS feed URL"""
        url_lower = url.lower()
        
        for keyword, category in self.category_mapping.items():
            if keyword in url_lower:
                return category
        
        # Source-specific defaults
        if 'tech' in source.lower():
            return 'technology'
        elif 'business' in source.lower():
            return 'business'
        
        return 'general'
    
    def clean_text(self, text: str) -> str:
        """Clean and validate article text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common RSS artifacts
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        text = re.sub(r'\[.*?\]', '', text)  # Bracketed content
        
        return text
    
    def scrape_article(self, url: str, source: str, category: str) -> Dict:
        """Scrape individual article with enhanced error handling"""
        for attempt in range(self.max_retries):
            try:
                article = Article(url)
                article.config.request_timeout = self.request_timeout
                article.config.browser_user_agent = self.headers['User-Agent']
                
                article.download()
                article.parse()
                
                # Quality validation
                title = self.clean_text(article.title)
                text = self.clean_text(article.text)
                
                if len(title) < self.min_title_length:
                    logger.debug(f"Title too short: {url}")
                    return None
                
                if len(text) < self.min_article_length:
                    logger.debug(f"Article too short: {url}")
                    return None
                
                # Extract publish date with fallback
                publish_date = ""
                if article.publish_date:
                    publish_date = article.publish_date.strftime('%Y-%m-%d')
                
                return {
                    'url': url,
                    'title': title,
                    'text': text,
                    'source': source,
                    'publish_date': publish_date,
                    'language': self.source_languages.get(source, 'en'),
                    'category': category,
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                continue
        
        logger.warning(f"Failed to scrape after {self.max_retries} attempts: {url}")
        return None
    
    def collect_from_rss(self, source: str, max_articles: int = 50) -> List[Dict]:
        """Collect articles from RSS feeds for a source"""
        articles = []
        feeds = self.reliable_feeds.get(source, [])
        
        if not feeds:
            logger.warning(f"No feeds configured for source: {source}")
            return articles
        
        logger.info(f"📰 Collecting from {source} ({len(feeds)} feeds)")
        
        for feed_url in feeds:
            try:
                # Validate feed first
                if not self.validate_feed(feed_url):
                    logger.warning(f"   ❌ Feed validation failed: {feed_url}")
                    continue
                
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                category = self.extract_category_from_url(feed_url, source)
                
                logger.info(f"   📡 {feed_url} -> {category} ({len(feed.entries)} entries)")
                
                # Process entries
                articles_from_feed = 0
                target_per_feed = max_articles // len(feeds)
                
                for entry in feed.entries[:target_per_feed * 2]:  # Get extra to account for failures
                    if articles_from_feed >= target_per_feed:
                        break
                    
                    if len(articles) >= max_articles:
                        break
                    
                    if not hasattr(entry, 'link') or not entry.link:
                        continue
                    
                    article_data = self.scrape_article(entry.link, source, category)
                    if article_data:
                        articles.append(article_data)
                        articles_from_feed += 1
                        logger.info(f"   ✅ {len(articles)}: {article_data['title'][:60]}...")
                    
                    # Rate limiting
                    time.sleep(self.delay)
                
                logger.info(f"   📊 Collected {articles_from_feed} articles from this feed")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to process feed {feed_url}: {str(e)}")
                continue
        
        return articles
    
    def collect_reliable_batch(self, articles_per_source: int = 30, batch_size: int = 3):
        """Collect articles from reliable sources only"""
        
        logger.info(f"🚀 Starting Reliable News Collection:")
        logger.info(f"   📊 {articles_per_source} articles per source")
        logger.info(f"   🔄 Processing {batch_size} sources at a time")
        logger.info(f"   ✅ Using only verified working RSS feeds")
        logger.info(f"   🎯 Focus: High-quality, recent articles")
        
        sources = list(self.reliable_feeds.keys())
        total_collected = 0
        
        # Process sources in batches
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            batch_articles = []
            
            logger.info(f"\n📦 Processing reliable batch {i//batch_size + 1}: {batch_sources}")
            
            for source in batch_sources:
                try:
                    articles = self.collect_from_rss(source, articles_per_source)
                    batch_articles.extend(articles)
                    
                    logger.info(f"✅ {source}: {len(articles)} high-quality articles collected")
                    
                    # Brief delay between sources
                    time.sleep(3)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to collect from {source}: {str(e)}")
                    continue
            
            # Save batch to CSV
            if batch_articles:
                df_batch = pd.DataFrame(batch_articles)
                
                # Remove duplicates within batch
                df_batch = df_batch.drop_duplicates(subset=['url'])
                
                # Append to existing file
                self.append_to_csv(df_batch, "data/raw/news_articles.csv")
                
                total_collected += len(df_batch)
                logger.info(f"💾 Reliable batch saved: {len(df_batch)} articles")
                logger.info(f"📊 Total reliable collected so far: {total_collected}")
            
            # Delay between batches
            if i + batch_size < len(sources):
                logger.info(f"⏳ Waiting 10 seconds before next batch...")
                time.sleep(10)
        
        logger.info(f"\n🎉 Reliable collection complete!")
        logger.info(f"📊 Total high-quality articles collected: {total_collected}")
        
        # Show quality statistics
        self.show_quality_stats("data/raw/news_articles.csv")
        
        return total_collected
    
    def append_to_csv(self, df: pd.DataFrame, csv_path: str):
        """Append DataFrame to existing CSV file"""
        csv_path = Path(csv_path)
        
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                
                # Remove duplicates between new and existing data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
                
                # Calculate new articles added
                new_articles = len(combined_df) - len(existing_df)
                
                if new_articles > 0:
                    combined_df.to_csv(csv_path, index=False)
                    logger.info(f"   📝 Added {new_articles} new articles (duplicates removed)")
                else:
                    logger.info(f"   ⚠️ No new articles (all were duplicates)")
                    
            except Exception as e:
                logger.error(f"Error appending to CSV: {str(e)}")
                df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"   📁 Created new file with {len(df)} articles")
    
    def show_quality_stats(self, csv_path: str):
        """Show quality statistics about the dataset"""
        try:
            df = pd.read_csv(csv_path)
            
            logger.info(f"\n📊 QUALITY STATISTICS:")
            logger.info(f"   Total articles: {len(df):,}")
            
            # Articles with dates
            articles_with_dates = df[df['publish_date'].notna() & (df['publish_date'] != '')].shape[0]
            date_percentage = (articles_with_dates / len(df)) * 100
            logger.info(f"   Articles with dates: {articles_with_dates:,} ({date_percentage:.1f}%)")
            
            # Recent articles (last 30 days)
            if articles_with_dates > 0:
                df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
                recent_cutoff = datetime.now() - pd.Timedelta(days=30)
                recent_articles = df[df['publish_date'] > recent_cutoff].shape[0]
                logger.info(f"   Recent articles (30 days): {recent_articles:,}")
            
            # Content quality
            if 'text' in df.columns:
                avg_length = df['text'].str.len().mean()
                logger.info(f"   Average article length: {avg_length:.0f} characters")
            
            # Source diversity
            logger.info(f"   Unique sources: {df['source'].nunique()}")
            
            # Top performing sources
            top_sources = df['source'].value_counts().head(5)
            logger.info(f"   Top sources: {dict(top_sources)}")
            
        except Exception as e:
            logger.error(f"Error showing quality stats: {str(e)}")

def main():
    """Test the reliable collector"""
    collector = ReliableNewsCollector()
    
    logger.info("🧪 Testing Reliable News Scraper")
    articles_collected = collector.collect_reliable_batch(
        articles_per_source=20,  # Moderate test
        batch_size=2
    )
    
    logger.info(f"✅ Test complete: {articles_collected} reliable articles collected")

if __name__ == "__main__":
    main()