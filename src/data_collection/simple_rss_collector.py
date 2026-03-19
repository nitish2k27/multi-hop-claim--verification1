"""
Simple RSS Collector - Direct RSS Content Extraction
Bypasses newspaper3k issues by using RSS content directly
"""

import logging
import time
import requests
import feedparser
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import re
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRSSCollector:
    def __init__(self):
        """Initialize with working RSS feeds and simple extraction"""
        
        # VERIFIED WORKING FEEDS
        self.working_feeds = {
            'guardian': [
                'https://www.theguardian.com/world/rss',
                'https://www.theguardian.com/business/rss',
                'https://www.theguardian.com/technology/rss',
            ],
            'times_of_india': [
                'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
            ],
            'the_hindu': [
                'https://www.thehindu.com/news/national/feeder/default.rss',
            ],
            'al_jazeera': [
                'https://www.aljazeera.com/xml/rss/all.xml',
            ],
            'techcrunch': [
                'https://techcrunch.com/feed/',
            ],
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def clean_html(self, text):
        """Remove HTML tags and clean text"""
        if not text:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_from_rss_entry(self, entry, source):
        """Extract article data directly from RSS entry"""
        try:
            # Get title
            title = getattr(entry, 'title', '')
            title = self.clean_html(title)
            
            if len(title) < 10:
                return None
            
            # Get content from RSS (summary or content)
            content = ""
            if hasattr(entry, 'content') and entry.content:
                content = entry.content[0].value if isinstance(entry.content, list) else entry.content
            elif hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description
            
            content = self.clean_html(content)
            
            if len(content) < 50:
                return None
            
            # Get publish date
            publish_date = ""
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    dt = datetime(*entry.published_parsed[:6])
                    publish_date = dt.strftime('%Y-%m-%d')
                except:
                    pass
            
            # Get URL
            url = getattr(entry, 'link', '')
            if not url:
                return None
            
            return {
                'url': url,
                'title': title,
                'text': content,
                'source': source,
                'publish_date': publish_date,
                'language': 'en',
                'category': 'general',
                'extraction_method': 'rss_direct'
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract from RSS entry: {str(e)}")
            return None
    
    def collect_from_feed(self, feed_url, source, max_articles=20):
        """Collect articles from a single RSS feed"""
        articles = []
        
        try:
            logger.info(f"   📡 Fetching {feed_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"   ⚠️ Feed has parsing issues: {feed_url}")
            
            logger.info(f"   📊 Found {len(feed.entries)} entries")
            
            # Process entries
            for i, entry in enumerate(feed.entries[:max_articles]):
                article_data = self.extract_from_rss_entry(entry, source)
                
                if article_data:
                    articles.append(article_data)
                    logger.info(f"   ✅ {len(articles)}: {article_data['title'][:60]}...")
                
                # Small delay
                time.sleep(0.5)
            
            logger.info(f"   📈 Extracted {len(articles)} articles from this feed")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to process feed {feed_url}: {str(e)}")
        
        return articles
    
    def collect_simple_batch(self, articles_per_source=15):
        """Collect articles using simple RSS extraction"""
        
        logger.info(f"🚀 Starting Simple RSS Collection:")
        logger.info(f"   📊 {articles_per_source} articles per source")
        logger.info(f"   📡 Direct RSS content extraction (no web scraping)")
        logger.info(f"   ✅ Fast and reliable method")
        
        all_articles = []
        
        for source, feeds in self.working_feeds.items():
            logger.info(f"\n📰 Collecting from {source}")
            
            source_articles = []
            articles_per_feed = articles_per_source // len(feeds)
            
            for feed_url in feeds:
                feed_articles = self.collect_from_feed(feed_url, source, articles_per_feed)
                source_articles.extend(feed_articles)
            
            logger.info(f"✅ {source}: {len(source_articles)} articles collected")
            all_articles.extend(source_articles)
            
            # Brief pause between sources
            time.sleep(2)
        
        # Save to CSV
        if all_articles:
            df = pd.DataFrame(all_articles)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['url'])
            
            # Save/append to file
            self.save_to_csv(df, "data/raw/news_articles.csv")
            
            logger.info(f"\n🎉 Collection complete!")
            logger.info(f"📊 Total articles collected: {len(df)}")
            
            # Show sample
            self.show_sample_articles(df)
            
            return len(df)
        
        return 0
    
    def save_to_csv(self, df, csv_path):
        """Save or append DataFrame to CSV"""
        csv_path = Path(csv_path)
        
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
                
                new_articles = len(combined_df) - len(existing_df)
                combined_df.to_csv(csv_path, index=False)
                
                logger.info(f"💾 Added {new_articles} new articles to existing dataset")
                logger.info(f"📊 Total dataset size: {len(combined_df)} articles")
                
            except Exception as e:
                logger.error(f"Error appending: {str(e)}")
                df.to_csv(csv_path, index=False)
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"📁 Created new dataset with {len(df)} articles")
    
    def show_sample_articles(self, df):
        """Show sample of collected articles"""
        logger.info(f"\n📋 SAMPLE ARTICLES:")
        
        for i, row in df.head(3).iterrows():
            logger.info(f"   {i+1}. {row['title']}")
            logger.info(f"      Source: {row['source']} | Date: {row['publish_date']}")
            logger.info(f"      Content: {row['text'][:100]}...")
            logger.info("")

def main():
    """Test the simple collector"""
    collector = SimpleRSSCollector()
    
    logger.info("🧪 Testing Simple RSS Collector")
    articles = collector.collect_simple_batch(articles_per_source=10)
    
    logger.info(f"✅ Test complete: {articles} articles collected using RSS direct extraction")

if __name__ == "__main__":
    main()