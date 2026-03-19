"""
Entertainment & Celebrity News Scraper
Collects 2000+ articles about Hollywood, Bollywood, celebrities, music industry
"""

import requests
from newspaper import Article
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import feedparser
from typing import List, Dict
import random
from urllib.parse import urljoin, urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntertainmentNewsCollector:
    def __init__(self):
        """Initialize the entertainment news collector"""
        
        # Entertainment-focused RSS feeds
        self.entertainment_feeds = {
            # Hollywood & International Entertainment
            'variety': [
                'https://variety.com/feed/',
                'https://variety.com/c/film/feed/',
                'https://variety.com/c/tv/feed/',
                'https://variety.com/c/music/feed/',
                'https://variety.com/c/digital/feed/'
            ],
            'hollywood_reporter': [
                'https://www.hollywoodreporter.com/feed/',
                'https://www.hollywoodreporter.com/c/movies/feed/',
                'https://www.hollywoodreporter.com/c/tv/feed/',
                'https://www.hollywoodreporter.com/c/music/feed/'
            ],
            'entertainment_weekly': [
                'https://ew.com/feed/',
                'https://ew.com/movies/feed/',
                'https://ew.com/tv/feed/',
                'https://ew.com/music/feed/',
                'https://ew.com/celebrity/feed/'
            ],
            'deadline': [
                'https://deadline.com/feed/',
                'https://deadline.com/category/film/feed/',
                'https://deadline.com/category/tv/feed/',
                'https://deadline.com/category/music/feed/'
            ],
            'people_magazine': [
                'https://people.com/feed/',
                'https://people.com/movies/feed/',
                'https://people.com/tv/feed/',
                'https://people.com/music/feed/',
                'https://people.com/celebrity/feed/'
            ],
            'us_weekly': [
                'https://www.usmagazine.com/feed/',
                'https://www.usmagazine.com/celebrity-news/feed/',
                'https://www.usmagazine.com/entertainment/feed/'
            ],
            'e_news': [
                'https://www.eonline.com/feeds/rss/topstories.xml',
                'https://www.eonline.com/feeds/rss/movies.xml',
                'https://www.eonline.com/feeds/rss/tv.xml',
                'https://www.eonline.com/feeds/rss/music.xml'
            ],
            
            # Music Industry
            'rolling_stone': [
                'https://www.rollingstone.com/feed/',
                'https://www.rollingstone.com/music/feed/',
                'https://www.rollingstone.com/culture/feed/',
                'https://www.rollingstone.com/tv-movies/feed/'
            ],
            'billboard': [
                'https://www.billboard.com/feed/',
                'https://www.billboard.com/c/music/feed/',
                'https://www.billboard.com/c/charts/feed/',
                'https://www.billboard.com/c/pop/feed/'
            ],
            'pitchfork': [
                'https://pitchfork.com/rss/news/',
                'https://pitchfork.com/rss/reviews/albums/',
                'https://pitchfork.com/rss/features/'
            ],
            
            # Bollywood & Indian Entertainment
            'bollywood_hungama': [
                'https://www.bollywoodhungama.com/rss/news.xml',
                'https://www.bollywoodhungama.com/rss/movie-reviews.xml',
                'https://www.bollywoodhungama.com/rss/celebrity.xml'
            ],
            'filmfare': [
                'https://www.filmfare.com/feeds/latest-news.xml',
                'https://www.filmfare.com/feeds/bollywood.xml',
                'https://www.filmfare.com/feeds/celebrity.xml'
            ],
            'pinkvilla': [
                'https://www.pinkvilla.com/rss.xml',
                'https://www.pinkvilla.com/entertainment/rss.xml',
                'https://www.pinkvilla.com/entertainment/bollywood/rss.xml'
            ],
            'india_today_entertainment': [
                'https://www.indiatoday.in/rss/1206514',  # Entertainment
                'https://www.indiatoday.in/rss/1206578',  # Movies
                'https://www.indiatoday.in/rss/1206577'   # Television
            ],
            'times_entertainment': [
                'https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms',  # Entertainment
                'https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms', # Bollywood
                'https://timesofindia.indiatimes.com/rssfeeds/1899270.cms'      # TV
            ],
            'hindustan_times_entertainment': [
                'https://www.hindustantimes.com/feeds/rss/entertainment/rssfeed.xml',
                'https://www.hindustantimes.com/feeds/rss/bollywood/rssfeed.xml',
                'https://www.hindustantimes.com/feeds/rss/tv/rssfeed.xml'
            ],
            'ndtv_entertainment': [
                'https://feeds.feedburner.com/ndtv/entertainment',
                'https://feeds.feedburner.com/ndtv/movies',
                'https://feeds.feedburner.com/ndtv/celebrity'
            ],
            
            # Regional Entertainment
            'tamil_cinema': [
                'https://www.behindwoods.com/rss/tamil-cinema-news.xml',
                'https://www.sify.com/rss/movies-tamil-news.xml'
            ],
            'telugu_cinema': [
                'https://www.123telugu.com/rss.xml',
                'https://www.sify.com/rss/movies-telugu-news.xml'
            ],
            
            # International Entertainment
            'bbc_entertainment': [
                'http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml'
            ],
            'cnn_entertainment': [
                'http://rss.cnn.com/rss/edition_entertainment.rss'
            ],
            'reuters_entertainment': [
                'https://feeds.reuters.com/reuters/entertainment'
            ],
            
            # Celebrity & Gossip
            'tmz': [
                'https://www.tmz.com/rss.xml'
            ],
            'page_six': [
                'https://pagesix.com/feed/'
            ],
            'just_jared': [
                'https://www.justjared.com/feed/'
            ],
            
            # Fashion & Lifestyle (Celebrity focused)
            'vogue_celebrity': [
                'https://www.vogue.com/feed/celebrity',
                'https://www.vogue.com/feed/culture'
            ],
            'elle_celebrity': [
                'https://www.elle.com/rss/celebrity.xml',
                'https://www.elle.com/rss/culture.xml'
            ]
        }
        
        # Language mapping
        self.source_languages = {
            # English sources
            'variety': 'en', 'hollywood_reporter': 'en', 'entertainment_weekly': 'en',
            'deadline': 'en', 'people_magazine': 'en', 'us_weekly': 'en', 'e_news': 'en',
            'rolling_stone': 'en', 'billboard': 'en', 'pitchfork': 'en',
            'bollywood_hungama': 'en', 'filmfare': 'en', 'pinkvilla': 'en',
            'india_today_entertainment': 'en', 'times_entertainment': 'en',
            'hindustan_times_entertainment': 'en', 'ndtv_entertainment': 'en',
            'tamil_cinema': 'en', 'telugu_cinema': 'en',
            'bbc_entertainment': 'en', 'cnn_entertainment': 'en', 'reuters_entertainment': 'en',
            'tmz': 'en', 'page_six': 'en', 'just_jared': 'en',
            'vogue_celebrity': 'en', 'elle_celebrity': 'en'
        }
        
        # Category mapping for entertainment
        self.category_mapping = {
            'movies': 'entertainment', 'film': 'entertainment', 'cinema': 'entertainment',
            'tv': 'entertainment', 'television': 'entertainment',
            'music': 'music', 'celebrity': 'celebrity', 'bollywood': 'bollywood',
            'hollywood': 'hollywood', 'entertainment': 'entertainment',
            'culture': 'entertainment', 'fashion': 'lifestyle',
            'gossip': 'celebrity', 'news': 'entertainment'
        }
        
        # Rate limiting
        self.delay = 1.5  # Slightly faster for entertainment sites
        
    def extract_category_from_url(self, url: str, source: str) -> str:
        """Extract category from RSS feed URL"""
        url_lower = url.lower()
        
        # Check for specific entertainment categories
        if 'bollywood' in url_lower:
            return 'bollywood'
        elif 'hollywood' in url_lower:
            return 'hollywood'
        elif 'music' in url_lower:
            return 'music'
        elif 'celebrity' in url_lower:
            return 'celebrity'
        elif any(word in url_lower for word in ['movie', 'film', 'cinema']):
            return 'entertainment'
        elif 'tv' in url_lower or 'television' in url_lower:
            return 'entertainment'
        elif 'fashion' in url_lower or 'lifestyle' in url_lower:
            return 'lifestyle'
        
        return 'entertainment'  # Default for entertainment scraper
    
    def scrape_article(self, url: str, source: str, category: str) -> Dict:
        """Scrape individual entertainment article"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Skip if article is too short
            if len(article.text) < 150:
                return None
                
            # Skip if it's not entertainment related (basic filter)
            text_lower = article.text.lower()
            title_lower = article.title.lower()
            
            entertainment_keywords = [
                'movie', 'film', 'actor', 'actress', 'celebrity', 'music', 'singer',
                'bollywood', 'hollywood', 'entertainment', 'tv show', 'series',
                'album', 'concert', 'premiere', 'award', 'oscar', 'grammy',
                'director', 'producer', 'star', 'celebrity', 'fashion', 'red carpet'
            ]
            
            if not any(keyword in text_lower or keyword in title_lower for keyword in entertainment_keywords):
                return None
                
            return {
                'url': url,
                'title': article.title.strip(),
                'text': article.text.strip(),
                'source': source,
                'publish_date': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else '',
                'language': self.source_languages.get(source, 'en'),
                'category': category
            }
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {str(e)}")
            return None
    
    def collect_from_rss(self, source: str, max_articles: int = 100) -> List[Dict]:
        """Collect entertainment articles from RSS feeds for a source"""
        articles = []
        feeds = self.entertainment_feeds.get(source, [])
        
        logger.info(f"🎬 Collecting from {source} ({len(feeds)} feeds)")
        
        for feed_url in feeds:
            try:
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                category = self.extract_category_from_url(feed_url, source)
                
                logger.info(f"   📡 {feed_url} -> {category}")
                
                # Process entries
                articles_per_feed = max_articles // len(feeds) if len(feeds) > 0 else max_articles
                
                for entry in feed.entries[:articles_per_feed + 5]:  # Get a few extra in case some fail
                    if len(articles) >= max_articles:
                        break
                        
                    article_data = self.scrape_article(entry.link, source, category)
                    if article_data:
                        articles.append(article_data)
                        logger.info(f"   ✅ {len(articles)}: {article_data['title'][:50]}...")
                    
                    # Rate limiting
                    time.sleep(self.delay)
                    
            except Exception as e:
                logger.error(f"Failed to process feed {feed_url}: {str(e)}")
                continue
                
        return articles
    
    def collect_entertainment_batch(self, articles_per_source: int = 50, batch_size: int = 3):
        """Collect entertainment articles in batches"""
        
        logger.info(f"🎭 ENTERTAINMENT NEWS COLLECTION")
        logger.info(f"=" * 50)
        logger.info(f"   📊 {articles_per_source} articles per source")
        logger.info(f"   🔄 Processing {batch_size} sources at a time")
        logger.info(f"   🎯 Target: ~2000 entertainment articles")
        
        sources = list(self.entertainment_feeds.keys())
        total_collected = 0
        
        # Process sources in batches
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            batch_articles = []
            
            logger.info(f"\n🎪 Processing batch {i//batch_size + 1}: {batch_sources}")
            
            for source in batch_sources:
                try:
                    articles = self.collect_from_rss(source, articles_per_source)
                    batch_articles.extend(articles)
                    
                    logger.info(f"✅ {source}: {len(articles)} articles collected")
                    
                    # Small delay between sources
                    time.sleep(3)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to collect from {source}: {str(e)}")
                    continue
            
            # Save batch to CSV (append mode)
            if batch_articles:
                df_batch = pd.DataFrame(batch_articles)
                
                # Remove duplicates within batch
                df_batch = df_batch.drop_duplicates(subset=['url'])
                
                # Append to existing file
                self.append_to_csv(df_batch, "data/raw/news_articles.csv")
                
                total_collected += len(df_batch)
                logger.info(f"💾 Batch saved: {len(df_batch)} articles")
                logger.info(f"🎬 Total entertainment articles: {total_collected}")
            
            # Longer delay between batches
            if i + batch_size < len(sources):
                logger.info(f"⏳ Waiting 8 seconds before next batch...")
                time.sleep(8)
        
        logger.info(f"\n🎉 ENTERTAINMENT COLLECTION COMPLETE!")
        logger.info(f"🎭 Total entertainment articles collected: {total_collected}")
        
        # Show final statistics
        self.show_entertainment_stats("data/raw/news_articles.csv")
        
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
                    logger.info(f"   📝 Added {new_articles} new entertainment articles")
                else:
                    logger.info(f"   ⚠️ No new articles (all were duplicates)")
                    
            except Exception as e:
                logger.error(f"Error appending to CSV: {str(e)}")
                df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"   📁 Created new file with {len(df)} articles")
    
    def show_entertainment_stats(self, csv_path: str):
        """Show statistics about entertainment articles in dataset"""
        try:
            df = pd.read_csv(csv_path)
            
            # Filter entertainment categories
            entertainment_categories = ['entertainment', 'bollywood', 'hollywood', 'music', 'celebrity', 'lifestyle']
            entertainment_df = df[df['category'].isin(entertainment_categories)]
            
            logger.info(f"\n🎭 ENTERTAINMENT DATASET STATISTICS:")
            logger.info(f"   Total articles: {len(df):,}")
            logger.info(f"   Entertainment articles: {len(entertainment_df):,}")
            logger.info(f"   Entertainment categories: {list(entertainment_df['category'].unique())}")
            
            # Top entertainment sources
            if len(entertainment_df) > 0:
                top_sources = entertainment_df['source'].value_counts().head(5)
                logger.info(f"   Top entertainment sources: {dict(top_sources)}")
            
        except Exception as e:
            logger.error(f"Error showing stats: {str(e)}")

def main():
    """Main entertainment collection function"""
    collector = EntertainmentNewsCollector()
    
    # Collect entertainment articles
    # With 30+ sources × 50 articles each = ~1500+ articles
    # Adjust articles_per_source to reach 2000 target
    total_collected = collector.collect_entertainment_batch(
        articles_per_source=70,  # Higher count for entertainment
        batch_size=3            # Process 3 sources at a time
    )
    
    logger.info(f"\n🌟 ENTERTAINMENT MISSION ACCOMPLISHED!")
    logger.info(f"   Entertainment articles added: {total_collected}")
    logger.info(f"   Dataset location: data/raw/news_articles.csv")
    logger.info(f"   Ready for diverse fact-checking!")

if __name__ == "__main__":
    main()