"""
Multi-source News Scraper for RAG Data Collection
Collects 5,000-10,000 news articles from credible sources
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

class NewsDataCollector:
    def __init__(self):
        """Initialize the news collector"""
        
        # RSS feeds from credible sources
        self.rss_feeds = {
            # Major English Sources
            'reuters': [
                'https://feeds.reuters.com/reuters/topNews',
                'https://feeds.reuters.com/reuters/businessNews',
                'https://feeds.reuters.com/reuters/technologyNews',
                'https://feeds.reuters.com/reuters/worldNews',
                'https://feeds.reuters.com/reuters/politicsNews'
            ],
            'bbc': [
                'http://feeds.bbci.co.uk/news/rss.xml',
                'http://feeds.bbci.co.uk/news/world/rss.xml',
                'http://feeds.bbci.co.uk/news/business/rss.xml',
                'http://feeds.bbci.co.uk/news/technology/rss.xml',
                'http://feeds.bbci.co.uk/news/health/rss.xml'
            ],
            'guardian': [
                'https://www.theguardian.com/world/rss',
                'https://www.theguardian.com/business/rss',
                'https://www.theguardian.com/technology/rss',
                'https://www.theguardian.com/science/rss',
                'https://www.theguardian.com/politics/rss'
            ],
            'ap_news': [
                'https://feeds.apnews.com/rss/apf-topnews',
                'https://feeds.apnews.com/rss/apf-business',
                'https://feeds.apnews.com/rss/apf-technology',
                'https://feeds.apnews.com/rss/apf-science'
            ],
            'cnn': [
                'http://rss.cnn.com/rss/edition.rss',
                'http://rss.cnn.com/rss/money_latest.rss',
                'http://rss.cnn.com/rss/edition_technology.rss'
            ],
            
            # Indian English Sources
            'times_of_india': [
                'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
                'https://timesofindia.indiatimes.com/rssfeeds/1898055.cms',  # India news
                'https://timesofindia.indiatimes.com/rssfeeds/1898024.cms',  # Business
                'https://timesofindia.indiatimes.com/rssfeeds/5880659.cms'   # Tech
            ],
            'the_hindu': [
                'https://www.thehindu.com/news/national/feeder/default.rss',
                'https://www.thehindu.com/business/feeder/default.rss',
                'https://www.thehindu.com/sci-tech/feeder/default.rss',
                'https://www.thehindu.com/news/international/feeder/default.rss'
            ],
            'economic_times': [
                'https://economictimes.indiatimes.com/rssfeedsdefault.cms',
                'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
                'https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms',
                'https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms'
            ],
            'mint': [
                'https://www.livemint.com/rss/news',
                'https://www.livemint.com/rss/markets',
                'https://www.livemint.com/rss/technology',
                'https://www.livemint.com/rss/politics'
            ],
            'deccan_herald': [
                'https://www.deccanherald.com/rss/national.xml',
                'https://www.deccanherald.com/rss/business.xml',
                'https://www.deccanherald.com/rss/international.xml'
            ],
            'indian_express': [
                'https://indianexpress.com/section/india/feed/',
                'https://indianexpress.com/section/business/feed/',
                'https://indianexpress.com/section/technology/feed/'
            ],
            'hindustan_times': [
                'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
                'https://www.hindustantimes.com/feeds/rss/business-news/rssfeed.xml',
                'https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml'
            ],
            'ndtv': [
                'https://feeds.feedburner.com/ndtvnews-top-stories',
                'https://feeds.feedburner.com/ndtvnews-india-news',
                'https://feeds.feedburner.com/ndtvprofit-latest'
            ],
            
            # Regional Indian Sources
            'news18': [
                'https://www.news18.com/rss/india.xml',
                'https://www.news18.com/rss/business.xml',
                'https://www.news18.com/rss/tech.xml'
            ],
            'zee_news': [
                'https://zeenews.india.com/rss/india-national-news.xml',
                'https://zeenews.india.com/rss/business.xml',
                'https://zeenews.india.com/rss/technology-news.xml'
            ],
            
            # International Business/Economic
            'bloomberg': [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://feeds.bloomberg.com/technology/news.rss'
            ],
            'financial_times': [
                'https://www.ft.com/rss/home/uk',
                'https://www.ft.com/rss/companies',
                'https://www.ft.com/rss/technology'
            ],
            'wsj': [
                'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
                'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
                'https://feeds.a.dj.com/rss/RSSWSJD.xml'
            ],
            
            # Middle Eastern Sources
            'al_jazeera': [
                'https://www.aljazeera.com/xml/rss/all.xml',
                'https://www.aljazeera.com/xml/rss/business.xml',
                'https://www.aljazeera.com/xml/rss/technology.xml'
            ],
            'arab_news': [
                'https://www.arabnews.com/rss.xml',
                'https://www.arabnews.com/taxonomy/term/46/feed',  # Business
                'https://www.arabnews.com/taxonomy/term/166/feed'  # Technology
            ],
            
            # European Sources
            'dw_english': [
                'https://rss.dw.com/rdf/rss-en-all',
                'https://rss.dw.com/rdf/rss-en-bus',
                'https://rss.dw.com/rdf/rss-en-sci'
            ],
            'france24': [
                'https://www.france24.com/en/rss',
                'https://www.france24.com/en/business/rss',
                'https://www.france24.com/en/technology/rss'
            ],
            'euronews': [
                'https://www.euronews.com/rss?format=mrss',
                'https://www.euronews.com/rss?format=mrss&level=theme&theme=business',
                'https://www.euronews.com/rss?format=mrss&level=theme&theme=tech'
            ],
            
            # Canadian Sources
            'cbc': [
                'https://rss.cbc.ca/lineup/topstories.xml',
                'https://rss.cbc.ca/lineup/business.xml',
                'https://rss.cbc.ca/lineup/technology.xml'
            ],
            'globe_mail': [
                'https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/news/',
                'https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/business/',
                'https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/technology/'
            ],
            
            # Australian Sources
            'abc_australia': [
                'https://www.abc.net.au/news/feed/45910/rss.xml',  # Top stories
                'https://www.abc.net.au/news/feed/51120/rss.xml',  # Business
                'https://www.abc.net.au/news/feed/51892/rss.xml'   # Technology
            ],
            
            # Japanese Sources (English)
            'japan_times': [
                'https://www.japantimes.co.jp/feed/',
                'https://www.japantimes.co.jp/news/business/feed/',
                'https://www.japantimes.co.jp/news/science-health/feed/'
            ],
            'nikkei_asia': [
                'https://asia.nikkei.com/rss/feed/nar',
                'https://asia.nikkei.com/rss/feed/nar-business',
                'https://asia.nikkei.com/rss/feed/nar-tech'
            ],
            
            # Spanish Sources
            'el_pais': [
                'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada',
                'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/economia/portada',
                'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/tecnologia/portada'
            ],
            'bbc_mundo': [
                'https://feeds.bbci.co.uk/mundo/rss.xml',
                'https://feeds.bbci.co.uk/mundo/temas/economia/rss.xml',
                'https://feeds.bbci.co.uk/mundo/temas/tecnologia/rss.xml'
            ],
            
            # French Sources
            'le_monde': [
                'https://www.lemonde.fr/rss/une.xml',
                'https://www.lemonde.fr/economie/rss_full.xml',
                'https://www.lemonde.fr/pixels/rss_full.xml'
            ],
            'france_info': [
                'https://www.francetvinfo.fr/titres.rss',
                'https://www.francetvinfo.fr/economie.rss',
                'https://www.francetvinfo.fr/internet.rss'
            ],
            
            # German Sources
            'dw_german': [
                'https://rss.dw.com/rdf/rss-de-all',
                'https://rss.dw.com/rdf/rss-de-wirtschaft',
                'https://rss.dw.com/rdf/rss-de-wissenschaft'
            ],
            
            # Tech-focused Sources
            'techcrunch': [
                'https://techcrunch.com/feed/',
                'https://techcrunch.com/category/startups/feed/',
                'https://techcrunch.com/category/artificial-intelligence/feed/'
            ],
            'wired': [
                'https://www.wired.com/feed/rss',
                'https://www.wired.com/feed/category/business/rss',
                'https://www.wired.com/feed/category/science/rss'
            ],
            'ars_technica': [
                'http://feeds.arstechnica.com/arstechnica/index',
                'http://feeds.arstechnica.com/arstechnica/technology-lab',
                'http://feeds.arstechnica.com/arstechnica/business'
            ]
        }
        
        # Language detection mapping
        self.source_languages = {
            # English Sources
            'reuters': 'en', 'bbc': 'en', 'guardian': 'en', 'ap_news': 'en', 'cnn': 'en',
            'times_of_india': 'en', 'the_hindu': 'en', 'economic_times': 'en', 'mint': 'en',
            'deccan_herald': 'en', 'indian_express': 'en', 'hindustan_times': 'en', 'ndtv': 'en',
            'news18': 'en', 'zee_news': 'en', 'bloomberg': 'en', 'financial_times': 'en', 'wsj': 'en',
            'al_jazeera': 'en', 'arab_news': 'en', 'dw_english': 'en', 'france24': 'en', 'euronews': 'en',
            'cbc': 'en', 'globe_mail': 'en', 'abc_australia': 'en', 'japan_times': 'en', 'nikkei_asia': 'en',
            'techcrunch': 'en', 'wired': 'en', 'ars_technica': 'en',
            
            # Spanish Sources
            'el_pais': 'es', 'bbc_mundo': 'es',
            
            # French Sources  
            'le_monde': 'fr', 'france_info': 'fr',
            
            # German Sources
            'dw_german': 'de'
        }
        
        # Category mapping
        self.category_mapping = {
            'topnews': 'general',
            'world': 'world',
            'business': 'business', 
            'technology': 'technology',
            'science': 'science',
            'national': 'politics',
            'markets': 'business'
        }
        
        # Rate limiting
        self.delay = 2  # seconds between requests
        
    def extract_category_from_url(self, url: str, source: str) -> str:
        """Extract category from RSS feed URL"""
        url_lower = url.lower()
        
        for keyword, category in self.category_mapping.items():
            if keyword in url_lower:
                return category
                
        return 'general'
    
    def scrape_article(self, url: str, source: str, category: str) -> Dict:
        """Scrape individual article"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Skip if article is too short
            if len(article.text) < 200:
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
        """Collect articles from RSS feeds for a source"""
        articles = []
        feeds = self.rss_feeds.get(source, [])
        
        logger.info(f"📰 Collecting from {source} ({len(feeds)} feeds)")
        
        for feed_url in feeds:
            try:
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                category = self.extract_category_from_url(feed_url, source)
                
                logger.info(f"   📡 {feed_url} -> {category}")
                
                # Process entries
                for entry in feed.entries[:max_articles//len(feeds)]:
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
    
    def collect_and_append(self, articles_per_source: int = 50, batch_size: int = 5):
        """Collect articles in batches and continuously append to existing CSV"""
        
        logger.info(f"🚀 Starting incremental collection:")
        logger.info(f"   📊 {articles_per_source} articles per source")
        logger.info(f"   🔄 Processing {batch_size} sources at a time")
        logger.info(f"   📁 Appending to: data/raw/news_articles.csv")
        
        sources = list(self.rss_feeds.keys())
        total_collected = 0
        
        # Process sources in batches
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            batch_articles = []
            
            logger.info(f"\n📦 Processing batch {i//batch_size + 1}: {batch_sources}")
            
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
                logger.info(f"📊 Total collected so far: {total_collected}")
            
            # Longer delay between batches
            if i + batch_size < len(sources):
                logger.info(f"⏳ Waiting 10 seconds before next batch...")
                time.sleep(10)
        
        logger.info(f"\n🎉 Collection complete!")
        logger.info(f"📊 Total new articles collected: {total_collected}")
        
        # Show final statistics
        self.show_dataset_stats("data/raw/news_articles.csv")
        
        return total_collected
    
    def append_to_csv(self, df: pd.DataFrame, csv_path: str):
        """Append DataFrame to existing CSV file"""
        csv_path = Path(csv_path)
        
        if csv_path.exists():
            # Load existing data to check for duplicates
            try:
                existing_df = pd.read_csv(csv_path)
                
                # Remove duplicates between new and existing data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
                
                # Calculate new articles added
                new_articles = len(combined_df) - len(existing_df)
                
                if new_articles > 0:
                    # Save combined data
                    combined_df.to_csv(csv_path, index=False)
                    logger.info(f"   📝 Added {new_articles} new articles (duplicates removed)")
                else:
                    logger.info(f"   ⚠️ No new articles (all were duplicates)")
                    
            except Exception as e:
                logger.error(f"Error appending to CSV: {str(e)}")
                # Fallback: just append without duplicate checking
                df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # Create new file
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"   📁 Created new file with {len(df)} articles")
    
    def show_dataset_stats(self, csv_path: str):
        """Show statistics about the collected dataset"""
        try:
            df = pd.read_csv(csv_path)
            
            logger.info(f"\n📊 DATASET STATISTICS:")
            logger.info(f"   Total articles: {len(df):,}")
            logger.info(f"   Sources: {df['source'].nunique()}")
            logger.info(f"   Languages: {list(df['language'].unique())}")
            logger.info(f"   Categories: {list(df['category'].unique())}")
            
            # Top sources
            top_sources = df['source'].value_counts().head(5)
            logger.info(f"   Top sources: {dict(top_sources)}")
            
        except Exception as e:
            logger.error(f"Error showing stats: {str(e)}")
    
    def save_articles(self, df: pd.DataFrame, output_path: str = "data/raw/news_articles_new.csv"):
        """Save articles to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved {len(df)} articles to {output_path}")
        
        return output_path
    
    def merge_with_existing(self, new_csv: str, existing_csv: str = "data/raw/news_articles.csv"):
        """Merge new articles with existing ones"""
        try:
            # Load existing data
            existing_df = pd.read_csv(existing_csv)
            logger.info(f"📂 Existing articles: {len(existing_df)}")
            
            # Load new data
            new_df = pd.read_csv(new_csv)
            logger.info(f"📥 New articles: {len(new_df)}")
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['url'])
            
            # Save combined data
            combined_df.to_csv(existing_csv, index=False)
            
            logger.info(f"🔄 Merged dataset: {len(combined_df)} total articles")
            logger.info(f"💾 Saved to {existing_csv}")
            
            return len(combined_df)
            
        except FileNotFoundError:
            logger.warning(f"No existing file found at {existing_csv}")
            # Just rename new file
            Path(new_csv).rename(existing_csv)
            new_df = pd.read_csv(existing_csv)
            return len(new_df)

def main():
    """Main collection function - continuously append to existing dataset"""
    collector = NewsDataCollector()
    
    logger.info("🌍 MULTI-SOURCE NEWS COLLECTION")
    logger.info("=" * 50)
    
    # Collect articles in batches and append to existing CSV
    # Adjust articles_per_source based on your needs:
    # - 20-30 for quick testing
    # - 50-100 for regular collection  
    # - 200+ for large-scale collection
    
    total_collected = collector.collect_and_append(
        articles_per_source=30,  # Start conservative
        batch_size=3            # Process 3 sources at a time
    )
    
    logger.info(f"\n� MISSION ACCOMPLISHED!")
    logger.info(f"   New articles added: {total_collected}")
    logger.info(f"   Dataset location: data/raw/news_articles.csv")

if __name__ == "__main__":
    main()