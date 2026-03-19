"""
AI & Technology News Scraper
Collects 3000+ articles about AI, technology, startups, cybersecurity, innovation
Multiple languages: English, Spanish, French, German, Japanese, Chinese
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

class AITechNewsCollector:
    def __init__(self):
        """Initialize the AI & Technology news collector"""
        
        # AI & Technology-focused RSS feeds
        self.ai_tech_feeds = {
            
           
            
            # International Tech Sources
           
            'cnn_tech': [
                'http://rss.cnn.com/rss/edition_technology.rss'
            ],
            'guardian_tech': [
                'https://www.theguardian.com/technology/rss'
            ],
            
            # European Tech Sources
            'tech_eu': [
                'https://tech.eu/feed/'
            ],
            'sifted': [
                'https://sifted.eu/feed'
            ],
            
            # Asian Tech Sources
            'nikkei_tech': [
                'https://asia.nikkei.com/rss/feed/nar-tech'
            ],
            'tech_in_asia': [
                'https://www.techinasia.com/rss'
            ],
            'kr_asia': [
                'https://kr-asia.com/feed'
            ],
            
            # Spanish Tech Sources
            'xataka': [
                'https://www.xataka.com/index.xml'
            ],
            'hipertextual': [
                'https://hipertextual.com/feed'
            ],
            
            # French Tech Sources
            'numerama': [
                'https://www.numerama.com/rss/news.rss'
            ],
            'frandroid': [
                'https://www.frandroid.com/feed'
            ],
            
            # German Tech Sources
            'heise_online': [
                'https://www.heise.de/rss/heise-atom.xml'
            ],
            'golem': [
                'https://rss.golem.de/rss.php?feed=ATOM1.0'
            ],
            
            # Research & Academic
            'nature_tech': [
                'https://www.nature.com/subjects/computer-science.rss',
                'https://www.nature.com/subjects/machine-learning.rss'
            ],
            'science_robotics': [
                'https://robotics.sciencemag.org/rss/current.xml'
            ],
            'ieee_spectrum': [
                'https://spectrum.ieee.org/rss/blog/tech-talk',
                'https://spectrum.ieee.org/rss/robotics'
            ],
            
            # Mobile & Gaming
            'android_authority': [
                'https://www.androidauthority.com/feed/'
            ],
            'macrumors': [
                'https://feeds.macrumors.com/MacRumors-All'
            ],
            'polygon': [
                'https://www.polygon.com/rss/index.xml'
            ],
            'gamesindustry': [
                'https://www.gamesindustry.biz/feed'
            ]
        }
        
        # Language mapping
        self.source_languages = {
            # English sources
            'techcrunch': 'en', 'the_verge': 'en', 'wired': 'en', 'ars_technica': 'en',
            'engadget': 'en', 'gizmodo': 'en', 'mashable_tech': 'en', 'ai_news': 'en',
            'venturebeat_ai': 'en', 'mit_tech_review': 'en', 'ai_magazine': 'en',
            'towards_data_science': 'en', 'bloomberg_tech': 'en', 'reuters_tech': 'en',
            'wsj_tech': 'en', 'financial_times_tech': 'en', 'techstartups': 'en',
            'startup_grind': 'en', 'product_hunt': 'en', 'krebs_security': 'en',
            'dark_reading': 'en', 'security_week': 'en', 'threatpost': 'en',
            'hacker_news': 'en', 'github_blog': 'en', 'stack_overflow_blog': 'en',
            'dev_to': 'en', 'aws_news': 'en', 'google_cloud_blog': 'en',
            'microsoft_azure': 'en', 'yourstory': 'en', 'inc42': 'en', 'medianama': 'en',
            'economic_times_tech': 'en', 'livemint_tech': 'en', 'bbc_tech': 'en',
            'cnn_tech': 'en', 'guardian_tech': 'en', 'tech_eu': 'en', 'sifted': 'en',
            'nikkei_tech': 'en', 'tech_in_asia': 'en', 'kr_asia': 'en',
            'nature_tech': 'en', 'science_robotics': 'en', 'ieee_spectrum': 'en',
            'android_authority': 'en', 'macrumors': 'en', 'polygon': 'en', 'gamesindustry': 'en',
            
            # Spanish sources
            'xataka': 'es', 'hipertextual': 'es',
            
            # French sources
            'numerama': 'fr', 'frandroid': 'fr',
            
            # German sources
            'heise_online': 'de', 'golem': 'de'
        }
        
        # Category mapping for AI & Tech
        self.category_mapping = {
            'ai': 'artificial_intelligence', 'artificial': 'artificial_intelligence',
            'machine': 'artificial_intelligence', 'deep': 'artificial_intelligence',
            'neural': 'artificial_intelligence', 'robotics': 'robotics',
            'startup': 'startups', 'venture': 'startups', 'funding': 'startups',
            'security': 'cybersecurity', 'cyber': 'cybersecurity', 'hack': 'cybersecurity',
            'cloud': 'cloud_computing', 'aws': 'cloud_computing', 'azure': 'cloud_computing',
            'mobile': 'mobile_tech', 'android': 'mobile_tech', 'ios': 'mobile_tech',
            'gaming': 'gaming_tech', 'game': 'gaming_tech', 'vr': 'gaming_tech',
            'blockchain': 'blockchain', 'crypto': 'blockchain', 'bitcoin': 'blockchain',
            'tech': 'technology', 'innovation': 'technology', 'digital': 'technology'
        }
        
        # Rate limiting
        self.delay = 1.5  # Faster for tech sites
        
    def extract_category_from_url(self, url: str, source: str) -> str:
        """Extract category from RSS feed URL"""
        url_lower = url.lower()
        
        # Check for specific tech categories
        if any(word in url_lower for word in ['ai', 'artificial', 'machine', 'neural']):
            return 'artificial_intelligence'
        elif any(word in url_lower for word in ['security', 'cyber', 'hack']):
            return 'cybersecurity'
        elif any(word in url_lower for word in ['startup', 'venture', 'funding']):
            return 'startups'
        elif any(word in url_lower for word in ['cloud', 'aws', 'azure']):
            return 'cloud_computing'
        elif any(word in url_lower for word in ['mobile', 'android', 'ios']):
            return 'mobile_tech'
        elif any(word in url_lower for word in ['game', 'gaming', 'vr']):
            return 'gaming_tech'
        elif any(word in url_lower for word in ['blockchain', 'crypto', 'bitcoin']):
            return 'blockchain'
        elif any(word in url_lower for word in ['robot', 'robotics']):
            return 'robotics'
        
        return 'technology'  # Default for tech scraper
    
    def scrape_article(self, url: str, source: str, category: str) -> Dict:
        """Scrape individual AI/tech article"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Skip if article is too short
            if len(article.text) < 150:
                return None
                
            # Skip if it's not tech related (basic filter)
            text_lower = article.text.lower()
            title_lower = article.title.lower()
            
            tech_keywords = [
                'technology', 'tech', 'ai', 'artificial intelligence', 'machine learning',
                'software', 'hardware', 'computer', 'digital', 'internet', 'app',
                'startup', 'innovation', 'cybersecurity', 'data', 'algorithm',
                'programming', 'coding', 'developer', 'cloud', 'mobile', 'smartphone',
                'robotics', 'automation', 'blockchain', 'cryptocurrency', 'gaming',
                'virtual reality', 'augmented reality', 'iot', 'internet of things',
                'silicon valley', 'tech company', 'platform', 'api', 'database'
            ]
            
            if not any(keyword in text_lower or keyword in title_lower for keyword in tech_keywords):
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
    
    def collect_from_rss(self, source: str, max_articles: int = 50) -> List[Dict]:
        """Collect AI/tech articles from RSS feeds for a source"""
        articles = []
        feeds = self.ai_tech_feeds.get(source, [])
        
        logger.info(f"🤖 Collecting from {source} ({len(feeds)} feeds)")
        
        for feed_url in feeds:
            try:
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                category = self.extract_category_from_url(feed_url, source)
                
                logger.info(f"   📡 {feed_url} -> {category}")
                
                # Process entries
                articles_per_feed = max_articles // len(feeds) if len(feeds) > 0 else max_articles
                
                for entry in feed.entries[:articles_per_feed + 3]:  # Get a few extra in case some fail
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
    
    def collect_ai_tech_batch(self, articles_per_source: int = 50, batch_size: int = 4):
        """Collect AI/tech articles in batches"""
        
        logger.info(f"🤖 AI & TECHNOLOGY NEWS COLLECTION")
        logger.info(f"=" * 50)
        logger.info(f"   📊 {articles_per_source} articles per source")
        logger.info(f"   🔄 Processing {batch_size} sources at a time")
        logger.info(f"   🎯 Target: ~3000 AI & tech articles")
        logger.info(f"   🌐 Languages: English, Spanish, French, German")
        
        sources = list(self.ai_tech_feeds.keys())
        total_collected = 0
        
        # Process sources in batches
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            batch_articles = []
            
            logger.info(f"\n💻 Processing batch {i//batch_size + 1}: {batch_sources}")
            
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
                logger.info(f"🤖 Total AI/tech articles: {total_collected}")
            
            # Longer delay between batches
            if i + batch_size < len(sources):
                logger.info(f"⏳ Waiting 8 seconds before next batch...")
                time.sleep(8)
        
        logger.info(f"\n🎉 AI & TECH COLLECTION COMPLETE!")
        logger.info(f"🤖 Total AI/tech articles collected: {total_collected}")
        
        # Show final statistics
        self.show_ai_tech_stats("data/raw/news_articles.csv")
        
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
                    logger.info(f"   📝 Added {new_articles} new AI/tech articles")
                else:
                    logger.info(f"   ⚠️ No new articles (all were duplicates)")
                    
            except Exception as e:
                logger.error(f"Error appending to CSV: {str(e)}")
                df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"   📁 Created new file with {len(df)} articles")
    
    def show_ai_tech_stats(self, csv_path: str):
        """Show statistics about AI/tech articles in dataset"""
        try:
            df = pd.read_csv(csv_path)
            
            # Filter AI/tech categories
            tech_categories = ['artificial_intelligence', 'technology', 'cybersecurity', 'startups', 
                             'cloud_computing', 'mobile_tech', 'gaming_tech', 'blockchain', 'robotics']
            tech_df = df[df['category'].isin(tech_categories)]
            
            logger.info(f"\n🤖 AI & TECH DATASET STATISTICS:")
            logger.info(f"   Total articles: {len(df):,}")
            logger.info(f"   AI/Tech articles: {len(tech_df):,}")
            logger.info(f"   Tech categories: {list(tech_df['category'].unique())}")
            
            # Top tech sources
            if len(tech_df) > 0:
                top_sources = tech_df['source'].value_counts().head(5)
                logger.info(f"   Top tech sources: {dict(top_sources)}")
                
                # Language breakdown
                lang_breakdown = tech_df['language'].value_counts()
                logger.info(f"   Language breakdown: {dict(lang_breakdown)}")
                
                # Category breakdown
                cat_breakdown = tech_df['category'].value_counts()
                logger.info(f"   Category breakdown: {dict(cat_breakdown)}")
            
        except Exception as e:
            logger.error(f"Error showing stats: {str(e)}")

def main():
    """Main AI/tech collection function"""
    collector = AITechNewsCollector()
    
    # Collect AI/tech articles
    # With 60+ sources × 50 articles each = ~3000+ articles
    total_collected = collector.collect_ai_tech_batch(
        articles_per_source=50,  # Balanced for comprehensive coverage
        batch_size=4            # Process 4 sources at a time
    )
    
    logger.info(f"\n🌟 AI & TECH MISSION ACCOMPLISHED!")
    logger.info(f"   AI/Tech articles added: {total_collected}")
    logger.info(f"   Dataset location: data/raw/news_articles.csv")
    logger.info(f"   Ready for cutting-edge fact-checking!")

if __name__ == "__main__":
    main()