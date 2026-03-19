"""
Geopolitics & International Relations News Scraper
Collects 2000+ articles about geopolitics, international relations, diplomacy, conflicts
Multiple languages: English, Spanish, French, German, Arabic, Hindi
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

class GeopoliticsNewsCollector:
    def __init__(self):
        """Initialize the geopolitics news collector"""
        
        # Geopolitics-focused RSS feeds
        self.geopolitics_feeds = {
            # Major International Relations Sources
            'foreign_affairs': [
                'https://www.foreignaffairs.com/rss.xml',
                'https://www.foreignaffairs.com/regions/rss.xml'
            ],
            'foreign_policy': [
                'https://foreignpolicy.com/feed/',
                'https://foreignpolicy.com/category/politics/feed/',
                'https://foreignpolicy.com/category/security/feed/'
            ],
            'council_foreign_relations': [
                'https://www.cfr.org/rss-feeds/publication/blog',
                'https://www.cfr.org/rss-feeds/publication/backgrounder'
            ],
            'chatham_house': [
                'https://www.chathamhouse.org/rss.xml'
            ],
            'brookings': [
                'https://www.brookings.edu/feed/',
                'https://www.brookings.edu/topic/international-affairs/feed/'
            ],
            
            # International News Agencies
            'reuters_world': [
                'https://feeds.reuters.com/reuters/worldNews',
                'https://feeds.reuters.com/reuters/politicsNews',
                'https://feeds.reuters.com/reuters/topNews'
            ],
            'ap_world': [
                'https://feeds.apnews.com/rss/apf-intlnews',
                'https://feeds.apnews.com/rss/apf-politics',
                'https://feeds.apnews.com/rss/apf-usnews'
            ],
            'bbc_world': [
                'http://feeds.bbci.co.uk/news/world/rss.xml',
                'http://feeds.bbci.co.uk/news/world/africa/rss.xml',
                'http://feeds.bbci.co.uk/news/world/asia/rss.xml',
                'http://feeds.bbci.co.uk/news/world/europe/rss.xml',
                'http://feeds.bbci.co.uk/news/world/latin_america/rss.xml',
                'http://feeds.bbci.co.uk/news/world/middle_east/rss.xml'
            ],
            'cnn_world': [
                'http://rss.cnn.com/rss/edition_world.rss',
                'http://rss.cnn.com/rss/edition_africa.rss',
                'http://rss.cnn.com/rss/edition_asia.rss',
                'http://rss.cnn.com/rss/edition_europe.rss'
            ],
            
            # Regional Geopolitical Sources
            'al_jazeera_politics': [
                'https://www.aljazeera.com/xml/rss/all.xml',
                'https://www.aljazeera.com/xml/rss/politics.xml',
                'https://www.aljazeera.com/xml/rss/middleeast.xml',
                'https://www.aljazeera.com/xml/rss/africa.xml'
            ],
            'dw_politics': [
                'https://rss.dw.com/rdf/rss-en-pol',
                'https://rss.dw.com/rdf/rss-en-world',
                'https://rss.dw.com/rdf/rss-en-eu'
            ],
            'france24_politics': [
                'https://www.france24.com/en/rss',
                'https://www.france24.com/en/europe/rss',
                'https://www.france24.com/en/middle-east/rss',
                'https://www.france24.com/en/africa/rss'
            ],
            'euronews_politics': [
                'https://www.euronews.com/rss?format=mrss',
                'https://www.euronews.com/rss?format=mrss&level=theme&theme=world',
                'https://www.euronews.com/rss?format=mrss&level=theme&theme=europe'
            ],
            
            # Think Tanks & Analysis
            'csis': [
                'https://www.csis.org/rss.xml'
            ],
            'atlantic_council': [
                'https://www.atlanticcouncil.org/feed/'
            ],
            'carnegie_endowment': [
                'https://carnegieendowment.org/rss/publications.xml'
            ],
            'rand_corporation': [
                'https://www.rand.org/content/rand/news.rss.xml'
            ],
            
            # Regional Specialists
            'middle_east_eye': [
                'https://www.middleeasteye.net/rss.xml',
                'https://www.middleeasteye.net/news/rss.xml'
            ],
            'south_china_morning_post': [
                'https://www.scmp.com/rss/91/feed',  # China news
                'https://www.scmp.com/rss/2/feed',   # Asia news
                'https://www.scmp.com/rss/322291/feed' # World news
            ],
            'times_of_israel': [
                'https://www.timesofisrael.com/feed/'
            ],
            'haaretz': [
                'https://www.haaretz.com/cmlink/1.628752'
            ],
            
            # Indian Geopolitical Sources
            'observer_research': [
                'https://www.orfonline.org/feed/'
            ],
            'gateway_house': [
                'https://www.gatewayhouse.in/feed/'
            ],
            'the_diplomat': [
                'https://thediplomat.com/feed/',
                'https://thediplomat.com/regions/south-asia/feed/',
                'https://thediplomat.com/regions/east-asia/feed/'
            ],
            
            # European Sources
            'politico_eu': [
                'https://www.politico.eu/feed/',
                'https://www.politico.eu/section/politics/feed/'
            ],
            'european_council_foreign_relations': [
                'https://ecfr.eu/rss.xml'
            ],
            
            # Spanish Language Sources
            'el_pais_internacional': [
                'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/internacional/portada'
            ],
            'bbc_mundo_internacional': [
                'https://feeds.bbci.co.uk/mundo/rss.xml'
            ],
            
            # French Language Sources
            'le_monde_international': [
                'https://www.lemonde.fr/international/rss_full.xml',
                'https://www.lemonde.fr/politique/rss_full.xml'
            ],
            'le_figaro_international': [
                'https://www.lefigaro.fr/rss/figaro_international.xml',
                'https://www.lefigaro.fr/rss/figaro_politique.xml'
            ],
            
            # German Language Sources
            'dw_german_politics': [
                'https://rss.dw.com/rdf/rss-de-pol',
                'https://rss.dw.com/rdf/rss-de-welt'
            ],
            'spiegel_international': [
                'https://www.spiegel.de/international/index.rss'
            ],
            
            # Arabic Sources
            'al_arabiya': [
                'https://english.alarabiya.net/rss.xml'
            ],
            'arab_news_politics': [
                'https://www.arabnews.com/rss.xml'
            ],
            
            # Security & Defense
            'defense_news': [
                'https://www.defensenews.com/rss/top-news/',
                'https://www.defensenews.com/rss/global/'
            ],
            'janes': [
                'https://www.janes.com/feeds/news.xml'
            ],
            
            # Economic Geopolitics
            'financial_times_world': [
                'https://www.ft.com/rss/home/world',
                'https://www.ft.com/rss/world/global-economy'
            ],
            'wsj_world': [
                'https://feeds.a.dj.com/rss/RSSWorldNews.xml'
            ]
        }
        
        # Language mapping
        self.source_languages = {
            # English sources
            'foreign_affairs': 'en', 'foreign_policy': 'en', 'council_foreign_relations': 'en',
            'chatham_house': 'en', 'brookings': 'en', 'reuters_world': 'en', 'ap_world': 'en',
            'bbc_world': 'en', 'cnn_world': 'en', 'al_jazeera_politics': 'en', 'dw_politics': 'en',
            'france24_politics': 'en', 'euronews_politics': 'en', 'csis': 'en', 'atlantic_council': 'en',
            'carnegie_endowment': 'en', 'rand_corporation': 'en', 'middle_east_eye': 'en',
            'south_china_morning_post': 'en', 'times_of_israel': 'en', 'haaretz': 'en',
            'observer_research': 'en', 'gateway_house': 'en', 'the_diplomat': 'en',
            'politico_eu': 'en', 'european_council_foreign_relations': 'en', 'defense_news': 'en',
            'janes': 'en', 'financial_times_world': 'en', 'wsj_world': 'en', 'al_arabiya': 'en',
            'arab_news_politics': 'en', 'spiegel_international': 'en',
            
            # Spanish sources
            'el_pais_internacional': 'es', 'bbc_mundo_internacional': 'es',
            
            # French sources
            'le_monde_international': 'fr', 'le_figaro_international': 'fr',
            
            # German sources
            'dw_german_politics': 'de'
        }
        
        # Category mapping for geopolitics
        self.category_mapping = {
            'international': 'geopolitics', 'world': 'geopolitics', 'politics': 'geopolitics',
            'diplomacy': 'geopolitics', 'foreign': 'geopolitics', 'global': 'geopolitics',
            'security': 'security', 'defense': 'security', 'military': 'security',
            'conflict': 'conflict', 'war': 'conflict', 'peace': 'geopolitics',
            'trade': 'economic_geopolitics', 'economy': 'economic_geopolitics',
            'europe': 'geopolitics', 'asia': 'geopolitics', 'africa': 'geopolitics',
            'middle_east': 'geopolitics', 'america': 'geopolitics'
        }
        
        # Rate limiting
        self.delay = 2  # Slightly slower for quality sources
        
    def extract_category_from_url(self, url: str, source: str) -> str:
        """Extract category from RSS feed URL"""
        url_lower = url.lower()
        
        # Check for specific geopolitical categories
        if any(word in url_lower for word in ['security', 'defense', 'military']):
            return 'security'
        elif any(word in url_lower for word in ['conflict', 'war']):
            return 'conflict'
        elif any(word in url_lower for word in ['economy', 'trade', 'economic']):
            return 'economic_geopolitics'
        elif any(word in url_lower for word in ['international', 'world', 'global', 'foreign']):
            return 'geopolitics'
        elif any(word in url_lower for word in ['politics', 'political']):
            return 'geopolitics'
        
        return 'geopolitics'  # Default for geopolitics scraper
    
    def scrape_article(self, url: str, source: str, category: str) -> Dict:
        """Scrape individual geopolitics article"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Skip if article is too short
            if len(article.text) < 200:
                return None
                
            # Skip if it's not geopolitics related (basic filter)
            text_lower = article.text.lower()
            title_lower = article.title.lower()
            
            geopolitics_keywords = [
                'government', 'politics', 'international', 'diplomacy', 'foreign policy',
                'geopolitics', 'security', 'defense', 'military', 'conflict', 'war',
                'peace', 'treaty', 'alliance', 'sanctions', 'trade war', 'summit',
                'president', 'minister', 'ambassador', 'parliament', 'congress',
                'election', 'democracy', 'authoritarian', 'regime', 'coup',
                'terrorism', 'intelligence', 'espionage', 'cyber warfare',
                'nato', 'un', 'eu', 'g7', 'g20', 'brics', 'asean'
            ]
            
            if not any(keyword in text_lower or keyword in title_lower for keyword in geopolitics_keywords):
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
    
    def collect_from_rss(self, source: str, max_articles: int = 60) -> List[Dict]:
        """Collect geopolitics articles from RSS feeds for a source"""
        articles = []
        feeds = self.geopolitics_feeds.get(source, [])
        
        logger.info(f"🌍 Collecting from {source} ({len(feeds)} feeds)")
        
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
    
    def collect_geopolitics_batch(self, articles_per_source: int = 60, batch_size: int = 3):
        """Collect geopolitics articles in batches"""
        
        logger.info(f"🌍 GEOPOLITICS NEWS COLLECTION")
        logger.info(f"=" * 50)
        logger.info(f"   📊 {articles_per_source} articles per source")
        logger.info(f"   🔄 Processing {batch_size} sources at a time")
        logger.info(f"   🎯 Target: ~2000 geopolitics articles")
        logger.info(f"   🌐 Languages: English, Spanish, French, German")
        
        sources = list(self.geopolitics_feeds.keys())
        total_collected = 0
        
        # Process sources in batches
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            batch_articles = []
            
            logger.info(f"\n🏛️ Processing batch {i//batch_size + 1}: {batch_sources}")
            
            for source in batch_sources:
                try:
                    articles = self.collect_from_rss(source, articles_per_source)
                    batch_articles.extend(articles)
                    
                    logger.info(f"✅ {source}: {len(articles)} articles collected")
                    
                    # Small delay between sources
                    time.sleep(4)
                    
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
                logger.info(f"🌍 Total geopolitics articles: {total_collected}")
            
            # Longer delay between batches
            if i + batch_size < len(sources):
                logger.info(f"⏳ Waiting 10 seconds before next batch...")
                time.sleep(10)
        
        logger.info(f"\n🎉 GEOPOLITICS COLLECTION COMPLETE!")
        logger.info(f"🌍 Total geopolitics articles collected: {total_collected}")
        
        # Show final statistics
        self.show_geopolitics_stats("data/raw/news_articles.csv")
        
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
                    logger.info(f"   📝 Added {new_articles} new geopolitics articles")
                else:
                    logger.info(f"   ⚠️ No new articles (all were duplicates)")
                    
            except Exception as e:
                logger.error(f"Error appending to CSV: {str(e)}")
                df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"   📁 Created new file with {len(df)} articles")
    
    def show_geopolitics_stats(self, csv_path: str):
        """Show statistics about geopolitics articles in dataset"""
        try:
            df = pd.read_csv(csv_path)
            
            # Filter geopolitics categories
            geopolitics_categories = ['geopolitics', 'security', 'conflict', 'economic_geopolitics']
            geopolitics_df = df[df['category'].isin(geopolitics_categories)]
            
            logger.info(f"\n🌍 GEOPOLITICS DATASET STATISTICS:")
            logger.info(f"   Total articles: {len(df):,}")
            logger.info(f"   Geopolitics articles: {len(geopolitics_df):,}")
            logger.info(f"   Geopolitics categories: {list(geopolitics_df['category'].unique())}")
            
            # Top geopolitics sources
            if len(geopolitics_df) > 0:
                top_sources = geopolitics_df['source'].value_counts().head(5)
                logger.info(f"   Top geopolitics sources: {dict(top_sources)}")
                
                # Language breakdown
                lang_breakdown = geopolitics_df['language'].value_counts()
                logger.info(f"   Language breakdown: {dict(lang_breakdown)}")
            
        except Exception as e:
            logger.error(f"Error showing stats: {str(e)}")

def main():
    """Main geopolitics collection function"""
    collector = GeopoliticsNewsCollector()
    
    # Collect geopolitics articles
    # With 35+ sources × 60 articles each = ~2100+ articles
    total_collected = collector.collect_geopolitics_batch(
        articles_per_source=60,  # Higher count for comprehensive coverage
        batch_size=3            # Process 3 sources at a time
    )
    
    logger.info(f"\n🌟 GEOPOLITICS MISSION ACCOMPLISHED!")
    logger.info(f"   Geopolitics articles added: {total_collected}")
    logger.info(f"   Dataset location: data/raw/news_articles.csv")
    logger.info(f"   Ready for international fact-checking!")

if __name__ == "__main__":
    main()