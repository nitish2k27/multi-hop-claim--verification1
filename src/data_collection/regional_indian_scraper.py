"""
Regional Indian Language News Scraper
Targets vernacular portals for "drama" and political content across 10 major Indian languages.
Focus: Regional politics, cultural claims, entertainment drama, and local controversies.
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
from bs4 import BeautifulSoup
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegionalIndianNewsCollector:
    def __init__(self):
        """Initialize the regional Indian news collector"""
        
        # Vernacular portals with RSS feeds and direct scraping targets
        self.regional_sources = {
            # Hindi - Focus on UP/Bihar politics, cultural events
            'dainik_bhaskar': {
                'language': 'hi',
                'rss_feeds': [
                    'https://www.bhaskar.com/rss-feed/1061/',  # National
                    'https://www.bhaskar.com/rss-feed/1062/',  # Politics
                    'https://www.bhaskar.com/rss-feed/1063/',  # Entertainment
                ],
                'categories': ['politics', 'entertainment', 'regional']
            },
            'amar_ujala': {
                'language': 'hi', 
                'rss_feeds': [
                    'https://www.amarujala.com/rss/india-news.xml',
                    'https://www.amarujala.com/rss/uttar-pradesh.xml',
                    'https://www.amarujala.com/rss/entertainment.xml'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            'navbharat_times': {
                'language': 'hi',
                'rss_feeds': [
                    'https://navbharattimes.indiatimes.com/rssfeedsdefault.cms',
                    'https://navbharattimes.indiatimes.com/rss/india.cms',
                    'https://navbharattimes.indiatimes.com/rss/entertainment.cms'
                ],
                'categories': ['politics', 'entertainment', 'culture']
            },
            
            # Tamil - TN political rivalry, cinema-politics overlap
            'dinamalar': {
                'language': 'ta',
                'rss_feeds': [
                    'https://www.dinamalar.com/rss/rss_news_tamil.xml',
                    'https://www.dinamalar.com/rss/rss_cinema_tamil.xml'
                ],
                'categories': ['politics', 'entertainment', 'regional']
            },
            'thanthi_tv': {
                'language': 'ta',
                'rss_feeds': [
                    'https://www.thanthitv.com/rss/news',
                    'https://www.thanthitv.com/rss/politics',
                    'https://www.thanthitv.com/rss/entertainment'
                ],
                'categories': ['politics', 'entertainment', 'regional']
            },
            'vikatan': {
                'language': 'ta',
                'rss_feeds': [
                    'https://www.vikatan.com/rss/news',
                    'https://www.vikatan.com/rss/politics',
                    'https://www.vikatan.com/rss/cinema'
                ],
                'categories': ['politics', 'entertainment', 'culture']
            },
            
            # Telugu - AP/Telangana power shifts, caste dynamics
            'eenadu': {
                'language': 'te',
                'rss_feeds': [
                    'https://www.eenadu.net/rss/telangana-news.xml',
                    'https://www.eenadu.net/rss/andhra-pradesh-news.xml',
                    'https://www.eenadu.net/rss/entertainment.xml'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            'sakshi': {
                'language': 'te',
                'rss_feeds': [
                    'https://www.sakshi.com/rss/telangana',
                    'https://www.sakshi.com/rss/andhra-pradesh',
                    'https://www.sakshi.com/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            'abn_andhra_jyothy': {
                'language': 'te',
                'rss_feeds': [
                    'https://www.andhrajyothy.com/rss/politics',
                    'https://www.andhrajyothy.com/rss/telangana',
                    'https://www.andhrajyothy.com/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            
            # Marathi - Maharashtra coalition drama, local festivals
            'lokmat': {
                'language': 'mr',
                'rss_feeds': [
                    'https://www.lokmat.com/rss/maharashtra/',
                    'https://www.lokmat.com/rss/politics/',
                    'https://www.lokmat.com/rss/entertainment/'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            'sakal': {
                'language': 'mr',
                'rss_feeds': [
                    'https://www.esakal.com/rss/maharashtra',
                    'https://www.esakal.com/rss/politics',
                    'https://www.esakal.com/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'culture']
            },
            'abp_majha': {
                'language': 'mr',
                'rss_feeds': [
                    'https://majha.abplive.com/rss/maharashtra',
                    'https://majha.abplive.com/rss/politics',
                    'https://majha.abplive.com/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            
            # Bengali - WB political clashes, intellectual debates
            'anandabazar_patrika': {
                'language': 'bn',
                'rss_feeds': [
                    'https://www.anandabazar.com/rss/west-bengal',
                    'https://www.anandabazar.com/rss/politics',
                    'https://www.anandabazar.com/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'culture']
            },
            'sangbad_pratidin': {
                'language': 'bn',
                'rss_feeds': [
                    'https://www.sangbadpratidin.in/rss/west-bengal',
                    'https://www.sangbadpratidin.in/rss/politics',
                    'https://www.sangbadpratidin.in/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            
            # Malayalam - Kerala ideological debates, community news
            'mathrubhumi': {
                'language': 'ml',
                'rss_feeds': [
                    'https://www.mathrubhumi.com/rss/kerala',
                    'https://www.mathrubhumi.com/rss/politics',
                    'https://www.mathrubhumi.com/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'culture']
            },
            'malayala_manorama': {
                'language': 'ml',
                'rss_feeds': [
                    'https://www.manoramaonline.com/rss/kerala',
                    'https://www.manoramaonline.com/rss/politics',
                    'https://www.manoramaonline.com/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            
            # Kannada - Karnataka local governance, language activism
            'prajavani': {
                'language': 'kn',
                'rss_feeds': [
                    'https://www.prajavani.net/rss/karnataka',
                    'https://www.prajavani.net/rss/politics',
                    'https://www.prajavani.net/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'culture']
            },
            'tv9_kannada': {
                'language': 'kn',
                'rss_feeds': [
                    'https://kannada.tv9.com/rss/karnataka',
                    'https://kannada.tv9.com/rss/politics',
                    'https://kannada.tv9.com/rss/entertainment'
                ],
                'categories': ['politics', 'regional', 'entertainment']
            },
            
            # Gujarati - Business-politics links, regional pride
            'sandesh': {
                'language': 'gu',
                'rss_feeds': [
                    'https://www.sandesh.com/rss/gujarat',
                    'https://www.sandesh.com/rss/politics',
                    'https://www.sandesh.com/rss/business'
                ],
                'categories': ['politics', 'business', 'regional']
            },
            'gujarat_samachar': {
                'language': 'gu',
                'rss_feeds': [
                    'https://www.gujaratsamachar.com/rss/gujarat',
                    'https://www.gujaratsamachar.com/rss/politics',
                    'https://www.gujaratsamachar.com/rss/business'
                ],
                'categories': ['politics', 'business', 'regional']
            },
            
            # Punjabi - Farmers' issues, diaspora influence
            'ajit_jalandhar': {
                'language': 'pa',
                'rss_feeds': [
                    'https://www.ajitjalandhar.com/rss/punjab',
                    'https://www.ajitjalandhar.com/rss/politics',
                    'https://www.ajitjalandhar.com/rss/diaspora'
                ],
                'categories': ['politics', 'regional', 'diaspora']
            },
            'ptc_news': {
                'language': 'pa',
                'rss_feeds': [
                    'https://www.ptcnews.tv/rss/punjab',
                    'https://www.ptcnews.tv/rss/politics',
                    'https://www.ptcnews.tv/rss/agriculture'
                ],
                'categories': ['politics', 'agriculture', 'regional']
            },
            
            # Odia - Odisha state schemes, tribal culture claims
            'sambad': {
                'language': 'or',
                'rss_feeds': [
                    'https://sambad.in/rss/odisha',
                    'https://sambad.in/rss/politics',
                    'https://sambad.in/rss/culture'
                ],
                'categories': ['politics', 'regional', 'culture']
            },
            'dharitri': {
                'language': 'or',
                'rss_feeds': [
                    'https://www.dharitri.com/rss/odisha',
                    'https://www.dharitri.com/rss/politics',
                    'https://www.dharitri.com/rss/tribal'
                ],
                'categories': ['politics', 'regional', 'tribal']
            },
            
            # Multi-language vernacular giant (as recommended)
            'oneindia': {
                'language': 'multi',
                'rss_feeds': [
                    'https://hindi.oneindia.com/rss/hindi-news-fb.xml',
                    'https://tamil.oneindia.com/rss/tamil-news-fb.xml',
                    'https://telugu.oneindia.com/rss/telugu-news-fb.xml',
                    'https://kannada.oneindia.com/rss/kannada-news-fb.xml',
                    'https://malayalam.oneindia.com/rss/malayalam-news-fb.xml',
                    'https://bengali.oneindia.com/rss/bengali-news-fb.xml',
                    'https://marathi.oneindia.com/rss/marathi-news-fb.xml',
                    'https://gujarati.oneindia.com/rss/gujarati-news-fb.xml'
                ],
                'categories': ['politics', 'regional', 'entertainment', 'culture']
            }
        }
        
        # Drama and politics URL patterns (as recommended by Gemini)
        self.target_url_patterns = [
            '/politics/', '/rajkiya/',  # Regional Politics
            '/opinion/', '/sampadkiya/',  # Cultural/Political claims  
            '/entertainment/', '/cinema/',  # Regional movie/star drama
            '/culture/', '/sanskriti/',  # Cultural claims
            '/controversy/', '/vivad/',  # Controversies
            '/election/', '/chunav/',  # Election drama
            '/assembly/', '/vidhan-sabha/',  # Assembly politics
            '/chief-minister/', '/mukhyamantri/',  # CM related drama
            '/coalition/', '/gathbandhan/',  # Coalition politics
            '/caste/', '/jati/',  # Caste dynamics
            '/festival/', '/tyohar/',  # Festival controversies
            '/temple/', '/mandir/',  # Religious controversies
            '/language/', '/bhasha/'  # Language activism
        ]
        
        # Rate limiting
        self.delay = 3  # Longer delay for regional sites
        
    def is_drama_politics_url(self, url: str) -> bool:
        """Check if URL contains drama/politics patterns"""
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in self.target_url_patterns)
    
    def extract_category_from_url(self, url: str) -> str:
        """Extract category from URL patterns"""
        url_lower = url.lower()
        
        if any(p in url_lower for p in ['/politics/', '/rajkiya/', '/election/', '/chunav/']):
            return 'regional_politics'
        elif any(p in url_lower for p in ['/entertainment/', '/cinema/', '/bollywood/']):
            return 'regional_entertainment'
        elif any(p in url_lower for p in ['/culture/', '/sanskriti/', '/festival/', '/tyohar/']):
            return 'regional_culture'
        elif any(p in url_lower for p in ['/controversy/', '/vivad/', '/caste/', '/jati/']):
            return 'regional_controversy'
        elif any(p in url_lower for p in ['/opinion/', '/sampadkiya/']):
            return 'regional_opinion'
        elif any(p in url_lower for p in ['/temple/', '/mandir/', '/religion/']):
            return 'regional_religion'
        elif any(p in url_lower for p in ['/language/', '/bhasha/']):
            return 'regional_language'
        else:
            return 'regional_general'
    
    def scrape_article(self, url: str, source: str, language: str) -> Dict:
        """Scrape individual regional article"""
        try:
            # Skip if not drama/politics focused
            if not self.is_drama_politics_url(url):
                return None
                
            article = Article(url)
            article.download()
            article.parse()
            
            # Skip if article is too short (less content = less drama)
            if len(article.text) < 300:
                return None
            
            # Extract category
            category = self.extract_category_from_url(url)
            
            return {
                'url': url,
                'title': article.title.strip(),
                'text': article.text.strip(),
                'source': source,
                'publish_date': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else '',
                'language': language,
                'category': category,
                'region': self.get_region_from_source(source),
                'content_type': 'vernacular_drama'
            }
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {str(e)}")
            return None
    
    def get_region_from_source(self, source: str) -> str:
        """Map source to Indian region"""
        region_mapping = {
            # Hindi belt
            'dainik_bhaskar': 'North India', 'amar_ujala': 'North India', 'navbharat_times': 'North India',
            # South India
            'dinamalar': 'Tamil Nadu', 'thanthi_tv': 'Tamil Nadu', 'vikatan': 'Tamil Nadu',
            'eenadu': 'Andhra Pradesh', 'sakshi': 'Telangana', 'abn_andhra_jyothy': 'Andhra Pradesh',
            'mathrubhumi': 'Kerala', 'malayala_manorama': 'Kerala',
            'prajavani': 'Karnataka', 'tv9_kannada': 'Karnataka',
            # West India
            'lokmat': 'Maharashtra', 'sakal': 'Maharashtra', 'abp_majha': 'Maharashtra',
            'sandesh': 'Gujarat', 'gujarat_samachar': 'Gujarat',
            # East India
            'anandabazar_patrika': 'West Bengal', 'sangbad_pratidin': 'West Bengal',
            'sambad': 'Odisha', 'dharitri': 'Odisha',
            # North India
            'ajit_jalandhar': 'Punjab', 'ptc_news': 'Punjab',
            # Multi-regional
            'oneindia': 'Pan India'
        }
        return region_mapping.get(source, 'Unknown')
    
    def collect_from_rss(self, source: str, max_articles: int = 50) -> List[Dict]:
        """Collect regional articles from RSS feeds"""
        articles = []
        source_config = self.regional_sources.get(source, {})
        feeds = source_config.get('rss_feeds', [])
        language = source_config.get('language', 'hi')
        
        logger.info(f"🏛️ Collecting regional content from {source} ({language}) - {len(feeds)} feeds")
        
        for feed_url in feeds:
            try:
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                logger.info(f"   📡 {feed_url}")
                
                # Process entries with drama/politics filter
                drama_count = 0
                for entry in feed.entries[:max_articles//len(feeds) * 2]:  # Get more to filter
                    if len(articles) >= max_articles:
                        break
                    
                    # Only process drama/politics URLs
                    if self.is_drama_politics_url(entry.link):
                        article_data = self.scrape_article(entry.link, source, language)
                        if article_data:
                            articles.append(article_data)
                            drama_count += 1
                            logger.info(f"   🎭 {len(articles)}: {article_data['title'][:60]}...")
                    
                    # Rate limiting
                    time.sleep(self.delay)
                
                logger.info(f"   ✅ Found {drama_count} drama/politics articles from this feed")
                    
            except Exception as e:
                logger.error(f"Failed to process feed {feed_url}: {str(e)}")
                continue
                
        return articles
    
    def collect_regional_batch(self, articles_per_source: int = 40, batch_size: int = 3):
        """Collect regional articles in batches and append to CSV"""
        
        logger.info(f"🏛️ Starting Regional Indian Language Collection:")
        logger.info(f"   📊 {articles_per_source} articles per source")
        logger.info(f"   🔄 Processing {batch_size} sources at a time")
        logger.info(f"   🎯 Focus: Drama, Politics, Cultural Claims")
        logger.info(f"   🌍 Languages: Hindi, Tamil, Telugu, Marathi, Bengali, Malayalam, Kannada, Gujarati, Punjabi, Odia")
        
        sources = list(self.regional_sources.keys())
        total_collected = 0
        
        # Process sources in batches
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            batch_articles = []
            
            logger.info(f"\n📦 Processing regional batch {i//batch_size + 1}: {batch_sources}")
            
            for source in batch_sources:
                try:
                    articles = self.collect_from_rss(source, articles_per_source)
                    batch_articles.extend(articles)
                    
                    source_config = self.regional_sources[source]
                    language = source_config.get('language', 'unknown')
                    region = self.get_region_from_source(source)
                    
                    logger.info(f"✅ {source} ({language}, {region}): {len(articles)} drama/politics articles")
                    
                    # Longer delay between regional sources
                    time.sleep(5)
                    
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
                logger.info(f"💾 Regional batch saved: {len(df_batch)} articles")
                logger.info(f"📊 Total regional collected so far: {total_collected}")
                
                # Show language breakdown for this batch
                lang_breakdown = df_batch['language'].value_counts()
                logger.info(f"   🌍 Languages in batch: {dict(lang_breakdown)}")
            
            # Longer delay between batches for regional sites
            if i + batch_size < len(sources):
                logger.info(f"⏳ Waiting 15 seconds before next regional batch...")
                time.sleep(15)
        
        logger.info(f"\n🎉 Regional collection complete!")
        logger.info(f"📊 Total regional articles collected: {total_collected}")
        
        # Show final regional statistics
        self.show_regional_stats("data/raw/news_articles.csv")
        
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
                    logger.info(f"   📝 Added {new_articles} new regional articles (duplicates removed)")
                else:
                    logger.info(f"   ⚠️ No new regional articles (all were duplicates)")
                    
            except Exception as e:
                logger.error(f"Error appending regional data to CSV: {str(e)}")
                df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"   📁 Created new file with {len(df)} regional articles")
    
    def show_regional_stats(self, csv_path: str):
        """Show statistics about regional articles in the dataset"""
        try:
            df = pd.read_csv(csv_path)
            
            # Filter for regional content
            regional_df = df[df['category'].str.contains('regional', na=False) | 
                           df['content_type'].str.contains('vernacular', na=False)]
            
            if len(regional_df) == 0:
                logger.info("📊 No regional articles found in dataset")
                return
            
            logger.info(f"\n🏛️ REGIONAL INDIAN LANGUAGE STATISTICS:")
            logger.info(f"   Total regional articles: {len(regional_df):,}")
            logger.info(f"   Percentage of dataset: {(len(regional_df)/len(df)*100):.1f}%")
            
            # Language breakdown
            lang_counts = regional_df['language'].value_counts()
            logger.info(f"   🌍 Languages covered: {len(lang_counts)}")
            for lang, count in lang_counts.items():
                logger.info(f"      {lang}: {count:,} articles")
            
            # Regional breakdown
            if 'region' in regional_df.columns:
                region_counts = regional_df['region'].value_counts()
                logger.info(f"   🗺️ Regions covered: {len(region_counts)}")
                for region, count in region_counts.head(10).items():
                    logger.info(f"      {region}: {count:,} articles")
            
            # Category breakdown
            category_counts = regional_df['category'].value_counts()
            logger.info(f"   📂 Regional categories:")
            for category, count in category_counts.head(8).items():
                logger.info(f"      {category}: {count:,} articles")
            
            # Top sources
            source_counts = regional_df['source'].value_counts()
            logger.info(f"   📰 Top regional sources:")
            for source, count in source_counts.head(10).items():
                logger.info(f"      {source}: {count:,} articles")
            
        except Exception as e:
            logger.error(f"Error showing regional stats: {str(e)}")

def main():
    """Test the regional collector"""
    collector = RegionalIndianNewsCollector()
    
    # Test with a small batch
    logger.info("🧪 Testing Regional Indian Language Scraper")
    articles_collected = collector.collect_regional_batch(
        articles_per_source=10,  # Small test
        batch_size=2
    )
    
    logger.info(f"✅ Test complete: {articles_collected} regional articles collected")

if __name__ == "__main__":
    main()