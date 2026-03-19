"""
Master Script to Run All News Scrapers Sequentially
1. General news scraper (~1000 articles, 42-45 minutes)
2. Entertainment scraper (~2000 articles, 30-35 minutes)  
3. Geopolitics scraper (~2000 articles, 45-50 minutes)
4. AI & Technology scraper (~3000 articles, 40-45 minutes)
5. Regional Indian Languages scraper (~1500 articles, 35-40 minutes)
Total: ~9500 articles in 3-3.5 hours for ultra-comprehensive multilingual dataset
"""

import logging
import time
from datetime import datetime
import pandas as pd

# Import our scrapers
from news_scraper import NewsDataCollector
from entertainment_scraper import EntertainmentNewsCollector
from geopolitics_scraper import GeopoliticsNewsCollector
from ai_tech_scraper import AITechNewsCollector
from regional_indian_scraper import RegionalIndianNewsCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_final_statistics():
    """Show comprehensive statistics of the final dataset"""
    try:
        df = pd.read_csv("data/raw/news_articles.csv")
        
        logger.info(f"\n" + "="*60)
        logger.info(f"🎯 FINAL COMPREHENSIVE DATASET STATISTICS")
        logger.info(f"="*60)
        
        logger.info(f"📊 OVERALL STATS:")
        logger.info(f"   Total articles: {len(df):,}")
        logger.info(f"   Unique sources: {df['source'].nunique()}")
        logger.info(f"   Languages: {list(df['language'].unique())}")
        logger.info(f"   Date range: {df['publish_date'].min()} to {df['publish_date'].max()}")
        
        logger.info(f"\n📈 CATEGORY BREAKDOWN:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"   {category}: {count:,} ({percentage:.1f}%)")
        
        logger.info(f"\n🌍 LANGUAGE BREAKDOWN:")
        language_counts = df['language'].value_counts()
        for language, count in language_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"   {language}: {count:,} ({percentage:.1f}%)")
        
        logger.info(f"\n🏆 TOP 10 SOURCES:")
        top_sources = df['source'].value_counts().head(10)
        for source, count in top_sources.items():
            logger.info(f"   {source}: {count:,} articles")
        
        # Entertainment specific stats
        entertainment_categories = ['entertainment', 'bollywood', 'hollywood', 'music', 'celebrity', 'lifestyle']
        entertainment_df = df[df['category'].isin(entertainment_categories)]
        
        # Geopolitics specific stats
        geopolitics_categories = ['geopolitics', 'security', 'conflict', 'economic_geopolitics']
        geopolitics_df = df[df['category'].isin(geopolitics_categories)]
        
        # AI/Tech specific stats
        tech_categories = ['artificial_intelligence', 'technology', 'cybersecurity', 'startups', 
                         'cloud_computing', 'mobile_tech', 'gaming_tech', 'blockchain', 'robotics']
        tech_df = df[df['category'].isin(tech_categories)]
        
        # Regional Indian specific stats
        regional_categories = ['regional_politics', 'regional_entertainment', 'regional_culture', 
                             'regional_controversy', 'regional_opinion', 'regional_religion', 'regional_language']
        regional_df = df[df['category'].isin(regional_categories)]
        
        logger.info(f"\n🎭 SPECIALIZED CONTENT BREAKDOWN:")
        logger.info(f"   Entertainment articles: {len(entertainment_df):,} ({(len(entertainment_df)/len(df)*100):.1f}%)")
        logger.info(f"   Geopolitics articles: {len(geopolitics_df):,} ({(len(geopolitics_df)/len(df)*100):.1f}%)")
        logger.info(f"   AI/Tech articles: {len(tech_df):,} ({(len(tech_df)/len(df)*100):.1f}%)")
        logger.info(f"   Regional Indian articles: {len(regional_df):,} ({(len(regional_df)/len(df)*100):.1f}%)")
        
        # Show top categories in each domain
        if len(entertainment_df) > 0:
            ent_categories = entertainment_df['category'].value_counts().head(3)
            logger.info(f"   Top entertainment: {dict(ent_categories)}")
            
        if len(geopolitics_df) > 0:
            geo_categories = geopolitics_df['category'].value_counts().head(3)
            logger.info(f"   Top geopolitics: {dict(geo_categories)}")
            
        if len(tech_df) > 0:
            tech_categories_top = tech_df['category'].value_counts().head(3)
            logger.info(f"   Top tech: {dict(tech_categories_top)}")
            
        if len(regional_df) > 0:
            regional_categories_top = regional_df['category'].value_counts().head(3)
            logger.info(f"   Top regional: {dict(regional_categories_top)}")
            
            # Show regional language breakdown
            if 'language' in regional_df.columns:
                regional_langs = regional_df['language'].value_counts().head(5)
                logger.info(f"   Regional languages: {dict(regional_langs)}")
            
            # Show regional sources
            if 'region' in regional_df.columns:
                regional_regions = regional_df['region'].value_counts().head(5)
                logger.info(f"   Top regions: {dict(regional_regions)}")
        
        logger.info(f"\n💾 Dataset saved at: data/raw/news_articles.csv")
        logger.info(f"🚀 Ready for RAG pipeline integration!")
        
    except Exception as e:
        logger.error(f"Error showing final statistics: {str(e)}")

def main():
    """Run all scrapers in sequence"""
    
    start_time = datetime.now()
    
    logger.info("🌟" * 25)
    logger.info("🚀 ULTRA-COMPREHENSIVE NEWS DATA COLLECTION")
    logger.info("🌟" * 25)
    logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Target: ~9500 articles across 5 specialized domains")
    
    total_articles_collected = 0
    
    # Phase 1: General News Collection
    logger.info(f"\n" + "="*60)
    logger.info(f"📰 PHASE 1: GENERAL NEWS COLLECTION")
    logger.info(f"="*60)
    logger.info(f"🎯 Target: ~1000 articles from 32 diverse sources")
    logger.info(f"⏱️ Estimated time: 42-45 minutes")
    
    try:
        general_collector = NewsDataCollector()
        general_articles = general_collector.collect_and_append(
            articles_per_source=30,
            batch_size=3
        )
        total_articles_collected += general_articles
        logger.info(f"✅ Phase 1 complete: {general_articles} general articles collected")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 failed: {str(e)}")
        logger.info("🔄 Continuing to Phase 2...")
    
    # Short break between phases
    logger.info(f"\n⏳ Taking a 30-second break between phases...")
    time.sleep(30)
    
    # Phase 2: Entertainment News Collection  
    logger.info(f"\n" + "="*60)
    logger.info(f"🎭 PHASE 2: ENTERTAINMENT NEWS COLLECTION")
    logger.info(f"="*60)
    logger.info(f"🎯 Target: ~2000 entertainment articles")
    logger.info(f"📺 Focus: Hollywood, Bollywood, Music, Celebrities")
    logger.info(f"⏱️ Estimated time: 30-35 minutes")
    
    try:
        entertainment_collector = EntertainmentNewsCollector()
        entertainment_articles = entertainment_collector.collect_entertainment_batch(
            articles_per_source=70,  # Higher for entertainment diversity
            batch_size=3
        )
        total_articles_collected += entertainment_articles
        logger.info(f"✅ Phase 2 complete: {entertainment_articles} entertainment articles collected")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 failed: {str(e)}")
        logger.info("🔄 Continuing to Phase 3...")
    
    # Short break between phases
    logger.info(f"\n⏳ Taking a 30-second break between phases...")
    time.sleep(30)
    
    # Phase 3: Geopolitics News Collection
    logger.info(f"\n" + "="*60)
    logger.info(f"🌍 PHASE 3: GEOPOLITICS NEWS COLLECTION")
    logger.info(f"="*60)
    logger.info(f"🎯 Target: ~2000 geopolitics articles")
    logger.info(f"🏛️ Focus: International Relations, Diplomacy, Conflicts")
    logger.info(f"🌐 Languages: English, Spanish, French, German")
    logger.info(f"⏱️ Estimated time: 45-50 minutes")
    
    try:
        geopolitics_collector = GeopoliticsNewsCollector()
        geopolitics_articles = geopolitics_collector.collect_geopolitics_batch(
            articles_per_source=60,
            batch_size=3
        )
        total_articles_collected += geopolitics_articles
        logger.info(f"✅ Phase 3 complete: {geopolitics_articles} geopolitics articles collected")
        
    except Exception as e:
        logger.error(f"❌ Phase 3 failed: {str(e)}")
        logger.info("🔄 Continuing to Phase 4...")
    
    # Short break between phases
    logger.info(f"\n⏳ Taking a 30-second break between phases...")
    time.sleep(30)
    
    # Phase 4: AI & Technology News Collection
    logger.info(f"\n" + "="*60)
    logger.info(f"🤖 PHASE 4: AI & TECHNOLOGY NEWS COLLECTION")
    logger.info(f"="*60)
    logger.info(f"🎯 Target: ~3000 AI/tech articles")
    logger.info(f"💻 Focus: AI, Startups, Cybersecurity, Innovation")
    logger.info(f"🌐 Languages: English, Spanish, French, German")
    logger.info(f"⏱️ Estimated time: 40-45 minutes")
    
    try:
        ai_tech_collector = AITechNewsCollector()
        ai_tech_articles = ai_tech_collector.collect_ai_tech_batch(
            articles_per_source=50,
            batch_size=4
        )
        total_articles_collected += ai_tech_articles
        logger.info(f"✅ Phase 4 complete: {ai_tech_articles} AI/tech articles collected")
        
    except Exception as e:
        logger.error(f"❌ Phase 4 failed: {str(e)}")
        logger.info("📊 Showing results from completed phases...")
    
    # Short break between phases
    logger.info(f"\n⏳ Taking a 30-second break between phases...")
    time.sleep(30)
    
    # Phase 5: Regional Indian Languages Collection
    logger.info(f"\n" + "="*60)
    logger.info(f"🏛️ PHASE 5: REGIONAL INDIAN LANGUAGES COLLECTION")
    logger.info(f"="*60)
    logger.info(f"🎯 Target: ~1500 regional articles")
    logger.info(f"🌍 Focus: Political Drama, Cultural Claims, Regional Controversies")
    logger.info(f"📰 Sources: Vernacular portals across 10 major Indian languages")
    logger.info(f"⏱️ Estimated time: 35-40 minutes")
    
    try:
        regional_collector = RegionalIndianNewsCollector()
        regional_articles = regional_collector.collect_regional_batch(
            articles_per_source=40,
            batch_size=3
        )
        total_articles_collected += regional_articles
        logger.info(f"✅ Phase 5 complete: {regional_articles} regional articles collected")
        
    except Exception as e:
        logger.error(f"❌ Phase 5 failed: {str(e)}")
        logger.info("📊 Showing results from completed phases...")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"\n" + "🎉" * 20)
    logger.info(f"🏁 COLLECTION MISSION COMPLETE!")
    logger.info(f"🎉" * 20)
    logger.info(f"⏰ Started: {start_time.strftime('%H:%M:%S')}")
    logger.info(f"⏰ Finished: {end_time.strftime('%H:%M:%S')}")
    logger.info(f"⏱️ Total duration: {duration}")
    logger.info(f"📊 Total articles collected: {total_articles_collected:,}")
    
    # Show comprehensive statistics
    show_final_statistics()
    
    logger.info(f"\n🚀 NEXT STEPS:")
    logger.info(f"   1. Run RAG pipeline to index all articles")
    logger.info(f"   2. Test fact-checking across all domains")
    logger.info(f"   3. Evaluate multilingual claim verification")
    logger.info(f"   4. Test entertainment vs geopolitics vs tech claims")
    logger.info(f"   5. Benchmark against existing fact-checking datasets")

if __name__ == "__main__":
    main()