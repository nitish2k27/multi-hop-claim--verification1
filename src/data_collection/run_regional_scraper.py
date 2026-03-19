"""
Standalone Regional Indian Language Scraper
Test script for collecting vernacular drama and political content
"""

import logging
from datetime import datetime
from regional_indian_scraper import RegionalIndianNewsCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run regional Indian language scraper standalone"""
    
    start_time = datetime.now()
    
    logger.info("🏛️" * 25)
    logger.info("🚀 REGIONAL INDIAN LANGUAGE NEWS COLLECTION")
    logger.info("🏛️" * 25)
    logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Target: Regional drama, politics, and cultural claims")
    logger.info(f"🌍 Languages: Hindi, Tamil, Telugu, Marathi, Bengali, Malayalam, Kannada, Gujarati, Punjabi, Odia")
    
    try:
        regional_collector = RegionalIndianNewsCollector()
        
        # Collect regional articles
        articles_collected = regional_collector.collect_regional_batch(
            articles_per_source=50,  # More articles per source for comprehensive coverage
            batch_size=3
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n🎉 REGIONAL COLLECTION COMPLETE!")
        logger.info(f"⏰ Duration: {duration}")
        logger.info(f"📊 Articles collected: {articles_collected:,}")
        
        logger.info(f"\n🚀 NEXT STEPS:")
        logger.info(f"   1. Review collected regional content quality")
        logger.info(f"   2. Test language detection on vernacular articles")
        logger.info(f"   3. Validate political drama and cultural claims")
        logger.info(f"   4. Integrate with main RAG pipeline")
        logger.info(f"   5. Test multilingual fact-checking capabilities")
        
    except Exception as e:
        logger.error(f"❌ Regional collection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()