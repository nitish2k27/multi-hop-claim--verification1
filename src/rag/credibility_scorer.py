"""
Source Credibility Scoring
Assesses reliability of information sources
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List  # ← FIXED
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class CredibilityScorer:
    """
    Score source credibility based on multiple factors
    
    Factors:
    1. Domain authority
    2. Publication recency
    3. Source type (official, news, blog, etc.)
    """
    
    def __init__(self):
        """Initialize credibility scorer"""
        
        # Domain authority scores (manually curated)
        # Scale: 0.0 to 1.0
        self.domain_scores = {
            # Tier 1: High credibility (0.90-1.0)
            'reuters.com': 0.95,
            'apnews.com': 0.94,
            'bbc.com': 0.93,
            'bbc.co.uk': 0.93,
            'npr.org': 0.92,
            'nature.com': 0.95,
            'science.org': 0.95,
            'who.int': 0.96,
            'worldbank.org': 0.94,
            'imf.org': 0.94,
            'gov.uk': 0.92,
            'gov.in': 0.90,
            
            # Tier 2: Good credibility (0.80-0.89)
            'nytimes.com': 0.88,
            'wsj.com': 0.88,
            'theguardian.com': 0.87,
            'washingtonpost.com': 0.86,
            'economist.com': 0.87,
            'ft.com': 0.87,
            'bloomberg.com': 0.86,
            'aljazeera.com': 0.82,
            'dw.com': 0.83,
            'france24.com': 0.82,
            'thehindu.com': 0.84,
            'indianexpress.com': 0.83,
            'timesofindia.indiatimes.com': 0.80,
            
            # Tier 3: Moderate credibility (0.70-0.79)
            'cnn.com': 0.78,
            'foxnews.com': 0.70,
            'msnbc.com': 0.72,
            'ndtv.com': 0.76,
            'hindustantimes.com': 0.75,
            
            # Fact-checking sites (high credibility)
            'factcheck.org': 0.93,
            'snopes.com': 0.90,
            'politifact.com': 0.91,
            'boomlive.in': 0.88,
            'altnews.in': 0.87,

            'economic_times_tech': 0.82,
            'mint': 0.81,
            'ndtv': 0.76,
            'cnn_world': 0.78,
            'wired': 0.80,
            'indian_express': 0.83,
        }
        
        # Source type scores
        self.source_type_scores = {
            'government': 0.85,
            'academic': 0.90,
            'news_agency': 0.88,
            'newspaper': 0.80,
            'fact_checker': 0.90,
            'international_org': 0.88,
            'blog': 0.50,
            'social_media': 0.30,
            'unknown': 0.50
        }
        
        logger.info("✓ CredibilityScorer initialized")
    
    def score(
        self,
        url: Optional[str] = None,
        source: Optional[str] = None,
        publish_date: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate credibility score for a source
        
        Args:
            url: Source URL
            source: Source domain name
            publish_date: Publication date (ISO format)
            source_type: Type of source
        
        Returns:
            Credibility information
        """
        # Extract domain from URL or use provided source
        if url:
            domain = self._extract_domain(url)
        elif source:
            domain = source
        else:
            domain = 'unknown'
        
        # Get domain score
        domain_score = self.domain_scores.get(domain, 0.50)
        
        # Get recency score
        recency_score = self._calculate_recency_score(publish_date)
        
        # Get source type score
        type_score = self.source_type_scores.get(source_type, 0.50) if source_type else 0.50
        
        # Calculate weighted total
        # Domain: 50%, Recency: 30%, Type: 20%
        total_score = (
            domain_score * 0.5 +
            recency_score * 0.3 +
            type_score * 0.2
        )
        
        # Determine tier
        tier = self._get_tier(total_score)
        
        return {
            'total_score': total_score,
            'domain_score': domain_score,
            'recency_score': recency_score,
            'type_score': type_score,
            'domain': domain,
            'tier': tier,
            'is_high_credibility': total_score >= 0.80
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www.
            domain = domain.replace('www.', '')
            return domain
        except:
            return 'unknown'
    
    def _calculate_recency_score(self, publish_date: Optional[str]) -> float:
        """
        Calculate recency score
        
        More recent = higher score
        Decays over time (1 year half-life)
        """
        if not publish_date:
            return 0.50  # Default if no date
        
        try:
            # Parse date
            if isinstance(publish_date, str):
                pub_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
            else:
                pub_date = publish_date
            
            # Calculate age in days
            now = datetime.now(pub_date.tzinfo) if pub_date.tzinfo else datetime.now()
            age_days = (now - pub_date).days
            
            # Recency score (exponential decay)
            # 1.0 for today, 0.5 at 1 year, 0.25 at 2 years
            recency_score = max(0.2, 1.0 - (age_days / 365) * 0.5)
            
            return recency_score
            
        except Exception as e:
            logger.debug(f"Failed to parse date: {publish_date}, {str(e)}")
            return 0.50
    
    def _get_tier(self, score: float) -> str:
        """Get credibility tier from score"""
        if score >= 0.85:
            return 'HIGH'
        elif score >= 0.70:
            return 'MEDIUM'
        elif score >= 0.50:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def score_batch(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score multiple sources
        
        Args:
            sources: List of source dicts with url, source, publish_date, etc.
        
        Returns:
            List of sources with credibility scores added
        """
        scored = []
        
        for source in sources:
            credibility = self.score(
                url=source.get('url'),
                source=source.get('source'),
                publish_date=source.get('publish_date'),
                source_type=source.get('source_type')
            )
            
            # Add credibility to source
            source_copy = source.copy()
            source_copy['credibility'] = credibility
            scored.append(source_copy)
        
        return scored
    
    def add_domain(self, domain: str, score: float):
        """
        Add or update domain score
        
        PLUGIN POINT: Add custom domain scores
        
        Args:
            domain: Domain name
            score: Credibility score (0.0-1.0)
        """
        if not 0 <= score <= 1:
            raise ValueError("Score must be between 0 and 1")
        
        self.domain_scores[domain] = score
        logger.info(f"Added/updated domain: {domain} = {score}")


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scorer = CredibilityScorer()
    
    print("\n" + "="*60)
    print("CREDIBILITY SCORING TESTS")
    print("="*60)
    
    test_sources = [
        {
            'url': 'https://www.reuters.com/article/india-gdp',
            'publish_date': '2024-10-15'
        },
        {
            'source': 'randomBlog.com',
            'publish_date': '2020-01-01'
        },
        {
            'url': 'https://www.bbc.com/news/india',
            'publish_date': '2024-11-01'
        },
        {
            'source': 'unknown-site.net',
            'publish_date': None
        }
    ]
    
    for source in test_sources:
        result = scorer.score(**source)
        
        print(f"\nSource: {source.get('url') or source.get('source')}")
        print(f"  Total Score: {result['total_score']:.3f}")
        print(f"  Domain Score: {result['domain_score']:.3f}")
        print(f"  Recency Score: {result['recency_score']:.3f}")
        print(f"  Tier: {result['tier']}")
        print(f"  High Credibility: {result['is_high_credibility']}")