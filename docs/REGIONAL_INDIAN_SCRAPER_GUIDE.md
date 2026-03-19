# Regional Indian Language Scraper Guide

## Overview

The Regional Indian Language Scraper is designed to collect vernacular content focused on political drama, cultural claims, and regional controversies from top Indian language news portals. This scraper implements recommendations from Gemini AI for targeting high-quality vernacular sources.

## Key Features

### 🎯 Targeted Content Collection
- **Political Drama**: Regional politics, election controversies, coalition dynamics
- **Cultural Claims**: Festival disputes, temple controversies, tradition debates  
- **Entertainment Drama**: Cinema-politics overlap, celebrity controversies
- **Social Issues**: Caste dynamics, language activism, community tensions

### 🌍 Language Coverage
- **Hindi**: Dainik Bhaskar, Amar Ujala, Navbharat Times
- **Tamil**: Dinamalar, Thanthi TV, Vikatan
- **Telugu**: Eenadu, Sakshi, ABN Andhra Jyothy
- **Marathi**: Lokmat, Sakal, ABP Majha
- **Bengali**: Anandabazar Patrika, Sangbad Pratidin
- **Malayalam**: Mathrubhumi, Malayala Manorama
- **Kannada**: Prajavani, TV9 Kannada
- **Gujarati**: Sandesh, Gujarat Samachar
- **Punjabi**: Ajit Jalandhar, PTC News
- **Odia**: Sambad, Dharitri
- **Multi-language**: OneIndia (covers all major languages)

### 🔍 Smart Filtering
- URL pattern matching for drama/politics content
- Content length validation (minimum 300 characters)
- Duplicate detection and removal
- Language-specific category mapping

## Usage

### Standalone Regional Collection
```python
from src.data_collection.regional_indian_scraper import RegionalIndianNewsCollector

collector = RegionalIndianNewsCollector()
articles = collector.collect_regional_batch(
    articles_per_source=50,
    batch_size=3
)
```

### Command Line Usage
```bash
# Run regional scraper only
python src/data_collection/run_regional_scraper.py

# Run as part of full pipeline
python src/data_collection/run_all_scrapers.py
```

## Configuration

The scraper uses `configs/regional_scraper_config.yaml` for customization:

### Key Settings
- `articles_per_source`: Number of articles per news source (default: 40)
- `batch_size`: Sources processed simultaneously (default: 3)
- `delay_between_requests`: Rate limiting in seconds (default: 3)
- `min_article_length`: Minimum content length (default: 300 chars)

### Target Patterns
The scraper looks for specific URL patterns to identify drama/politics content:
- Politics: `/politics/`, `/rajkiya/`, `/election/`, `/chunav/`
- Culture: `/entertainment/`, `/cinema/`, `/controversy/`, `/vivad/`
- Opinion: `/opinion/`, `/sampadkiya/`, `/editorial/`
- Social: `/caste/`, `/jati/`, `/temple/`, `/mandir/`

## Output Format

Articles are saved with these fields:
- `url`: Article URL
- `title`: Article headline
- `text`: Full article content
- `source`: News source name
- `publish_date`: Publication date
- `language`: Language code (hi, ta, te, etc.)
- `category`: Content category (regional_politics, regional_entertainment, etc.)
- `region`: Indian region (North India, Tamil Nadu, etc.)
- `content_type`: Always "vernacular_drama"

## Regional Categories

### Political Content
- `regional_politics`: State-level political news
- `regional_controversy`: Political scandals and disputes
- `regional_opinion`: Editorial and opinion pieces

### Cultural Content  
- `regional_entertainment`: Cinema and celebrity news
- `regional_culture`: Festival and tradition coverage
- `regional_religion`: Temple and religious controversies

### Social Issues
- `regional_language`: Language activism and disputes
- `regional_general`: Other regional content

## Quality Assurance

### Content Validation
- Minimum article length enforcement
- Drama/politics URL pattern matching
- Duplicate URL detection
- Language consistency checking

### Rate Limiting
- 3-second delay between article requests
- 5-second delay between sources
- 15-second delay between batches
- Respectful crawling practices

### Error Handling
- Graceful failure for individual articles
- Batch-level error recovery
- Comprehensive logging
- Fallback source mechanisms

## Integration with RAG Pipeline

### Metadata Enhancement
Regional articles include additional metadata for RAG retrieval:
- **Region**: Geographic classification for location-based queries
- **Language**: Enables multilingual fact-checking
- **Category**: Allows domain-specific retrieval
- **Content Type**: Identifies vernacular drama content

### Multilingual Support
- Language detection integration
- Unicode text handling
- Script-specific processing
- Cross-language claim verification

## Performance Metrics

### Expected Collection Rates
- **Articles per source**: 30-50 drama/politics articles
- **Total sources**: 25+ vernacular portals
- **Collection time**: 35-40 minutes for full run
- **Success rate**: 70-80% (due to filtering)

### Quality Indicators
- High drama content percentage
- Regional political coverage
- Cultural controversy inclusion
- Language distribution balance

## Troubleshooting

### Common Issues
1. **Low article count**: Check RSS feed availability
2. **Language detection errors**: Verify source language mapping
3. **Duplicate content**: Review URL normalization
4. **Rate limiting**: Adjust delay settings

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Feed Validation
Test individual RSS feeds:
```python
collector = RegionalIndianNewsCollector()
articles = collector.collect_from_rss('dainik_bhaskar', max_articles=5)
```

## Future Enhancements

### Planned Features
- **Drama Scoring**: ML-based content drama intensity scoring
- **Political Heat Index**: Automated political controversy detection
- **Sentiment Analysis**: Regional sentiment tracking
- **Trend Detection**: Emerging controversy identification

### Advanced Filtering
- **Claim Detection**: Automatic factual claim extraction
- **Bias Analysis**: Source bias identification
- **Credibility Scoring**: Source reliability metrics
- **Cross-reference Validation**: Multi-source claim verification

## Best Practices

### For RAG Integration
1. Use regional metadata for geographic queries
2. Leverage language tags for multilingual retrieval
3. Filter by category for domain-specific searches
4. Consider content type for drama/controversy queries

### For Fact-Checking
1. Cross-reference claims across languages
2. Validate regional political claims with multiple sources
3. Check cultural claims against authoritative sources
4. Monitor for coordinated misinformation campaigns

### For Performance
1. Run during off-peak hours for better success rates
2. Monitor RSS feed health regularly
3. Adjust batch sizes based on network conditions
4. Use backup sources for critical languages

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Validate RSS feed accessibility
3. Review configuration settings
4. Test with smaller batch sizes

The Regional Indian Language Scraper provides comprehensive coverage of vernacular political drama and cultural controversies, essential for building robust multilingual fact-checking systems.