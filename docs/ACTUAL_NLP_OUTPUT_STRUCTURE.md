# Actual NLP Pipeline Output Structure

## Real Output Format from Your System

Based on the code analysis, here's the **exact JSON structure** your NLP pipeline produces:

### NLP Pipeline Output (`nlp.analyze()`)
```json
{
  "text": "The Indian government launched a new scheme in 2024",
  "language": "en",
  "analysis": {
    "claim_detection": {
      "is_claim": true,
      "confidence": 0.892,
      "label": "CLAIM",
      "model_type": "trained"
    },
    "entities": {
      "total_entities": 3,
      "counts": {
        "ORG": 1,
        "DATE": 1,
        "MISC": 1
      },
      "entities": {
        "ORG": [
          {
            "text": "Indian government",
            "start": 4,
            "end": 21,
            "score": 0.9998,
            "entity_group": "ORG"
          }
        ],
        "DATE": [
          {
            "text": "2024",
            "start": 50,
            "end": 54,
            "score": 0.9995,
            "entity_group": "DATE"
          }
        ],
        "MISC": [
          {
            "text": "scheme",
            "start": 36,
            "end": 42,
            "score": 0.8876,
            "entity_group": "MISC"
          }
        ]
      }
    },
    "linked_entities": [
      {
        "word": "Indian government",
        "entity_group": "ORG",
        "score": 0.9998,
        "wikidata_id": "Q668",
        "wikidata_label": "India",
        "confidence": 0.95
      }
    ],
    "temporal": {
      "dates": [
        {
          "text": "2024",
          "normalized": "2024-01-01",
          "type": "year",
          "start": 50,
          "end": 54
        }
      ],
      "total_count": 1
    }
  }
}
```

### RAG Pipeline Output (`rag.verify_claim()`)
```json
{
  "claim": "The Indian government launched a new scheme in 2024",
  "verdict": "SUPPORTED",
  "confidence": 78.5,
  "evidence": [
    {
      "document": "The Indian government announced the launch of the PM-KISAN scheme extension in March 2024, providing direct benefit transfers to farmers across the country...",
      "source": "pib.gov.in",
      "url": "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=2024123",
      "stance": "SUPPORTS",
      "stance_confidence": 0.89,
      "credibility": {
        "total_score": 0.92,
        "domain_score": 0.90,
        "recency_score": 0.98,
        "type_score": 0.85,
        "domain": "pib.gov.in",
        "tier": "HIGH",
        "is_high_credibility": true
      },
      "metadata": {
        "publish_date": "2024-03-15T10:30:00Z",
        "collection": "news_articles",
        "chunk_id": "pib_scheme_2024_chunk_1"
      }
    },
    {
      "document": "Economic Times reported that several new government schemes were launched in 2024 as part of the budget announcements...",
      "source": "economictimes.indiatimes.com",
      "url": "https://economictimes.indiatimes.com/news/economy/policy/new-schemes-2024",
      "stance": "SUPPORTS",
      "stance_confidence": 0.76,
      "credibility": {
        "total_score": 0.80,
        "domain_score": 0.80,
        "recency_score": 0.85,
        "type_score": 0.75,
        "domain": "economictimes.indiatimes.com",
        "tier": "MEDIUM",
        "is_high_credibility": true
      }
    }
  ],
  "aggregation": {
    "verdict": "SUPPORTED",
    "confidence": 78.5,
    "support_percentage": 85.0,
    "refute_percentage": 5.0,
    "neutral_percentage": 10.0,
    "num_supports": 2,
    "num_refutes": 0,
    "num_neutral": 1
  },
  "llm_context": "CLAIM TO VERIFY:\nThe Indian government launched a new scheme in 2024\n\nEVIDENCE ANALYSIS:\n1. [SUPPORTS] (Credibility: 0.92, Source: pib.gov.in)\n   The Indian government announced the launch of the PM-KISAN scheme extension in March 2024...\n\n2. [SUPPORTS] (Credibility: 0.80, Source: economictimes.indiatimes.com)\n   Economic Times reported that several new government schemes were launched in 2024...\n\nVERDICT CALCULATION:\n- Support: 85.0% (2 pieces)\n- Refute: 5.0% (0 pieces)\n- Neutral: 10.0% (1 piece)\n- Preliminary Verdict: SUPPORTED\n- Confidence: 78.5%",
  "ready_for_llm": true
}
```

### Enhanced Main Pipeline Output (`pipeline.verify_claim()`)
```json
{
  "verdict": "SUPPORTED",
  "verdict_translated": "SUPPORTED",
  "confidence": 78.5,
  "explanation": "Based on evidence from government sources and news reports, the claim is supported. The Indian government did launch new schemes in 2024.",
  "evidence": [
    {
      "document": "The Indian government announced the launch of the PM-KISAN scheme extension...",
      "source": "pib.gov.in",
      "stance": "SUPPORTS",
      "stance_confidence": 0.89,
      "credibility": {
        "total_score": 0.92,
        "tier": "HIGH"
      }
    }
  ],
  "metadata": {
    "user_language": "en",
    "original_claim": "The Indian government launched a new scheme in 2024",
    "english_claim": "The Indian government launched a new scheme in 2024",
    "processing_language": "en"
  },
  "input_metadata": {
    "input_type": "text",
    "original_input": "The Indian government launched a new scheme in 2024",
    "detected_language": "en",
    "processing_timestamp": "2024-04-03T15:30:45Z"
  },
  "ready_for_llm": true
}
```

## Key Field Names for Integration

### Core Fields
- `verdict`: "SUPPORTED" | "REFUTED" | "INSUFFICIENT" | "NOT_A_CLAIM"
- `confidence`: Float (0-100)
- `evidence`: Array of evidence objects
- `ready_for_llm`: Boolean flag

### NLP Analysis Fields
- `analysis.claim_detection.is_claim`: Boolean
- `analysis.entities.entities`: Object with entity types as keys
- `analysis.linked_entities`: Array of Wikidata-linked entities
- `analysis.temporal.dates`: Array of temporal expressions

### Evidence Fields
- `evidence[].document`: Text content
- `evidence[].source`: Domain name
- `evidence[].stance`: "SUPPORTS" | "REFUTES" | "NEUTRAL"
- `evidence[].credibility.total_score`: Float (0-1)
- `evidence[].credibility.tier`: "HIGH" | "MEDIUM" | "LOW"

### RAG-Specific Fields
- `llm_context`: Pre-formatted string for LLM input
- `aggregation.support_percentage`: Float
- `aggregation.num_supports`: Integer

## Gen AI Integration Code

```python
def process_nlp_output(nlp_result):
    """Process actual NLP pipeline output for Gen AI"""
    
    # Extract key information
    verdict = nlp_result.get('verdict', 'UNKNOWN')
    confidence = nlp_result.get('confidence', 0)
    evidence = nlp_result.get('evidence', [])
    
    # Get high-credibility sources
    high_cred_sources = [
        ev for ev in evidence 
        if ev.get('credibility', {}).get('is_high_credibility', False)
    ]
    
    # Build context for LLM
    if 'llm_context' in nlp_result:
        # Use pre-built context
        context = nlp_result['llm_context']
    else:
        # Build context manually
        context = f"Claim: {nlp_result.get('metadata', {}).get('original_claim', '')}\n"
        context += f"Verdict: {verdict} (Confidence: {confidence}%)\n"
        context += f"Evidence: {len(evidence)} pieces found\n"
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'context': context,
        'sources': [ev.get('source') for ev in high_cred_sources],
        'language': nlp_result.get('metadata', {}).get('user_language', 'en')
    }
```

## Error Handling

### Error Response Format
```json
{
  "verdict": "ERROR",
  "confidence": 0.0,
  "explanation": "Processing failed: Model not found",
  "error": "FileNotFoundError: Model file not found",
  "input_metadata": {
    "input_type": "text",
    "error_timestamp": "2024-04-03T15:30:45Z"
  },
  "ready_for_llm": false
}
```

This is the **exact structure** your pipeline produces. Use these field names in your Gen AI integration code to avoid any mismatches.