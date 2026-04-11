# NLP Pipeline Output Format for Gen AI Integration

## Overview
This document defines the exact output structure of the NLP fact verification pipeline, designed to feed directly into Gen AI models for response generation.

## Complete Output Structure

### Main Pipeline Output
```json
{
  "verdict": "SUPPORTED | REFUTED | INSUFFICIENT | NOT_A_CLAIM | ERROR",
  "verdict_translated": "समर्थित | खंडित | अपर्याप्त साक्ष्य | दावा नहीं",
  "confidence": 0.85,
  "explanation": "Based on evidence from Reuters and BBC, the claim is supported...",
  "evidence": [
    {
      "document": "India's GDP grew by 8.2% in Q3 2024 according to official statistics...",
      "source": "reuters.com",
      "url": "https://reuters.com/article/india-gdp-2024",
      "stance": "SUPPORTS",
      "stance_confidence": 0.92,
      "credibility": {
        "total_score": 0.95,
        "domain_score": 0.95,
        "recency_score": 0.98,
        "type_score": 0.88,
        "tier": "HIGH",
        "is_high_credibility": true
      },
      "metadata": {
        "publish_date": "2024-10-15T10:30:00Z",
        "author": "Economic Correspondent",
        "collection": "news_articles",
        "chunk_id": "reuters_gdp_2024_chunk_1"
      }
    }
  ],
  "aggregation": {
    "verdict": "SUPPORTED",
    "confidence": 85.2,
    "support_percentage": 75.0,
    "refute_percentage": 10.0,
    "neutral_percentage": 15.0,
    "num_supports": 3,
    "num_refutes": 1,
    "num_neutral": 1,
    "weighted_score": 0.852
  },
  "llm_context": "CLAIM TO VERIFY:\nIndia's GDP grew 8% in 2024\n\nEVIDENCE ANALYSIS:\n1. [SUPPORTS] (Credibility: 0.95, Source: reuters.com)\n   India's GDP grew by 8.2% in Q3 2024...",
  "metadata": {
    "user_language": "hi",
    "original_claim": "भारत की जीडीपी 2024 में 8% बढ़ी",
    "english_claim": "India's GDP grew 8% in 2024",
    "processing_language": "en"
  },
  "input_metadata": {
    "input_type": "text",
    "original_input": "भारत की जीडीपी 2024 में 8% बढ़ी",
    "detected_language": "hi",
    "processing_timestamp": "2024-04-03T15:30:45Z"
  },
  "audio_response": "/tmp/response_audio_12345.mp3",
  "ready_for_llm": true
}
```

## Key Components for Gen AI

### 1. Core Verification Results
```json
{
  "verdict": "SUPPORTED",           // Primary classification
  "confidence": 0.85,              // Overall confidence (0-1)
  "explanation": "...",             // Human-readable explanation
  "ready_for_llm": true            // Processing completion flag
}
```

### 2. Evidence Array (Rich Context)
```json
{
  "evidence": [
    {
      "document": "Full text excerpt...",
      "source": "reuters.com",
      "stance": "SUPPORTS",
      "stance_confidence": 0.92,
      "credibility": {
        "total_score": 0.95,
        "tier": "HIGH"
      }
    }
  ]
}
```

### 3. LLM-Ready Context String
```text
CLAIM TO VERIFY:
India's GDP grew 8% in 2024

EVIDENCE ANALYSIS:
1. [SUPPORTS] (Credibility: 0.95, Source: reuters.com)
   India's GDP grew by 8.2% in Q3 2024 according to official statistics...

2. [REFUTES] (Credibility: 0.70, Source: blog.example.com)
   Some analysts question the GDP figures citing methodology concerns...

VERDICT CALCULATION:
- Support: 75.0% (3 pieces)
- Refute: 10.0% (1 piece)
- Neutral: 15.0% (1 piece)
- Preliminary Verdict: SUPPORTED
- Confidence: 85.2%
```

## Input Type Variations

### Text Input
```json
{
  "input_metadata": {
    "input_type": "text",
    "original_input": "भारत की जीडीपी 2024 में 8% बढ़ी",
    "detected_language": "hi"
  }
}
```

### Voice Input
```json
{
  "input_metadata": {
    "input_type": "voice",
    "original_input": "/tmp/audio_input_12345.wav",
    "detected_language": "hi",
    "transcription_confidence": 0.94,
    "audio_duration": 3.2
  }
}
```

### Document Input
```json
{
  "input_metadata": {
    "input_type": "document",
    "original_input": "company_report_2024.pdf",
    "detected_language": "en",
    "document_metadata": {
      "filename": "company_report_2024.pdf",
      "file_size": 2048576,
      "pages_processed": 45,
      "claims_extracted": 12
    }
  }
}
```

## Gen AI Integration Patterns

### Pattern 1: Direct Response Generation
```python
def generate_response(nlp_output: Dict) -> str:
    """Generate response using NLP pipeline output"""
    
    prompt = f"""
    Based on the fact verification analysis:
    
    Claim: {nlp_output['metadata']['original_claim']}
    Verdict: {nlp_output['verdict']} (Confidence: {nlp_output['confidence']:.1f})
    
    Evidence Summary:
    {nlp_output['llm_context']}
    
    Generate a comprehensive response in {nlp_output['metadata']['user_language']}.
    """
    
    return llm.generate(prompt)
```

### Pattern 2: Structured Response with Citations
```python
def generate_structured_response(nlp_output: Dict) -> Dict:
    """Generate structured response with citations"""
    
    # Extract high-credibility evidence
    high_cred_evidence = [
        ev for ev in nlp_output['evidence'] 
        if ev['credibility']['is_high_credibility']
    ]
    
    response = {
        'summary': llm.generate_summary(nlp_output),
        'verdict': nlp_output['verdict_translated'],
        'confidence': nlp_output['confidence'],
        'citations': [
            {
                'source': ev['source'],
                'url': ev.get('url'),
                'credibility': ev['credibility']['tier']
            }
            for ev in high_cred_evidence
        ],
        'language': nlp_output['metadata']['user_language']
    }
    
    return response
```

### Pattern 3: Multi-Modal Response
```python
def generate_multimodal_response(nlp_output: Dict) -> Dict:
    """Generate text + audio response"""
    
    # Generate text response
    text_response = llm.generate(nlp_output['llm_context'])
    
    # Use existing audio if available, or generate new
    audio_response = nlp_output.get('audio_response')
    if not audio_response:
        audio_response = tts.generate(
            text_response, 
            language=nlp_output['metadata']['user_language']
        )
    
    return {
        'text': text_response,
        'audio': audio_response,
        'verdict': nlp_output['verdict_translated'],
        'confidence': nlp_output['confidence']
    }
```

## Error Handling

### Error Response Format
```json
{
  "verdict": "ERROR",
  "confidence": 0.0,
  "explanation": "Processing failed: Unable to extract text from document",
  "error": "PyPDF2.errors.PdfReadError: Invalid PDF structure",
  "input_metadata": {
    "input_type": "document",
    "error_timestamp": "2024-04-03T15:30:45Z"
  },
  "ready_for_llm": false
}
```

### Gen AI Error Handling
```python
def handle_nlp_error(nlp_output: Dict) -> str:
    """Handle NLP pipeline errors gracefully"""
    
    if nlp_output['verdict'] == 'ERROR':
        return f"""
        I apologize, but I encountered an issue processing your request: 
        {nlp_output['explanation']}
        
        Please try:
        - Rephrasing your question
        - Uploading a different file format
        - Checking your internet connection
        """
    
    return generate_normal_response(nlp_output)
```

## Performance Metrics

### Response Time Tracking
```json
{
  "performance": {
    "total_processing_time": 2.34,
    "language_detection_time": 0.12,
    "translation_time": 0.45,
    "rag_search_time": 1.23,
    "stance_detection_time": 0.34,
    "response_preparation_time": 0.20
  }
}
```

### Quality Indicators
```json
{
  "quality_indicators": {
    "evidence_count": 5,
    "high_credibility_sources": 3,
    "language_detection_confidence": 0.98,
    "translation_quality": "high",
    "stance_detection_accuracy": 0.91
  }
}
```

## Usage Examples

### Example 1: Simple Fact Check
```python
# Input
claim = "The Earth is flat"

# NLP Pipeline Output
nlp_result = pipeline.verify_claim(claim)

# Gen AI Integration
response = f"""
Claim: {claim}
Verdict: {nlp_result['verdict']} 
Confidence: {nlp_result['confidence']:.1f}

{nlp_result['explanation']}

Sources: {', '.join([ev['source'] for ev in nlp_result['evidence']])}
"""
```

### Example 2: Multilingual Processing
```python
# Input (Hindi)
claim = "पृथ्वी चपटी है"

# NLP Pipeline Output (includes translation)
nlp_result = pipeline.verify_claim(claim, input_type='text')

# Gen AI generates response in original language
response = llm.generate(f"""
Original claim: {nlp_result['metadata']['original_claim']}
English translation: {nlp_result['metadata']['english_claim']}
Analysis: {nlp_result['llm_context']}

Respond in Hindi with the verification result.
""")
```

### Example 3: Document Analysis
```python
# Input
document_path = "research_paper.pdf"

# NLP Pipeline Output (includes extracted claims)
nlp_result = pipeline.verify_claim(document_path, input_type='document')

# Gen AI generates comprehensive analysis
response = llm.generate(f"""
Document: {nlp_result['input_metadata']['document_metadata']['filename']}
Claims found: {len(nlp_result['evidence'])}

Analysis:
{nlp_result['llm_context']}

Provide a summary of the document's factual accuracy.
""")
```

## Integration Checklist

### Before Gen AI Processing
- ✅ Check `ready_for_llm` flag
- ✅ Validate `verdict` is not "ERROR"
- ✅ Ensure `llm_context` is present
- ✅ Check `confidence` threshold (e.g., > 0.3)

### During Gen AI Processing
- ✅ Use `llm_context` as primary input
- ✅ Include `verdict` and `confidence` in response
- ✅ Respect `user_language` for output
- ✅ Cite high-credibility sources from `evidence`

### After Gen AI Processing
- ✅ Include original `verdict` and `confidence`
- ✅ Maintain source attribution
- ✅ Handle multilingual output correctly
- ✅ Provide audio response if requested

This output format provides everything needed for sophisticated Gen AI integration while maintaining transparency, accuracy, and multilingual support.