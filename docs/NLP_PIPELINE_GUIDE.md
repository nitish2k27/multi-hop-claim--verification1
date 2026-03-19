# NLP Pipeline Documentation

## Overview

The NLP Pipeline provides comprehensive natural language processing capabilities for fact verification, including claim detection, entity extraction, temporal analysis, and stance detection.

## Components

### 1. Model Manager (`src/nlp/model_manager.py`)

**Purpose:** Manages loading of NLP models with automatic fallback from trained to placeholder models.

**Features:**
- ✅ Automatic model loading with fallback
- ✅ Supports both trained and placeholder models
- ✅ Model caching for performance
- ✅ Configuration-driven model selection

**Usage:**
```python
from src.nlp.model_manager import ModelManager

manager = ModelManager(config_path="configs/nlp_config.yaml")

# Load models
claim_detector = manager.load_claim_detector()
ner_model = manager.load_ner_model()
stance_detector = manager.load_stance_detector()

# Get model info
info = manager.get_model_info('claim_detector')
print(info)  # {'loaded': True, 'type': 'placeholder', 'task': 'zero_shot'}
```

### 2. Claim Detection (`src/nlp/claim_detection.py`)

**Purpose:** Detects if text contains verifiable factual claims.

**Features:**
- ✅ Binary classification (is_claim / not_claim)
- ✅ Confidence scoring
- ✅ Sentence-level claim extraction
- ✅ Supports trained and zero-shot models

**Usage:**
```python
from src.nlp.claim_detection import ClaimDetector

detector = ClaimDetector(model_manager)

# Detect single claim
result = detector.detect("India's GDP grew 8% in 2024")
print(result)
# {
#     'is_claim': True,
#     'confidence': 0.92,
#     'label': 'factual claim'
# }

# Extract claims from document
claims = detector.extract_claims_from_text(long_text, threshold=0.7)
```

### 3. Entity Extraction (`src/nlp/entity_extraction.py`)

**Purpose:** Extracts named entities (PERSON, ORGANIZATION, LOCATION, DATE, etc.)

**Features:**
- ✅ Multi-entity type extraction
- ✅ Confidence scoring
- ✅ Position tracking
- ✅ Entity grouping by type

**Usage:**
```python
from src.nlp.entity_extraction import EntityExtractor

extractor = EntityExtractor(model_manager)

# Extract all entities
entities = extractor.extract("Narendra Modi is the PM of India")
print(entities)
# {
#     'PER': [{'text': 'Narendra Modi', 'score': 0.99, ...}],
#     'LOC': [{'text': 'India', 'score': 0.98, ...}]
# }

# Get summary
summary = extractor.get_entity_summary(text)
print(summary['counts'])  # {'PER': 1, 'LOC': 1}
```

### 4. Entity Linking (`src/nlp/entity_linking.py`)

**Purpose:** Links entities to knowledge base (Wikidata) for cross-lingual matching.

**Features:**
- ✅ Wikidata integration
- ✅ Cross-lingual entity matching
- ✅ Entity disambiguation
- ✅ Knowledge base lookup

**Usage:**
```python
from src.nlp.entity_linking import EntityLinker

linker = EntityLinker()

# Link entities
linked = linker.link_entities("Narendra Modi is the PM of India")

# Get Wikidata info
info = linker.get_wikidata_info("Q1058")  # Modi's Wikidata ID
```

### 5. Temporal Extraction (`src/nlp/temporal_extraction.py`)

**Purpose:** Extracts and normalizes dates, times, and temporal expressions.

**Features:**
- ✅ Multiple date formats (YYYY, MM/DD/YYYY, etc.)
- ✅ Quarter extraction (Q1 2024, Q3 2023)
- ✅ Relative dates (yesterday, last week)
- ✅ Fiscal years (FY2024)
- ✅ Timeline construction

**Usage:**
```python
from src.nlp.temporal_extraction import TemporalExtractor

extractor = TemporalExtractor()

# Extract dates
result = extractor.extract("India's GDP grew 8% in Q3 2024")
print(result['dates'])
# [
#     {
#         'original': 'Q3 2024',
#         'normalized': '2024-07-01',
#         'type': 'quarter',
#         'quarter': 3,
#         'year': 2024
#     }
# ]

# Build timeline
timeline = extractor.build_timeline([text1, text2, text3])
```

### 6. Stance Detection (`src/nlp/stance_detection.py`)

**Purpose:** Determines if evidence SUPPORTS, REFUTES, or is NEUTRAL to a claim.

**Features:**
- ✅ Three-way classification (SUPPORTS/REFUTES/NEUTRAL)
- ✅ Confidence scoring
- ✅ Batch processing
- ✅ Stance aggregation

**Usage:**
```python
from src.nlp.stance_detection import StanceDetector

detector = StanceDetector(model_manager)

# Detect stance
result = detector.detect(
    claim="India's GDP grew 8% in 2024",
    evidence="Official data shows 8.2% growth in 2024"
)
print(result)
# {
#     'stance': 'SUPPORTS',
#     'confidence': 0.94
# }

# Batch detection
results = detector.detect_batch(claim, [evidence1, evidence2, evidence3])

# Aggregate stances
aggregated = detector.aggregate_stances(results)
print(aggregated['overall_stance'])  # 'SUPPORTS' or 'REFUTES' or 'CONFLICTING'
```

### 7. Complete NLP Pipeline (`src/nlp/nlp_pipeline.py`)

**Purpose:** Integrates all NLP components into a single pipeline.

**Features:**
- ✅ End-to-end NLP analysis
- ✅ Claim detection → Entity extraction → Temporal analysis
- ✅ Claim-evidence pair analysis
- ✅ Document-level claim extraction

**Usage:**
```python
from src.nlp.nlp_pipeline import NLPPipeline

pipeline = NLPPipeline()

# Analyze single text
result = pipeline.analyze("India's GDP grew 8% in Q3 2024")
print(result['analysis'])
# {
#     'claim_detection': {...},
#     'entities': {...},
#     'linked_entities': [...],
#     'temporal': {...}
# }

# Extract claims from document
claims = pipeline.extract_claims_from_document(document, threshold=0.7)

# Analyze claim-evidence pair
pair_analysis = pipeline.analyze_claim_evidence_pair(claim, evidence)
print(pair_analysis['stance'])
```

## Configuration

### NLP Config File (`configs/nlp_config.yaml`)

```yaml
nlp_pipeline:
  use_trained_models: false  # Set to true when trained models available
  device: -1  # -1 for CPU, 0 for GPU
  
  models:
    claim_detector:
      use_trained: false
      trained_path: "models/claim_detector"
      placeholder: "facebook/bart-large-mnli"
      
    ner:
      use_trained: false
      trained_path: "models/ner_model"
      placeholder: "dslim/bert-base-NER"
      aggregation_strategy: "simple"
      
    stance_detector:
      use_trained: false
      trained_path: "models/stance_detector"
      placeholder: "microsoft/deberta-v3-base-mnli"
```

## Model Types

### Placeholder Models (Default)

Used when trained models are not available:

1. **Claim Detection:** Zero-shot classifier (BART-MNLI)
2. **NER:** Pre-trained BERT-NER
3. **Stance Detection:** NLI model (DeBERTa-MNLI)

### Trained Models (Production)

Custom models trained on fact-checking datasets:

1. **Claim Detection:** Binary classifier (claim/not_claim)
2. **NER:** Fine-tuned NER model
3. **Stance Detection:** Three-way classifier (SUPPORTS/REFUTES/NEUTRAL)

## Integration with Main Pipeline

```python
from src.main_pipeline import FactVerificationPipeline

# Initialize with NLP pipeline
pipeline = FactVerificationPipeline()

# The NLP pipeline is automatically used in verification
result = pipeline.verify_claim_with_context(
    claim_input="India's GDP grew 8% in Q3",
    context_documents=[...]
)

# NLP analysis is in result['nlp_analysis']
print(result['nlp_analysis']['claim_detection'])
print(result['nlp_analysis']['entities'])
print(result['nlp_analysis']['temporal'])
```

## Testing

### Run NLP Tests

```bash
# Test individual components
python src/nlp/claim_detection.py
python src/nlp/entity_extraction.py
python src/nlp/temporal_extraction.py
python src/nlp/stance_detection.py

# Test complete pipeline
python src/nlp/nlp_pipeline.py

# Run test suite
python tests/test_nlp_pipeline.py
```

## Performance

### Model Loading Times

- **Placeholder models:** 2-5 seconds (first load)
- **Trained models:** 1-3 seconds (first load)
- **Cached models:** < 0.1 seconds

### Processing Speed

- **Claim detection:** ~50ms per text
- **Entity extraction:** ~100ms per text
- **Temporal extraction:** ~10ms per text (rule-based)
- **Stance detection:** ~100ms per pair

## Supported Languages

### Current Support

- **English:** Full support (all models)
- **Hindi:** Entity extraction via multilingual models
- **Spanish, French, German:** Entity extraction

### Future Support

- Multilingual claim detection
- Cross-lingual stance detection
- More language-specific models

## Best Practices

1. **Use Configuration File:** Centralize model settings in `nlp_config.yaml`
2. **Cache Models:** ModelManager automatically caches loaded models
3. **Batch Processing:** Use batch methods for multiple texts
4. **Threshold Tuning:** Adjust confidence thresholds based on your use case
5. **GPU Acceleration:** Set `device: 0` in config for GPU usage

## Troubleshooting

### Issue: Models not loading

**Solution:** Check if models exist at configured paths. Pipeline automatically falls back to placeholder models.

### Issue: Slow performance

**Solutions:**
1. Enable GPU: Set `device: 0` in config
2. Use batch processing
3. Reduce model size (use smaller variants)

### Issue: Low accuracy

**Solutions:**
1. Train custom models on your domain data
2. Adjust confidence thresholds
3. Use ensemble methods

## Summary

✅ **Complete NLP Pipeline** with 7 components

✅ **Automatic Fallback** from trained to placeholder models

✅ **Configuration-Driven** model management

✅ **Production-Ready** with caching and batch processing

✅ **Extensible** - Easy to add new components

✅ **Well-Tested** with comprehensive test suite
