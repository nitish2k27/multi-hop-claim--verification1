# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download Spacy model:**
```bash
python -m spacy download en_core_web_sm
```

## Basic Usage

### 1. Process Claim Only

```python
from src.preprocessing.input_processor import InputProcessor

processor = InputProcessor()

result = processor.process(
    "Our company's revenue grew 50% in Q3",
    'text'
)

print(result['text'])
print(result['language'])
```

### 2. Process Claim + Context Documents ⭐

```python
from src.preprocessing.input_processor import InputProcessor

processor = InputProcessor()

result = processor.process_with_context(
    claim_input="Our company's revenue grew 50% in Q3",
    context_documents=[
        {
            'data': 'data/financial_report.pdf',
            'type': 'pdf',
            'name': 'Q3 Financial Report'
        },
        {
            'data': 'data/earnings_call.docx',
            'type': 'docx',
            'name': 'Earnings Call Transcript'
        }
    ]
)

# Access claim
print("CLAIM:", result['claim']['text'])

# Access context documents (marked as high priority)
for doc in result['context_documents']:
    print(f"\nDocument: {doc['metadata']['document_name']}")
    print(f"Priority: {doc['priority']}")  # 'high'
    print(f"User Provided: {doc['is_user_provided']}")  # True
    print(f"Preview: {doc['text'][:100]}...")
```

### 3. Complete Verification Pipeline

```python
from src.main_pipeline import FactVerificationPipeline

pipeline = FactVerificationPipeline()

result = pipeline.verify_claim_with_context(
    claim_input="Our revenue grew 50% in Q3",
    context_documents=[
        {'data': 'data/financial_report.pdf', 'type': 'pdf'}
    ]
)

print(f"Claim: {result['claim']}")
print(f"Verdict: {result['verification']['verdict']}")
print(f"Confidence: {result['verification']['confidence']}")

# Evidence (user docs appear FIRST with highest priority)
for ev in result['evidence']:
    print(f"\n[{ev['source_type']}] Priority: {ev['priority']}")
    print(f"Source: {ev['source']}")
    print(f"Text: {ev['text'][:100]}...")
```

## Supported Input Types

### Text
```python
result = processor.process("Some claim text", 'text')
```

### Voice/Audio
```python
result = processor.process('audio.wav', 'voice')
```

### PDF
```python
result = processor.process('document.pdf', 'pdf')
```

### Word Document
```python
result = processor.process('document.docx', 'docx')
```

### Excel Spreadsheet
```python
result = processor.process('data.xlsx', 'xlsx')
```

### CSV File
```python
result = processor.process('data.csv', 'csv')
```

### Image (OCR)
```python
result = processor.process('screenshot.png', 'image')
```

### URL/Web Page
```python
result = processor.process('https://example.com/article', 'url')
```

## Batch Processing

```python
inputs = [
    {'data': 'claim1.txt', 'type': 'text'},
    {'data': 'report.pdf', 'type': 'pdf'},
    {'data': 'audio.wav', 'type': 'voice'}
]

results = processor.process_batch(inputs)

for result in results:
    if 'error' in result:
        print(f"Failed: {result['error']}")
    else:
        print(f"Success: {result['text'][:50]}...")
```

## Examples

Run the examples:

```bash
# Example with context documents
python examples/claim_with_context_example.py

# Test suite
python tests/test_input_with_context.py
```

## Key Features

✅ **8 Input Types:** Text, voice, PDF, DOCX, XLSX, CSV, images, URLs

✅ **Context Support:** Process claim + user-provided documents

✅ **Priority System:** User documents get highest priority (1.0)

✅ **Batch Processing:** Process multiple inputs efficiently

✅ **Language Detection:** Automatic language detection for all inputs

✅ **OCR Support:** Handle scanned PDFs and images

✅ **Table Extraction:** Extract tables from PDF, DOCX, XLSX

## Documentation

- **Complete Workflow:** `docs/WORKFLOW_WITH_CONTEXT.md`
- **Capabilities:** `docs/INPUT_PROCESSOR_CAPABILITIES.md`
- **Implementation:** `IMPLEMENTATION_SUMMARY.md`

## Troubleshooting

### Issue: fasttext installation fails on Windows
**Solution:** Comment out fasttext in requirements.txt (already done)

### Issue: Whisper model download is slow
**Solution:** Use smaller model: `InputProcessor(whisper_model_size="tiny")`

### Issue: OCR languages not loading
**Solution:** Reduce languages: `InputProcessor(ocr_languages=['en'])`

## Next Steps

1. Implement NLP pipeline for claim analysis
2. Implement verification model for stance detection
3. Add vector database integration (ChromaDB/FAISS)
4. Add web search integration
5. Implement explanation generation

## Support

For issues or questions, refer to:
- `docs/WORKFLOW_WITH_CONTEXT.md` - Complete workflow guide
- `examples/claim_with_context_example.py` - Working examples
- `tests/test_input_with_context.py` - Test cases
