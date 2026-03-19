# Input Processor Capabilities

## Overview
Your `InputProcessor` is now fully capable of handling the use case where users provide BOTH a claim AND supporting documents for verification.

## Supported Input Types

### 1. Text
- Direct text input
- Encoding fix and cleaning
- Language detection

### 2. Voice/Audio
- Formats: .wav, .mp3, .m4a, .flac, .ogg
- Transcription using Whisper
- Automatic language detection
- Confidence scoring

### 3. PDF Documents
- Text extraction
- Table extraction
- OCR for scanned PDFs
- Multi-page support

### 4. Word Documents (.docx)
- Paragraph extraction
- Table extraction
- Structure preservation

### 5. Excel Spreadsheets (.xlsx, .xls) ✨ NEW
- Multi-sheet support
- Data extraction
- Column and row statistics
- Text representation of data

### 6. CSV Files ✨ NEW
- Data extraction
- Column statistics
- Text representation

### 7. Images
- Formats: .jpg, .jpeg, .png, .bmp, .tiff
- OCR using EasyOCR
- Multi-language support
- Confidence scoring
- Image preprocessing for better accuracy

### 8. URLs/Web Pages
- Article extraction using newspaper3k
- BeautifulSoup fallback
- Metadata extraction (title, authors, publish date)

## Key Methods

### 1. `process(input_data, input_type, metadata=None)`
Process a single input of any supported type.

**Example:**
```python
result = processor.process('claim.pdf', 'pdf')
```

### 2. `process_with_context(claim_input, context_documents, claim_type='text')` ⭐
Process a claim with user-provided context documents.

**Example:**
```python
result = processor.process_with_context(
    claim_input="Our revenue grew 50% in Q3",
    context_documents=[
        {'data': 'report.pdf', 'type': 'pdf', 'name': 'Q3 Report'},
        {'data': 'transcript.docx', 'type': 'docx', 'name': 'Earnings Call'},
        {'data': 'data.xlsx', 'type': 'xlsx', 'name': 'Financial Data'}
    ],
    claim_type='text'
)
```

**Returns:**
```python
{
    'claim': {
        'text': str,
        'language': str,
        'source_type': str,
        'metadata': dict
    },
    'context_documents': [
        {
            'text': str,
            'language': str,
            'source_type': str,
            'metadata': dict,
            'priority': 'high',  # ⭐ User-provided = high priority
            'is_user_provided': True  # ⭐ Distinguishes from auto-retrieved
        }
    ],
    'processing_mode': 'claim_with_context',
    'metadata': {
        'num_context_docs': int,
        'context_languages': list,
        'total_context_length': int
    }
}
```

### 3. `process_batch(inputs)` 
Process multiple inputs in batch.

**Example:**
```python
results = processor.process_batch([
    {'data': 'claim.txt', 'type': 'text'},
    {'data': 'report.pdf', 'type': 'pdf'},
    {'data': 'audio.wav', 'type': 'voice'}
])
```

## Use Case: Claim + Supporting Documents

### Scenario
User provides:
- **CLAIM:** "Our company's revenue grew 50% in Q3"
- **DOCUMENTS:**
  - financial_report_Q3.pdf
  - earnings_call_transcript.docx
  - competitor_analysis.xlsx

### How It Works

1. **Process Claim Separately**
   - Claim is extracted and cleaned
   - Language detected
   - Stored in `result['claim']`

2. **Process Each Document**
   - Each document processed based on its type
   - Text extracted and cleaned
   - Language detected
   - Stored in `result['context_documents']`

3. **Mark Documents as High Priority**
   - Each document gets `priority: 'high'`
   - Each document gets `is_user_provided: True`
   - Documents indexed in order with `context_doc_index`

4. **Pass to RAG System**
   - RAG system can prioritize user-provided documents
   - Higher weight in retrieval
   - Used first for verification

## Integration with RAG

```python
# After processing
result = processor.process_with_context(claim, documents)

# Index context documents with high priority
for doc in result['context_documents']:
    if doc['is_user_provided']:
        rag_system.index(
            text=doc['text'],
            metadata=doc['metadata'],
            priority='high'  # ⭐ Prioritize in retrieval
        )

# Retrieve evidence for claim
evidence = rag_system.retrieve(
    query=result['claim']['text'],
    prioritize_user_docs=True
)

# Verify claim
verification = verify_claim(
    claim=result['claim']['text'],
    evidence=evidence
)
```

## Features

✅ **Multiple Input Types** - Text, voice, PDF, DOCX, XLSX, CSV, images, URLs

✅ **Claim + Context Processing** - Separate processing with priority marking

✅ **Batch Processing** - Process multiple inputs efficiently

✅ **Language Detection** - Automatic language detection for all inputs

✅ **Error Handling** - Graceful error handling with detailed logging

✅ **Metadata Extraction** - Rich metadata for each processed input

✅ **Priority Marking** - User-provided documents marked as high priority

✅ **Multi-format Tables** - Extract tables from PDF, DOCX, XLSX

✅ **OCR Support** - Handle scanned documents and images

✅ **Web Scraping** - Extract content from URLs

## Requirements

All required packages are in `requirements.txt`:
- ftfy (text cleaning)
- langdetect (language detection)
- whisper (voice transcription)
- soundfile (audio handling)
- pdfplumber (PDF processing)
- python-docx (Word documents)
- openpyxl (Excel files) ✨ NEW
- pandas (data processing)
- easyocr (OCR)
- opencv-python (image processing)
- newspaper3k (web scraping)
- beautifulsoup4 (HTML parsing)

## Example Usage

See `examples/claim_with_context_example.py` for a complete working example.
