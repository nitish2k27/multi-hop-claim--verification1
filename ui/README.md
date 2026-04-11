# NLP Pipeline UI

Simple user interface to simulate the complete NLP pipeline from input to analysis for LLM generation.

## Features

### Input Types Supported
- **Text Input**: Direct text/claim entry
- **Document Upload**: PDF, DOCX, TXT files
- **Voice Input**: Transcribed text simulation
- **URL/Link**: Web content fetching (demo mode)

### Analysis Pipeline
1. **Language Detection**: Auto-detect or specify source language
2. **Text Preprocessing**: Clean and prepare text
3. **Claim Detection**: Identify factual claims
4. **Stance Detection**: Analyze stance towards claims
5. **Entity Extraction**: Extract named entities
6. **RAG Search**: Retrieve relevant context
7. **LLM Preparation**: Format data for generation

### Output
- Complete analysis results in JSON format
- Ready-to-use data for LLM generation
- Downloadable analysis reports

## Quick Start

### Option 1: Web UI (Streamlit)
```bash
# From project root
cd ui
python run_ui.py
```

Then open: http://localhost:8501

### Option 2: Command Line
```bash
# From project root
cd ui
python simple_cli_demo.py
```

### Option 3: Manual Streamlit
```bash
# Install requirements
pip install -r ui/requirements.txt

# Run UI
streamlit run ui/nlp_simulator.py
```

## Usage Examples

### Text Analysis
1. Select "Text" input type
2. Enter your text/claim
3. Configure analysis options
4. Click "Run NLP Analysis"
5. View results and download JSON

### Document Processing
1. Select "Document Upload"
2. Upload PDF/DOCX/TXT file
3. System extracts and processes text
4. Run analysis on extracted content
5. Get structured results

### Voice Input Simulation
1. Select "Voice Input"
2. Paste transcribed text
3. Process as voice input type
4. Get analysis with voice metadata

## Configuration Options

### Language Settings
- **Source Language**: Auto-detect or specify (en, hi, es, fr, de, zh, ar)
- **Target Language**: Output language for results

### Analysis Components
- **Claim Detection**: Enable/disable claim identification
- **Stance Detection**: Enable/disable stance analysis
- **Entity Extraction**: Enable/disable NER
- **RAG Search**: Enable/disable context retrieval

## Output Format

```json
{
  "timestamp": "2026-04-03T...",
  "input_metadata": {
    "type": "text|document|voice|url",
    "source": "...",
    "filename": "..." // for documents
  },
  "analysis_steps": [
    {
      "step": "language_detection",
      "detected_language": "en",
      "confidence": 0.95
    },
    {
      "step": "claim_detection",
      "result": {...}
    }
  ],
  "final_output": {
    "input_text": "...",
    "detected_language": "en",
    "ready_for_llm": true
  }
}
```

## Integration with Main Pipeline

The UI integrates with:
- `src/enhanced_main_pipeline.py` - Main fact verification
- `src/document_processing/document_handler.py` - Document processing
- `src/voice_processing/speech_handler.py` - Voice processing
- `src/multilingual/multilingual_pipeline.py` - Language support

## Troubleshooting

### Import Errors
Make sure you're running from the project root and all dependencies are installed:
```bash
pip install -r requirements_enhanced.txt
pip install -r ui/requirements.txt
```

### Component Initialization
If components fail to initialize:
1. Check that models are downloaded
2. Verify configuration files exist
3. Run system status check: `python scripts/check_system_status.py`

### Performance
For better performance:
- Use smaller documents for testing
- Disable unused analysis components
- Check system resources

## Development

### Adding New Input Types
1. Add option to `input_type` selectbox
2. Implement handler in `run_analysis()` method
3. Update metadata structure

### Adding Analysis Steps
1. Add checkbox to sidebar
2. Implement analysis in `run_analysis()`
3. Update results structure
4. Add display logic

### Customizing Output
Modify the `final_output` structure in `run_analysis()` to match your LLM requirements.