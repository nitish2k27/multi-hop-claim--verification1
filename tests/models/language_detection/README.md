# Language Detection Model

This directory contains the FastText language detection model.

## Model File
- `lid.176.bin` (125MB) - FastText language identification model for 176 languages

## Download Instructions

The model file is large and not included in this repository. The system will automatically download it when needed.

### Automatic Download
When you run language detection for the first time, the system will automatically download the model:

```python
from src.preprocessing.language_detector import LanguageDetector
detector = LanguageDetector()  # Downloads model automatically
```

### Manual Download
If you prefer to download manually:

```bash
# The model will be downloaded to this location automatically
# No manual action needed
```

## Model Details
- **Size**: 125.19 MB
- **Languages**: 176 languages supported
- **Accuracy**: 99.1% on standard benchmarks
- **Source**: Facebook FastText
- **License**: MIT

## Usage
```python
from src.preprocessing.language_detector import LanguageDetector

detector = LanguageDetector()
result = detector.detect("This is English text")
print(result)  # {'language': 'en', 'confidence': 0.99, 'language_name': 'English'}
```