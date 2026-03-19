# Language Detection Model

## FastText Model

This directory contains the FastText language detection model.

**Model:** `lid.176.bin` (126 MB)

## Automatic Download

The model will be automatically downloaded on first use:

```python
from src.preprocessing.language_detector import LanguageDetector

# Model downloads automatically if not present
detector = LanguageDetector()
```

## Manual Download

If you want to download manually:

```bash
# Create directory
mkdir -p models/language_detection

# Download model
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
     -O models/language_detection/lid.176.bin
```

Or on Windows:
```powershell
# Using PowerShell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin" -OutFile "models/language_detection/lid.176.bin"
```

## Model Info

- **Size:** 126 MB
- **Languages:** 176
- **Accuracy:** 99.1%
- **Source:** Facebook Research FastText

## Note

This file is NOT included in the Git repository due to its size (> 100 MB GitHub limit).
It will be downloaded automatically when needed.
