# OCR Language Configuration Guide

## EasyOCR Language Compatibility

EasyOCR has specific language combination rules. Not all languages can be combined together.

## Compatible Language Combinations

### ✅ Safe Combinations (Recommended)

**English + Hindi:**
```python
processor = InputProcessor(ocr_languages=['en', 'hi'])
```

**English + Chinese:**
```python
processor = InputProcessor(ocr_languages=['en', 'ch_sim'])  # Simplified Chinese
processor = InputProcessor(ocr_languages=['en', 'ch_tra'])  # Traditional Chinese
```

**English + Japanese:**
```python
processor = InputProcessor(ocr_languages=['en', 'ja'])
```

**English + Korean:**
```python
processor = InputProcessor(ocr_languages=['en', 'ko'])
```

**English + Thai:**
```python
processor = InputProcessor(ocr_languages=['en', 'th'])
```

**English + Vietnamese:**
```python
processor = InputProcessor(ocr_languages=['en', 'vi'])
```

### ⚠️ Special Cases

**Arabic (ONLY with English):**
```python
# ✅ Correct
processor = InputProcessor(ocr_languages=['ar', 'en'])

# ❌ Wrong - will fail
processor = InputProcessor(ocr_languages=['ar', 'hi', 'en'])
```

**Persian/Farsi (ONLY with English):**
```python
processor = InputProcessor(ocr_languages=['fa', 'en'])
```

**Urdu (ONLY with English):**
```python
processor = InputProcessor(ocr_languages=['ur', 'en'])
```

## Default Configuration

The default configuration uses **English + Hindi** which is safe and covers many use cases:

```python
processor = InputProcessor()  # Uses ['en', 'hi'] by default
```

## Supported Languages

EasyOCR supports 80+ languages:

**Latin Script:**
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)

**Asian Languages:**
- Chinese Simplified (ch_sim)
- Chinese Traditional (ch_tra)
- Japanese (ja)
- Korean (ko)
- Thai (th)
- Vietnamese (vi)

**Indian Languages:**
- Hindi (hi)
- Bengali (bn)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)

**Middle Eastern:**
- Arabic (ar) - Only with English
- Persian/Farsi (fa) - Only with English
- Urdu (ur) - Only with English

**And 60+ more...**

## Custom Configuration Examples

### For Indian Documents (English + Hindi)
```python
processor = InputProcessor(ocr_languages=['en', 'hi'])
```

### For Arabic Documents
```python
processor = InputProcessor(ocr_languages=['ar', 'en'])
```

### For Chinese Documents
```python
processor = InputProcessor(ocr_languages=['en', 'ch_sim'])
```

### For Japanese Documents
```python
processor = InputProcessor(ocr_languages=['en', 'ja'])
```

### For European Documents (English + French + German)
```python
processor = InputProcessor(ocr_languages=['en', 'fr', 'de'])
```

## Error Handling

The InputProcessor automatically falls back to English-only if the language combination fails:

```python
# If this fails due to incompatible languages
processor = InputProcessor(ocr_languages=['ar', 'hi', 'en'])

# It automatically falls back to
# processor = InputProcessor(ocr_languages=['en'])
```

## Best Practices

1. **Use English + One Other Language** - Most reliable
2. **Test Your Combination** - Some combinations may not work
3. **Use Default for General Use** - `['en', 'hi']` works for most cases
4. **Specify Languages Based on Your Data** - If you know your documents are in specific languages

## Performance Notes

- **More languages = Slower OCR** - Each additional language increases processing time
- **Recommended: 2-3 languages maximum** - Balance between coverage and speed
- **GPU Recommended** - OCR is much faster with GPU (set `gpu=True`)

## Troubleshooting

### Error: "Arabic is only compatible with English"

**Problem:** Trying to combine Arabic with other languages besides English

**Solution:**
```python
# ❌ Wrong
processor = InputProcessor(ocr_languages=['ar', 'hi', 'en'])

# ✅ Correct
processor = InputProcessor(ocr_languages=['ar', 'en'])
```

### Error: Language combination not supported

**Solution:** Use English + one other language, or use default configuration

### Slow OCR Performance

**Solutions:**
1. Reduce number of languages
2. Use GPU: `InputProcessor(ocr_languages=['en', 'hi'])` with GPU enabled
3. Preprocess images to improve quality

## Summary

✅ **Default:** `['en', 'hi']` - Safe and covers many use cases

✅ **Arabic/Persian/Urdu:** Only combine with English

✅ **Best Practice:** English + 1-2 other languages maximum

✅ **Automatic Fallback:** Falls back to English-only if combination fails
