"""
Requirements Checker
Checks if all packages in requirements.txt are installed
"""

import importlib
import sys

# Map package names to import names (they differ sometimes)
PACKAGE_TO_IMPORT = {
    'torch': 'torch',
    'transformers': 'transformers',
    'datasets': 'datasets',
    'sentence-transformers': 'sentence_transformers',
    'accelerate': 'accelerate',
    'safetensors': 'safetensors',
    'scikit-learn': 'sklearn',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'chromadb': 'chromadb',
    'rank-bm25': 'rank_bm25',
    'spacy': 'spacy',
    'nltk': 'nltk',
    'textstat': 'textstat',
    'httpx': 'httpx',
    'requests': 'requests',
    'deep-translator': 'deep_translator',
    'openai-whisper': 'whisper',
    'SpeechRecognition': 'speech_recognition',
    'gtts': 'gtts',
    'pygame': 'pygame',
    'librosa': 'librosa',
    'soundfile': 'soundfile',
    'PyPDF2': 'PyPDF2',
    'python-docx': 'docx',
    'Pillow': 'PIL',
    'pytesseract': 'pytesseract',
    'python-dotenv': 'dotenv',
    'pyyaml': 'yaml',
    'tqdm': 'tqdm',
    'python-dateutil': 'dateutil',
    'pytest': 'pytest',
    'pytest-cov': 'pytest_cov',
    'tf-keras': 'tf_keras',
}

# Which ones are critical vs optional
CRITICAL = {
    'torch', 'transformers', 'datasets', 'sentence-transformers',
    'accelerate', 'scikit-learn', 'pandas', 'numpy', 'chromadb',
    'rank-bm25', 'nltk', 'requests', 'deep-translator',
    'python-dotenv', 'pyyaml', 'tqdm', 'python-dateutil'
}

OPTIONAL = {
    'openai-whisper', 'SpeechRecognition', 'gtts', 'pygame',
    'librosa', 'soundfile', 'PyPDF2', 'python-docx', 'Pillow',
    'pytesseract', 'spacy', 'textstat', 'scipy', 'safetensors',
    'httpx', 'pytest', 'pytest-cov', 'tf-keras'
}


def check_package(package_name):
    import_name = PACKAGE_TO_IMPORT.get(package_name, package_name)
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def main():
    print("\n" + "="*60)
    print("REQUIREMENTS CHECKER")
    print("="*60)

    installed_critical   = []
    missing_critical     = []
    installed_optional   = []
    missing_optional     = []

    # Check critical
    print("\n📦 CRITICAL PACKAGES:")
    print("-"*60)
    for pkg in sorted(CRITICAL):
        ok, version = check_package(pkg)
        if ok:
            print(f"  ✅ {pkg:<30} {version}")
            installed_critical.append(pkg)
        else:
            print(f"  ❌ {pkg:<30} NOT INSTALLED")
            missing_critical.append(pkg)

    # Check optional
    print("\n📦 OPTIONAL PACKAGES:")
    print("-"*60)
    for pkg in sorted(OPTIONAL):
        ok, version = check_package(pkg)
        if ok:
            print(f"  ✅ {pkg:<30} {version}")
            installed_optional.append(pkg)
        else:
            print(f"  ⚠️  {pkg:<30} not installed (optional)")
            missing_optional.append(pkg)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Critical:  {len(installed_critical)}/{len(CRITICAL)} installed")
    print(f"Optional:  {len(installed_optional)}/{len(OPTIONAL)} installed")

    if missing_critical:
        print(f"\n❌ MISSING CRITICAL PACKAGES — install these:")
        for pkg in missing_critical:
            print(f"   pip install {pkg}")

    if missing_optional:
        print(f"\n⚠️  Missing optional packages (install if needed):")
        for pkg in missing_optional:
            print(f"   pip install {pkg}")

    if not missing_critical:
        print(f"\n✅ All critical packages are installed!")
        print(f"   Your core pipeline will work correctly.")
    else:
        print(f"\n❌ Fix missing critical packages before running the pipeline.")

    print("="*60)

    return len(missing_critical) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)