"""
Download required NLTK data
"""

import nltk

print("\n" + "="*80)
print("DOWNLOADING NLTK DATA")
print("="*80)

packages = [
    'punkt',
    'punkt_tab',
    'stopwords',
    'averaged_perceptron_tagger',
]

print("\nDownloading required packages...")

for package in packages:
    print(f"\n-> Downloading {package}...")
    try:
        nltk.download(package, quiet=False)
        print(f"  [OK] {package} downloaded")
    except Exception as e:
        print(f"  [ERROR] Failed to download {package}: {e}")

print("\n" + "="*80)
print("[SUCCESS] NLTK data setup complete")
print("="*80)