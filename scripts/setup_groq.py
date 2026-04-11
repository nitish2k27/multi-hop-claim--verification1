"""
scripts/setup_groq.py

One-time setup: saves your Groq API key and verifies it works.

Usage:
    python scripts/setup_groq.py "gsk_your_key_here"

Get free key at: console.groq.com (just email signup, instant access)
Free tier gives: 500 requests/day, 14400 tokens/minute
"""

import sys
import requests
from pathlib import Path

TOKEN_FILE = Path("configs/groq_token.txt")


def setup(api_key):
    api_key = api_key.strip()

    if not api_key.startswith("gsk_"):
        print("[WARN] Groq API keys usually start with gsk_")
        print("       Proceeding anyway...")

    # Save key
    TOKEN_FILE.parent.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(api_key, encoding="utf-8")
    print("[OK] Key saved to " + str(TOKEN_FILE))

    # Verify key works
    print("\nVerifying key with Groq API...")
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers,
            timeout=10,
        )

        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            print("[OK] API key is valid")
            print("\nAvailable models:")
            for m in sorted(models):
                print("  " + m)
            print("\nRecommended for fact verification: llama-3.1-70b-versatile")
            print("\nYou can now run:")
            print("  python scripts/test_groq_pipeline.py \"Your claim here\"")

        elif resp.status_code == 401:
            print("[FAIL] Invalid API key")
            print("       Check your key at console.groq.com")
            sys.exit(1)
        else:
            print("[FAIL] Groq API returned: " + str(resp.status_code))
            print("       Response: " + resp.text[:200])
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("[FAIL] Cannot reach Groq API")
        print("       Check your internet connection")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        key = sys.argv[1]
    else:
        print("Get your free Groq API key at: console.groq.com")
        key = input("Paste your Groq API key: ").strip()

    setup(key)