"""
scripts/set_inference_url.py
─────────────────────────────
Run this once after starting your Google Colab inference server.
Saves the Colab URL to configs/inference_url.txt so the pipeline
reads it automatically.

Usage:
    python scripts/set_inference_url.py "https://xxxx.googleusercontent.com"

Or run without argument and it will ask you to paste the URL:
    python scripts/set_inference_url.py
"""

import sys
import requests
from pathlib import Path

URL_FILE = Path("configs/inference_url.txt")


def save_and_verify(url: str):
    url = url.strip().rstrip("/")

    if not url.startswith("http"):
        print(f"[ERROR] URL must start with http or https. Got: {url}")
        sys.exit(1)

    URL_FILE.parent.mkdir(exist_ok=True)
    URL_FILE.write_text(url)
    print(f"\n[OK] URL saved to {URL_FILE}")
    print(f"     {url}")

    print("\nVerifying connection to server...")
    try:
        # ── Colab tunnels require this header ─────────────────────────────
        headers = {"ngrok-skip-browser-warning": "true"}

        resp = requests.get(f"{url}/health", headers=headers, timeout=15)
        if resp.status_code == 200:
            info = resp.json()
            print(f"[OK] Server is reachable and healthy")
            print(f"     Model:   {info.get('model', 'unknown')}")
            print(f"     Adapter: {info.get('adapter', 'unknown')}")
            print(f"     Device:  {info.get('device', 'unknown')}")
            print("\nYou can now run the pipeline:")
            print("  python scripts/test_full_pipeline.py")
        else:
            print(f"[WARN] Server responded with status {resp.status_code}")
            print("       URL saved but server may not be fully ready yet.")

    except requests.exceptions.Timeout:
        print("[WARN] Request timed out.")
        print("       Colab tunnel may still be initializing — wait 30s and retry.")
    except requests.exceptions.ConnectionError:
        print("[WARN] Could not connect to server.")
        print("       URL saved. Make sure the Colab notebook is running.")
    except Exception as e:
        print(f"[WARN] Connection check failed: {e}")
        print("       URL saved anyway.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        print("Paste the Colab public URL from your inference notebook:")
        print("(It looks like: https://xxxx-xx-xx-xx-xx.googleusercontent.com)")
        url = input("\nURL: ").strip()

    save_and_verify(url)