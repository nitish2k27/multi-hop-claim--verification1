"""
src/generation/report_generator_groq.py
Changes from original:
  - Uses prompt_builder for multilingual output
  - Model corrected to llama-3.3-70b-versatile
  - generate() accepts user_language parameter
"""
import logging, time, requests
from pathlib import Path
from typing import Any, Optional

logger     = logging.getLogger(__name__)
TOKEN_FILE = Path("configs/groq_token.txt")
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class ReportGeneratorGroq:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or self._load_key()
        self.model   = model
        self.headers = {"Authorization": f"Bearer {self.api_key}",
                        "Content-Type":  "application/json"}
        print(f"[OK] ReportGeneratorGroq ready — {self.model}")

    def _load_key(self) -> str:
        if TOKEN_FILE.exists():
            k = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if k: return k
        raise ValueError("No Groq key. Run: python scripts/setup_groq.py YOUR_KEY")

    def save_key(self, key: str):
        TOKEN_FILE.parent.mkdir(exist_ok=True)
        TOKEN_FILE.write_text(key.strip(), encoding="utf-8")
        self.api_key = key.strip()
        self.headers["Authorization"] = f"Bearer {self.api_key}"

    def health_check(self) -> bool:
        try:
            r = requests.get("https://api.groq.com/openai/v1/models",
                             headers=self.headers, timeout=10)
            if r.status_code == 200:
                print("[OK] Groq key valid — models: " +
                      ", ".join(m["id"] for m in r.json().get("data", [])[:5]))
                return True
            print(f"[FAIL] Groq returned {r.status_code}")
            return False
        except Exception as e:
            print(f"[FAIL] {e}"); return False

    def generate(self, nlp_output: Any, user_language: str = "en",
                 max_tokens: int = 1500, temperature: float = 0.2,
                 retries: int = 3) -> str:
        """
        Generate report from RAG pipeline output.
        user_language: ISO 639-1 code — report will be in this language.
        """
        from src.generation.prompt_builder import extract_llm_context, build_groq_messages
        ctx      = extract_llm_context(nlp_output)
        messages = build_groq_messages(ctx, user_language)
        payload  = {"model": self.model, "messages": messages,
                    "max_tokens": max_tokens, "temperature": temperature,
                    "top_p": 0.85, "stream": False}

        for attempt in range(1, retries + 2):
            try:
                r = requests.post(GROQ_URL, headers=self.headers,
                                  json=payload, timeout=60)
                if r.status_code == 429:
                    wait = int(r.headers.get("retry-after", 30))
                    print(f"[WAIT] Rate limited — {wait}s"); time.sleep(wait); continue
                if r.status_code == 401:
                    raise PermissionError("Invalid Groq key")
                r.raise_for_status()
                report = r.json()["choices"][0]["message"]["content"].strip()
                if not report: raise ValueError("Empty response")
                tokens = r.json().get("usage", {}).get("total_tokens", "?")
                print(f"[OK] Groq report: {len(report.split())} words, "
                      f"{tokens} tokens, lang={user_language}")
                return report
            except requests.exceptions.Timeout:
                if attempt <= retries:
                    print(f"[RETRY] Timeout {attempt}"); time.sleep(10); continue
                raise TimeoutError("Groq timed out")
            except requests.exceptions.ConnectionError:
                raise ConnectionError("Cannot reach Groq — check internet")
        raise RuntimeError("Groq failed after all retries")