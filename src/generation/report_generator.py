"""
src/generation/report_generator.py
Colab inference server client.
Change from original: generate() now accepts user_language and uses prompt_builder.
"""
import logging, requests, time
from pathlib import Path
from typing import Any, Optional

logger   = logging.getLogger(__name__)
URL_FILE = Path("configs/inference_url.txt")


class ReportGenerator:
    def __init__(self, inference_url: Optional[str] = None):
        self.inference_url = (inference_url or self._load_url()).rstrip("/")
        self._server_ok    = False

    def _load_url(self) -> str:
        if URL_FILE.exists():
            u = URL_FILE.read_text(encoding="utf-8").strip()
            if u: return u
        raise ValueError("No inference URL. Run: python scripts/set_inference_url.py URL")

    def health_check(self) -> bool:
        try:
            r = requests.get(self.inference_url + "/health",
                             headers={"ngrok-skip-browser-warning": "true"}, timeout=15)
            if r.status_code == 200:
                info = r.json()
                self._server_ok = True
                print(f"[OK] Colab server healthy — adapter: {info.get('adapter','?')}")
                return True
        except Exception as e:
            print(f"[FAIL] {e}")
        self._server_ok = False
        return False

    def generate(self, nlp_output: Any, user_language: str = "en",
                 max_new_tokens: int = 1000, temperature: float = 0.2,
                 retries: int = 2) -> str:
        """
        Generate report via Colab inference server.
        user_language: ISO code — prompt_builder instructs the model to respond in this language.
        """
        from src.generation.prompt_builder import extract_llm_context, build_mistral_prompt
        llm_context = extract_llm_context(nlp_output)
        # Build Mistral-formatted prompt with language instruction
        formatted   = build_mistral_prompt(llm_context, user_language)

        if not self._server_ok:
            if not self.health_check():
                raise ConnectionError(f"Cannot reach: {self.inference_url}\n"
                                      "Make sure Colab notebook is running.")

        for attempt in range(1, retries + 2):
            try:
                r = requests.post(
                    self.inference_url + "/generate",
                    json={"llm_context": llm_context, "prompt": formatted,
                          "max_new_tokens": max_new_tokens, "temperature": temperature},
                    headers={"ngrok-skip-browser-warning": "true"},
                    timeout=300,
                )
                r.raise_for_status()
                data   = r.json()
                report = data.get("report", "").strip()
                if not report: raise ValueError(f"Empty report: {data}")
                print(f"[OK] Colab report: {len(report.split())} words, lang={user_language}")
                return report
            except requests.exceptions.Timeout:
                if attempt <= retries:
                    time.sleep(20 * attempt); continue
                raise TimeoutError("Colab server timed out")
            except requests.exceptions.ConnectionError:
                self._server_ok = False
                raise ConnectionError("Lost connection. Restart Colab and update URL.")
        raise RuntimeError("Failed after all retries")