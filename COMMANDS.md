# Commands

Everything you need to run VerifAI, in the order you'd actually need it.

Repository root for every command below:

```
d:\fact-verification-system1\fact-verification-system2
```

---

## 1. Activate the virtual environment

Do this in **every new terminal**. All Python commands assume it.

```powershell
cd d:\fact-verification-system1\fact-verification-system2
.\venv\Scripts\Activate.ps1
```

Your prompt gains a `(venv)` prefix. Deactivate with `deactivate`.

<details>
<summary>Other shells</summary>

```cmd
:: Command Prompt (cmd.exe)
cd /d d:\fact-verification-system1\fact-verification-system2
venv\Scripts\activate.bat
```

```bash
# Git Bash
cd /d/fact-verification-system1/fact-verification-system2
source venv/Scripts/activate
```

```bash
# macOS / Linux
source venv/bin/activate
```
</details>

> **If PowerShell refuses with "running scripts is disabled":**
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Or skip activation entirely and call the interpreter directly — this always
> works and needs no policy change:
> ```powershell
> .\venv\Scripts\python.exe -m serve.cli "your claim"
> ```

---

## 2. Run the app (backend + frontend)

You need **two terminals**, both activated.

### Terminal 1 — backend (FastAPI, port 8000)

```powershell
uvicorn serve.api:app --reload
```

Wait for `Models and index loaded` — about 10 seconds while four CPU models
load. Startup does this deliberately so the first verification is as fast as
the tenth.

### Terminal 2 — frontend (Vite, port 5173)

```powershell
cd frontend
npm run dev
```

### Then open

**http://localhost:5173**

Vite proxies `/verify`, `/health` and `/download` to port 8000, so the browser
sees one origin and there's no CORS preflight on the multipart upload.

### Don't want Node?

Skip terminal 2. The backend serves a built React bundle from `ui/dist/`, or
falls back to a self-contained `ui/index.html` if you've never built it:

**http://127.0.0.1:8000**

### First time only

```powershell
cd frontend
npm install
```

### Build the frontend for the backend to serve

```powershell
cd frontend
npm run build          # writes ../ui/dist/
```

After this, `http://127.0.0.1:8000` serves the real React app and you only need
one terminal.

---

## 3. The two tiers, separately

The whole architecture is that these never talk to each other directly.
`ingest/` **writes** `index/`, `serve/` **reads** it, and `serve/` imports
nothing from `ingest/` — there's a test that enforces it.

That means you can rebuild the corpus while the API keeps serving the old
index, and switch over only when the new one validates.

---

### TIER 1 — `ingest/` · build the index

Needs **no Groq key**. Embeddings are local and nothing here calls an LLM.

#### Check what's published

```powershell
python -m ingest.run --status
```

```
published   index/v1
built at    2026-08-16T18:45:22
model       all-MiniLM-L6-v2 (dim 384)
chunking    700 / 100
documents   1,687
chunks      13,101  (39 duplicates collapsed)
dates       NOT RELIABLE  (1,622 placeholder dates)
```

#### Build from the CSV corpus

```powershell
python -m ingest.run --from-csv
```

~13 minutes the first time. After that an unchanged rebuild is a **1-second
no-op** — a corpus fingerprint short-circuits it. Force a rebuild anyway:

```powershell
python -m ingest.run --from-csv --force
```

#### Crawl live feeds

Written and tested, but **not yet run against live feeds**. Start cheap — one
request per feed, no article bodies fetched:

```powershell
python -m ingest.run --discover-only --limit 200
```

It reports what fraction of articles carry a real publisher date. The CSV
corpus manages 4%.

Then the real thing:

```powershell
python -m ingest.run --crawl --limit 500              # crawl only
python -m ingest.run --crawl --merge-csv              # crawl + keep the CSV corpus
python -m ingest.run --crawl --per-feed 10 --delay 2  # gentler, more spread out
```

| Flag | |
|---|---|
| `--limit N` | stop after N articles |
| `--per-feed N` | at most N per feed — stops one prolific outlet dominating |
| `--delay S` | minimum gap between requests to one host (default 1.0s) |
| `--merge-csv` | union with the CSV corpus, de-duplicated |
| `--force` | rebuild even when the fingerprint is unchanged |

#### Versions and rollback

```powershell
python -m ingest.run --list
```

```
  v1       1,687 docs   13,101 chunks  2026-08-16  1,622 placeholder dates <- CURRENT
```

```powershell
python -m ingest.run --rollback v1
```

`index/CURRENT` is a text file containing a version name. Publishing is one
atomic write; rollback is the same write with an older name. **Restart Tier 2
to pick it up.** Nothing is deleted — the old build stays on disk.

---

### TIER 2 — `serve/` · verify claims

Needs `GROQ_API_KEY` in `.env`.

#### Command line

```powershell
python -m serve.cli "India's software exports reached 222 billion dollars in 2024-25"
```

```powershell
python -m serve.cli --trace "your claim"      # per-node progress
python -m serve.cli --json "your claim"       # structured output
python -m serve.cli --no-web "your claim"     # index only, no web fallback
python -m serve.cli --graph                   # print the mermaid diagram
```

#### The three exits, to see them

```powershell
python -m serve.cli "India's software exports reached 222 billion dollars in 2024-25"
#  -> TRUE @ 92%          verified, 1 LLM call

python -m serve.cli --no-web "the moon is square in shape"
#  -> ABSTAINED           nothing cleared the relevance floor, 0 LLM calls

python -m serve.cli "what time is it"
#  -> REJECTED            not a checkable claim, 0 LLM calls, 0.5s
```

#### Voice, image, documents

```powershell
python -m serve.cli tests\fixtures\claim_en.mp3          # Whisper -> verify -> mp3 reply
python -m serve.cli tests\fixtures\claim_hi.mp3          # Hindi in, Hindi report out
python -m serve.cli tests\fixtures\insta_post.png        # OCR a screenshot
python -m serve.cli --document report.pdf "your claim"   # check a claim against a file
python -m serve.cli --export html,docx "your claim"      # also write those formats
```

#### The API alone

```powershell
uvicorn serve.api:app --reload                       # :8000
uvicorn serve.api:app --host 0.0.0.0 --port 8080     # different host/port
```

Check it's healthy and see what it can accept:

```powershell
curl.exe http://127.0.0.1:8000/health
```

```json
{ "index": "v1", "documents": 1687, "web_search": true,
  "capabilities": { "text": true, "voice": true, "image": true,
                    "document": true, "web_search": true } }
```

#### Exit codes

`0` verified · `2` abstained · `3` rejected · `4` index problem · `5` API quota

---

## 4. Tests, demo, evaluation

```powershell
pytest                                    # 68 tests, no network, no API calls
pytest tests\test_crawl.py -v             # just the crawl tests
python -m demo.run_demo                   # the full demo, ~80 seconds
python -m eval_harness.run_eval --all     # the full evaluation (uses Groq quota)
```

---

## 5. Setup, from scratch

```powershell
cd d:\fact-verification-system1\fact-verification-system2

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -e .

copy .env.example .env
```

Then edit `.env` and set **`GROQ_API_KEY`** — free at
[console.groq.com/keys](https://console.groq.com/keys).

First run only, so the two automatic models can download:

```powershell
$env:HF_OFFLINE = "false"
python -m ingest.run --from-csv
```

You also need the two fine-tuned BERTs (831 MB, gitignored) at
`models/claim_detector/final/` and `models/stance_detector/final/`. There is no
silent fallback — `core/models.py` raises with the missing path rather than
substituting a stock checkpoint.

### Optional — image OCR

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

**The Windows installer does not add it to PATH.** Put the full path in `.env`:

```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Every other input type works without it.

### Optional — web search fallback

Free tier at [app.tavily.com](https://app.tavily.com), 1,000 searches/month:

```
TAVILY_API_KEY=tvly-...
```

> **Set each key exactly once in `.env`.** python-dotenv keeps the *last*
> assignment, so a duplicate line further down — even an empty one — silently
> overrides the real value and the feature goes quiet without an error.

---

## Troubleshooting

**`ModuleNotFoundError`** — the venv isn't active. See step 1, or call
`.\venv\Scripts\python.exe` directly.

**`Address already in use`** — something is already on that port:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
Stop-Process -Id <THAT_ID>
```

**`OSError: The paging file is too small` / access violation** — not a code
bug. Loading two 400 MB BERTs plus torch needs headroom, and the backend, the
CLI and `pytest` each load their own copy. Don't run them at once. If it
persists, close Docker Desktop and any database services:

```powershell
Get-Process | Sort-Object PrivateMemorySize64 -Descending |
  Select-Object -First 8 Name, @{n='CommitMB';e={[int]($_.PrivateMemorySize64/1MB)}}
```

**`404 model_not_found` from Groq** — the configured model was retired. Groq
does this on a rolling basis. See what your key can actually use:

```powershell
python -c "from core.llm import available_models; print(sorted(available_models()))"
```

Then set `GROQ_MODEL` in `.env` to one of them. Nothing else needs changing.

**Frontend loads but requests fail** — the backend isn't running. Both
terminals need to be up; Vite only proxies, it doesn't serve the API.

**`UnicodeEncodeError` in a console** — the CLI forces UTF-8 on stdout, but if
you hit this anywhere else:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

## Quick reference

| | |
|---|---|
| Activate venv | `.\venv\Scripts\Activate.ps1` |
| Backend | `uvicorn serve.api:app --reload` → :8000 |
| Frontend | `cd frontend; npm run dev` → :5173 |
| Verify a claim | `python -m serve.cli "claim"` |
| Build index | `python -m ingest.run --from-csv` |
| Index status | `python -m ingest.run --status` |
| List versions | `python -m ingest.run --list` |
| Roll back | `python -m ingest.run --rollback v1` |
| Crawl (dry) | `python -m ingest.run --discover-only --limit 200` |
| Tests | `pytest` |
