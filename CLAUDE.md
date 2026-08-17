# CLAUDE.md — project memory

Working memory for VerifAI. **Read this first, then `BUILD_PLAN.md` for detail.**
Update the Status board at the end of every work session.

---

## What this is

Two-tier RAG fact verification system. **Portfolio / resume project** — not
production, not deployed, not multi-user. See `BUILD_PLAN.md` §0 for the
explicit "do not build" list (auth, rate limiting, CI, robots.txt engine...).

- **Tier 1 `ingest/`** — crawl → clean → chunk → embed → write a versioned index.
  Runs on demand, offline, write-only.
- **Tier 2 `serve/`** — LangGraph app. Reads the index, never writes to it.
- **The contract** — `index/current/{chroma/, parents/, bm25.pkl, manifest.json}`.
  Files, not imports.

Stack: LangChain 0.3 + LangGraph + Chroma + Groq. Two locally-trained BERTs
(claim detection, FEVER stance) stay on CPU. No local LLM — Groq only.

---

## Status board

Branch: `rebuild/two-tier` (created from `main`, nothing deleted yet)

| Milestone | State | Notes |
|---|---|---|
| **M0** Ground clearing | ✅ **DONE** | `python -m core.llm` prints `OK`. Live Groq call verified. |
| **M1** Tier 1 from CSV | ✅ **DONE** | `index/v1` published — 1,687 docs → 13,101 chunks. Re-run is a 0.8s no-op. |
| **M2** Tier 2 core | ✅ **DONE** | All three exits verified. Abstain costs 0 LLM calls. |
| **M3** Eval harness | ✅ **DONE** | recall@5 0.875→0.995. abstain_precision 1.000. false_confidence 0. |
| **M4** Web search fallback | ✅ **built** | Needs a `TAVILY_API_KEY` to go live. Degrades to abstain without one. |
| **M5** Modalities + languages | ✅ **built** | Voice round-trip verified. Image needs Tesseract. |
| **M6** Package it | ✅ **built** | README + 13 tests + demo. GIF and model hosting are yours. |

### Done so far

- [x] Branch `rebuild/two-tier` created
- [x] `pyproject.toml` — LangChain 0.3 pinned, optional extras for docs/voice/search/dev
- [x] `.env.example` — replaces `configs/groq_token.txt`
- [x] `core/config.py` — pydantic-settings, validates at import, paths resolved
      against project root (not cwd)
- [x] `core/llm.py` — `get_llm()`, `verify_models()` startup check against Groq's
      live catalogue
- [x] Package skeleton: `core/ ingest/ serve/ serve/nodes/ eval_harness/ demo/ index/`
- [x] LangChain stack installed into `venv/` — langchain 0.3.30, langgraph 1.0.1,
      langchain-chroma 0.2.6, langchain-groq 0.3.8, langchain-huggingface 0.3.1
- [x] **All 15 LangChain imports the plan depends on verified working** — no
      chromadb conflict, `BaseDocumentCompressor.compress_documents` signature
      matches BUILD_PLAN §6.5 exactly. M1/M2 can be written against known-good APIs.
- [x] Verified: config validates at import; bad `chunk_overlap` rejected; missing
      key produces an actionable message, not a traceback
- [x] `.gitignore` — **`.env` was not ignored**; added `.env`, `index/`,
      `data/.embed_cache/`, `*.egg-info/`. Verified with `git check-ignore`.
- [x] `.env` created from template and normalised
- [x] **M0 acceptance passed** — `python -m core.llm` → `[ok] model available:
      llama-3.3-70b-versatile` → `OK`
- [x] `python -m core.llm --list` added — dumps the live model catalogue

### M1 — done

- [x] `core/text.py` — `bm25_tokenize` (module-level so the pickled BM25 index
      can resolve it from `serve/` without importing `ingest/`), `sha256_id`
- [x] `ingest/clean.py` — CSV load, quality filter, URL + content dedup,
      `dateutil` parsing, Jan-1 `date_reliable` flag, domain normalisation
- [x] `ingest/chunk.py` — 700/100 splitter, `parent_id` = sha256(url) (hashed
      because `LocalFileStore` uses keys as filenames and URLs contain `/ ? :`),
      Chroma-safe metadata
- [x] `ingest/index.py` — `CacheBackedEmbeddings`, content-hash chunk ids,
      parent docstore, BM25 pickle, manifest, validate, pointer publish
- [x] `ingest/run.py` — `--from-csv` / `--status` / `--force`
- [x] `core/config.py` — version pointer (`index/CURRENT` text file, not a
      symlink: symlinks need Developer Mode on Windows)

**Build result (`index/v1`)**

```
documents  1,687      chunks 13,101   (39 duplicate chunks collapsed)
model      all-MiniLM-L6-v2, dim 384      chunking 700 / 100
dates      NOT RELIABLE — 1,622 placeholders (96%)
build      773s first run  ·  0.8s on re-run (fingerprint match)
```

Verified working: dense retrieval, parent expansion (570-char chunk → 3,070-char
article), BM25 unpickle + query, `data/chroma_db/` untouched.

**Early signal for M2's relevance floor** — dense scores on the published index:

```
"India GDP growth 2024"       -> +0.55 .. +0.42  (indianexpress, news18, economictimes)
"the moon is square in shape" -> +0.034, +0.030, +0.001
```

Nonsense queries already score near zero, so a floor in the 0.10–0.20 band should
separate them cleanly. **Still calibrate it properly at M3** — two queries is an
anecdote, not a calibration.

### M2 — done

- [x] `core/credibility.py` — ported, keyed off `metadata['domain']`, table
      rewritten to cover the 74 domains actually in this corpus
- [x] `core/models.py` — lazy `lru_cache` singletons; no placeholder fallback
      (the old manager could silently swap in a different model)
- [x] `core/compressors.py` — `ScoringCrossEncoderReranker`, `RelevanceFloor`,
      `CredibilityScorer`
- [x] `core/prompts.py` — ported; output-format section replaced by the schema
- [x] `serve/schemas.py` — `VerifyState`, `VerificationReport`
- [x] `serve/retriever.py` — manifest interlock + retriever assembly
- [x] `serve/nodes/` — `gate.py` `retrieve.py` `stance.py` `generate.py`
      `terminal.py` `render.py`
- [x] `serve/graph.py`, `serve/cli.py`
- [x] `core/config.py` — added `claim_threshold`, `stance_low_confidence`

**All three acceptance commands pass:**

```
"India's GDP grew 8% in 2024"  -> verified   11.1s  1 LLM call   exit 0
"the moon is square in shape"  -> abstained   7.0s  0 LLM calls  exit 2
"what time is it"              -> rejected    3.2s  0 LLM calls  exit 3
```

Also verified: manifest/embedder mismatch refuses to serve with an actionable
message (exit 4, no traceback); citation validator catches fabricated indices in
both `key_findings` and `evidence_analysis`; `serve/` imports nothing from
`ingest/`.

**Retrieval on the live index** (`"India's GDP grew 8% in 2024"`):
38 candidates → 8 chunks above the floor → 7 articles after parent dedup.

### 🔥 The relevance floor does NOT catch topical-but-wrong

Most important finding of M2, and it changes what M3 has to measure.

Three **factcheck.org** articles about *Trump's US economy claims* scored
**0.974 / 0.954 / 0.347** against "India's GDP grew 8% in 2024". The floor kept
them — correctly, they are about economic growth claims. The cross-encoder
measures topical relevance, and it is right that they are topical.

What caught them was the LLM's `directly_relevant: false`, and it then returned
UNVERIFIABLE at 30% rather than manufacturing a verdict out of adjacent
material. **That is the system working**, but it means:

- the floor separates *nonsense* (0.000) from *topical* (0.9+) and nothing else;
- **M3 cannot tune the floor on relevance alone.** Sweeping it to raise
  `abstain_precision` will not fix topical-but-wrong-subject retrieval, because
  those documents score at the very top. Measure `directly_relevant` rates too.
- the honest interview line: retrieval finds the right *topic*; the LLM decides
  relevance to the *claim*; the floor only guards the nonsense case.

Corpus skew is the underlying cause — 1,687 tech/entertainment RSS articles
contain no Indian GDP figure. Already a documented limitation; this quantifies it.

### M3 — done

- [x] `eval_harness/prepare_datasets.py` — builds all three datasets, offline
- [x] `eval_harness/datasets/known_item_200.jsonl` — 100 articles × head/tail
- [x] `eval_harness/datasets/fever_dev_200.jsonl` — KILT FEVER, balanced
- [x] `eval_harness/datasets/adversarial_20.jsonl` — hand-written, 4 categories
- [x] `eval_harness/baseline.py` — read-only adapter for `data/chroma_db`
- [x] `eval_harness/metrics.py` — recall@k, MRR, verdict, abstention
- [x] `eval_harness/run_eval.py` — `--retrieval` `--sweep` `--system`
- [x] `eval_harness/report.py` — generates `results/RESULTS.md` for the README
- [x] `RELEVANCE_FLOOR` calibrated 0.15 → **0.25** in `.env` and `.env.example`
- [x] LangSmith wiring in `core/config.py` (activates only if key + flag set)
- [x] `.gitignore` — eval datasets/results were caught by blanket `*.json(l)`;
      un-ignored, they are the deliverable

**Retrieval A/B** (200 known-item queries, no LLM):

|  | recall@1 | recall@5 | MRR |
|---|---|---|---|
| head — old | 0.95 | 0.98 | 0.965 |
| head — new | 1.00 | 1.00 | 1.000 |
| **tail — old** | **0.47** | **0.76** | **0.598** |
| **tail — new (dense only)** | **0.88** | **0.95** | **0.904** |
| tail — new (full pipeline) | 0.98 | 0.99 | 0.982 |
| overall — old | 0.71 | 0.87 | 0.781 |
| overall — new (full) | 0.985 | 0.995 | 0.988 |

⚠ **Two decimals only.** The eval was run twice: the old index moved 0.010 on
head recall@1 between runs, the new index reproduced exactly. Chroma's HNSW is
approximate, so the third decimal is noise. The measured effect (0.47 → 0.88) is
~40× that noise floor, so the conclusion is safe — but do not quote 3 decimals.

**Full system** (220 claims through the graph, ~16 LLM calls total):

```
adversarial   abstain_precision 1.000 (20/20)  — 16 via floor, 4 via LLM
              1.000 in every category, including all 8 near_miss
FEVER 200     abstain_rate 0.975 (195/200)     answered 5
              verdict_accuracy 1.000 (5/5)     false_confidence 0
```

### ⚠ How to state these numbers honestly

**`verdict_accuracy 1.000` is on n=5. Do NOT put "100% verdict accuracy" on the
resume** — an interviewer will ask for n and the answer destroys the bullet.
What is defensible:

- the retrieval A/B (n=200, deterministic, reproducible) — **this is the bullet**
- `abstain_precision 1.000` on 20 hand-built adversarial claims
- **`false_confidence 0` across 200 FEVER claims** — strong *because* n is 200
- 97.5% abstention on FEVER is correct behaviour, not a failure: FEVER's gold
  evidence is Wikipedia and this corpus is news. Say that before being asked.

The retrieval result also **falsified my own prediction**. I expected old-index
tail recall near zero "by construction" since that text was never embedded. It
was 0.760 @5. News articles are topically coherent, so the head vector still
retrieves the article by subject. Truncation did not destroy *findability* — it
destroyed *ranking precision*: recall@1 on tail queries, 0.470 → 0.880 with the
same embedder, chunking the only change. That is the accurate claim.

### 🔥 Two mechanisms do the abstaining, and only one is the floor

Proven, not inferred (`abstained_by_relevance_floor` / `..._by_llm_judgement`):

```
16 / 20  relevance floor    never reached the LLM, cost nothing
 4 / 20  LLM judgement      evidence retrieved and read, returned UNVERIFIABLE
```

The 4 are exactly the high-scoring `near_miss` claims — OpenAI 0.858,
Netflix 0.795, TechCrunch 0.647, Apple 0.593. The floor sweep shows why:

- every `absurd` / `absent_true` / `absent_false` claim scores **≤ 0.0058**
- every claim above 0.09 is `near_miss`, **without exception**

So the floor cleanly separates *absent from the corpus* from *present in the
corpus*, and cannot separate *supports this claim* from *is merely about the
same subject*. Raising the floor from 0.02 to 0.60 moves adversarial abstention
only 0.70 → 0.85, while costing recall on paraphrased real claims.

This is the answer to "why not just raise the threshold?" — and it is why the
schema has a `directly_relevant` field at all.

### M4 — built, one user action to go live

- [x] `serve/nodes/websearch.py` — Tavily as a `BaseRetriever`, fires only when
      the index returns nothing above the floor
- [x] `serve/graph.py` — `retrieve → web → abstain`, both edges conditional
- [x] `core/config.py` — `web_search_enabled` / `web_search_k` / `web_search_depth`
- [x] `serve/cli.py` — `--no-web`, plus the `web` node in `--trace`
- [x] `render.py` — evidence is labelled `· via web search`, and a banner says
      the corpus had nothing
- [x] `terminal.py` — an abstention now says whether the web was searched too
- [x] `eval_harness` — `--web` flag, results saved to a **separate** file
- [x] `.env` / `.env.example` / `pyproject.toml`

**Verified without a key** (mocked retriever, all checks pass): metadata
remapping, the shared compressor pipeline, floor filtering of off-topic results,
credibility via the same domain table, and graceful no-key degradation. Live
search itself is untested until a key exists.

🔴 **User action: get a free Tavily key** at https://app.tavily.com (1,000
searches/month) and put it in `.env`. Without it the `web` node logs once and
routes to `abstain` — which is exactly the pre-M4 behaviour, so nothing is
broken, the fallback just never fires.

### 🔥 `pip install langchain-tavily` breaks the whole app — do not do it

It requires `langchain-core>=1.0`, and installing it silently dragged the stack
from 0.3 to 1.x: `langchain 0.3.30 → 1.3.15`, `langchain-core 0.3.86 → 1.5.5`,
`langgraph 1.0.1 → 1.2.11`. LangChain 1.x moved `langchain.retrievers`, so
`serve/retriever.py` stopped importing and the app would not boot:

```
ModuleNotFoundError: No module named 'langchain.retrievers'
```

`pip` reported the conflict as a warning **and installed anyway**.

**The fix:** `tavily-python` — the thin HTTP client, zero langchain deps — with
`langchain_community.retrievers.TavilySearchAPIRetriever` (already on the pinned
0.3 line) wrapping it as a real `BaseRetriever`. Same LangChain-native
composition, no version conflict. `pyproject.toml` carries this warning inline.

Rollback that restored it, worth keeping:

```bash
pip uninstall -y langchain-tavily langchain-protocol
pip install "langchain==0.3.30" "langchain-core==0.3.86" "langgraph==1.0.1" \
            "langgraph-checkpoint==3.0.1" "langgraph-prebuilt==1.0.1" \
            "langgraph-sdk==0.2.15"
```

**Always run `pip check` and boot the CLI after adding any langchain-* package.**

### ⚠ M4 changes what `abstain_precision` measures — eval defaults to web OFF

The adversarial set is 20 claims *this corpus* cannot answer. The open web
answers most of them easily. With web search on, correctly answering "Napoleon
was born in Corsica" scores as a **failure** against a set whose expected
outcome is abstention — the metric would report a regression for behaviour that
actually improved.

So `run_eval --system` keeps web search **off by default** and measures the
property the architecture claims: given a fixed, inspectable corpus, does the
system decline when it should? `--web` measures end-to-end product behaviour
instead, and saves to `system_eval_web.json` so the two never overwrite each
other.

Practical reason too: ~211 of 220 claims abstain on the corpus, so a
web-enabled run spends ~211 Tavily searches — a fifth of the free monthly quota
per run.

### 🔥 Groq free tier is 100,000 tokens/DAY, not just per-minute

Hit mid-M4 while re-running the eval: `Used 98,892 / Limit 100,000`, reset in
43 minutes. The M3 runs had consumed the day's budget.

The retry logic made it worse — it classified the daily cap as "transient" and
retried twice on a 5s/10s backoff before raising, **losing the whole run**.
Fixed in `eval_harness/run_eval.py`:

- `QuotaExhausted` is raised immediately on `per day` / `TPD` / `RPD`, with the
  reset time parsed out of Groq's message — no pointless retries
- transient errors now honour the server's own `try again in Ns` hint
- both eval loops catch it and **save partial results** instead of discarding
  completed work

Budget roughly: a full `--system` run is ~16 LLM calls but ~66k tokens, so two
full runs per day is the free-tier ceiling.

### M5 — built

- [x] `serve/nodes/adapt.py` — text · voice (Groq Whisper) · image (pytesseract)
      · PDF/DOCX. **URL input cut** — a URL is a slower way to get text, and
      every URL worth pasting is either an article with an ambiguous claim or a
      social post you would screenshot instead.
- [x] `serve/nodes/language.py` — script-first detection, then `langdetect`;
      Groq translation to English for retrieval
- [x] `serve/nodes/export.py` — html · docx · mp3 (gTTS)
- [x] `serve/api.py` — FastAPI + SSE from `astream(stream_mode="updates")`
- [x] `ui/index.html` — rewritten for the new API, **with clipboard paste**
- [x] `render.py` — localised headings for 12 languages, upload/web labelling
- [x] `core/credibility.py` — the upload cap
- [x] 23 new tests (36 total, 6.9s)

**Verified end to end:**

```
Hindi audio → Whisper → detect hi → translate → retrieve → TRUE @ 80%
            → Hindi report → Hindi mp3        ← BUILD_PLAN's M5 criterion
Hindi text  → same, minus transcription
image (no Tesseract) → clean rejection, exit 3, no traceback
SSE          → one event per node, /health reports capabilities
```

✅ **The upload safety test PASSES.** Run 2026-08-17 on the new Groq key:

```
doc asserts "exports collapsed to 40 billion", claim asserts the same
-> FALSE @ 80%   (corpus says 222 billion; the upload was refuted)
```

The system used corpus evidence to contradict the user's own attachment. The
0.6 cap plus the `⚠ UNVERIFIED USER SUBMISSION` prompt label both did their job.

**Full flow verified end to end (7/7 explicit checks):**

```
PASS  text, verifiable          TRUE @ 80%            verified
PASS  text, topical but absent  UNVERIFIABLE @ 30%    verified
PASS  text, nothing relevant    abstained, 0 LLM calls
PASS  text, not a claim         rejected, 0.1s
PASS  Hindi text                TRUE @ 80%, hi report
PASS  Hindi VOICE               Whisper -> hi -> TRUE @ 80% + mp3
PASS  upload safety             FALSE @ 80%
```

**Not verifiable yet, both blocked on the user:** image OCR (Tesseract not
installed) and web fallback (`TAVILY_API_KEY` still empty). Both degrade
correctly — image rejects with an install message, web skips to abstain.

### Deviations from BUILD_PLAN §M5

- **Uploads are not written to a `user_uploads` collection.** The plan says to
  persist them; that would mean `serve/` writing to the index, breaking the
  read-only invariant and the tier boundary. Uploads are injected into the
  evidence list at request time and never persisted — same behaviour, boundary
  intact.
- **URL input cut** (see above).
- **Localised section headings** added for 12 languages. The LLM already
  generates the body in the user's language; English headings around Hindi prose
  read like a half-finished translation. Verdict labels stay English — the eval
  harness, the API and downstream code all key on those five tokens.

### M6 — built (two items are yours)

- [x] `tests/` — 13 tests, **9.6s, no API calls**. `conftest.py` skips cleanly
      when there is no index or no models, so `pytest` is useful on a fresh clone.
- [x] `demo/run_demo.py` — **78s**, 4 live scenarios + 2 that skip with a reason
- [x] `README.md` — full rewrite; every number cross-checked against the results
      JSON by script (39 assertions, all pass)
- [x] `pyproject.toml` — pytest config, `demo` added to packages

**Test list** — BUILD_PLAN §M6 names "upload credibility cap", but uploads are
M5 and do not exist. Swapped for **reranker score-stamping**, which is the bug
that would have made every claim abstain and faked a passing abstain test.

**Mutation-checked, not just green.** Each test was fed the broken
implementation to confirm it fails: stock `CrossEncoderReranker` (scores
discarded), a floor that filters nothing, credibility ignoring `date_reliable`,
a validator accepting any index, `.replace('www.','')`, and a `serve/ → ingest/`
import. **6 caught, 0 vacuous.**

**Demo narrative** — deliberately ordered so each scenario removes a different
way the system could be fooling you:

```
1. software exports claim   -> TRUE @ 70%        it works
2. "India's GDP grew 8%"    -> UNVERIFIABLE      on-topic evidence, still declines
3. "the moon is square"     -> ABSTAINED, 0 LLM  nothing relevant
4. "what time is it"        -> REJECTED, 0.2s    not a claim
5. web fallback             -> skipped (no Tavily key)
6. Hindi voice              -> skipped (M5)
```

Scenario 1 was changed from the plan's `"India's GDP grew 8% in 2024"` because
that claim always returns UNVERIFIABLE — a weak opener. The software-exports
claim is one the corpus genuinely supports, so the demo now *opens* with a real
cited verdict and keeps the GDP case as scenario 2, where its
topical-but-absent behaviour is the point rather than a disappointment.

🔴 **Yours to finish:**
1. **Record the GIF** — `python -m demo.run_demo`, ~78s, save to
   `demo/screenshots/demo.gif`. README has the slot marked.
2. **Host the two fine-tuned models** (871 MB, gitignored). Recommended:
   push to HuggingFace Hub — free, and "fine-tuned models published on HF" is a
   stronger portfolio line than a zip. Then replace the single `TODO:` block in
   README.md §Models with the download command. **Until this is done the
   clean-clone criterion cannot pass**, because a fresh clone has no models.

### Repo cleanup — done (BUILD_PLAN §2.1)

`python scripts/cleanup.py` — dry run by default, `--apply` to move,
`--restore` to undo. **Nothing was deleted.** 76 entries / 415 MB moved to
`_trash/` (gitignored), preserving relative paths, with `_trash/_manifest.json`
recording every move. That approach was chosen because **most targets were
untracked**, so `rm` would have been unrecoverable — `git checkout` could not
have brought back `app.py`, the root test scripts, the status docs, or
`models/mistral_fv_adapter/`.

Removed (77 entries): 5 root scratch scripts · 5 overlapping status docs · 13
stale architecture docs · 6 dead `src/` modules · 14 superseded `scripts/` · 13
old print-driven tests · 2 secret files · 4 unused model dirs (414 MB, incl. a
byte-identical 125 MB duplicate of `lid.176.bin` under `tests/models/`) ·
build artifacts · old launchers · **`requirements.txt`**.

`requirements.txt` was not merely stale — it pinned **none of the LangChain
stack** and still listed `openai-whisper`, `easyocr`, `selenium`, `gradio`,
`faiss-cpu` and `fasttext`, all cut in the rebuild. `pip install -r
requirements.txt` produced an environment that could not run this project.
**`pyproject.toml` is now the single dependency source.** The README quickstart
must say `pip install -e .`, never `pip install -r requirements.txt`.

**Kept on purpose:** the rest of `src/` (M4/M5 source material — the plan's
§2.2 move list), `app.py`, `data/processed` + `data/raw` (training provenance
for the two BERTs), `data/.embed_cache` (makes re-ingest a 1s no-op),
`data/chroma_db` (the M3 baseline), `notebooks/`, `ui/index.html`.

⚠ **`app.py` no longer imports cleanly** — it references
`src.generation.report_generator`, which §2.1 lists as dead (the Colab/ngrok
path) and which is now in `_trash/`. This is intended: `app.py` is kept only as
the **SSE structure reference for M5**, not as runnable code. M5 replaces it
with `serve/api.py` built on `graph.astream(stream_mode="updates")`.

Verified after cleanup: all three exits still run, `ingest.run --status` works,
`eval_harness.report` still generates.

### Frontend — built (React, Vite, no TypeScript)

`frontend/` → builds into `ui/dist/`, which `serve/api.py` serves. Node v20.18 /
npm 10.8 confirmed on this machine; 63 packages, 156 KB JS bundle.

```
frontend/src/
  api.js                 fetch + SSE parsing
  history.js             localStorage session history, capped at 40 entries
  App.jsx                shell and state
  components/ClaimInput  textarea · drag-drop · CLIPBOARD PASTE · examples
              Progress   one row per graph node, live
              Report     verdict · evidence · credibility meters · downloads
              History    re-openable past runs
```

**How it connects to FastAPI** — the answer to "or what to connect with the
backend": Vite proxies `/verify`, `/health`, `/download` to `127.0.0.1:8000` in
dev, so the browser sees one origin and there is **no CORS preflight** on the
multipart upload. `npm run build` → `ui/dist/` → FastAPI serves it directly.
CORS middleware is added anyway for anyone bypassing the proxy, localhost only.

**Fallback preserved:** `/` serves `ui/dist/index.html` if the React build
exists, else the self-contained `ui/index.html`. A clone with no Node still has
a working UI — React is an upgrade, not a requirement.

**History is localStorage, not a backend table.** There are no accounts to scope
a server-side history to, and it survives the backend restarting on a corpus
rebuild. Capped because a full result is tens of KB against a ~5 MB quota.

⚠ `.gitignore` needed two more negations: the blanket `*.json` rule was
swallowing `frontend/package.json` and `package-lock.json` — the two files a
clone needs to reproduce the build. `node_modules/` and `ui/dist/` are ignored.

### 👉 NEXT ACTION

**Build the crawl path** — `ingest/discover.py`, `ingest/fetch.py`,
`ingest/sources.yaml`. This is the one piece of the original plan never built:
M1 was made from the CSV deliberately (BUILD_PLAN §3) and `--crawl` was
deferred. All of M0–M6 is now done.

Agreed approach (surveyed 2026-08-17):

| Tool | Why |
|---|---|
| `feedparser` | RSS discovery. Pure Python, tiny, no key. **Not installed yet.** |
| `trafilatura` | Article extraction. Actively maintained, no ML model. **Not installed.** |
| ~~`newspaper3k`~~ | In `pyproject.toml`, but **unmaintained since 2021** — replace it |
| ~~Crawl4AI / Firecrawl / Scrapy~~ | Chromium, paid tiers, or overkill |

**You already have 363 distinct RSS feed URLs** in `src/data_collection/*.py` —
`sources.yaml` is mostly an extraction job, not authoring.

Shape (only two new files; everything downstream is unchanged because
`clean.py` already consumes an iterable of dicts):

```
sources.yaml → discover.py (feedparser) → fetch.py (trafilatura, 1 req/s)
             → clean.py ✅ → chunk.py ✅ → index.py ✅ → index/v2
```

Biggest win beyond freshness: **real publish dates**, which retires the 96%
placeholder-date limitation currently in the README.

---

#### Previously: M5 — done

1. `serve/nodes/adapt.py` — Groq Whisper (voice), PDF/DOCX, URL.
   **No image OCR** — this Groq key has no vision model (see the plan-change
   section below). Decide: pytesseract, or drop image input.
2. `serve/nodes/language.py` — `langdetect` + Groq translation. The report is
   **generated in** the target language, not back-translated —
   `core/prompts.py::LANG_INSTRUCTIONS` already does this, it just needs wiring.
3. Wire `adapt` and `translate` into `serve/graph.py` ahead of `gate`
   (currently the entry point).
4. Uploaded documents → credibility **capped at 0.6**, labelled as an unverified
   submission. `core/credibility.py` already has a `user_upload` source type.
5. `serve/nodes/render.py` — add html/docx, plus mp3 via gTTS for voice input.
6. `serve/api.py` — FastAPI + SSE from `graph.astream(stream_mode="updates")`
   (BUILD_PLAN §6.8). `app.py` is kept as the SSE reference; it no longer
   imports cleanly, so read it, do not run it.
7. Reconnect `ui/index.html`.

Then: enable demo scenario 6 (Hindi voice) — it is already written and skips
with a reason until M5 lands. Add the M5 tests (upload credibility cap, which
was swapped out of the M6 suite because uploads did not exist yet).

---

## Blocking on the user

- [ ] **🔴 COMMIT THE WORK.** `git status` shows 59 entries and **everything
      built in M0–M3 is untracked** — `core/ ingest/ serve/ eval_harness/
      CLAUDE.md BUILD_PLAN.md .env.example`. Last commit is `ecd9e1a`, from
      before the rebuild. A bad `git clean` or a disk problem loses all of it.
      Nothing has been committed on your behalf.
- [ ] **🔴 Rotate the Groq key — still outstanding.** `configs/groq_token.txt`
      held a live `gsk_…` key in plaintext and has been moved to `_trash/`, not
      destroyed. Note it was a **different key from the one now in `.env`**
      (both 56 chars, both `gsk_`), so there are two keys in play and at least
      one is disclosed. Rotate at console.groq.com, then delete `_trash/`.
- [ ] **Empty `_trash/`** once you are satisfied — 415 MB, 77 entries. See the
      cleanup section below.
- [ ] **Get a free Tavily key** — https://app.tavily.com, 1,000 searches/month.
      Paste into `TAVILY_API_KEY` in `.env`. M4 is built and its logic tested,
      but the live search path cannot fire without it.

---

## Invariants — do not break these

1. **`serve/` imports nothing from `ingest/`.** If it does, the tier boundary
   leaked and the architecture story is gone.
2. **No secret in any file under `configs/`.** Everything goes through
   `core.config.cfg`, which reads `.env`.
3. **Nothing reads `os.environ` directly.** One settings object.
4. **Paths resolve against `ROOT`, never cwd.** The old `/download` endpoint
   checked containment relative to the working directory — only correct when
   launched from the repo root.
5. **The relevance floor is calibrated, not guessed.** `RELEVANCE_FLOOR=0.25`,
   set by `eval_harness.run_eval --sweep` at M3. Re-sweep if the corpus,
   embedder, or reranker changes — all three move the score distribution.
6. **Abstain is a first-class outcome.** Three exits — verified, abstained,
   rejected — and all three must render a real response.
7. **`data/chroma_db/` stays untouched.** It is the M3 before/after baseline.

---

## Commands

```bash
# always use the venv interpreter on this machine
venv/Scripts/python.exe -m core.llm                  # M0 check
venv/Scripts/python.exe -m ingest.run --from-csv     # M1 build
venv/Scripts/python.exe -m ingest.run --status       # what is published
venv/Scripts/python.exe -m serve.cli "claim"         # M2 verify
venv/Scripts/python.exe -m serve.cli --trace "claim" # per-node progress
venv/Scripts/python.exe -m serve.cli --json  "claim" # structured output
venv/Scripts/python.exe -m serve.cli --graph         # mermaid, for the README
venv/Scripts/python.exe -m serve.cli --no-web "claim"   # index only (M4 off)

# M3 eval
venv/Scripts/python.exe -m eval_harness.prepare_datasets   # rebuild datasets
venv/Scripts/python.exe -m eval_harness.run_eval --retrieval  # A/B, no LLM, ~8m
venv/Scripts/python.exe -m eval_harness.run_eval --sweep      # floor, no LLM, ~8m
venv/Scripts/python.exe -m eval_harness.run_eval --system     # uses Groq, ~20m
venv/Scripts/python.exe -m eval_harness.run_eval --system --skip-fever  # ~3m
venv/Scripts/python.exe -m eval_harness.report --write        # -> results/RESULTS.md

# M6
venv/Scripts/python.exe -m pytest                    # 36 tests, ~7s, no API calls
venv/Scripts/python.exe -m demo.run_demo             # the demo
venv/Scripts/python.exe -m demo.run_demo --list      # what runs vs skips

# M5 multimodal
venv/Scripts/python.exe -m serve.cli tests/fixtures/claim_hi.mp3      # voice
venv/Scripts/python.exe -m serve.cli tests/fixtures/insta_post.png    # image
venv/Scripts/python.exe -m serve.cli --document report.docx "claim"   # upload
venv/Scripts/python.exe -m serve.cli --export html,docx "claim"       # exports
venv/Scripts/python.exe -m uvicorn serve.api:app --port 8000          # web UI
```

⚠ **Do not run uvicorn and the CLI at once.** Each loads all four models (~1 GB)
and this box has ~1.3 GB free — the second one dies with
`OSError: The paging file is too small` (error 1455), which reads like a
corruption bug and is not one.

`--retrieval` and `--sweep` make **no LLM calls** — re-run them freely after any
retrieval change. Only `--system` costs API calls (~16 for the full 220 claims,
because abstention is free).

`serve.cli` exit codes: `0` verified · `2` abstained · `3` rejected ·
`4` index missing/incompatible · `5` Groq quota/rate limit · `1` other.

---

## Decisions already made — do not re-litigate

| Decision | Why |
|---|---|
| No local LLM | Device can't run a 7B. Groq only. Mistral LoRA archived, unused. |
| Whisper → Groq API | Same key. Removes 140 MB model + ffmpeg. |
| OCR → Groq vision | One call does OCR + claim extraction. |
| Translation → Groq | Drops `deep-translator`'s scraped endpoint from the hot path. |
| Embeddings stay **local** | Free and rate-limit-free during bulk ingest of ~12.7k chunks. 80 MB. |
| Reranker stays **local** | Runs on ~20 candidates per query. Milliseconds. |
| Both trained BERTs stay | Yours, accurate, CPU-fine. Stance via LLM would be +8 calls/verification. |
| Chroma, local file | Not worth a hosted vector DB at this scale. |
| No Wikipedia ingestion | Tavily fallback covers the demo need in half a day vs two. |
| No re-crawl for v1 index | The existing CSV has full text and zero nulls. |

### Deviations from BUILD_PLAN.md

- **`eval/` → `eval_harness/`.** `eval` shadows the Python builtin and is a poor
  package name. Same contents, same role.
- **Building alongside `src/`, not replacing it yet.** New packages sit at repo
  root next to the old `src/`. Deletions (§2.1) happen after M2 passes, so
  there is always a working system on disk.
- **M2: ensemble runs at chunk level; parents expand *after* compression.**
  BUILD_PLAN §6.6 puts `MultiVectorRetriever` in the dense leg, so parents come
  back *before* reranking. That reintroduces the exact bug M1 exists to fix —
  the cross-encoder truncates at 512 tokens and the median article is 2,996
  chars, so it would rerank on the first third and score the rest blind. It also
  fuses a parent-level ranking with a chunk-level BM25 ranking. Same
  "search on chunks, reason on parents" idea, in the only order where each model
  gets input it can handle. `serve/retriever.py` documents this at length.
- **M3: recall is measured on a corpus-derived known-item set, not on FEVER.**
  BUILD_PLAN §M3 defines `recall@5` as "fraction of FEVER claims where gold
  evidence is in the top 5". Not measurable here — FEVER's gold evidence is
  Wikipedia pages and this corpus contains none, so the metric would read 0.00
  for both the old and new index and prove nothing. Replaced with known-item
  retrieval (query = a real passage, gold = its source article), which measures
  the truncation fix directly and supports a real A/B. FEVER is kept for
  abstention rate, verdict accuracy on what it answers, and false confidence.
- **M2: `CrossEncoderReranker` subclassed, not used directly.** The stock class
  computes scores and throws them away, so `RelevanceFloor` (which reads
  `metadata['relevance_score']`) would have seen `0.0` for every document and
  dropped everything — **every claim would abstain, and the abstain test would
  have passed for entirely the wrong reason.** See the next section.

---

## ⚠ Plan change: no Groq vision model exists on this key

`python -m core.llm --list` returns 15 models, **none multimodal**:

```
allam-2-7b · canopylabs/orpheus-{arabic-saudi,v1-english}
groq/compound · groq/compound-mini
llama-3.1-8b-instant · llama-3.3-70b-versatile
meta-llama/llama-prompt-guard-2-{22m,86m}
openai/gpt-oss-{120b,20b} · openai/gpt-oss-safeguard-20b
qwen/qwen3.6-27b · whisper-large-v3 · whisper-large-v3-turbo
```

**Voice is safe** — `whisper-large-v3` is there, so the important modality works
as planned. **Image OCR via Groq is not possible.** Decide at M5:

| Option | Cost |
|---|---|
| `pytesseract` (installed) + system Tesseract | ~50 MB Windows installer; a system dep in the README |
| `easyocr` | no system install, but ~100 MB models + slow on CPU |
| Second provider (Gemini free tier) | another API key |
| **Drop image input** | image is the least impressive of the four modalities; voice is the wow factor |

Leaning: drop it, or pytesseract if the README dep is acceptable. Revisit at M5.

Also noted: `groq/compound` has built-in web search and could substitute for
Tavily at M4 — but Tavily stays, because explicit retrieval is inspectable and
traceable, and a black-box search step weakens the architecture story.

## 🔥 Import order: sentence_transformers BEFORE chromadb (hard crash)

Cost the better part of an hour at M1. **Read this before writing any `serve/` module.**

```python
import sentence_transformers; import chromadb   # works
import chromadb; import sentence_transformers   # exit 0xC0000005, process dead
```

The bad order kills the interpreter **during the import itself** — access
violation, no Python traceback, no `MemoryError`, nothing catchable. It presents
as "the script produced no output and exited 5". Their native runtimes
(onnxruntime vs torch) conflict and whichever initialises first wins.

**Ruled out:** `KMP_DUPLICATE_LIB_OK=TRUE` (no effect), importing `torch` first
(no effect — only the *full* `sentence_transformers` import works), low memory
(a red herring, though the box only has 8 GB with ~1.6 GB free), and a corrupt
HF cache (blobs/ being empty is normal on Windows — files are copied into
snapshots/ rather than symlinked).

**The fix:** `core.init_native_libs()`, called at the top of any module that
imports chromadb directly or via `langchain_chroma`. `ingest/index.py` does this;
every `serve/` module that opens the vector store must too.

Diagnosis technique worth reusing: `python -X faulthandler script.py` prints a C
stack for these — that is what located it (crash inside
`sentence_transformers/__init__.py` line 15, not at model load as it appeared).

## 🔥 sentence_transformers breaks socket.getaddrinfo (hard crash)

Found at M2. **Same root cause as the "HF hub crash" recorded below** — that
entry's diagnosis was incomplete, and this is the real one.

```
baseline                       -> getaddrinfo OK
import torch                   -> getaddrinfo OK
import sentence_transformers   -> getaddrinfo CRASHES  (0xC0000005)
```

After `sentence_transformers` is imported, **any** DNS lookup kills the process.
It surfaced as the `generate` node dying mid-run with no traceback — the C stack
from `-X faulthandler` ended in `socket.getaddrinfo`, reached through
`httpx → httpcore → socket.create_connection` inside the Groq client.

Going offline (the M1 "fix") removed the *lookup*, not the *fault*. It could not
work here: calling Groq is the point.

**Ruled out:** warming the OS resolver cache first (crash is inside the call,
not in the lookup); `torch` alone.

**The fix:** `core/__init__.py` resolves the API hosts *before* importing
sentence_transformers, then installs a caching `socket.getaddrinfo` so the
broken path is never re-entered. Unknown hosts still fall through to the real
resolver. `init_native_libs()` now does both jobs — DNS pre-warm, then the
ordered import — so every existing call site got the fix for free.

Hosts are `VERIFAI_DNS_PREWARM` (default: Groq, Tavily, LangSmith). **Add any
new outbound host there**, or its first call will crash the process.

## 🔥 HF hub network calls crash on load — run offline

Separate crash, same session. `sentence-transformers` pings the Hub for updated
files on every model load; that httpx call intermittently access-violated too.
`core/__init__.py` now sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` at
import (before `huggingface_hub` reads them — it caches them at *its* import,
so setting them later is a no-op).

Correct regardless of the crash: Tier 1 should never need the network to rebuild
an index from a local CSV. **`HF_OFFLINE=false` for a first run on a cold model
cache**, then leave it alone.

## Gotchas discovered

- **`.gitignore` had blanket `*.json` and `*.jsonl` rules** (line 102-103, for
  bulk data dumps). They swallowed the entire M3 deliverable — datasets and
  results. Un-ignored explicitly at the bottom of the file. Also note
  `git check-ignore -v` prints the matching rule for **negations too**, so its
  output alone does not tell you whether a file is ignored; use the exit code
  (`git check-ignore -q path`; 0 = ignored).
- **The `fever` dataset in the HF cache is a loading *script*, not data.**
  `datasets--fever/` contains only `fever.py`, so `load_dataset("fever")` needs
  the network. `datasets--kilt_tasks/*/fever/*.parquet` has real rows and works
  offline — that is what `prepare_datasets.py` reads. KILT's FEVER split drops
  NOT ENOUGH INFO, so the gold labels are two-class only.
- **`models/*/final/` is gitignored (871 MB).** Correct for git, but it means a
  clean clone cannot run anything. Must be solved at M6 — see NEXT ACTION.

- **`CrossEncoderReranker` discards its scores.** LangChain's stock reranker
  sorts by score, slices `top_n`, returns bare documents — it never writes
  `relevance_score` to metadata, which is what BUILD_PLAN §6.5's `RelevanceFloor`
  reads. Chained as the plan wrote it, the floor sees `0.0` everywhere and drops
  every document. `core/compressors.py::ScoringCrossEncoderReranker` subclasses
  it to stamp the score. **Verify assumed metadata contracts before relying on
  them** — this one fails silently and in the direction that looks like success.
- **Cross-encoder output is an unbounded logit, not a 0–1 score.**
  `ms-marco-MiniLM-L-6-v2` has `num_labels=1` and sentence-transformers leaves
  `activation_fn=Identity()`. Measured range on this corpus: **-11.4 to +7.1**.
  A floor of 0.15 against a raw logit is meaningless. Sigmoid is applied before
  the floor sees it. Measured: relevant `+7.10 → 0.9992`, irrelevant
  `-11.14 → 0.00001`, **refuting `-0.06 → 0.485`** — that last one matters,
  contradicting evidence must survive the floor.
- **Two HuggingFace caches on this machine.** `HF_HOME=D:\hf-cache` is set as a
  user env var, so `C:\Users\user\.cache\huggingface` is **not** the active one.
  The cross-encoder was only in the C: cache and failed to load offline;
  copied it to `D:\hf-cache\hub`. Note `D:\hf-cache\hub\models--BAAI--bge-reranker-base`
  is entirely **zero-byte files** (aborted download) — do not switch to it.
- **The claim detector's 1.00 accuracy is not what it looks like.** It separates
  declarative sentences from questions/opinions; it does not judge verifiability.
  `"asdkjh askjdh"` scores 0.997 as a claim. Fine for the gate's actual job
  (keep questions out), but **do not claim "claim detection" on the resume
  without that caveat** — an interviewer will probe it.
- **The stance detector gets direct contradictions wrong.** Probed: `"India's
  economy contracted by 2 percent"` vs a claim of 8% growth → **SUPPORTS at
  0.519**. 73.6% overall, REFUTES F1 0.74. Its labels are passed to the LLM as
  hints with confidence attached, and `core/prompts.py` instructs the model to
  follow the evidence text where the two disagree. Observed working: the model
  overrode two stance labels in the live run.

- **chromadb conflict — checked, none.** `chromadb` 1.5.8 + `langchain-chroma`
  0.2.6 coexist; all imports resolve. Was a suspected risk, now ruled out.
- **VSCode shows "package not installed" hints in `pyproject.toml`.** The IDE has
  a different interpreter selected. Point it at `venv/Scripts/python.exe`
  (Ctrl+Shift+P → Python: Select Interpreter). The packages are installed.
- **`langgraph` resolved to 1.0.1**, not the 0.2 line the plan assumed. The
  `StateGraph` / `add_conditional_edges` / `astream` API used in BUILD_PLAN §6.7
  is unchanged. `pyproject.toml` now pins `>=1.0,<2` to match what was verified.
- **`publish_date` is fabricated** — 1,231 of 1,687 rows are `2024-01-01`
  placeholders from `extract_year_from_url`; only 15 distinct dates exist, and
  `data/raw/scraped_news.csv` already has them. Not recoverable by re-cleaning.
  → flag `date_reliable: false`, weight domain 0.7 / type 0.3, skip recency,
  record `"dates_reliable": false` in the manifest, disclose in the README.
- **MiniLM caps at 256 tokens (~1,000 chars).** Median article is 2,996 chars, so
  the old index silently dropped ~65% of the median document. This is the core
  bug the rebuild fixes and the best interview story in the project.
- **Old `tests/`** — 13 files, **zero `assert` statements**. They are print-driven
  demos. Do not count them as coverage.
- **dotenv + empty value + inline comment = the comment becomes the value.**
  `GROQ_VISION_MODEL=    # note` parsed as the literal string `# note` and blew up
  the startup check. Fixed in the template (comments on their own lines) and
  guarded in `core/config.py::_strip_inline_comment`. **Never put a `#` comment
  after a value in `.env`.**

---

## Repo map

Post-cleanup. `[brackets]` = planned, not yet written.

```
core/           __init__.py config.py llm.py text.py models.py
                credibility.py compressors.py prompts.py
ingest/         run.py clean.py chunk.py index.py [discover.py fetch.py sources.yaml]
serve/          cli.py graph.py schemas.py retriever.py [api.py]
                nodes/  gate.py retrieve.py stance.py generate.py
                        terminal.py render.py [adapt.py language.py websearch.py]
eval_harness/   prepare_datasets.py run_eval.py metrics.py baseline.py report.py
                datasets/{known_item_200,fever_dev_200,adversarial_20}.jsonl
                results/{retrieval_ab,floor_sweep,system_eval}.json RESULTS.md
tests/          conftest.py  test_silent_failures.py   13 tests, mutation-checked
demo/           run_demo.py  screenshots/              GIF goes here
scripts/        cleanup.py  prepare_fever_data_fixed.py (BERT training provenance)
notebooks/      claim-detection.ipynb  stance-detection.ipynb  (model provenance)
ui/             index.html          — reconnected at M5

index/          CURRENT -> v1/                     122 MB, gitignored
models/         claim_detector/final (418 MB)  stance_detector/final (414 MB)
data/           processed/news_articles_rag.csv    the M1 input
                chroma_db/                         the M3 baseline — DO NOT TOUCH
                .embed_cache/                      makes re-ingest a 1s no-op
                processed/*_train.csv, raw/fever_* BERT training provenance
_trash/         415 MB of removed files — delete once verified

src/            OLD — M4/M5 source material only; goes after M5
app.py          OLD — SSE reference for M5; does NOT import cleanly any more
README.md       OLD — describes the pre-rebuild system; M6 replaces it
```

**Top-level files (8):** `.env` `.env.example` `.gitignore` `BUILD_PLAN.md`
`CLAUDE.md` `pyproject.toml` `README.md` `app.py`

Down from 22 files + 5 stray dirs before cleanup.
