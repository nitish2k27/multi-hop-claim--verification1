# VerifAI — Build Plan

Two-tier, LangChain-backed fact verification system.
**Scope: portfolio / resume project.** Not deployed, not multi-user, not hardened.

Target: ~7 working days solo. You can stop after M3 and still have a strong
project — M4–M6 add demo surface to something already correct.

---

## 0. What "done" means here

This is not a production system, and pretending otherwise wastes days on work
nobody reviewing your resume will ever check. What actually gets judged:

1. **It runs.** Clean clone → working verification in under 5 minutes, following
   only the README. If it doesn't run on someone else's machine, nothing else counts.
2. **It has numbers.** "Improved retrieval recall@5 from 0.34 to 0.71 by fixing
   chunk-level truncation" is a resume bullet. "Built a RAG system" is not.
3. **It demos in 90 seconds.** Four canned claims that each show a different
   capability, runnable from one command.
4. **It has a story.** Interviewers ask "what was hard?" You need a real answer.
5. **It reads well.** Someone will skim the repo for 60 seconds. Structure and
   README carry that.

### Explicitly cut — do not build these

| Cut | Why |
|---|---|
| Auth, rate limiting, CORS lockdown | Single-user local demo. One README line covers it. |
| Opaque report IDs, upload size caps | Same. |
| CI pipelines | Run the eval manually; paste numbers in the README. |
| Resumable crawl checkpointing | Your corpus is 1,700 docs. A re-run is 2 minutes. |
| `robots.txt` engine, per-domain politeness tuning | A 1-second delay between requests is enough at this scale. |
| Dual `PROFILE=lean\|cloud` config | Over-engineering. Pick lean, move on. |
| Full pytest suite | ~8 tests on the parts that break silently. That's it. |
| Qdrant / hosted vector DB | Chroma is a file. Keep it. |
| Wikipedia full-dump ingestion | Web search fallback covers the same demo need for a fraction of the work. |

### Locked technical decisions

| Question | Decision |
|---|---|
| Local LLM? | **No.** Groq only. Archive the Mistral LoRA off-repo. |
| Speech-to-text | Groq Whisper API — same key you already have |
| Image OCR | Groq vision model (OCR + claim extraction in one call) |
| Translation | Groq (drops `deep-translator`) |
| Language detect | `langdetect` (drop the 125 MB FastText model) |
| Embeddings | `all-MiniLM-L6-v2`, local — free, unlimited, ~80 MB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, local |
| Claim + stance | Your two trained BERTs — keep. Best asset in the repo. |
| Vector store | Chroma, local file |
| Web search | Tavily |
| Orchestration | LangGraph |
| Re-crawl for the first index? | **No.** Re-index from the CSV you already have. |

Local footprint: four small CPU models, ~1 GB, ~2 s startup. No GPU.

---

## 1. Architecture

```
TIER 1 · ingest/          TIER 2 · serve/
run on demand             runs while you demo
write-only                read-only

discover                          adapt input
   |                                  |
fetch                            translate -> EN
   |                                  |
clean + dedupe                    claim gate ---------> reject
   |                                  |
chunk (700/100)                   retrieve --+
   |                                  |      | none
embed + upsert                        |    web search --+
   |                                  |      |          | none
   v                                  |    hits         v
+---------------------------+         |      |       abstain
|      THE CONTRACT         | <-------+------+          |
|  index/current/           |  reads   |                |
|    chroma/                |          v                |
|    parents/               |    stance + credibility   |
|    bm25.pkl               |          |                |
|    manifest.json          |     Groq -> verdict       |
+---------------------------+          |                |
                                       v                |
                                    render <------------+
```

**The two-tier split is the architectural story.** Say it in interviews like
this: *"Ingestion and serving share files, not imports. The index is a versioned
artifact — I can rebuild the corpus or swap the embedding model without touching
serving code, and the API restarts without re-crawling anything."*

The manifest is a safety interlock: it records the embedding model and dimension,
and Tier 2 refuses to boot on a mismatch. That catches the classic silent RAG
failure — index with one model, query with another, get garbage and no error.

---

## 2. Restructure

### 2.1 Delete

| Path | Why |
|---|---|
| `src/rag/retrieval.py` | Dead second `RAGPipeline`; its API can't work against `VectorDatabase` |
| `src/main_pipeline.py` | Imports that dead class — broken since written |
| `src/rag/enhanced_rag_pipeline.py` | Keyword stance engine that overrides your trained model |
| `src/data_processing/input_processor.py` | Duplicate of the `preprocessing/` one |
| `src/generation/report_exprotex_xx.py` | Zero bytes |
| `src/generation/report_generator.py`, `scripts/set_inference_url.py` | Colab/ngrok path — gone |
| `scripts/prepare_fever_data.py`, `scripts/export_pipeline_simple.py` | Keep one of each pair |
| `test_environment*.py`, `test_pipeline_*.py`, `test_single_export.py` | Root scratch files |
| `configs/groq_token.txt`, `configs/inference_url.txt` | **Rotate the key first**, then delete |
| `models/language_detection/lid.176.bin` | 125 MB, unused on the live path |
| `PIPELINE_FIXES_SUMMARY.md`, `SETUP_COMPLETE.md`, `VIRTUAL_ENV_SETUP.md`, `implementation_status_report.md`, `technical_decisions_status.md`, `docs/project_staus.md` | Six overlapping status files. One README instead. |
| `data/reports/`, `data/reports_groq/` | Build artifacts committed to the tree |

Archive off-repo (keep the weights, remove from project): `models/mistral_fv_adapter/`

### 2.2 Move — this is ~60% of the codebase surviving

| From | To | Change |
|---|---|---|
| `src/generation/prompt_builder.py` | `core/prompts.py` | **Verbatim.** Best file you have. |
| `src/rag/credibility_scorer.py` | `core/credibility.py` | Rewire to `metadata['domain']` |
| `src/nlp/claim_detection.py` | `serve/nodes/gate.py` | Wrap as `RunnableLambda` |
| `src/nlp/stance_detection.py` | `serve/nodes/stance.py` | Wrap as `RunnableLambda` |
| `src/nlp/model_manager.py` | `core/models.py` | Lazy singletons |
| `src/data_processing/clean_news_data.py` | `ingest/clean.py` | Date fixes only |
| `src/document_processing/document_handler.py` | `serve/nodes/adapt.py` | Cap upload credibility |
| `src/generation/report_exporter.py` | `serve/nodes/render.py` | Delete its regex verdict parsers |
| `app.py` | `serve/api.py` | Keep SSE, drop the inline pipeline |
| RSS URLs in `src/data_collection/*_scraper.py` | `ingest/sources.yaml` | Config, not code |
| `scripts/prepare_fever_data_fixed.py` | `eval/prepare_datasets.py` | — |
| `models/claim_detector/`, `models/stance_detector/`, `ui/index.html` | unchanged | — |

### 2.3 Target tree

```
verifai/
├── ingest/                  # TIER 1
│   ├── run.py               # python -m ingest.run [--from-csv] [--crawl]
│   ├── discover.py          # RSSFeedLoader
│   ├── fetch.py             # simple polite fetch
│   ├── clean.py             # <- clean_news_data.py
│   ├── chunk.py             # splitter + metadata stamping
│   ├── index.py             # embed, upsert, parents, bm25, manifest
│   └── sources.yaml
│
├── serve/                   # TIER 2
│   ├── api.py               # FastAPI + SSE from graph.astream
│   ├── cli.py               # python -m serve.cli "claim"
│   ├── graph.py             # LangGraph wiring
│   ├── schemas.py           # VerifyState, VerificationReport
│   └── nodes/
│       ├── adapt.py  language.py  gate.py  retrieve.py
│       └── websearch.py  stance.py  generate.py  render.py
│
├── core/                    # the ONLY shared surface
│   ├── config.py  models.py  llm.py
│   ├── credibility.py  compressors.py  prompts.py
│
├── eval/
│   ├── prepare_datasets.py  run_eval.py
│   └── datasets/{fever_dev_200.jsonl, adversarial_20.jsonl}
│
├── index/                   # THE CONTRACT — gitignored
│   ├── current -> v1/
│   └── v1/{chroma/, parents/, bm25.pkl, manifest.json}
│
├── demo/
│   ├── run_demo.py          # 4 canned claims, one command
│   └── screenshots/
│
├── data/  models/  ui/  tests/
├── .env.example
├── README.md                # THE DELIVERABLE
└── pyproject.toml
```

**Rule:** `serve/` imports nothing from `ingest/`. If it does, the boundary leaked.

---

## 3. Data plan

**Do not re-crawl for the first index.** `data/processed/news_articles_rag.csv`
has 1,687 rows, full text, **zero nulls** across `url, text, title, source,
publish_date, language, domain, category`. That's everything the pipeline needs.

**You must re-embed.** Current vectors captured only the first ~1,000 characters
of each article — MiniLM truncates at 256 tokens. That information was never
stored, so there is nothing to repair.

```
7,338,249 chars  ->  ~12,744 chunks @ 700/100
mean 7.6 chunks/article, median 5, max 75
runtime: ~2 minutes, CPU, no network
```

**7.5× more of your corpus becomes searchable.** That 75-chunk article was ~97%
invisible before. This is your best "what was hard" interview answer — see §7.

### Known defect: publish dates are fabricated

```
2024-01-01  1231   <- placeholder from extract_year_from_url
2026-01-01   320   <- placeholder
2026-03-10    31   <- real
distinct dates across 1,687 articles: 15
```

`data/raw/scraped_news.csv` already has the placeholders, so this happened at
scrape time. Re-running the cleaner cannot recover them.

**Handle it, don't hide it:**
- Flag Jan-1 dates as `date_reliable: false` in chunk metadata.
- When false, drop the recency term and weight domain 0.7 / type 0.3 instead of
  0.5/0.3/0.2. Never let a fabricated date drive 30% of a credibility score.
- Put `"dates_reliable": false` in the manifest and mention it in the README's
  limitations section. **Documenting a known data defect reads as maturity.**

---

## 4. Milestones

### M0 — Ground clearing · half a day

- [ ] `git checkout -b rebuild/two-tier`
- [ ] **Rotate the Groq API key** — it's in plaintext in `configs/groq_token.txt`
- [ ] `.env` with the new key; commit `.env.example` with empty values
- [ ] Delete §2.1, move §2.2, create the §2.3 tree
- [ ] `pyproject.toml` (§6.1), install
- [ ] `core/config.py` — pydantic-settings, validates on import
- [ ] `core/llm.py` — `init_chat_model` + startup check against `GET /openai/v1/models`

**Check:** `python -c "from core.llm import llm; print(llm.invoke('say OK').content)"`
prints `OK`. With a wrong `GROQ_MODEL`, it fails at import with a clear message.

---

### M1 — Tier 1 from the CSV · 1 day

- [ ] `ingest/chunk.py` — splitter + metadata stamping (§6.2)
- [ ] `ingest/clean.py` — port `clean_news_data.py`; add the Jan-1 flag; `dateutil`
- [ ] `ingest/index.py` — sha256 chunk IDs, parent docstore, BM25 pickle, manifest (§6.3)
- [ ] `ingest/run.py --from-csv`

**Check:** `python -m ingest.run --from-csv`
- `index/v1/` has `chroma/`, `parents/`, `bm25.pkl`, `manifest.json`
- manifest: ~12,700 chunks, 1,687 docs, `embed_dim: 384`, `dates_reliable: false`
- **run it twice** — second run upserts nothing, finishes in seconds
- `data/chroma_db/` untouched (that's your A/B baseline for M3)

---

### M2 — Tier 2 core · 2 days ← **the money milestone**

- [ ] `serve/schemas.py` — `VerifyState`, `VerificationReport` (§6.4)
- [ ] `core/compressors.py` — `RelevanceFloor`, `CredibilityScorer` (§6.5)
- [ ] `serve/nodes/retrieve.py` — MultiVector + Ensemble + pipeline (§6.6)
- [ ] `gate.py`, `stance.py` — trained BERTs as `RunnableLambda`
- [ ] `core/prompts.py` — `prompt_builder.py` as a `ChatPromptTemplate`
- [ ] `generate.py` — `with_structured_output` + citation validation
- [ ] `serve/graph.py` — full graph, **including abstain and reject edges** (§6.7)
- [ ] `serve/cli.py`

**Check — all three must pass:**
```bash
python -m serve.cli "India's GDP grew 8% in 2024"   # verdict + evidence
python -m serve.cli "the moon is square in shape"   # ABSTAINED, zero LLM calls
python -m serve.cli "what time is it"               # REJECTED at the gate
```

The abstain case is the whole project. Today that claim produces a 700-word
report analysing Rashmika Mandanna's bridal henna and a foldable phone review.
After M2 it must return nothing and cost nothing.

Also: startup fails loudly on a manifest/embedder mismatch, and every
`evidence_index` in `key_findings` resolves to a real retrieved document.

---

### M3 — Eval harness · 1 day ← **your resume bullets live here**

- [ ] `eval/prepare_datasets.py` — 200 FEVER claims → `fever_dev_200.jsonl`
- [ ] Hand-write `adversarial_20.jsonl` — 20 claims your corpus genuinely cannot
      answer. Correct behaviour is abstention. "The moon is square" is #1.
- [ ] `eval/run_eval.py`
- [ ] LangSmith tracing on (free, one env var — and the trace view screenshots well)

| Metric | Definition |
|---|---|
| `recall@5` | fraction of FEVER claims where gold evidence is in the top 5 |
| `verdict_accuracy` | 3-class: SUPPORTS / REFUTES / NOT ENOUGH INFO |
| `abstain_precision` | of the adversarial set, fraction correctly abstained |
| `false_confidence` | wrong verdicts issued above 70% confidence — target 0 |

**Run it twice: once against `data/chroma_db/` (the old truncated index) and once
against `index/v1/`.** That before/after delta is the most valuable output of this
entire project. Put the table in the README.

**Calibrate `RELEVANCE_FLOOR` here** — sweep 0.05–0.40, pick the value maximising
`abstain_precision` without dropping `recall@5`. The `0.15` in the config is a
placeholder. Do not ship a guessed number.

> **You could stop here.** Two-tier architecture, LangGraph, a real eval harness
> with before/after numbers, and correct abstention. That is already a better
> portfolio project than most. Everything below is demo surface.

---

### M4 — Web search fallback · half a day

- [ ] `serve/nodes/websearch.py` — Tavily, fires only when the index abstains

Small change, large demo payoff: it means **any** claim someone throws at your
demo produces a sensible answer instead of an abstention, while still showing the
abstain path when search also comes up empty. This is why Wikipedia ingestion is
cut — Tavily covers the same need in half a day instead of two.

**Check:** a claim about something not in your news corpus now resolves via web
search, and the trace shows the fallback edge firing.

---

### M5 — Modalities and languages · 2 days

- [ ] `adapt.py` — Groq Whisper (voice), Groq vision (image), PDF/DOCX, URL
- [ ] `language.py` — `langdetect` + Groq translation. Report is **generated in**
      the target language, not back-translated — your `LANG_INSTRUCTIONS` table
      already does this.
- [ ] Uploaded docs → `user_uploads` collection, **credibility capped at 0.6**,
      labelled as an unverified submission
- [ ] `render.py` — md / html / docx, plus mp3 via gTTS for voice input
- [ ] `serve/api.py` — FastAPI, SSE via `graph.astream(stream_mode="updates")` (§6.8)
- [ ] Reconnect `ui/index.html`

| Input | Comes back |
|---|---|
| Text | report in detected language · md + html + docx · JSON |
| Voice | transcript · report · **mp3 summary** in that language · JSON |
| Image | extracted text · report · JSON |
| PDF/DOCX (no claim) | extracted claim · report |
| PDF/DOCX (+ claim) | report; doc ranked first, capped at 0.6 credibility |
| URL | extracted claim · report |

**Check:** speak a Hindi claim into the UI → Hindi report + Hindi audio back.
Upload a PDF asserting something false, with a matching claim — the verdict must
**not** be TRUE just because your document says so.

---

### M6 — Package it · 1 day ← **what recruiters actually see**

- [ ] `demo/run_demo.py` — the four claims below, one command, ~90 seconds
- [ ] **Record a 60–90 second screen capture.** Embed the GIF at the top of the README.
- [ ] README (§6.9 outline) — architecture diagram, the M3 numbers table, quickstart,
      limitations
- [ ] ~8 tests on the things that break silently: manifest/embedder mismatch,
      idempotent re-ingest, relevance floor filtering, citation resolution,
      credibility with unreliable dates, abstain routing, reject routing,
      upload credibility cap
- [ ] Clean-clone test: fresh directory, follow only the README, time it

**The four demo claims:**

| Claim | Shows |
|---|---|
| `"India's GDP grew 8% in 2024"` | normal path — retrieval, stance, cited verdict |
| `"the moon is square in shape"` | **abstention** — the differentiator |
| a Hindi voice clip | multilingual + multimodal, audio out |
| a claim about this week's news | web search fallback |

**Check:** someone else clones it, follows the README, and gets a verdict in under
5 minutes without asking you anything.

---

## 5. For the resume itself

### Bullets — fill in your real numbers from M3

> **Fact Verification System** — Python, LangChain, LangGraph, ChromaDB, Groq
> - Built a two-tier RAG system separating offline ingestion from serving via a
>   versioned index artifact, allowing corpus rebuilds without redeploying the API.
> - Diagnosed and fixed silent embedding truncation where 65% of the median
>   document was never indexed; raised retrieval recall@5 from **X to Y** across
>   200 FEVER claims.
> - Implemented an abstention path so claims without supporting evidence return
>   "insufficient evidence" instead of a fabricated analysis — **N%** correct
>   abstention on a hand-built adversarial set.
> - Fine-tuned BERT classifiers for claim detection and FEVER stance; combined
>   with hybrid dense/BM25 retrieval and cross-encoder reranking.
> - Multimodal input (text, voice, image, PDF) across 20 languages, with
>   schema-constrained LLM output enforcing that every claim cites retrieved evidence.

### Interview answers to have ready

**"What was the hardest bug?"**
> The retriever looked fine but returned irrelevant results. The embedding model
> caps at 256 tokens and I was passing whole articles — median 3,000 characters —
> so two thirds of every document was silently truncated at embed time. No error
> anywhere. Worse, BM25 indexed the full text, so dense and sparse retrieval
> disagreed about what a document even contained. Fixed it by chunking at 700
> characters with parent-document expansion, and validated with a before/after
> recall measurement.

**"How do you know it works?"**
> 200 FEVER claims for accuracy, plus 20 hand-written claims the corpus can't
> answer, where correct behaviour is abstention. I measure recall@5, 3-class
> verdict accuracy, abstention precision, and specifically the count of wrong
> verdicts issued above 70% confidence — because a confident wrong answer is
> worse than no answer.

**"Why LangChain?"**
> Retriever composition and structured output, mainly. `DocumentCompressorPipeline`
> lets me chain reranking, a relevance floor and credibility scoring as one object.
> `with_structured_output` means the verdict is a validated Pydantic model instead
> of something I regex out of prose — which was a real bug in my first version,
> where the exported report could contradict the API response. I didn't force
> everything through it: crawling and file rendering are plain Python.

**"What would you do differently?"**
> Build the eval harness first. I spent time tuning retrieval before I could
> measure it. Also, my news corpus is the wrong shape for general fact-checking —
> it's mostly tech and entertainment RSS, so web search fallback does more work
> than it should. A Wikipedia tier would fix that.

---

## 6. Reference code

### 6.1 `pyproject.toml`

```toml
[project]
name = "verifai"
requires-python = ">=3.11"
dependencies = [
  "langchain~=0.3", "langchain-core~=0.3", "langchain-community~=0.3",
  "langgraph~=0.2", "langchain-chroma", "langchain-huggingface",
  "langchain-text-splitters", "langchain-groq", "langchain-tavily",
  "chromadb", "rank-bm25", "sentence-transformers", "transformers", "torch",
  "fastapi", "uvicorn", "python-multipart",
  "pydantic-settings", "python-dotenv", "pyyaml",
  "pandas", "numpy", "python-dateutil", "langdetect",
  "feedparser", "newspaper3k", "beautifulsoup4", "requests",
]

[project.optional-dependencies]
docs  = ["PyPDF2", "pdfplumber", "python-docx"]
voice = ["gtts"]
dev   = ["pytest", "langsmith"]
```

Pin exact versions once it works — LangChain's package split moves between minors
and import paths shift with it.

### 6.2 Chunking with correct metadata

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def to_chunks(row):
    parent = Document(page_content=row.text, metadata={"url": row.url})
    for i, ch in enumerate(splitter.split_documents([parent])):
        ch.metadata.update({
            "chunk_id":      f"{row.url}#{i}",
            "parent_id":     row.url,
            "url":           row.url,
            "domain":        row.domain,        # credibility keys off THIS
            "source":        row.source,
            "publish_date":  row.publish_date,
            "date_reliable": not is_placeholder(row.publish_date),
        })
        yield ch
```

### 6.3 Index build

```python
from langchain.storage import LocalFileStore, create_kv_docstore

# sha256 ids -> re-running upserts identical rows, a no-op
store.add_documents(chunks, ids=[sha256(c.page_content) for c in chunks])

parents = create_kv_docstore(LocalFileStore(build_dir / "parents"))
parents.mset([(d.metadata["url"], d) for d in parent_docs])

pickle.dump(BM25Retriever.from_documents(chunks), open(build_dir/"bm25.pkl","wb"))

manifest = {
    "version": version, "built_at": utcnow_iso(),
    "embed_model": cfg.embed_model, "embed_dim": 384,
    "chunk_size": 700, "chunk_overlap": 100,
    "documents": n_docs, "chunks": n_chunks,
    "dates_reliable": False,
}
# validate (chunks > 0, dim matches, smoke query returns hits), THEN swap `current`
```

### 6.4 Output schema

```python
from pydantic import BaseModel, Field
from typing import Literal

class Citation(BaseModel):
    evidence_index: int
    quote: str

class EvidenceAssessment(BaseModel):
    source: str
    credibility: float
    stance: Literal["SUPPORTS", "REFUTES", "NEUTRAL"]
    reasoning: str
    directly_relevant: bool

class VerificationReport(BaseModel):
    claim: str
    claim_type: Literal["statistical","event","policy","scientific","biographical","other"]
    evidence_analysis: list[EvidenceAssessment]
    contradictions: list[str]
    verdict: Literal["TRUE","MOSTLY_TRUE","UNVERIFIABLE","MOSTLY_FALSE","FALSE"]
    confidence: int = Field(ge=0, le=100)
    key_findings: list[Citation] = Field(max_length=5)
    limitations: list[str]
    conclusion: str
```

Enforce in code what the prompt only requests: reject any generation whose
`evidence_index` values don't resolve to retrieved documents, retry once.

### 6.5 Post-retrieval compressors

```python
from langchain_core.documents import BaseDocumentCompressor

class RelevanceFloor(BaseDocumentCompressor):
    floor: float = 0.15
    def compress_documents(self, documents, query, callbacks=None):
        return [d for d in documents
                if d.metadata.get("relevance_score", 0.0) >= self.floor]

class CredibilityScorer(BaseDocumentCompressor):
    def compress_documents(self, documents, query, callbacks=None):
        for d in documents:
            d.metadata["credibility"] = score_domain(d.metadata)
        return documents
```

`score_domain` reads `metadata['domain']` (e.g. `www.bbc.com`), strips a leading
`www.` with `removeprefix`, looks it up in the curated table. When `date_reliable`
is false: domain 0.7 / type 0.3, skip recency.

### 6.6 Retriever assembly (once, at startup)

```python
from langchain.retrievers import MultiVectorRetriever, EnsembleRetriever, \
                                 ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, CrossEncoderReranker)

parents = create_kv_docstore(LocalFileStore(idx / "parents"))
dense   = MultiVectorRetriever(vectorstore=store, docstore=parents,
                               id_key="parent_id", search_kwargs={"k": 20})
sparse  = pickle.load(open(idx / "bm25.pkl", "rb"))

pipeline = DocumentCompressorPipeline(transformers=[
    CrossEncoderReranker(model=cross_encoder, top_n=8),
    RelevanceFloor(floor=cfg.relevance_floor),
    CredibilityScorer(),
])

retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5]),
)
```

### 6.7 The graph

```python
from langgraph.graph import StateGraph, END

g = StateGraph(VerifyState)
for name, fn in [
    ("adapt", adapt_input), ("translate", to_english), ("gate", claim_gate),
    ("retrieve", retrieve_evidence), ("web", web_search),
    ("stance", stance_and_credibility), ("generate", generate_report),
    ("abstain", abstain_report), ("reject", reject_report), ("render", render_output),
]:
    g.add_node(name, fn)

g.set_entry_point("adapt")
g.add_edge("adapt", "translate")
g.add_edge("translate", "gate")
g.add_conditional_edges("gate",     lambda s: "retrieve" if s["is_claim"] else "reject")
g.add_conditional_edges("retrieve", lambda s: "stance" if s["evidence"] else "web")
g.add_conditional_edges("web",      lambda s: "stance" if s["evidence"] else "abstain")
g.add_edge("stance", "generate")
for terminal in ("generate", "abstain", "reject"):
    g.add_edge(terminal, "render")
g.add_edge("render", END)

app_graph = g.compile()
```

`app_graph.get_graph().draw_mermaid_png()` renders this — **use it as the
architecture diagram in your README.** Generated from the real code, so it can't
drift from what the system does.

### 6.8 SSE from the graph

```python
async def event_stream(state):
    async for chunk in app_graph.astream(state, stream_mode="updates"):
        for node, update in chunk.items():
            yield sse({"type": "step_done", "step_id": node,
                       "detail": summarise(node, update)})
```

Node names become step IDs. Deletes the hand-rolled queue, executor and
`call_soon_threadsafe` plumbing in the current `app.py`.

### 6.9 README outline — the actual deliverable

```
# VerifAI
One line: what it does.
[demo.gif]

## What makes it different
The abstention behaviour. Lead with this.

## Results          <- the M3 before/after table
## Architecture     <- the mermaid diagram from 6.7
## Quickstart       <- 4 commands, must work on a clean clone
## How it works     <- the two-tier contract, ~200 words
## Tech stack
## Limitations      <- fabricated dates, corpus skew. Be honest; it reads well.
```

### 6.10 `.env.example`

```
GROQ_API_KEY=
TAVILY_API_KEY=
LANGSMITH_API_KEY=

GROQ_MODEL=llama-3.3-70b-versatile      # verify against GET /openai/v1/models
GROQ_WHISPER_MODEL=whisper-large-v3
GROQ_VISION_MODEL=

INDEX_PATH=index/current
EMBED_MODEL=all-MiniLM-L6-v2
RELEVANCE_FLOOR=0.15                    # calibrated at M3, not guessed
TOP_K=8
```

---

## 7. Definition of done

- [ ] Clean clone → working verification in under 5 minutes, README only
- [ ] `python -m ingest.run --from-csv` twice → identical output
- [ ] Tier 2 fails loudly at startup on a manifest/embedder mismatch
- [ ] All three exits render — verified, abstained, rejected. No path returns a trace.
- [ ] Every citation resolves to a retrieved document, enforced in code
- [ ] `serve/` imports nothing from `ingest/`
- [ ] No secret in any file under `configs/`
- [ ] `demo/run_demo.py` runs all four claims in ~90 seconds
- [ ] README has the before/after numbers table and the demo GIF
- [ ] You can answer all four interview questions in §5 without notes

---

## 8. Start here

```bash
git checkout -b rebuild/two-tier

# 1. rotate the key at console.groq.com, then:
cp .env.example .env          # paste the NEW key

# 2. confirm the model id is current
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     https://api.groq.com/openai/v1/models | grep -o '"id":"[^"]*"'

# 3. M0 -> M6 in order. Do not skip M3.
```

**Priority if time runs short:** M0 → M1 → M2 → M3 → M6. Ship with numbers, a
README and a demo, even without voice or images. A small project that runs and
has measured results beats a large one that doesn't.
