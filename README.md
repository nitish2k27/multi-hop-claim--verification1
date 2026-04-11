# VerifAI — Multilingual Fact Verification System

> An end-to-end AI pipeline that takes a claim in any language (text, audio, image, PDF, DOCX), verifies it against a knowledge base using NLP and RAG, and generates a structured credibility report via a Large Language Model.

---

## Table of Contents

1. [What Is This System?](#1-what-is-this-system)
2. [How to Run the System](#2-how-to-run-the-system)
3. [The Mother File — Entry Points and Import Order](#3-the-mother-file--entry-points-and-import-order)
4. [Full Pipeline Flow — Step by Step](#4-full-pipeline-flow--step-by-step)
5. [Complete File-by-File Breakdown](#5-complete-file-by-file-breakdown)
   - [A. Root Level Files](#a-root-level-files)
   - [B. NLP Module](#b-nlp-module-srcnlp)
   - [C. RAG Module](#c-rag-module-srcrag)
   - [D. Generation Module](#d-generation-module-srcgeneration)
   - [E. Multilingual Module](#e-multilingual-module-srcmultilingual)
   - [F. Document Processing](#f-document-processing-srcdocument_processing)
   - [G. Voice Processing](#g-voice-processing-srcvoice_processing)
   - [H. Data Collection](#h-data-collection-srcdata_collection)
   - [I. Preprocessing](#i-preprocessing-srcpreprocessing)
   - [J. Data Processing](#j-data-processing-srcdata_processing)
   - [K. Scripts](#k-scripts)
   - [L. Tests](#l-tests)
6. [NLP Part vs GenAI Part — Clear Classification](#6-nlp-part-vs-genai-part--clear-classification)
7. [All Technologies and Libraries Used](#7-all-technologies-and-libraries-used)
8. [Configuration Files](#8-configuration-files)
9. [Data Directories](#9-data-directories)
10. [Known Issues and Limitations](#10-known-issues-and-limitations)
11. [Quick Reference — Important Methods Table](#11-quick-reference--important-methods-table)

---

## 1. What Is This System?

VerifAI is a **fact verification system** designed to answer the question:
> *"Is this claim true or false — and how confident are we?"*

**Core capabilities:**
- Accepts claims as **text, audio recording, image/screenshot, PDF, or DOCX**
- Supports **20+ languages** — detects language automatically and translates to English for NLP processing
- Runs the claim through a **4-stage NLP analysis**: claim detection → named entity recognition → entity linking → temporal extraction
- Searches a **ChromaDB vector knowledge base** using hybrid vector+BM25 retrieval to find supporting/refuting evidence
- Scores each evidence source's **credibility** (Reuters = 0.95, blogs = lower)
- Detects **stance** of each evidence piece: SUPPORTS / REFUTES / NEUTRAL
- Calls a **Large Language Model** (Groq API / Colab-hosted Mistral) to generate a structured report
- Exports the report as **HTML, PDF, and DOCX**
- For voice input: generates an **MP3 audio response**
- Streams **live step-by-step progress** to the browser UI via Server-Sent Events (SSE)

---

## 2. How to Run the System

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (for LLM report generation)
python scripts/setup_groq.py YOUR_GROQ_KEY

# 3. Start the backend
uvicorn app:app --host 0.0.0.0 --port 8000

# 4. Open the UI
# Open ui/index.html in your browser
```

For Python-only usage (no web UI):
```python
from src.fact_verification_service import FactVerificationService

svc = FactVerificationService()
result = svc.verify(claim="India GDP grew 8%", llm_mode="groq")
print(result["report"])
```

---

## 3. The Mother File — Entry Points and Import Order

### The Mother File (System Orchestrator)

```
src/fact_verification_service.py   <- THE MOTHER FILE
```

This is the **single entry point** for the entire pipeline. Everything flows through `FactVerificationService.verify()`. All components are **lazy-loaded** (they only initialize when first called — no 2-minute wait on startup).

### How the System Boots Up (Import Order)

When `app.py` (the FastAPI server) starts, here is the order in which modules get imported and initialized:

```
app.py
 └── (on first /verify/stream request)
      └── src/fact_verification_service.py          <- imported for helpers
           |
           ├── src/nlp/nlp_pipeline.py              <- LAZY LOAD on Step 4
           |    ├── src/nlp/model_manager.py         <- loads YAML config, decides model
           |    ├── src/nlp/claim_detection.py       <- zero-shot or trained classifier
           |    ├── src/nlp/entity_extraction.py     <- BERT NER model
           |    ├── src/nlp/entity_linking.py        <- Wikidata API calls
           |    ├── src/nlp/temporal_extraction.py   <- regex + dateutil rules
           |    └── src/nlp/stance_detection.py      <- NLI model
           |
           ├── src/rag/vector_database.py            <- LAZY LOAD on Step 5
           |    └── chromadb + sentence-transformers
           |
           ├── src/rag/rag_pipeline.py               <- LAZY LOAD on Step 5
           |    ├── src/rag/hybrid_retrieval.py      <- vector + BM25 fusion
           |    |    ├── src/rag/vector_database.py
           |    |    └── src/rag/sparse_retrieval.py <- BM25 on all docs
           |    ├── src/rag/reranker.py              <- cross-encoder reranking
           |    ├── src/rag/credibility_scorer.py    <- domain authority scores
           |    └── src/nlp/stance_detection.py      <- shared from NLP
           |
           ├── src/multilingual/translator.py        <- LAZY LOAD on non-English input
           |
           ├── src/generation/report_generator_groq.py  <- LAZY LOAD on Step 6
           |    └── src/generation/prompt_builder.py
           |
           └── src/generation/report_exporter.py    <- LAZY LOAD on Step 7
```

### Import Trigger Summary

| Component | When It Loads |
|-----------|--------------|
| `NLPPipeline` | First NLP analysis call |
| `VectorDatabase` | First RAG retrieval call |
| `RAGPipeline` | First `verify_claim()` call |
| `Translator` | First non-English input |
| `DocumentHandler` | First PDF/DOCX upload |
| `SpeechHandler` | First audio input |
| `ReportGeneratorGroq` | First LLM generation call |
| `ReportExporter` | First export call |

---

## 4. Full Pipeline Flow — Step by Step

```
User Input (text / image / audio / PDF / DOCX)
        |
        v
+-----------------------------------------------------+
|  STEP 1 — INPUT EXTRACTION                         |
|  File type detected -> OCR / Whisper STT / PDF     |
|  Result: raw text string + detected language        |
+-----------------------------------------------------+
        |
        v
+-----------------------------------------------------+
|  STEP 2 — DOCUMENT INGESTION (if PDF/DOCX given)   |
|  Chunks document -> generates embeddings ->         |
|  stores in ChromaDB "uploaded_documents" collection |
+-----------------------------------------------------+
        |
        v
+-----------------------------------------------------+
|  STEP 3 — LANGUAGE DETECTION + TRANSLATION         |
|  langdetect / script-based detection               |
|  deep-translator: Hindi/Tamil/... -> English        |
|  English claim ready for NLP                        |
+-----------------------------------------------------+
        |
        v
+-----------------------------------------------------+
|  STEP 4 — NLP ANALYSIS PIPELINE  [NLP TERRITORY]  |
|  4a. Claim Detection (BERT zero-shot classifier)   |
|  4b. Named Entity Recognition (dslim/bert-base-NER)|
|  4c. Entity Linking (Wikidata REST API)            |
|  4d. Temporal Extraction (regex + dateutil)        |
|  Result: structured NLP analysis dict              |
+-----------------------------------------------------+
        |
        v
+-----------------------------------------------------+
|  STEP 5 — RAG PIPELINE  [NLP + AI TERRITORY]       |
|  5a. Hybrid Retrieval: vector search (ChromaDB) +  |
|      BM25 keyword search, fused via RRF            |
|  5b. Re-ranking: cross-encoder scores candidates   |
|  5c. Credibility Scoring: domain authority tiers   |
|  5d. Stance Detection: NLI model per evidence piece|
|  5e. Evidence Aggregation: weighted verdict calc   |
|  5f. LLM Context Preparation: structured string    |
|  Result: verdict + confidence + evidence + context |
+-----------------------------------------------------+
        |
        v
+-----------------------------------------------------+
|  STEP 6 — LLM REPORT GENERATION [GEN AI TERRITORY]|
|  Prompt built from RAG context (prompt_builder.py) |
|  Sent to Groq API (llama-3.3-70b-versatile)        |
|  OR Colab-hosted Mistral (fallback)                |
|  Anti-hallucination rules enforced in system prompt|
|  Report: 7 structured sections in user's language  |
+-----------------------------------------------------+
        |
        v
+-----------------------------------------------------+
|  STEP 7 — EXPORT                                   |
|  HTML: dark-themed, verdict badge, evidence charts |
|  PDF: via weasyprint from HTML                     |
|  DOCX: Word document with bold/heading formatting  |
|  MP3: gTTS audio response (if voice input)         |
+-----------------------------------------------------+
        |
        v
   Final result dict streamed to browser via SSE
```

---

## 5. Complete File-by-File Breakdown

---

### A. Root Level Files

---

#### `app.py` — FastAPI Backend Server

**Role:** The web entry point. Runs the HTTP server, handles file uploads, streams pipeline progress to the browser using Server-Sent Events (SSE).

**Why it exists:** The original Streamlit UI timed out on long NLP/RAG operations. FastAPI + SSE lets the browser receive live step-by-step updates without timeouts.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `detect_input_type(filename, has_claim)` | Maps file extension to input type: `.pdf` -> `"pdf"`, `.mp3` -> `"audio"`, etc. If user also typed a claim, returns `"pdf_claim"` (doc as evidence context) |
| `sse(event_dict)` | Serializes a dict to the `data: {...}\n\n` SSE wire format |
| `step_start(step_id)` | Emits `{"type":"step_start"}` event so the UI shows a spinner |
| `step_done(step_id, detail)` | Emits `{"type":"step_done"}` — UI marks step green |
| `step_error(step_id, detail)` | Emits `{"type":"step_error"}` — UI marks step red |
| `upload_file()` | `POST /upload` — saves uploaded file to `data/uploads_temp/`, returns temp path |
| `verify_stream(req)` | `POST /verify/stream` — returns `StreamingResponse` of SSE events |
| `_run_pipeline(req)` | Async generator: runs the blocking pipeline in a thread pool via `asyncio.run_in_executor`, queues SSE events |
| `_pipeline_with_callbacks(req, queue, loop)` | The actual synchronous pipeline execution — imports and calls NLP, RAG, generation steps one by one, emitting SSE at each step boundary |
| `download_file(path)` | `GET /download?path=...` — serves HTML/DOCX/MP3 files securely (path must be inside `data/`) |
| `root()` | `GET /` — serves `ui/index.html` |
| `health()` | `GET /health` — returns `{"status":"ok"}` for UI demo mode detection |

**Technologies used:** FastAPI, Uvicorn, asyncio, Server-Sent Events, python-multipart, pydantic

---

#### `src/fact_verification_service.py` — Pipeline Orchestrator (THE MOTHER FILE)

**Role:** The unified pipeline class that wires all components together. Contains the `FactVerificationService` class plus all helper functions for input extraction, language handling, and LLM dispatch.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `_get_nlp()` | Lazy-loads `NLPPipeline` — only initializes once, cached in `_cache` dict |
| `_get_vector_db()` | Lazy-loads `VectorDatabase` (ChromaDB connection) |
| `_get_rag(collection_name)` | Lazy-loads `RAGPipeline` for a given collection |
| `_get_translator()` | Lazy-loads `Translator` with Google backend |
| `_get_doc_handler()` | Lazy-loads `DocumentHandler` for PDF/DOCX ingestion |
| `_get_speech_handler()` | Lazy-loads `SpeechHandler` (Whisper STT + gTTS TTS) |
| `_detect_language(text)` | Uses `langdetect` library; falls back to Unicode script ranges (Devanagari -> Hindi, Arabic script -> Arabic, etc.) |
| `_translate_to_english(text, source_lang)` | Calls `translator.to_english()` if not already English |
| `_translate_from_english(text, target_lang)` | Back-translates report to user's original language |
| `_extract_text_from_image(file_path)` | OCR via `easyocr` (GPU=False); falls back to `pytesseract` |
| `_extract_text_from_pdf(file_path)` | Extracts text via `PyPDF2`, falls back to `pdfplumber` |
| `_extract_text_from_docx(file_path)` | Extracts paragraphs via `python-docx` |
| `_transcribe_audio(file_path)` | Calls `SpeechHandler.speech_to_text()` (Whisper model) |
| `_generate_audio_output(text, language)` | Creates MP3 from report excerpt using `gTTS` |
| `FactVerificationService.verify()` | **The main 7-step pipeline method.** Takes claim + options, returns complete result dict |
| `FactVerificationService.verify_streaming()` | Runs `verify()` in a background thread, yields partial result snapshots every 0.5 seconds for UI display |
| `_generate_report(rag_result, llm_mode, ...)` | Dispatches to `ReportGeneratorGroq` or `ReportGenerator` based on `llm_mode` param |

**Input types supported by `verify()`:**

| `input_type` | Behaviour |
|---|---|
| `"text"` | Plain text claim — direct processing |
| `"image"` | OCR extracts text from image/screenshot |
| `"audio"` | Whisper transcribes, language auto-detected |
| `"pdf"` | Full PDF extracted; first 500 chars become the claim |
| `"docx"` | Same as PDF but for Word documents |
| `"pdf_claim"` | User types a claim; PDF is chunked as evidence context for RAG |
| `"docx_claim"` | User types a claim; DOCX is chunked as evidence context |

---

#### `src/enhanced_main_pipeline.py` — Backward Compatibility Wrapper

**Role:** Legacy wrapper that forwards calls to `FactVerificationService`. Kept so older scripts that import the old pipeline class do not break.

---

#### `src/main_pipeline.py` — Original Pipeline (Deprecated)

**Role:** The original pipeline before `fact_verification_service.py` was written. No longer the primary entry point but preserved for reference.

---

#### `start_nlp_ui.py` — NLP-Only Quick Launcher

**Role:** A lightweight script that starts just the NLP analysis UI (without GenAI). Useful for testing NLP components in isolation.

---

### B. NLP Module (`src/nlp/`)

The NLP module contains **5 analysis components** wired together by `nlp_pipeline.py`. All components support two modes: a **trained model** (if fine-tuned weights exist) and a **placeholder** (zero-shot / rule-based fallback).

---

#### `src/nlp/nlp_pipeline.py` — NLP Orchestrator

**Role:** The NLP pipeline controller. Initializes all 5 NLP components and runs them in sequence via `analyze()`.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(config_path)` | Loads `ModelManager`, then instantiates `ClaimDetector`, `EntityExtractor`, `WikidataEntityLinker`, `TemporalExtractor`, `StanceDetector` |
| `analyze(text, language)` | Runs all 4 analysis steps (claim detection -> NER -> entity linking -> temporal extraction) and returns a nested result dict |
| `extract_claims_from_document(text, threshold)` | Splits document into sentences, runs `detect()` on each, returns only sentences that pass the claim threshold |
| `analyze_claim_evidence_pair(claim, evidence)` | Runs full analysis on both texts + stance detection between them |
| `get_pipeline_info()` | Returns dict of which model version is loaded for each component |

**Output structure of `analyze()`:**
```python
{
  "text": "...",
  "language": "en",
  "analysis": {
    "claim_detection": {"is_claim": True, "confidence": 0.91, "label": "factual claim"},
    "entities": {"total_entities": 3, "counts": {"ORG": 1, "LOC": 2}, "entities": {...}},
    "linked_entities": [{"word": "India", "wikidata_id": "Q668", ...}],
    "temporal": {"dates": [...], "total_count": 1}
  }
}
```

---

#### `src/nlp/model_manager.py` — Model Loader

**Role:** Reads `configs/nlp_config.yaml` and loads the appropriate Hugging Face model for each NLP task. Automatically falls back to zero-shot classifiers when trained fine-tuned models are not present.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(config_path)` | Reads YAML config, sets device (cpu/cuda), caches loaded models |
| `load_claim_detector()` | Loads binary text classifier or `facebook/bart-large-mnli` as zero-shot fallback |
| `load_ner_model()` | Loads NER model — default: `dslim/bert-base-NER` |
| `load_stance_detector()` | Loads NLI model for stance — default: `cross-encoder/nli-deberta-v3-small` |
| `get_model_info(component_name)` | Returns metadata about a loaded model (type, path, task) |

**Placeholder vs Trained logic:**
- If `models/language_detection/` exists -> use fine-tuned model
- Otherwise -> use `facebook/bart-large-mnli` for zero-shot classification
- This lets the system work immediately without any fine-tuning

---

#### `src/nlp/claim_detection.py` — Claim Detector

**Role:** Determines whether a given text is a **verifiable factual claim** (vs an opinion, question, or statement).

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(model_manager)` | Calls `model_manager.load_claim_detector()`, stores model type |
| `detect(text, threshold)` | Returns `{"is_claim": bool, "confidence": float, "label": str}` |
| `_detect_with_trained_model(text, threshold)` | Uses binary classifier: `LABEL_1` = is claim |
| `_detect_with_placeholder(text, threshold)` | Uses zero-shot: candidate labels = `["factual claim", "opinion", "question", "statement"]`; returns is_claim=True only if top label is "factual claim" |
| `extract_claims_from_text(text, threshold)` | NLTK sentence tokenizes -> runs `detect()` on each sentence -> returns list of sentences that ARE claims |

---

#### `src/nlp/entity_extraction.py` — Named Entity Recognition

**Role:** Identifies and labels named entities in the text: persons (PER), organizations (ORG), locations (LOC), dates (DATE), etc.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(model_manager)` | Calls `model_manager.load_ner_model()` |
| `extract(text)` | Runs BERT NER pipeline, groups results by `entity_group` type, returns `{"PER": [...], "ORG": [...]}` |
| `extract_specific_type(text, entity_type)` | Returns only entities of a given type (e.g., only `"PER"`) |
| `get_entity_summary(text)` | Returns `{"total_entities": N, "entity_types": [...], "counts": {...}, "entities": {...}}` — used by the NLP pipeline |

**Model used:** `dslim/bert-base-NER` (BERT fine-tuned on CoNLL-2003)
Each entity: `{"text": "India", "score": 0.99, "start": 15, "end": 20}`

---

#### `src/nlp/entity_linking.py` — Entity Linking to Knowledge Base

**Role:** Takes the NER output and links recognized entities to their Wikidata entries, adding context like description, Wikipedia URL, and types.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `link_entities(entities, language)` | For each entity from NER output, calls Wikidata REST API to get QID, description, and Wikipedia link |
| `_search_wikidata(entity_text, language)` | HTTP GET to `wikidata.org/w/api.php?action=wbsearchentities` |
| `_get_entity_details(qid)` | Fetches label, description, sitelinks for a Wikidata QID |

**Output per entity:**
```python
{"word": "India", "entity_group": "LOC", "wikidata_id": "Q668",
 "description": "country in South Asia", "wikipedia_url": "https://..."}
```

---

#### `src/nlp/temporal_extraction.py` — Temporal/Date Extraction

**Role:** Extracts all temporal expressions from text using regex patterns and `dateutil` normalization.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(reference_date)` | Sets reference date for resolving relative expressions; calls `_compile_patterns()` |
| `_compile_patterns()` | Compiles 8 regex patterns: ISO date, US date, written date, year-only, month+year, relative day, relative week, duration |
| `extract(text)` | Runs all patterns, normalizes each match to ISO 8601, returns list of `{"text": "in 2024", "normalized": "2024", "type": "year_only"}` dicts |
| `_normalize_date(match, pattern_name)` | Converts raw regex match to a normalized date string using `dateutil.parser` |

**Supported patterns:**
- Absolute: `2024-01-15`, `01/15/2024`, `January 15, 2024`, `January 2024`, `in 2024`
- Relative: `yesterday`, `last week`, `3 days ago`, `next month`
- Durations: `2 hours`, `3 months`

---

#### `src/nlp/stance_detection.py` — Stance Classifier

**Role:** For a given (claim, evidence) pair, determines whether the evidence **SUPPORTS**, **REFUTES**, or is **NEUTRAL** toward the claim. Used both in the NLP pipeline (pair analysis) and in the RAG pipeline (per evidence piece).

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(model_manager)` | Calls `model_manager.load_stance_detector()`; if trained model exists, also calls `_load_trained_components()` |
| `_load_trained_components()` | Loads `AutoTokenizer` + `AutoModelForSequenceClassification` from the fine-tuned path; reads `labels.json` |
| `detect(claim, evidence)` | Routes to trained or placeholder depending on `model_type` |
| `_detect_with_trained_model(claim, evidence)` | Tokenizes as sentence pair `[CLS] claim [SEP] evidence [SEP]`, runs model, applies softmax, maps to SUPPORTS/REFUTES/NEUTRAL |
| `_detect_with_placeholder(claim, evidence)` | Uses NLI model: `entailment -> SUPPORTS`, `contradiction -> REFUTES`, `neutral -> NEUTRAL` |
| `detect_batch(claim, evidence_list)` | Runs `detect()` over a list of evidence strings |
| `aggregate_stances(stance_results)` | Computes weighted percentages and overall verdict from a list of stance results |

---

### C. RAG Module (`src/rag/`)

The RAG module handles everything from **document storage -> retrieval -> ranking -> credibility -> context preparation**. The boundary between NLP and GenAI passes through this module: retrieval and scoring are NLP-adjacent; the final context preparation feeds the GenAI step.

---

#### `src/rag/rag_pipeline.py` — RAG Orchestrator

**Role:** The main RAG controller. Wires together hybrid retrieval -> reranking -> credibility scoring -> stance detection -> aggregation -> LLM context building.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(vector_db, collection_name, nlp_model_manager)` | Initializes `HybridRetriever`, `Reranker`, `CredibilityScorer`, `StanceDetector`. Warns if collection is empty |
| `retrieve(query, top_k, filters, user_context_docs)` | Runs hybrid search on ChromaDB, adds user-provided docs with priority=1.0 (highest), knowledge base docs with priority=0.7 |
| `detect_stances(claim, evidence_list)` | Runs `StanceDetector.detect()` on each evidence piece, adds `stance` and `stance_confidence` to each evidence dict |
| `aggregate_evidence(claim, evidence_list)` | Calculates weighted verdict: `weight = stance_confidence x credibility x priority`. Applies threshold logic to determine final verdict |
| `prepare_context_for_llm(claim, evidence_list, aggregation, max_context_length)` | Formats evidence + aggregation stats into a structured string that becomes the LLM prompt's user message |
| `verify_claim(claim, top_k, user_context_docs)` | **The main RAG method.** Calls retrieve -> detect_stances -> aggregate_evidence -> prepare_context_for_llm in sequence. Returns the full verification result dict |

**Verdict thresholds in `aggregate_evidence()`:**

| Support % | Refute % | Verdict |
|-----------|---------|---------|
| > 70% | — | TRUE |
| > 50% | — | MOSTLY TRUE |
| — | > 70% | FALSE |
| — | > 50% | MOSTLY FALSE |
| Within 10% of each other | — | CONFLICTING |
| Otherwise | — | UNVERIFIABLE |

---

#### `src/rag/vector_database.py` — ChromaDB Vector Store

**Role:** Manages all interactions with the ChromaDB persistent vector database. Handles collection creation, document embedding, and semantic search.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(persist_directory, embedding_model)` | Opens ChromaDB at `data/chroma_db`, loads `all-MiniLM-L6-v2` from sentence-transformers |
| `create_collection(collection_name, metadata)` | Creates or retrieves existing collection |
| `get_collection(collection_name)` | Returns collection object |
| `add_documents(collection_name, documents, metadatas, ids, batch_size)` | Generates embeddings in batches using `SentenceTransformer.encode()`, adds to ChromaDB |
| `add_from_dataframe(collection_name, df, text_column, ...)` | Ingests data from a pandas DataFrame |
| `add_from_csv(collection_name, csv_path, text_column, ...)` | Reads CSV, calls `add_from_dataframe` |
| `search(collection_name, query, top_k, filters)` | Embeds query -> calls `collection.query()` -> returns top-k semantically similar documents |
| `get_collection_stats(collection_name)` | Returns `{"name", "count", "exists"}` |
| `DataIngestionHelper.ingest_news_articles(csv_path)` | Convenience wrapper for ingesting news data CSVs |
| `DataIngestionHelper.ingest_user_documents(documents)` | Convenience wrapper for user-provided context docs |

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional dense vectors
**Storage:** Persistent on disk at `data/chroma_db/` — survives restarts

---

#### `src/rag/hybrid_retrieval.py` — Hybrid Dense+Sparse Retriever

**Role:** Combines ChromaDB vector search (dense) with BM25 keyword search (sparse) using **Reciprocal Rank Fusion** to produce a single ranked result list.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(vector_db, collection_name, k)` | Loads all documents from collection for BM25 indexing; initializes `BM25Retriever`; stores RRF parameter `k=60` |
| `search(query, top_k, dense_weight, sparse_weight)` | Runs both retrievers at `top_k x 3` depth, then fuses |
| `_reciprocal_rank_fusion(dense_results, sparse_results, dense_weight, sparse_weight)` | RRF formula: `score(d) = sum(weight / (k + rank(d)))`. Documents appearing in both lists get their scores summed. Sorted descending |

**Why hybrid?**
- Dense (vector) search catches semantic similarity — "India's economy expanded" matches "GDP grew"
- Sparse (BM25) search catches exact keywords — "8%" or "Q3 2024" exact matches
- RRF fusion gives the best of both worlds

---

#### `src/rag/sparse_retrieval.py` — BM25 Keyword Retriever

**Role:** Implements BM25 (Best Match 25) — a classic TF-IDF variant — for keyword-based document retrieval.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(documents, metadatas)` | Tokenizes all documents (lowercased), builds BM25 index using `rank_bm25.BM25Okapi` |
| `search(query, top_k)` | Tokenizes query, calls `bm25.get_scores()`, returns top-k document indices with scores |

**Library:** `rank-bm25`

---

#### `src/rag/reranker.py` — Cross-Encoder Reranker

**Role:** After hybrid retrieval gives N candidates, the reranker uses a **cross-encoder** to more accurately score how relevant each document is to the query. Cross-encoders are slower but more accurate than bi-encoders for ranking.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(model_name)` | Loads `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers |
| `rerank(query, documents, top_k)` | Creates (query, document) pairs -> `CrossEncoder.predict()` gives relevance scores -> sort descending -> return top-k |
| `rerank_with_metadata(query, results, top_k, document_key)` | Same as `rerank()` but accepts the full result dicts from hybrid retrieval, preserving metadata |

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — trained on MS-MARCO passage ranking

---

#### `src/rag/credibility_scorer.py` — Source Credibility Scorer

**Role:** Assigns a credibility score (0.0–1.0) to each evidence source based on domain authority, publication date recency, and source type.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__()` | Defines a curated `domain_scores` dict with ~40 news sources |
| `score(url, source, publish_date, source_type)` | Combines domain score + recency penalty + source type bonus. Returns `{"total_score": float, "tier": str, "domain_score": float, ...}` |
| `_score_domain(url, source)` | Looks up domain in `domain_scores`; unknown domains get 0.5 |
| `_score_recency(publish_date)` | Penalizes old articles; articles > 2 years old lose up to 0.15 points |
| `_classify_tier(score)` | Maps numeric score to tier string: HIGH_CREDIBILITY / MEDIUM / LOW |

**Domain tiers:**
- Tier 1 (0.90–0.96): `reuters.com`, `who.int`, `apnews.com`, `nature.com`
- Tier 2 (0.80–0.89): `nytimes.com`, `theguardian.com`, `thehindu.com`
- Tier 3 (0.70–0.79): `cnn.com`, `ndtv.com`
- Fact-checkers: `snopes.com` (0.90), `factcheck.org` (0.93)

---

#### `src/rag/enhanced_rag_pipeline.py` — Multi-Collection RAG

**Role:** An enhanced variant of `RAGPipeline` that can search across multiple ChromaDB collections simultaneously (e.g., `news_articles` + `wikipedia` + `uploaded_documents`) and merge results.

---

#### `src/rag/retrieval.py` — Simple Vector Retrieval

**Role:** A simpler retrieval utility that only uses dense vector search (no BM25). Used as a fallback or for quick single-collection lookups.

---

### D. Generation Module (`src/generation/`)

The generation module is the **pure GenAI territory** — it takes the RAG pipeline's output and turns it into a human-readable report using an LLM.

---

#### `src/generation/prompt_builder.py` — LLM Prompt Constructor

**Role:** Builds the system prompt and user message that get sent to the LLM. Enforces anti-hallucination rules, multilingual output instructions, and a strict 7-section output format.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `build_system_prompt(user_language)` | Returns the full system prompt string with language-specific instruction injected. Contains 9 anti-hallucination rules and the required output format |
| `build_user_message(llm_context, user_language)` | Wraps the RAG context with "FACT VERIFICATION CONTEXT TO ANALYSE:" header + language reminder |
| `build_groq_messages(llm_context, user_language)` | Returns `[{"role":"system",...}, {"role":"user",...}]` list for Groq/OpenAI API |
| `build_mistral_prompt(llm_context, user_language)` | Returns `<s>[INST] ... [/INST]` format for Mistral/Colab inference |
| `extract_llm_context(pipeline_output)` | Extracts or reconstructs the `llm_context` string from RAG pipeline output. Falls back to building it manually from `evidence` + `aggregation` if `llm_context` key is missing |

**Anti-hallucination rules in system prompt (9 rules):**
1. Only use facts from provided evidence — no training knowledge
2. Never add names, dates, statistics not in evidence
3. If evidence is insufficient, say so — no speculation
4. Highlight evidence contradictions explicitly
5. Weight evidence by credibility score
6. Analyze each evidence piece individually
7. Verdict must be exactly one of: `TRUE | MOSTLY TRUE | UNVERIFIABLE | MOSTLY FALSE | FALSE`
8. Never invent citations
9. Every bullet in Key Findings must reference specific evidence

**Supported output languages:** English, Hindi, Tamil, Telugu, Marathi, Bengali, Gujarati, Kannada, Malayalam, Punjabi, Urdu, Spanish, French, German, Arabic, Chinese, Japanese, Korean, Russian, Portuguese

---

#### `src/generation/report_generator_groq.py` — Groq API Client

**Role:** Sends the structured prompt to the Groq API and returns the generated report text. Handles rate limiting, retries, and key management.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(api_key, model)` | Loads API key from `configs/groq_token.txt` if not passed directly. Sets `Authorization` header |
| `_load_key()` | Reads key from `configs/groq_token.txt`; raises `ValueError` if missing |
| `save_key(key)` | Persists a new key to `configs/groq_token.txt` |
| `health_check()` | GET `/v1/models` to verify the key is valid and list available models |
| `generate(nlp_output, user_language, max_tokens, temperature, retries)` | Calls `prompt_builder.extract_llm_context()` + `build_groq_messages()`, POSTs to Groq, handles 429 rate limit with `retry-after` header, returns report string |

**Model used:** `llama-3.3-70b-versatile`
**API endpoint:** `https://api.groq.com/openai/v1/chat/completions`
**Parameters:** `max_tokens=1500`, `temperature=0.2`, `top_p=0.85`

---

#### `src/generation/report_generator.py` — Colab/Mistral Client (Fallback)

**Role:** Sends prompts to a self-hosted Mistral model running on Google Colab (via `ngrok` tunnel URL). Used when Groq API is unavailable.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(inference_url)` | Loads Colab URL from `configs/inference_url.txt` if not passed |
| `generate(pipeline_output, user_language)` | Builds Mistral-format prompt, POSTs to `{inference_url}/generate`, returns report |

---

#### `src/generation/report_exporter.py` — Report File Exporter

**Role:** Takes the LLM-generated Markdown report and exports it to styled HTML, PDF (via weasyprint), and DOCX.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(output_dir)` | Creates output directory at `data/reports/` |
| `_filename(claim, ext)` | Generates timestamped filename: `report_India_GDP_20240115_143022.html` |
| `to_markdown(report_md, claim)` | Saves raw `.md` file, returns path |
| `to_html(report_md, claim, rag_result)` | Converts Markdown -> HTML with python-markdown, wraps in full CSS template (dark theme, verdict badge, confidence bar, evidence stance bar chart, summary stats chart). Uses DM Serif Display + DM Sans fonts |
| `to_pdf(report_md, claim, rag_result)` | Calls `to_html()`, then renders HTML -> PDF via `weasyprint` |
| `to_docx(report_md, claim, rag_result)` | Parses Markdown headings/bullets/bold and builds a Word document via `python-docx` |
| `export_all(report_md, claim, rag_result)` | Calls all four export methods, returns dict with paths. PDF/DOCX failures are caught and reported as `"SKIPPED: ..."` without crashing the pipeline |
| `_parse_verdict(report_md, rag_result)` | Extracts verdict string from report text using keyword matching |
| `_parse_confidence(report_md, rag_result)` | Extracts confidence percentage via regex `[Cc]onfidence: XX%` |

**HTML report features:**
- Dark-themed responsive design
- Verdict badge with color: green (TRUE), amber (UNVERIFIABLE), red (FALSE), purple (CONFLICTING)
- Confidence progress bar
- Evidence stance breakdown bar chart (Supports / Refutes / Neutral %)
- Verification summary statistics table
- Full markdown-rendered report body
- Print-friendly CSS media query

---

#### `src/generation/report_exprotex_xx.py` — Experimental Exporter

**Role:** An experimental version of the exporter with additional export format experiments. Not currently in the active pipeline.

---

### E. Multilingual Module (`src/multilingual/`)

---

#### `src/multilingual/translator.py` — Translation Engine

**Role:** Wraps the `deep-translator` library to provide clean `to_english()` and `from_english()` methods.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(backend)` | Initializes with `"google"` backend (Google Translate via deep-translator) |
| `to_english(text, source_lang)` | Translates text from `source_lang` -> English |
| `from_english(text, target_lang)` | Translates English text -> `target_lang` |
| `detect_language(text)` | Language detection via deep-translator's detection |

---

#### `src/multilingual/multilingual_pipeline.py` — Multilingual Wrapper

**Role:** An earlier wrapper class (`MultilingualVerificationPipeline`) that combines NLP + RAG + Translator. Now superseded by `FactVerificationService` but useful for standalone multilingual testing.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(nlp_pipeline, rag_pipeline, translator)` | Takes pre-initialized component instances |
| `verify_claim(claim, user_language)` | Detect language -> translate to English -> NLP -> RAG -> translate result back |

---

### F. Document Processing (`src/document_processing/`)

---

#### `src/document_processing/document_handler.py` — Document Ingestor

**Role:** Handles uploading of user PDF and DOCX files into the ChromaDB knowledge base as searchable chunks.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `process_upload(file_path)` | Extracts text from PDF/DOCX, splits into 500-character chunks with 50-char overlap, returns `{"chunks": [...], "metadata": {...}}` |
| `add_to_rag(doc_data, vector_db, collection_name)` | Calls `vector_db.add_documents()` with chunks + filename metadata |
| `_chunk_text(text, chunk_size, overlap)` | Splits text at sentence boundaries where possible |

---

#### `src/document_processing/claim_extractor.py` — Document Claim Extractor

**Role:** Extracts verifiable claims from longer documents. Uses `NLPPipeline.extract_claims_from_document()` under the hood.

---

### G. Voice Processing (`src/voice_processing/`)

---

#### `src/voice_processing/speech_handler.py` — Speech-to-Text / Text-to-Speech

**Role:** Handles audio input transcription and audio output generation.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `__init__(stt_backend, tts_backend)` | Initializes with `"whisper"` STT and `"gtts"` TTS |
| `speech_to_text(audio_file_path)` | Loads Whisper model (`openai-whisper`), transcribes audio, returns `{"text": str, "language": str}` with auto-detected language |
| `text_to_speech(text, language, output_path)` | Uses `gTTS` to generate MP3 from text in the detected language |

**Note:** There is a known import path bug in the original `enhanced_main_pipeline.py` that imports `src.voice.speech_handler` — the correct path is `src.voice_processing.speech_handler`. This is fixed in `fact_verification_service.py`.

---

### H. Data Collection (`src/data_collection/`)

These scripts were used to build the knowledge base. They are not part of the live pipeline but were used to populate ChromaDB with ~1600 news articles.

| File | What it does |
|------|-------------|
| `news_scraper.py` | Scrapes news articles from RSS feeds |
| `reliable_news_scraper.py` | Focused scraper for high-credibility sources (Reuters, BBC, AP) |
| `ai_tech_scraper.py` | Scrapes AI/technology news for the knowledge base |
| `geopolitics_scraper.py` | Scrapes geopolitics news articles |
| `entertainment_scraper.py` | Scrapes entertainment news |
| `regional_indian_scraper.py` | Scrapes regional Indian language news |
| `simple_rss_collector.py` | Lightweight RSS feed collector |
| `generate_synthetic_data.py` | Generates synthetic fact-check pairs for testing |
| `download_claim_datasets.py` | Downloads FEVER and other public fact-checking datasets |
| `download_stance_dataset.py` | Downloads stance detection training datasets |
| `process_scraped_data.py` | Cleans and normalizes raw scraped articles |
| `run_all_scrapers.py` | Orchestrates all scrapers in sequence |
| `run_regional_scraper.py` | Runs only the regional scraper |
| `collect_raw.py` / `collect_processed.py` | Batch collection utilities |

---

### I. Preprocessing (`src/preprocessing/`)

---

#### `src/preprocessing/input_processor.py`

**Role:** Handles initial input normalization — text cleaning, encoding fixes, length validation.

#### `src/preprocessing/language_detector.py`

**Role:** An earlier standalone language detection module using `langdetect`. Now replaced by the inline `_detect_language()` function in `fact_verification_service.py`.

---

### J. Data Processing (`src/data_processing/`)

| File | What it does |
|------|-------------|
| `clean_news_data.py` | Removes HTML tags, normalizes whitespace, deduplicates articles |
| `combine_datasets.py` | Merges multiple scraped datasets into a single CSV for ingestion |
| `input_processor.py` | Pre-processing of user input text |

---

### K. Scripts

Utility scripts for setup, testing, and data operations:

| Script | What it does |
|--------|-------------|
| `scripts/setup_groq.py` | Saves your Groq API key to `configs/groq_token.txt` |
| `scripts/set_inference_url.py` | Saves your Colab ngrok URL to `configs/inference_url.txt` |
| `scripts/ingest_to_rag.py` | Ingests a CSV of news articles into ChromaDB `news_articles` collection |
| `scripts/test_full_pipeline.py` | End-to-end test: runs a claim through the full 7-step pipeline |
| `scripts/test_groq_pipeline.py` | Tests specifically the Groq API connection and report generation |
| `scripts/test_my_claim_nlp.py` | Tests NLP analysis on a custom claim |
| `scripts/test_single_claim.py` | Runs a single claim through NLP + RAG (no LLM) |
| `scripts/test_system.py` | System health check: ChromaDB connection, model loading, API key |
| `scripts/check_system_status.py` | Prints status of all components |
| `scripts/setup_nltk.py` | Downloads required NLTK data (punkt tokenizer) |
| `scripts/prepare_fever_data.py` | Prepares FEVER dataset for ingestion |
| `scripts/prepare_fever_data_fixed.py` | Fixed version of FEVER data preparation |
| `scripts/generate_synthetic_pipeline_outputs.py` | Creates dummy pipeline outputs for UI testing |
| `scripts/export_pipeline_outputs.py` | Exports saved pipeline outputs to HTML/DOCX |
| `scripts/export_pipeline_simple.py` | Simpler export utility |

---

### L. Tests

| File | What it tests |
|------|-------------|
| `tests/test_complete_nlp_pipeline.py` | Full NLP pipeline integration test |
| `tests/test_nlp_pipeline.py` | Individual NLP component unit tests |
| `tests/test_fact_verification_e2e.py` | End-to-end pipeline smoke test |
| `tests/test_language_detection.py` | Language detection accuracy across multiple languages |
| `tests/test_pretrained_ner.py` | NER model output validation |
| `tests/test_input_with_context.py` | Tests `pdf_claim` / `docx_claim` input modes |
| `tests/check_chromadb_structure.py` | Inspects ChromaDB collections, counts, and sample documents |
| `tests/checkrequirment.py` | Verifies all required packages are installed |
| `tests/finaltest.py` | Final system readiness test |

---

## 6. NLP Part vs GenAI Part — Clear Classification

This is the most important conceptual boundary in the system.

### NLP Part (Steps 1–5)

Everything from input extraction through RAG context preparation is **NLP territory**. These steps use classical ML, pre-trained transformers, and rule-based methods. No generative LLM is involved.

| Step | Module | Technology |
|------|--------|-----------|
| Input extraction (OCR, STT) | `fact_verification_service.py` | easyOCR, Whisper (OpenAI) |
| Language detection | `fact_verification_service.py` | langdetect, Unicode ranges |
| Translation to English | `multilingual/translator.py` | deep-translator (Google Translate API) |
| **Claim Detection** | `nlp/claim_detection.py` | BERT zero-shot classification (`facebook/bart-large-mnli`) |
| **Named Entity Recognition** | `nlp/entity_extraction.py` | BERT NER (`dslim/bert-base-NER`) |
| **Entity Linking** | `nlp/entity_linking.py` | Wikidata REST API |
| **Temporal Extraction** | `nlp/temporal_extraction.py` | Regex + `dateutil` (rule-based) |
| Vector search | `rag/vector_database.py` | ChromaDB + `all-MiniLM-L6-v2` embeddings |
| BM25 search | `rag/sparse_retrieval.py` | `rank-bm25` library |
| Hybrid fusion | `rag/hybrid_retrieval.py` | Reciprocal Rank Fusion algorithm |
| Re-ranking | `rag/reranker.py` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Credibility scoring | `rag/credibility_scorer.py` | Rule-based domain authority lookup |
| **Stance Detection** | `nlp/stance_detection.py` | NLI model (`cross-encoder/nli-deberta-v3-small`) |
| Evidence aggregation & verdict | `rag/rag_pipeline.py` | Weighted scoring formula |
| LLM context preparation | `rag/rag_pipeline.py` | String formatting |

**NLP pipeline output (handed to GenAI):**
```python
{
  "claim": "India GDP grew 8%",
  "verdict": "MOSTLY TRUE",   # preliminary verdict before LLM
  "confidence": 62.3,
  "evidence": [
    {"document": "...", "stance": "SUPPORTS", "credibility": {"total_score": 0.95}, ...}
  ],
  "aggregation": {"support_percentage": 68, "refute_percentage": 15, ...},
  "llm_context": "CLAIM TO VERIFY:\n..."   # structured string ready for LLM
}
```

### GenAI Part (Steps 6–7)

Everything from prompt construction through report export is **GenAI territory**. These steps use a Large Language Model to synthesize findings into natural language.

| Step | Module | Technology |
|------|--------|-----------|
| Prompt construction | `generation/prompt_builder.py` | Prompt engineering, multilingual instructions |
| LLM API call | `generation/report_generator_groq.py` | Groq API, `llama-3.3-70b-versatile` |
| LLM fallback | `generation/report_generator.py` | Colab-hosted Mistral via HTTP |
| Back-translation of report | `multilingual/translator.py` | deep-translator |
| HTML/PDF/DOCX export | `generation/report_exporter.py` | python-markdown, weasyprint, python-docx |
| Audio report | `voice_processing/speech_handler.py` | gTTS |

**GenAI prompt structure:**
```
[SYSTEM]: You are a fact-checking analyst.
          Language: Write in Hindi.
          9 anti-hallucination rules.
          Required 7-section output format.

[USER]:   FACT VERIFICATION CONTEXT:
          Claim: "India GDP grew 8%"
          Evidence [1] [SUPPORTS] (source: reuters.com, credibility: 0.95)
            "India's GDP expanded 8.2% in FY2024..."
          Evidence [2] [NEUTRAL] (source: ndtv.com, credibility: 0.76)
            "Government releases quarterly data..."
          Verdict Calculation: Support 68%, Refute 15%, Neutral 17%
          Preliminary Verdict: MOSTLY TRUE (62.3%)
```

**GenAI report output — 7 required sections:**
```
## Claim
## Initial Assessment
## Evidence Analysis
## Contradictions
## Verdict        <- contains [VERDICT: MOSTLY TRUE], Confidence: 65%
## Key Findings
## Limitations
## Conclusion
```

### The RAG Bridge

RAG sits between the two worlds:
- **NLP side of RAG**: retrieval, reranking, credibility scoring, stance detection, aggregation — all deterministic, no generation
- **GenAI side of RAG**: the `llm_context` string that RAG prepares becomes the LLM's user message — this is where NLP findings become GenAI input

---

## 7. All Technologies and Libraries Used

### Core Frameworks

| Library | Used For |
|---------|---------|
| `fastapi` | REST API + SSE streaming backend |
| `uvicorn` | ASGI server for FastAPI |
| `python-multipart` | File upload parsing in FastAPI |
| `pydantic` | Request/response data validation |

### NLP and ML Models

| Library | Used For |
|---------|---------|
| `transformers` (HuggingFace) | BERT NER, zero-shot classification, NLI stance |
| `torch` (PyTorch) | Model inference backend |
| `sentence-transformers` | `all-MiniLM-L6-v2` embeddings + CrossEncoder reranker |
| `spaCy` | Optional NLP preprocessing |
| `nltk` | Sentence tokenization in claim extractor |

### RAG and Vector Search

| Library | Used For |
|---------|---------|
| `chromadb` | Vector database for document storage and retrieval |
| `rank-bm25` | BM25Okapi sparse keyword retrieval |

### Multilingual Support

| Library | Used For |
|---------|---------|
| `deep-translator` | Hindi/Tamil/... to/from English translation |
| `langdetect` | Language identification from text |

### Input Processing

| Library | Used For |
|---------|---------|
| `openai-whisper` | Audio speech-to-text transcription |
| `gtts` | Text-to-speech MP3 generation |
| `easyocr` | Image/screenshot OCR (primary) |
| `pytesseract` | OCR fallback |
| `Pillow` | Image loading for OCR |
| `PyPDF2` | PDF text extraction (primary) |
| `pdfplumber` | PDF text extraction (fallback) |
| `python-docx` | DOCX text extraction and DOCX report writing |

### Report Export

| Library | Used For |
|---------|---------|
| `markdown` (python-markdown) | Converts LLM markdown output to HTML |
| `weasyprint` | Renders HTML to PDF |
| `python-docx` | Creates formatted Word documents |

### Data and Config

| Library | Used For |
|---------|---------|
| `pandas` | DataFrame operations in data collection and ingestion |
| `numpy` | Array operations in embeddings |
| `pyyaml` | Reading `configs/nlp_config.yaml` |
| `tqdm` | Progress bars during batch embedding |
| `requests` | HTTP calls to Groq API, Wikidata, Colab |
| `python-dateutil` | Parsing and normalizing date strings |
| `beautifulsoup4` | HTML parsing in scrapers |
| `feedparser` | RSS feed parsing |

### Deployment

| Tool | Used For |
|------|---------|
| `uvicorn` | Production ASGI server |
| Vanilla JS + SSE | Browser frontend (no framework) |
| Google Colab + ngrok | Optional: self-hosted Mistral inference |

---

## 8. Configuration Files

```
configs/
├── groq_token.txt        <- Your Groq API key (starts with gsk_...)
├── inference_url.txt     <- Your Colab ngrok URL (for Mistral fallback)
└── nlp_config.yaml       <- Controls which NLP models are used
```

**`configs/nlp_config.yaml` structure:**
```yaml
nlp_pipeline:
  device: cpu                    # or "cuda:0" if GPU available
  use_trained_models: false      # true = use fine-tuned, false = use placeholders
  models:
    claim_detector:
      placeholder: "facebook/bart-large-mnli"
      trained_path: "models/claim_detector/"
      use_trained: false
    ner:
      model: "dslim/bert-base-NER"
    stance_detector:
      placeholder: "cross-encoder/nli-deberta-v3-small"
      trained_path: "models/stance_detector/"
      use_trained: false
```

---

## 9. Data Directories

```
data/
├── chroma_db/            <- Persistent ChromaDB storage (~1600 articles ingested)
|   └── news_articles/    <- Main collection for fact verification
├── reports/              <- Generated HTML/PDF/DOCX reports
├── reports_groq/         <- Reports generated via Groq API
├── audio/                <- Generated MP3 audio responses
└── uploads_temp/         <- Temporary storage for uploaded files

models/
├── mistral_fv_adapter/   <- Placeholder (empty — fine-tuned Mistral adapter weights)
├── language_detection/   <- Placeholder (empty — uses zero-shot fallback)
├── llm_finetuned/        <- Placeholder (empty — Colab model path)
└── stance_detector/      <- Placeholder (uses NLI fallback)
```

---

## 10. Known Issues and Limitations

| Issue | Status | Details |
|-------|--------|---------|
| Import path bug in `enhanced_main_pipeline.py` | Fixed in `fact_verification_service.py` | Old file has `from src.voice.speech_handler` — correct path is `src.voice_processing.speech_handler` |
| Fine-tuned models missing | Expected fallback | `models/` directories are empty — system uses zero-shot classifier + NLI placeholders. Works correctly but with lower accuracy than fine-tuned models |
| PDF generation (weasyprint) | May need system libs | weasyprint requires `libpango` on Linux/Mac. On Windows, PDF export may fall back to HTML-only |
| Whisper startup time | First call only | First audio transcription loads the Whisper model (~30 seconds). Subsequent calls are fast |
| Groq rate limits | Handled automatically | 429 responses include `retry-after` header; the client sleeps and retries automatically |

---

## 11. Quick Reference — Important Methods Table

| You want to... | Call this |
|---|---|
| Verify a text claim end-to-end | `FactVerificationService().verify(claim="...", llm_mode="groq")` |
| Stream pipeline steps to UI | `FactVerificationService().verify_streaming(claim="...", llm_mode="groq")` |
| Run only NLP analysis | `NLPPipeline().analyze("text")` |
| Search the knowledge base | `RAGPipeline.verify_claim("text", top_k=5)` |
| Detect if text is a claim | `ClaimDetector.detect("text")` |
| Extract entities from text | `EntityExtractor.get_entity_summary("text")` |
| Check stance of evidence | `StanceDetector.detect(claim, evidence)` |
| Add documents to ChromaDB | `VectorDatabase().add_from_csv(collection, csv_path, text_col)` |
| Generate LLM report | `ReportGeneratorGroq().generate(rag_result)` |
| Export report to HTML | `ReportExporter().to_html(report_md, claim, rag_result)` |
| Export all formats | `ReportExporter().export_all(report_md, claim, rag_result)` |
| Check Groq API key | `ReportGeneratorGroq().health_check()` |
| Translate text to English | `Translator("google").to_english(text, "hi")` |

---

*Built for the VerifAI project. NLP pipeline fully operational. GenAI generation requires valid Groq API key in `configs/groq_token.txt`.*
