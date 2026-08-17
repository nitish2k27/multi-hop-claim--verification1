# Data

Nothing in here is loaded at request time. The serving index lives in
[`index/`](../index/), and `data/` holds only training provenance and the
old vector store the eval harness measures against.

```
data/
├── raw/            FEVER downloads and scraped news — training provenance
├── processed/      Cleaned training sets for the two BERTs
├── chroma_db/      v1 vector store — the eval baseline, read-only
├── .embed_cache/   CacheBackedEmbeddings store (regenerable)
└── outputs/        Generated html/docx/mp3 exports (regenerable)
```

All of it is gitignored.

## Building the serving index

The index is the contract between the two tiers, and it is rebuildable in
about two minutes:

```bash
python -m ingest.run
```

That writes `index/v<n>/` — `chroma/`, `parents/`, `bm25.pkl`,
`manifest.json` — and flips `index/CURRENT` when it completes. Tier 2 reads
`CURRENT`, checks the manifest against its own embedder, and refuses to boot
on a mismatch rather than serving nonsense from mixed dimensions.

`serve/` imports nothing from `ingest/`. Files are the entire interface, and
`tests/` enforces it.

## `chroma_db/` is deliberately kept

It is the v1 index: dense-only Chroma over the same embedding model, built
before chunking existed. `eval_harness/baseline.py` reads it to isolate what
the chunking change bought — tail recall@1 went 0.47 → 0.88 from chunking
alone, then → 0.98 with the full pipeline. Delete it and that comparison is no
longer reproducible.

## Feed list

The 363 RSS/Atom feeds the v1 scrapers used are preserved as data in
[`ingest/sources.yaml`](../ingest/sources.yaml). The scraper code they lived
in was superseded by `ingest/` and removed; the curated list was worth keeping.
