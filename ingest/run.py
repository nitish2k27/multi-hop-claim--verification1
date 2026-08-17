"""
Tier 1 entry point.

    python -m ingest.run --from-csv           # build from the processed CSV
    python -m ingest.run --from-csv --force   # rebuild even if unchanged
    python -m ingest.run --status             # what is published right now

Needs no Groq key — embeddings are local, and nothing here calls an LLM. That is
the point of the tier split: the corpus can be rebuilt offline while the serving
app keeps running on the previously published index.

The `--crawl` path (discover → fetch → live sources) lands in M5.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict

from core.config import cfg
from ingest import index as index_mod
from ingest.chunk import to_documents
from ingest.clean import CleanStats, load_csv, summarise


def _log_setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These two are chatty at INFO and drown the build progress.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def cmd_status() -> int:
    pointer = cfg.index_root / cfg.POINTER
    if not pointer.exists():
        print("No index published yet. Run: python -m ingest.run --from-csv")
        return 1

    manifest_path = cfg.manifest_path(cfg.active_index)
    if not manifest_path.exists():
        print(f"CURRENT points at {cfg.active_index.name}, but its manifest is missing.")
        return 1

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"published   index/{m['version']}")
    print(f"built at    {m['built_at']}")
    print(f"model       {m['embed_model']} (dim {m['embed_dim']})")
    print(f"chunking    {m['chunk_size']} / {m['chunk_overlap']}")
    print(f"documents   {m['documents']:,}")
    print(f"chunks      {m['chunks']:,}"
          f"  ({m['chunks_collapsed']:,} duplicates collapsed)")
    print(f"dates       {'reliable' if m['dates_reliable'] else 'NOT RELIABLE'}"
          f"  ({m['dates_unreliable_count']:,} placeholder dates)")
    return 0


def cmd_from_csv(force: bool) -> int:
    started = time.monotonic()

    if not cfg.source_csv.exists():
        print(f"Source CSV not found: {cfg.source_csv}", file=sys.stderr)
        return 1

    # ── Stages 3 + 4: clean, then chunk ───────────────────────────────────
    stats = CleanStats()
    parents, chunks = [], []
    for parent, parent_chunks in to_documents(load_csv(cfg.source_csv, stats)):
        parents.append(parent)
        chunks.extend(parent_chunks)

    print("\nClean:")
    print(summarise(stats))
    print(f"\nChunked into {len(chunks):,} pieces "
          f"({len(chunks) / max(len(parents), 1):.1f} per document)\n")

    if not parents:
        print("Nothing to index.", file=sys.stderr)
        return 1

    # ── Idempotency gate ──────────────────────────────────────────────────
    # Same inputs and same parameters would reproduce the same artifacts, so a
    # re-run exits here in seconds rather than rebuilding identical files.
    fingerprint = index_mod.corpus_fingerprint(parents)
    if not force and fingerprint == index_mod.live_fingerprint():
        print(f"No change since index/{cfg.active_index.name} "
              f"— nothing to rebuild. Use --force to override.")
        print(f"\nDone in {time.monotonic() - started:.1f}s")
        return 0

    # ── Stage 5: build, validate, publish ─────────────────────────────────
    build_dir = index_mod.build(parents, chunks, clean_stats=asdict(stats))
    index_mod.publish(build_dir)

    print(f"\nBuilt and published index/{build_dir.name} "
          f"in {time.monotonic() - started:.1f}s")
    print(f"Artifacts: {build_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="ingest.run", description=__doc__)
    ap.add_argument("--from-csv", action="store_true",
                    help="build from data/processed/news_articles_rag.csv")
    ap.add_argument("--status", action="store_true",
                    help="show the currently published index")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when the corpus is unchanged")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _log_setup(args.verbose)
    cfg.index_root.mkdir(parents=True, exist_ok=True)

    if args.status:
        return cmd_status()
    if args.from_csv:
        return cmd_from_csv(force=args.force)

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
