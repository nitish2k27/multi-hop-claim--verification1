"""
Tier 1 entry point.

    python -m ingest.run --status                    what is published now
    python -m ingest.run --list                      every built version
    python -m ingest.run --from-csv                  build from the processed CSV
    python -m ingest.run --from-csv --force          rebuild even if unchanged
    python -m ingest.run --discover-only --limit 50  list feed articles, fetch nothing
    python -m ingest.run --crawl --limit 200         crawl, then build
    python -m ingest.run --crawl --merge-csv         crawl + CSV, merged
    python -m ingest.run --rollback v1               point CURRENT back at v1

Needs no Groq key — embeddings are local, and nothing here calls an LLM. That is
the point of the tier split: the corpus can be rebuilt offline while the serving
app keeps running on the previously published index.


HOW A REBUILD MERGES WITH THE EXISTING CORPUS
=============================================
A build is **never** an in-place edit. Every run writes a complete new
`index/v{n}/` directory and leaves the live one untouched:

    index/
      CURRENT      -> a text file containing "v1"
      v1/          the live index, being served right now
      v2/          the new build, invisible until CURRENT says otherwise

"Merging with the existing corpus" therefore means merging *records*, not
merging index files. The union happens before anything is embedded:

    crawl records ─┐
                   ├─> clean.merge_records() ─> chunk ─> build v2 ─> validate ─> publish
    CSV records ───┘        dedup by URL,
                            then by body text

`--crawl` alone builds from the crawl only. `--crawl --merge-csv` builds from
both, with the crawl stream taking priority: when the same article arrives from
both sources, the crawled copy wins because it carries a real publisher date
rather than the CSV's fabricated January-1 placeholder.

Because the whole record set is rebuilt each time, a build is reproducible from
its inputs and there is no accumulated state to drift. The cost is that a crawl
you want to *keep* must be merged with `--merge-csv` on every subsequent run —
a crawl-only build does not inherit the CSV corpus just because v1 had it.


WHAT ROLLBACK IS, AND WHY IT IS ONE LINE
========================================
`index/CURRENT` is a text file whose entire contents are a version name. Tier 2
reads it at startup to find the index, and validates the manifest before serving.

Publishing is therefore a single atomic file write, and rolling back is the same
write with an older name:

    python -m ingest.run --list           # see what exists
    python -m ingest.run --rollback v1    # CURRENT now reads "v1"

Restart Tier 2 and it is serving the old index again. Nothing was deleted,
nothing was copied, and the bad build is still on disk to inspect.

Three properties make this safe, and all three are deliberate:

  * A build that raises **deletes its own directory** (`index.build`), so a
    half-written index can never be rolled forward to by accident.
  * A build is **validated before publishing** — presence, non-emptiness, a real
    smoke query, and a BM25 pickle round-trip. An index that exists but
    retrieves nothing is the failure mode that otherwise shows up as "the model
    can't answer anything".
  * The manifest records the embedding model and dimension, and **Tier 2 refuses
    to boot on a mismatch**. Rolling back to an index built with a different
    embedder fails loudly at startup instead of silently returning nonsense.

Old versions cost disk and nothing else. Delete them by hand when you are sure.
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


def _force_utf8_stdio() -> None:
    """
    Make stdout/stderr UTF-8 regardless of the console's code page.

    Must run before argparse can print anything. This module's docstring is the
    `--help` text (RawDescriptionHelpFormatter), it contains box-drawing
    characters, and a Windows console is cp1252 — so without this, `--help`
    itself dies with UnicodeEncodeError. The same guard is in serve/cli.py for
    the same reason; there it was report text rather than help text.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


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


def cmd_list() -> int:
    """Every index version on disk, with the live one marked."""
    versions = sorted(
        (p for p in cfg.index_root.glob("v*") if p.is_dir()),
        key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0,
    )
    if not versions:
        print("No index versions built yet.")
        return 1

    live = cfg.active_index.name if (cfg.index_root / cfg.POINTER).exists() else None
    for path in versions:
        manifest_path = cfg.manifest_path(path)
        marker = " <- CURRENT" if path.name == live else ""
        if not manifest_path.exists():
            print(f"  {path.name:6s}  (no manifest — incomplete build){marker}")
            continue
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        dates = "real dates" if m["dates_reliable"] else \
                f"{m['dates_unreliable_count']:,} placeholder dates"
        print(f"  {path.name:6s}  {m['documents']:>6,} docs  "
              f"{m['chunks']:>7,} chunks  {m['built_at'][:10]}  {dates}{marker}")

    print("\nRoll back with:  python -m ingest.run --rollback <version>")
    return 0


def cmd_rollback(version: str) -> int:
    """
    Point CURRENT at an existing version.

    Validates before switching. Rolling back to a broken index would turn a bad
    deploy into two bad deploys, and the whole value of keeping old versions is
    that the one you fall back to is known-good.
    """
    target = cfg.index_root / version
    if not target.is_dir():
        print(f"No such version: index/{version}", file=sys.stderr)
        print("Run --list to see what exists.", file=sys.stderr)
        return 1

    if not cfg.manifest_path(target).exists():
        print(f"index/{version} has no manifest — incomplete build, refusing.",
              file=sys.stderr)
        return 1

    try:
        index_mod.validate(target)
    except Exception as exc:
        print(f"index/{version} failed validation, refusing to publish: {exc}",
              file=sys.stderr)
        return 1

    previous = cfg.active_index.name if (cfg.index_root / cfg.POINTER).exists() else "none"
    index_mod.publish(target)
    print(f"CURRENT: {previous} -> {version}")
    print("Restart Tier 2 (uvicorn / the CLI) to pick it up.")
    return 0


def cmd_discover_only(limit: int | None, per_feed: int | None,
                      delay: float) -> int:
    """
    Read the feeds and report what is there. Fetches no article bodies.

    Worth running before a real crawl: it is one request per feed instead of one
    per article, and it answers the question the crawl exists to answer — how
    many articles actually carry a publisher date.
    """
    from ingest.discover import date_coverage, discover, load_feeds

    feeds = load_feeds()
    print(f"Feeds in sources.yaml: {len(feeds):,}\n")

    started = time.monotonic()
    entries = list(discover(limit=limit, per_feed=per_feed, delay=delay))
    if not entries:
        print("No articles discovered.", file=sys.stderr)
        return 1

    coverage = date_coverage(entries)
    domains: dict[str, int] = {}
    for e in entries:
        domains[e.domain] = domains.get(e.domain, 0) + 1

    print(f"\nDiscovered {coverage['entries']:,} unique articles "
          f"across {len(domains):,} domains in {time.monotonic() - started:.0f}s")
    print(f"With a real publisher date: {coverage['with_reliable_date']:,} "
          f"({coverage['coverage']:.0%})")
    print("  — the CSV corpus manages 4%\n")

    # Alongside the CSV corpus — same directory, same role: a source record set.
    out = cfg.source_csv.parent / "discovered.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    print(f"Wrote {out}")

    print("\nTop domains:")
    for domain, n in sorted(domains.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:>5,}  {domain}")
    return 0


def cmd_crawl(force: bool, merge_csv: bool, limit: int | None,
              per_feed: int | None, delay: float) -> int:
    """
    Crawl the feeds and build an index, optionally merged with the CSV corpus.

    Does not touch the live index until the new build validates — see the
    module docstring.
    """
    from ingest.clean import merge_records
    from ingest.discover import discover
    from ingest.fetch import FetchStats, fetch_all
    from ingest.fetch import summarise as fetch_summary

    started = time.monotonic()

    fetch_stats = FetchStats()
    clean_stats = CleanStats()

    print(f"Crawling (delay {delay}s per host"
          + (f", limit {limit:,} articles" if limit else "")
          + (", merging the CSV corpus" if merge_csv else "")
          + ")\n")

    crawl_stream = fetch_all(
        discover(limit=limit, per_feed=per_feed, delay=delay),
        stats=fetch_stats,
        delay=delay,
    )

    # Crawl first: on a URL or content collision the crawled copy wins, and it
    # is the one carrying a real publisher date.
    streams = [crawl_stream]
    if merge_csv:
        if not cfg.source_csv.exists():
            print(f"--merge-csv given but {cfg.source_csv} is missing",
                  file=sys.stderr)
            return 1
        streams.append(load_csv(cfg.source_csv, clean_stats))

    parents, chunks = [], []
    for parent, parent_chunks in to_documents(
        merge_records(*streams, stats=clean_stats)
    ):
        parents.append(parent)
        chunks.extend(parent_chunks)

    print("\nCrawl:")
    print(fetch_summary(fetch_stats))

    if not parents:
        print("\nNothing to index — no articles survived the crawl.",
              file=sys.stderr)
        return 1

    print(f"\nChunked into {len(chunks):,} pieces "
          f"({len(chunks) / max(len(parents), 1):.1f} per document)\n")

    fingerprint = index_mod.corpus_fingerprint(parents)
    if not force and fingerprint == index_mod.live_fingerprint():
        print(f"No change since index/{cfg.active_index.name} "
              f"— nothing to rebuild. Use --force to override.")
        return 0

    build_dir = index_mod.build(
        parents, chunks,
        clean_stats={**asdict(clean_stats), "crawl": asdict(fetch_stats)},
    )
    index_mod.publish(build_dir)

    print(f"\nBuilt and published index/{build_dir.name} "
          f"in {time.monotonic() - started:.0f}s")
    print(f"Roll back with:  python -m ingest.run --rollback "
          f"{cfg.active_index.name}")
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
    # Before the parser exists: --help prints this module's docstring, which
    # contains box-drawing characters a cp1252 console cannot encode.
    _force_utf8_stdio()

    ap = argparse.ArgumentParser(
        prog="ingest.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--from-csv", action="store_true",
                    help="build from data/processed/news_articles_rag.csv")
    ap.add_argument("--crawl", action="store_true",
                    help="crawl the feeds in ingest/sources.yaml, then build")
    ap.add_argument("--discover-only", action="store_true",
                    help="list what the feeds advertise; fetch no article bodies")
    ap.add_argument("--merge-csv", action="store_true",
                    help="with --crawl: also include the CSV corpus, de-duplicated")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="stop after N articles (crawl paths only)")
    ap.add_argument("--per-feed", type=int, metavar="N",
                    help="take at most N articles from each feed")
    ap.add_argument("--delay", type=float, default=1.0, metavar="SECONDS",
                    help="minimum gap between requests to one host (default 1.0)")
    ap.add_argument("--status", action="store_true",
                    help="show the currently published index")
    ap.add_argument("--list", action="store_true",
                    help="list every index version on disk")
    ap.add_argument("--rollback", metavar="VERSION",
                    help="point CURRENT at an existing version, e.g. v1")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when the corpus is unchanged")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _log_setup(args.verbose)
    cfg.index_root.mkdir(parents=True, exist_ok=True)

    if args.status:
        return cmd_status()
    if args.list:
        return cmd_list()
    if args.rollback:
        return cmd_rollback(args.rollback)
    if args.discover_only:
        return cmd_discover_only(args.limit, args.per_feed, args.delay)
    if args.crawl:
        return cmd_crawl(force=args.force, merge_csv=args.merge_csv,
                         limit=args.limit, per_feed=args.per_feed,
                         delay=args.delay)
    if args.from_csv:
        return cmd_from_csv(force=args.force)

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
