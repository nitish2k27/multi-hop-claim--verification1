"""
Repo cleanup — BUILD_PLAN §2.1 / §2.2, run after M3.

    python scripts/cleanup.py            # dry run, shows what would move
    python scripts/cleanup.py --apply    # actually move
    python scripts/cleanup.py --restore  # put everything back

NOTHING IS DELETED. Everything moves into `_trash/`, which is gitignored,
preserving the original relative path. That matters here because most of the
targets are **untracked** — `git checkout` cannot bring them back, so a real
`rm` would be irreversible. Empty `_trash/` by hand once the repo is verified.

WHAT IS DELIBERATELY *NOT* REMOVED
----------------------------------
`src/` is not deleted wholesale. BUILD_PLAN §2.2 still lists several of its
modules as source material for M4/M5 (`document_handler.py`, `report_exporter.py`,
the scrapers, `input_processor.py`, `multilingual/`). Only the modules §2.1 names
as dead are moved. The rest of `src/` goes after M5 lands.

Training data (`data/processed/*_train.csv`, `data/raw/fever_*`) stays: it is the
provenance for the two fine-tuned BERTs and deleting it would make them
unreproducible. `data/.embed_cache/` stays because it is what makes a re-ingest
take 1 second instead of 13 minutes. Both are already gitignored, so neither
pollutes the repo.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# This file lives in scripts/, so the repo root is one level up.
ROOT = Path(__file__).resolve().parent.parent
TRASH = ROOT / "_trash"
MANIFEST = TRASH / "_manifest.json"

# ── What moves, and why ──────────────────────────────────────────────────────
# Grouped so the dry run reads as an argument rather than a list of paths.

TARGETS: dict[str, list[str]] = {
    "Root scratch files — ad-hoc test scripts never part of the build": [
        "test_environment.py",
        "test_environment_safe.py",
        "test_pipeline_fix.py",
        "test_pipeline_output.py",
        "test_single_export.py",
    ],
    "Overlapping status documents — six files, one README replaces them": [
        "PIPELINE_FIXES_SUMMARY.md",
        "SETUP_COMPLETE.md",
        "VIRTUAL_ENV_SETUP.md",
        "implementation_status_report.md",
        "technical_decisions_status.md",
        "docs/project_staus.md",
    ],
    "Docs describing the OLD architecture — actively misleading now": [
        "docs/ACTUAL_NLP_OUTPUT_STRUCTURE.md",
        "docs/DEPENDENCY_FIX_SUMMARY.md",
        "docs/ENHANCED_SYSTEM_GUIDE.md",
        "docs/FEVER_DATA_FIX.md",
        "docs/INPUT_PROCESSOR_CAPABILITIES.md",
        "docs/LANGUAGE_DETECTION.md",
        "docs/NLP_PIPELINE_GUIDE.md",
        "docs/NLP_PIPELINE_OUTPUT_FORMAT.md",
        "docs/OCR_LANGUAGE_GUIDE.md",
        "docs/QUICK_START.md",
        "docs/RAG_ARCHITECTURE_EXPLAINED.md",
        "docs/RAG_PIPELINE_GUIDE.md",
        "docs/WORKFLOW_WITH_CONTEXT.md",
    ],
    "Dead src/ modules (BUILD_PLAN §2.1) — the rest of src/ survives until M5": [
        # Second RAGPipeline whose API cannot work against VectorDatabase.
        "src/rag/retrieval.py",
        # Imports that dead class — broken since it was written.
        "src/main_pipeline.py",
        # Keyword stance engine that overrode the trained model.
        "src/rag/enhanced_rag_pipeline.py",
        # Duplicate of the preprocessing/ one.
        "src/data_processing/input_processor.py",
        # Zero bytes.
        "src/generation/report_exprotex_xx.py",
        # Colab/ngrok inference path — removed with the local-LLM decision.
        "src/generation/report_generator.py",
    ],
    "Colab/ngrok + superseded scripts": [
        "scripts/set_inference_url.py",
        "scripts/export_pipeline_simple.py",          # 0 bytes
        "scripts/generate_synthetic_pipeline_outputs.py",  # 0 bytes
        "scripts/prepare_fever_data.py",              # keep _fixed only
        "scripts/setup_groq.py",
        "scripts/setup_nltk.py",
        "scripts/check_system_status.py",
        "scripts/ingest_to_rag.py",                   # superseded by ingest/
        "scripts/export_pipeline_outputs.py",
        "scripts/test_full_pipeline.py",
        "scripts/test_groq_pipeline.py",
        "scripts/test_my_claim_nlp.py",
        "scripts/test_single_claim.py",
        "scripts/test_system.py",
    ],
    "Old test suite — 13 files, zero assert statements (print-driven demos)": [
        "tests/checkrequirment.py",
        "tests/check_chromadb_structure.py",
        "tests/finaltest.py",
        "tests/test_complete_nlp_pipeline.py",
        "tests/test_fact_verification_e2e.py",
        "tests/test_input_with_context.py",
        "tests/test_language_detection.py",
        "tests/test_nlp_pipeline.py",
        "tests/test_pretrained_ner.py",
        "tests/test_rag_pipeline.py",
        "tests/test_stance_detector.py",
        "tests/test_stance_detector_fever.py",
        "tests/test_trained_claim_detector.py",
    ],
    "SECRETS — plaintext credentials in the tree": [
        "configs/groq_token.txt",
        "configs/inference_url.txt",
    ],
    "Config for the replaced model manager": [
        "configs/nlp_config.yaml",
    ],
    "Unused models — 414 MB": [
        # 125 MB fastText language ID, replaced by langdetect.
        "models/language_detection",
        # Byte-identical duplicate of the above, 125 MB.
        "tests/models",
        # 164 MB LoRA adapter. No local LLM — archive it off-repo if wanted.
        "models/mistral_fv_adapter",
        "models/llm_finetuned",
    ],
    "Build artifacts committed into the tree": [
        "data/reports",
        "data/reports_groq",
        "data/test_output.json",
        "data/uploads_temp",
    ],
    "Superseded launchers and requirements (pyproject.toml replaces them)": [
        "activate_env.bat",
        "run_app.bat",
        "run_app.ps1",
        "start_nlp_ui.py",
        "requirements_enhanced.txt",
        # Actively harmful, not merely stale: it pins none of the LangChain
        # stack and still lists openai-whisper, easyocr, selenium, gradio,
        # faiss-cpu and fasttext — all cut in the rebuild. Installing from it
        # produces an environment that cannot run this project.
        "requirements.txt",
        "examples",
        "api",            # only a .gitkeep; serve/api.py is the API
    ],
    "Caches": [
        "__pycache__",
    ],
}

# Moved but still needed later — called out separately so the dry run can warn.
NEEDED_LATER = {
    "app.py": "M5 — becomes serve/api.py (keep its SSE handling as reference)",
}


def _iter_targets():
    for reason, paths in TARGETS.items():
        for rel in paths:
            yield reason, rel


def plan() -> list[tuple[str, Path, int]]:
    """Existing targets only, with their size on disk."""
    found = []
    for reason, rel in _iter_targets():
        path = ROOT / rel
        if not path.exists():
            continue
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            size = path.stat().st_size
        found.append((reason, path, size))
    return found


def do_move(items, apply: bool) -> None:
    moved: list[dict] = []
    total = 0
    current_reason = None

    for reason, path, size in items:
        if reason != current_reason:
            print(f"\n{reason}")
            current_reason = reason
        rel = path.relative_to(ROOT)
        total += size
        print(f"    {str(rel):<52} {size / 1024 / 1024:>8.1f} MB")

        if apply:
            destination = TRASH / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            shutil.move(str(path), str(destination))
            moved.append({"path": str(rel).replace("\\", "/"), "bytes": size,
                          "reason": reason})

    print(f"\n  {len(items)} entries, {total / 1024 / 1024:.1f} MB")

    if apply:
        TRASH.mkdir(exist_ok=True)
        # Merge with any previous run. Overwriting would orphan everything moved
        # earlier — the files would still be in _trash/ but --restore would no
        # longer know where they came from.
        existing: list[dict] = []
        if MANIFEST.exists():
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        already = {entry["path"] for entry in existing}
        merged = existing + [m for m in moved if m["path"] not in already]
        MANIFEST.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"  manifest now records {len(merged)} entries "
              f"({len(moved)} added this run)")
        print(f"  moved to {TRASH.relative_to(ROOT)}/ — manifest written")
        print("  restore with:  python scripts/cleanup.py --restore")
    else:
        print("  DRY RUN — nothing moved. Re-run with --apply")


def restore() -> None:
    if not MANIFEST.exists():
        print("No manifest — nothing to restore.")
        return
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in entries:
        source = TRASH / entry["path"]
        destination = ROOT / entry["path"]
        if not source.exists():
            print(f"  missing in trash: {entry['path']}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        print(f"  restored {entry['path']}")
    MANIFEST.unlink()
    print(f"\nRestored {len(entries)} entries.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually move files")
    parser.add_argument("--restore", action="store_true", help="undo a previous run")
    args = parser.parse_args()

    if args.restore:
        restore()
        return 0

    items = plan()
    if not items:
        print("Nothing to clean — already done.")
        return 0

    do_move(items, apply=args.apply)

    print("\nKept deliberately:")
    print("    src/ (minus the dead modules)  — M4/M5 source material")
    print("    app.py                         — M5 reference for SSE")
    print("    data/processed, data/raw       — training data for the two BERTs")
    print("    data/.embed_cache              — makes re-ingest a 1s no-op")
    print("    data/chroma_db                 — the M3 before/after baseline")
    print("    notebooks/                     — provenance for the fine-tuned models")
    print("    ui/index.html                  — reconnected at M5")
    print("    configs/regional_scraper_config.yaml — M5 crawl config")

    for path, why in NEEDED_LATER.items():
        if (ROOT / path).exists():
            print(f"\n  NOTE: {path} kept — {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
