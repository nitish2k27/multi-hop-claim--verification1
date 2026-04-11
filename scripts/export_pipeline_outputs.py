"""
export_pipeline_outputs.py
──────────────────────────
Runs a list of real claims through YOUR existing NLP + RAG pipeline
and exports the llm_context strings as JSONL.

These become the domain-specific half of your training data.

Run locally:
    python scripts/export_pipeline_outputs.py

Output: data/training/pipeline_outputs.jsonl
        Each line: {"llm_context": "...", "report": null}
"""

import sys
import json
import logging
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR  = project_root / "data" / "training"
OUTPUT_FILE = OUTPUT_DIR / "pipeline_outputs.jsonl"

logging.basicConfig(level=logging.WARNING)   # suppress pipeline INFO noise

# ── Real claims to run through the pipeline ───────────────────────────────────
# These are India-focused since your RAG DB has Indian news articles.
# Add or replace with claims relevant to your actual article collection.

CLAIMS = [
    # Economy
    "India's GDP grew 8% in 2024",
    "India's inflation rate crossed 6% in 2023",
    "India became the fifth largest economy in the world",
    "India's forex reserves crossed 600 billion dollars",
    "The Indian rupee depreciated against the US dollar in 2024",
    "India's software exports reached 220 billion dollars in 2024",
    "India's unemployment rate is below 5%",
    "India's stock market reached an all-time high in 2024",
    "India's retail inflation fell to 4% in early 2025",
    "India's trade deficit widened in 2024",

    # Technology
    "Apple manufactures iPhones in India",
    "India launched a semiconductor manufacturing policy in 2023",
    "India has the second largest number of internet users in the world",
    "Generative AI will impact Indian IT jobs significantly",
    "India's IT sector revenue crossed 250 billion dollars",

    # Government & Policy
    "The Indian government launched a new scheme in 2024",
    "India increased its defence budget in 2024",
    "India's renewable energy capacity crossed 150 GW",
    "India launched a digital currency pilot in 2023",
    "India signed a free trade agreement with the UAE",

    # Global context
    "India overtook China in population in 2023",
    "India is a member of the G20",
    "India sent a mission to the Moon in 2023",
    "India's PM Modi visited the United States in 2023",
    "India abstained on the UN vote on Russia's invasion of Ukraine",

    # Mixed/tricky claims (tests REFUTES and UNVERIFIABLE)
    "India's GDP growth rate was higher than China's in 2023",
    "India has achieved 100% rural electrification",
    "India's literacy rate is above 90%",
    "India has the largest army in the world",
    "India's COVID vaccination programme covered all adults by 2022",
]


def load_pipeline():
    """
    Load your existing NLP + RAG pipeline.
    Uses the enhanced main pipeline for better integration.
    """
    try:
        from src.enhanced_main_pipeline import EnhancedFactVerificationPipeline

        print("[OK] Importing enhanced pipeline...")
        
        # Use the enhanced pipeline which handles all the complexity
        pipeline = EnhancedFactVerificationPipeline()
        
        return pipeline

    except ImportError as e:
        print(f"\n[ERROR] Could not import enhanced pipeline: {e}")
        print("Trying individual components...")
        
        try:
            from src.rag.rag_pipeline import RAGPipeline
            from src.rag.vector_database import VectorDatabase
            from src.nlp.nlp_pipeline import NLPPipeline
            from src.nlp.model_manager import ModelManager

            print("[OK] Importing individual components...")
            
            # Initialize components with proper dependencies
            nlp = NLPPipeline()
            
            # Initialize vector database and RAG pipeline
            vector_db = VectorDatabase()
            model_manager = ModelManager()
            rag = RAGPipeline(
                vector_db=vector_db,
                collection_name='news_articles',
                nlp_model_manager=model_manager
            )
            
            return nlp, rag

        except ImportError as e2:
            print(f"\n[ERROR] Could not import pipeline components: {e2}")
            print("\nMake sure all dependencies are installed and models are available.\n")
            raise


def run_claim(claim: str, pipeline) -> dict | None:
    """
    Run one claim through the enhanced pipeline and extract llm_context.
    Returns None if the pipeline fails for this claim.
    """
    try:
        # Use enhanced pipeline directly
        result = pipeline.verify_claim(claim, input_type='text')
        
        # Check if processing was successful
        if result.get('verdict') == 'ERROR' or 'error' in result:
            error_msg = result.get('error', result.get('explanation', 'Unknown error'))
            print(f"  [ERROR] Pipeline error: {error_msg}")
            return None
        
        # Extract or build llm_context from the result
        llm_context = result.get("llm_context", "")
        
        # If no llm_context, build it from the result
        if not llm_context:
            evidence_summary = ""
            evidence = result.get('evidence', [])
            if evidence:
                evidence_summary = f"\nEVIDENCE ({len(evidence)} pieces):"
                for i, ev in enumerate(evidence[:3], 1):  # Show first 3
                    source = ev.get('source', 'unknown')
                    stance = ev.get('stance', 'NEUTRAL')
                    evidence_summary += f"\n  [{i}] [{stance}] (source: {source})"
                    evidence_summary += f"\n      {ev.get('document', '')[:100]}..."
            
            llm_context = f"""=== FACT VERIFICATION CONTEXT ===

CLAIM:
  {claim}

VERDICT: {result.get('verdict', 'UNKNOWN')}
CONFIDENCE: {result.get('confidence', 0)}%

EXPLANATION:
{result.get('explanation', 'No explanation available')}
{evidence_summary}

TASK:
Based on the analysis above, provide a detailed fact-verification report with:
1. Final verdict (TRUE / FALSE / MOSTLY TRUE / MOSTLY FALSE / UNVERIFIABLE)
2. Confidence score (0-100%)
3. Key evidence supporting your verdict
4. Any conflicting evidence
5. Important caveats or limitations
=== END CONTEXT ==="""

        return {
            "llm_context": llm_context,
            "report": None,
            "metadata": {
                "source": "enhanced_pipeline",
                "verdict": result.get("verdict", "UNKNOWN"),
                "confidence": result.get("confidence", 0.0),
                "claim": claim,
                "evidence_count": len(result.get('evidence', [])),
                "language": result.get('metadata', {}).get('user_language', 'en')
            }
        }

    except Exception as e:
        print(f"  [ERROR] Failed on '{claim[:60]}': {e}")
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("EXPORTING REAL PIPELINE OUTPUTS")
    print("="*60)
    print(f"\nClaims to process: {len(CLAIMS)}")
    print("Loading pipeline (this may take 30-60 seconds)...\n")

    pipeline = load_pipeline()

    print(f"\nRunning {len(CLAIMS)} claims through pipeline...\n")

    examples = []
    for i, claim in enumerate(CLAIMS, 1):
        print(f"[{i:02d}/{len(CLAIMS)}] {claim[:65]}")
        result = run_claim(claim, pipeline)
        if result:
            examples.append(result)
            print(f"         → {result['metadata']['verdict']} "
                  f"({result['metadata']['confidence']:.1f}%)")

    print(f"\n[OK] Processed: {len(examples)} / {len(CLAIMS)} claims")

    # Write JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[OK] Saved to: {OUTPUT_FILE.resolve()}")
    print(f"     File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print("\nNext: upload this file + fever_converted.jsonl to Kaggle dataset\n")


if __name__ == "__main__":
    main()
