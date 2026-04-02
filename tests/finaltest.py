"""
Complete End-to-End Pipeline Test
Input → NLP → RAG → Stance → LLM-ready context

Run from project root:
    python tests/test_e2e_to_llm.py
"""

import sys
import json
import logging
from pathlib import Path

# Project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# ── Helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def ok(msg):   print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def fail(msg): print(f"  ❌ {msg}")


def build_llm_context(claim, nlp_analysis, rag_result):
    """
    Build the final context string that will be passed to the LLM.
    This is what your finetuned model will receive as input.
    """
    analysis  = nlp_analysis['analysis']
    claim_det = analysis['claim_detection']
    entities  = analysis['entities']
    temporal  = analysis['temporal']['dates']
    linked    = nlp_analysis['analysis'].get('linked_entities', [])

    # Entity summary
    entity_lines = []
    for etype, ents in entities['entities'].items():
        names = [e['text'] for e in ents]
        entity_lines.append(f"  - {etype}: {', '.join(names)}")

    # Temporal summary
    temporal_lines = []
    for t in temporal:
        temporal_lines.append(f"  - {t.get('text','')} → {t.get('normalized','')} ({t.get('type','')})")

    # Wikidata links
    linked_lines = []
    for e in linked:
        if e.get('wikidata_id'):
            linked_lines.append(f"  - {e['word']} → {e['wikidata_id']}")

    # Evidence summary
    evidence_lines = []
    if rag_result and rag_result.get('evidence'):
        for i, ev in enumerate(rag_result['evidence'][:5], 1):
            doc     = ev.get('document', '')[:200]
            stance  = ev.get('stance', 'UNKNOWN')
            source  = ev.get('source', 'unknown')
            cred    = ev.get('credibility', {})
            cred_score = cred.get('total_score', 0) if isinstance(cred, dict) else cred
            evidence_lines.append(
                f"  [{i}] [{stance}] (source: {source}, credibility: {cred_score:.2f})\n"
                f"      {doc}..."
            )

    # Aggregation summary
    agg_lines = []
    if rag_result and rag_result.get('aggregation'):
        agg = rag_result['aggregation']
        agg_lines = [
            f"  - Preliminary verdict:  {agg.get('verdict', 'N/A')}",
            f"  - Support score:        {agg.get('support_percentage', 0):.1f}%",
            f"  - Refute score:         {agg.get('refute_percentage', 0):.1f}%",
            f"  - Neutral score:        {agg.get('neutral_percentage', 0):.1f}%",
            f"  - Evidence pieces:      {agg.get('num_evidence', 0)}",
            f"  - Supports:             {agg.get('num_supports', 0)}",
            f"  - Refutes:              {agg.get('num_refutes', 0)}",
        ]

    context = f"""=== FACT VERIFICATION CONTEXT ===

CLAIM:
  {claim}

CLAIM ANALYSIS:
  Is verifiable claim: {claim_det['is_claim']}
  Claim confidence:    {claim_det['confidence']:.3f}

NAMED ENTITIES:
{chr(10).join(entity_lines) if entity_lines else '  None found'}

TEMPORAL EXPRESSIONS:
{chr(10).join(temporal_lines) if temporal_lines else '  None found'}

KNOWLEDGE BASE LINKS:
{chr(10).join(linked_lines) if linked_lines else '  None found'}

RETRIEVED EVIDENCE:
{chr(10).join(evidence_lines) if evidence_lines else '  No evidence found'}

EVIDENCE AGGREGATION:
{chr(10).join(agg_lines) if agg_lines else '  Not available'}

TASK:
Based on the claim, entities, temporal context, and retrieved evidence above,
provide a detailed fact-verification analysis with:
1. Final verdict (TRUE / FALSE / MOSTLY TRUE / MOSTLY FALSE / UNVERIFIABLE)
2. Confidence score (0-100%)
3. Key evidence supporting your verdict
4. Any conflicting evidence
5. Important caveats or limitations

=== END CONTEXT ==="""

    return context


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(claim: str):
    """Run complete pipeline on a single claim"""

    print(f"\n{'='*60}")
    print(f"CLAIM: {claim}")
    print(f"{'='*60}")

    nlp_result = None
    rag_result = None

    # ── Step 1: NLP Analysis ─────────────────────────────────────────────────
    section("STEP 1: NLP Analysis")
    try:
        from src.nlp.nlp_pipeline import NLPPipeline
        nlp = NLPPipeline()
        nlp_result = nlp.analyze(claim)
        analysis   = nlp_result['analysis']

        claim_info = analysis['claim_detection']
        ok(f"Claim detection: {'YES' if claim_info['is_claim'] else 'NO'} "
           f"(confidence: {claim_info['confidence']:.3f})")

        entity_summary = analysis['entities']
        if entity_summary['total_entities'] > 0:
            for etype, ents in entity_summary['entities'].items():
                ok(f"{etype}: {', '.join(e['text'] for e in ents)}")
        else:
            warn("No entities found")

        temporal_list = analysis['temporal']['dates']
        if temporal_list:
            ok(f"Temporal: {', '.join(t.get('text','') for t in temporal_list)}")
        else:
            warn("No temporal expressions")

        linked = analysis.get('linked_entities', [])
        wikidata_linked = [e for e in linked if e.get('wikidata_id')]
        if wikidata_linked:
            for e in wikidata_linked:
                ok(f"Wikidata: {e['word']} → {e['wikidata_id']}")

    except Exception as e:
        fail(f"NLP failed: {e}")
        import traceback; traceback.print_exc()
        return None

    # ── Step 2: Check if it's a claim ────────────────────────────────────────
    is_claim = nlp_result['analysis']['claim_detection']['is_claim']
    if not is_claim:
        warn("Not a verifiable claim — skipping RAG")
        llm_context = build_llm_context(claim, nlp_result, None)
        section("LLM CONTEXT (No RAG)")
        print(llm_context)
        return {'claim': claim, 'is_claim': False, 'llm_context': llm_context}

    # ── Step 3: RAG Retrieval ────────────────────────────────────────────────
    section("STEP 2: RAG Retrieval + Stance Detection")
    try:
        from src.rag.vector_database import VectorDatabase
        from src.rag.rag_pipeline import RAGPipeline
        from src.nlp.model_manager import ModelManager

        vdb   = VectorDatabase()
        stats = vdb.get_collection_stats('news_articles')

        if not stats['exists'] or stats.get('count', 0) == 0:
            warn("RAG database empty — skipping RAG step")
        else:
            ok(f"RAG database: {stats['count']:,} articles")
            model_manager = ModelManager()
            rag           = RAGPipeline(
                vector_db=vdb,
                collection_name='news_articles',
                nlp_model_manager=model_manager
            )
            rag_result = rag.verify_claim(claim, top_k=5)

            ok(f"Verdict:    {rag_result['verdict']}")
            ok(f"Confidence: {rag_result['confidence']:.1f}%")
            ok(f"Evidence:   {len(rag_result.get('evidence', []))} pieces")

            # Show evidence stances
            evidence = rag_result.get('evidence', [])
            supports = sum(1 for e in evidence if e.get('stance') == 'SUPPORTS')
            refutes  = sum(1 for e in evidence if e.get('stance') == 'REFUTES')
            neutral  = sum(1 for e in evidence if e.get('stance') == 'NEUTRAL')
            ok(f"Stances:    {supports} SUPPORTS | {refutes} REFUTES | {neutral} NEUTRAL")

    except Exception as e:
        fail(f"RAG failed: {e}")
        import traceback; traceback.print_exc()

    # ── Step 4: Build LLM Context ────────────────────────────────────────────
    section("STEP 3: LLM-Ready Context")
    llm_context = build_llm_context(claim, nlp_result, rag_result)
    print(llm_context)

    # ── Step 5: Summary ──────────────────────────────────────────────────────
    section("SUMMARY")
    ok(f"Input:        {claim}")
    ok(f"Is Claim:     {is_claim}")
    if rag_result:
        ok(f"RAG Verdict:  {rag_result.get('verdict', 'N/A')}")
        ok(f"Confidence:   {rag_result.get('confidence', 0):.1f}%")
    ok("LLM context: ready ✅")
    print(f"\n  👉 This context is ready to be passed to your finetuned LLM")
    print(f"     for final verdict generation.\n")

    return {
        'claim':       claim,
        'is_claim':    is_claim,
        'nlp_result':  nlp_result,
        'rag_result':  rag_result,
        'llm_context': llm_context
    }


# ── Test cases ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  END-TO-END PIPELINE TEST: INPUT → LLM CONTEXT")
    print("="*60)
    print("  Testing complete flow for your finetuned LLM input")

    test_claims = [
        # Should be claim + RAG finds evidence
        "India's GDP grew 8% in 2024",

        # Should be claim + named entities
        "Narendra Modi is the Prime Minister of India",

        # Should NOT be a claim
        "What is the current GDP of India?",

        # Should be claim + temporal
        "Apple announced record profits in Q4 2024",

        # Climate claim
        "Climate change is accelerating globally",
    ]

    results = []
    for claim in test_claims:
        result = run_pipeline(claim)
        results.append(result)

    # Final scorecard
    print("\n" + "="*60)
    print("  FINAL SCORECARD")
    print("="*60)
    for r in results:
        if r:
            verdict = r.get('rag_result', {}).get('verdict', 'NO RAG') if r.get('rag_result') else ('SKIP' if not r['is_claim'] else 'NO RAG')
            print(f"  {'✅' if r['is_claim'] else '❌'} {r['claim'][:50]:<50} → {verdict}")

    print(f"\n  Pipeline complete. LLM contexts generated for all claims.")
    print(f"  Next step: Pass llm_context to your finetuned model.\n")


if __name__ == "__main__":
    main()