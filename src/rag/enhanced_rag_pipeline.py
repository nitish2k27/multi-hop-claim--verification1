"""
Enhanced RAG Pipeline with Multi-Collection Support
Handles both news articles and uploaded documents with smart prioritization
"""

import logging
from typing import Dict, Any, List, Optional

from src.rag.vector_database import VectorDatabase
from src.rag.reranker import Reranker
from src.rag.credibility_scorer import CredibilityScorer

logger = logging.getLogger(__name__)


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with multi-collection support

    Features:
    - Search multiple collections (news_articles, uploaded_documents)
    - Context-aware prioritization
    - Credibility scoring per source type
    - Smart result merging and reranking
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        model_manager=None,
        collections: List[str] = None,
        search_strategy: str = 'context_aware'
    ):
        self.vector_db        = vector_db
        self.model_manager    = model_manager
        self.collections      = collections or ['news_articles', 'uploaded_documents']
        self.search_strategy  = search_strategy

        # Reranker takes model_name string, not model_manager
        self.reranker          = Reranker()
        self.credibility_scorer = CredibilityScorer()

        self.collection_config = {
            'news_articles': {
                'credibility_base': 0.85,
                'weight': 1.0,
                'description': 'Scraped news articles from reliable sources'
            },
            'uploaded_documents': {
                'credibility_base': 0.70,
                'weight': 0.8,
                'description': 'User-uploaded documents'
            }
        }

        logger.info(f"✓ EnhancedRAGPipeline initialized")
        logger.info(f"  Collections: {self.collections}")
        logger.info(f"  Strategy: {search_strategy}")

    def verify_claim(
        self,
        claim: str,
        top_k: int = 5,
        collection_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Verify claim against multiple collections"""

        logger.info(f"Verifying claim across {len(self.collections)} collections")

        search_weights    = self._determine_search_weights(claim, collection_weights)
        all_results       = []
        collection_results = {}

        for collection in self.collections:
            if collection not in search_weights:
                continue
            try:
                results = self._search_collection(collection, claim, top_k * 2)
                for result in results:
                    result['collection']        = collection
                    result['collection_weight'] = search_weights[collection]

                collection_results[collection] = results
                all_results.extend(results)
                logger.info(f"  {collection}: {len(results)} results")

            except Exception as e:
                logger.error(f"Search failed for {collection}: {e}")
                collection_results[collection] = []

        if not all_results:
            return self._no_evidence_response(claim)

        final_results = self._merge_and_rerank(all_results, claim, top_k)
        return self._generate_verdict(claim, final_results, collection_results)

    def _determine_search_weights(self, claim, custom_weights):
        if custom_weights:
            return custom_weights
        if self.search_strategy == 'equal_weight':
            return {col: 1.0 for col in self.collections}
        elif self.search_strategy == 'prioritize_uploads':
            return {'news_articles': 0.7, 'uploaded_documents': 1.0}
        elif self.search_strategy == 'context_aware':
            return self._context_aware_weights(claim)
        return {col: 1.0 for col in self.collections}

    def _context_aware_weights(self, claim: str) -> Dict[str, float]:
        claim_lower = claim.lower()
        public_keywords  = ['gdp', 'economy', 'government', 'country', 'nation',
                            'global', 'world', 'international', 'market', 'stock',
                            'election', 'president', 'minister', 'policy', 'law']
        private_keywords = ['our', 'we', 'company', 'revenue', 'profit', 'quarter',
                            'fiscal', 'internal', 'department', 'team', 'project',
                            'budget', 'expenses', 'sales', 'customers']

        public_score  = sum(1 for k in public_keywords  if k in claim_lower)
        private_score = sum(1 for k in private_keywords if k in claim_lower)

        if private_score > public_score:
            return {'news_articles': 0.6, 'uploaded_documents': 1.0}
        elif public_score > private_score:
            return {'news_articles': 1.0, 'uploaded_documents': 0.7}
        return {'news_articles': 1.0, 'uploaded_documents': 0.9}

    def _search_collection(self, collection: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using VectorDatabase directly"""
        try:
            raw = self.vector_db.search(
                collection_name=collection,
                query=query,
                top_k=top_k
            )

            config   = self.collection_config.get(collection, {})
            results  = []

            for i, doc in enumerate(raw['documents'][0]):
                results.append({
                    'text':             doc,
                    'metadata':         raw['metadatas'][0][i],
                    'score':            raw['distances'][0][i],
                    'source_type':      collection,
                    'base_credibility': config.get('credibility_base', 0.75),
                    'source_description': config.get('description', 'Unknown source')
                })

            return results

        except Exception as e:
            logger.error(f"Collection search failed for '{collection}': {e}")
            return []

    def _merge_and_rerank(self, all_results, claim, top_k):
        logger.info(f"Merging and reranking {len(all_results)} results")

        # Apply collection weights
        for result in all_results:
            result['weighted_score'] = result.get('score', 0.0) * result.get('collection_weight', 1.0)

        # Deduplicate
        unique_results = self._deduplicate_results(all_results)

        # Rerank with cross-encoder using rerank_with_metadata
        try:
            reranked = self.reranker.rerank_with_metadata(
                query=claim,
                results=unique_results,
                top_k=top_k,
                document_key='text'
            )
        except Exception as e:
            logger.warning(f"Reranking failed: {e} — falling back to score sort")
            reranked = sorted(unique_results, key=lambda x: x['weighted_score'], reverse=True)

        # Apply credibility scoring using score() method
        for result in reranked:
            meta     = result.get('metadata', {})
            cred     = self.credibility_scorer.score(
                url=meta.get('url'),
                source=meta.get('source'),
                publish_date=meta.get('publish_date'),
                source_type=meta.get('source_type')
            )
            result['credibility_score'] = cred['total_score']

            rerank_score       = result.get('rerank_score', result['weighted_score'])
            result['final_score'] = (rerank_score * 0.7) + (cred['total_score'] * 0.3)

        final = sorted(reranked, key=lambda x: x['final_score'], reverse=True)[:top_k]
        logger.info(f"✓ Final results: {len(final)}")
        return final

    def _deduplicate_results(self, results):
        seen_texts     = set()
        unique_results = []

        for result in results:
            text         = result['text']
            is_duplicate = any(self._text_similarity(text, seen) > 0.9 for seen in seen_texts)

            if not is_duplicate:
                seen_texts.add(text)
                unique_results.append(result)

        logger.info(f"Deduplication: {len(results)} → {len(unique_results)}")
        return unique_results

    def _text_similarity(self, text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        union  = len(words1 | words2)
        return len(words1 & words2) / union if union > 0 else 0.0

    def _generate_verdict(self, claim, evidence, collection_results):
        if not evidence:
            return self._no_evidence_response(claim)

        support_count  = 0
        refute_count   = 0
        evidence_summary = []

        for result in evidence:
            stance = self._detect_stance(claim, result['text'])
            result['stance']            = stance['label']
            result['stance_confidence'] = stance['confidence']

            if stance['label'] == 'SUPPORTS':
                support_count += stance['confidence']
            elif stance['label'] == 'REFUTES':
                refute_count  += stance['confidence']

            evidence_summary.append({
                'text':        result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],
                'source':      result.get('collection', 'unknown'),
                'stance':      stance['label'],
                'confidence':  result['final_score'],
                'credibility': result['credibility_score']
            })

        if support_count > refute_count and support_count > 0.5:
            verdict    = 'SUPPORTED'
            confidence = min(support_count / len(evidence), 0.95)
        elif refute_count > support_count and refute_count > 0.5:
            verdict    = 'REFUTED'
            confidence = min(refute_count / len(evidence), 0.95)
        else:
            verdict    = 'INSUFFICIENT'
            confidence = 0.3

        explanation = self._generate_explanation(verdict, evidence_summary, collection_results)

        return {
            'verdict':          verdict,
            'confidence':       confidence * 100,
            'explanation':      explanation,
            'evidence':         evidence_summary,
            'evidence_sources': {col: len(res) for col, res in collection_results.items()},
            'search_metadata':  {
                'total_results_found': sum(len(r) for r in collection_results.values()),
                'collections_searched': list(collection_results.keys()),
                'search_strategy': self.search_strategy
            }
        }

    def _detect_stance(self, claim, evidence):
        claim_words    = set(claim.lower().split())
        evidence_lower = evidence.lower()
        evidence_words = set(evidence_lower.split())
        overlap_ratio  = len(claim_words & evidence_words) / len(claim_words) if claim_words else 0

        if overlap_ratio > 0.3:
            negations = ['not', 'no', 'never', 'false', 'incorrect', 'wrong']
            if any(w in evidence_lower for w in negations):
                return {'label': 'REFUTES',   'confidence': 0.6}
            return {'label': 'SUPPORTS',      'confidence': 0.7}
        return {'label': 'NOT_ENOUGH_INFO',   'confidence': 0.4}

    def _generate_explanation(self, verdict, evidence, collection_results):
        parts = []
        if verdict == 'SUPPORTED':
            parts.append("The claim is supported by available evidence.")
        elif verdict == 'REFUTED':
            parts.append("The claim is contradicted by available evidence.")
        else:
            parts.append("There is insufficient evidence to verify this claim.")

        source_info = []
        for col, results in collection_results.items():
            if results:
                source_info.append(f"{len(results)} {col.replace('_', ' ')}")
        if source_info:
            parts.append(f"Evidence found in: {', '.join(source_info)}.")

        if evidence:
            parts.append(f"Key evidence: {evidence[0]['text'][:150]}...")

        return " ".join(parts)

    def _no_evidence_response(self, claim):
        return {
            'verdict':          'INSUFFICIENT',
            'confidence':       0.0,
            'explanation':      "No relevant evidence found in the available knowledge base.",
            'evidence':         [],
            'evidence_sources': {col: 0 for col in self.collections},
            'search_metadata':  {
                'total_results_found':  0,
                'collections_searched': self.collections,
                'search_strategy':      self.search_strategy
            }
        }

    def add_collection(self, collection_name, config=None):
        if collection_name not in self.collections:
            self.collections.append(collection_name)
        if config:
            self.collection_config[collection_name] = config
        logger.info(f"Added collection: {collection_name}")

    def get_collection_stats(self):
        stats = {}
        for collection in self.collections:
            try:
                obj   = self.vector_db.get_collection(collection)
                count = len(obj.get()['ids'])
                stats[collection] = {'document_count': count, 'status': 'available'}
            except Exception as e:
                stats[collection] = {'document_count': 0, 'status': f'error: {str(e)}'}
        return stats