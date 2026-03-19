"""
Entity Linking using Wikidata
Links entities to knowledge base IDs for disambiguation and cross-lingual matching
"""

import logging
import requests
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
import time

logger = logging.getLogger(__name__)


class WikidataEntityLinker:
    """
    Link entities to Wikidata IDs
    
    Features:
    - Entity disambiguation
    - Cross-lingual matching
    - Knowledge base integration
    - Caching for performance
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        timeout: int = 5,
        max_retries: int = 3
    ):
        """
        Initialize Wikidata entity linker
        
        Args:
            cache_size: LRU cache size
            timeout: API request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.wikidata_api = "https://www.wikidata.org/w/api.php"
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Statistics
        self.stats = {
            'searches': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'failures': 0
        }
        
        logger.info("✓ WikidataEntityLinker initialized")
        logger.info(f"  Cache size: {cache_size}")
        logger.info(f"  Timeout: {timeout}s")
    
    @lru_cache(maxsize=1000)
    def search_entity(
        self,
        entity_text: str,
        entity_type: Optional[str] = None,
        language: str = 'en'
    ) -> Optional[str]:
        """
        Search for entity in Wikidata
        
        Args:
            entity_text: Entity text (e.g., "Narendra Modi")
            entity_type: Entity type hint (PER, ORG, LOC, MISC)
            language: Language code
        
        Returns:
            Wikidata ID (e.g., "Q1058") or None
        """
        self.stats['searches'] += 1
        
        try:
            # Search Wikidata
            params = {
                'action': 'wbsearchentities',
                'format': 'json',
                'language': language,
                'search': entity_text,
                'limit': 5,
                'type': 'item'
            }
            
            self.stats['api_calls'] += 1
            response = self._make_request(self.wikidata_api, params)
            
            if not response or 'search' not in response:
                return None
            
            results = response['search']
            
            if not results:
                return None
            
            # Filter by entity type if provided
            if entity_type:
                filtered = self._filter_by_type(results, entity_type)
                if filtered:
                    results = filtered
            
            # Get top result
            top_result = results[0]
            wikidata_id = top_result['id']
            
            logger.debug(f"Linked '{entity_text}' → {wikidata_id}")
            
            return wikidata_id
            
        except Exception as e:
            self.stats['failures'] += 1
            logger.debug(f"Failed to link '{entity_text}': {str(e)}")
            return None
    
    @lru_cache(maxsize=1000)
    def get_entity_info(
        self,
        wikidata_id: str,
        languages: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get entity information from Wikidata
        
        Args:
            wikidata_id: Wikidata ID (e.g., "Q1058")
            languages: List of language codes
        
        Returns:
            Entity info dict with labels, description, aliases
        """
        if languages is None:
            languages = ['en', 'hi', 'es', 'ar', 'fr', 'de']
        
        try:
            params = {
                'action': 'wbgetentities',
                'format': 'json',
                'ids': wikidata_id,
                'languages': '|'.join(languages),
                'props': 'labels|descriptions|aliases|claims'
            }
            
            self.stats['api_calls'] += 1
            response = self._make_request(self.wikidata_api, params)
            
            if not response or 'entities' not in response:
                return None
            
            if wikidata_id not in response['entities']:
                return None
            
            entity = response['entities'][wikidata_id]
            
            # Extract labels in multiple languages
            labels = {}
            if 'labels' in entity:
                for lang, label_data in entity['labels'].items():
                    labels[lang] = label_data['value']
            
            # Extract description
            description = ''
            if 'descriptions' in entity and 'en' in entity['descriptions']:
                description = entity['descriptions']['en']['value']
            
            # Extract aliases
            aliases = []
            if 'aliases' in entity and 'en' in entity['aliases']:
                aliases = [a['value'] for a in entity['aliases']['en']]
            
            # Extract type (instance of)
            entity_types = []
            if 'claims' in entity and 'P31' in entity['claims']:  # P31 = instance of
                for claim in entity['claims']['P31']:
                    if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                        entity_types.append(
                            claim['mainsnak']['datavalue']['value'].get('id', '')
                        )
            
            return {
                'id': wikidata_id,
                'labels': labels,
                'description': description,
                'aliases': aliases,
                'types': entity_types
            }
            
        except Exception as e:
            logger.debug(f"Failed to get info for {wikidata_id}: {str(e)}")
            return None
    
    def link_entities(
        self,
        entities: List[Dict[str, Any]],
        language: str = 'en'
    ) -> List[Dict[str, Any]]:
        """
        Link multiple entities to Wikidata
        
        Args:
            entities: List of entity dicts from NER
                     [{'word': 'Modi', 'entity_group': 'PER', ...}, ...]
            language: Language code
        
        Returns:
            Entities with Wikidata IDs and info added
        """
        linked = []
        
        for entity in entities:
            entity_copy = entity.copy()
            
            # Search Wikidata
            wikidata_id = self.search_entity(
                entity.get('word', ''),
                entity.get('entity_group'),
                language
            )
            
            if wikidata_id:
                entity_copy['wikidata_id'] = wikidata_id
                
                # Get additional info
                info = self.get_entity_info(wikidata_id)
                if info:
                    entity_copy['wikidata_info'] = info
            
            linked.append(entity_copy)
        
        return linked
    
    def match_entities_across_languages(
        self,
        entity1_text: str,
        entity2_text: str,
        lang1: str = 'en',
        lang2: str = 'en'
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if entities in different languages refer to same thing
        
        Args:
            entity1_text: Entity in language 1
            entity2_text: Entity in language 2
            lang1: Language code for entity 1
            lang2: Language code for entity 2
        
        Returns:
            (match: bool, wikidata_id: Optional[str])
        """
        # Link both entities
        id1 = self.search_entity(entity1_text, language=lang1)
        id2 = self.search_entity(entity2_text, language=lang2)
        
        if not id1 or not id2:
            # Fallback to text matching
            return entity1_text.lower() == entity2_text.lower(), None
        
        # Compare Wikidata IDs
        return id1 == id2, id1 if id1 == id2 else None
    
    def disambiguate(
        self,
        entity_text: str,
        context: str,
        candidates: List[str] = None
    ) -> Optional[str]:
        """
        Disambiguate entity using context
        
        Args:
            entity_text: Ambiguous entity (e.g., "Washington")
            context: Surrounding text
            candidates: List of candidate Wikidata IDs
        
        Returns:
            Best matching Wikidata ID
        """
        # Get candidates
        if not candidates:
            wikidata_id = self.search_entity(entity_text)
            return wikidata_id
        
        # Simple disambiguation: choose based on context keywords
        best_match = None
        best_score = 0
        
        for candidate_id in candidates:
            info = self.get_entity_info(candidate_id)
            if not info:
                continue
            
            # Score based on description match
            description = info.get('description', '').lower()
            context_lower = context.lower()
            
            score = sum(1 for word in description.split() if word in context_lower)
            
            if score > best_score:
                best_score = score
                best_match = candidate_id
        
        return best_match
    
    def get_statistics(self) -> Dict[str, int]:
        """Get linker statistics"""
        cache_info = self.search_entity.cache_info()
        self.stats['cache_hits'] = cache_info.hits
        
        return {
            **self.stats,
            'cache_size': cache_info.currsize,
            'cache_maxsize': cache_info.maxsize
        }
    
    def _make_request(
        self,
        url: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retries"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                
                # Rate limiting
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.debug(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
            except requests.exceptions.Timeout:
                logger.debug(f"Timeout on attempt {attempt + 1}")
                continue
            except Exception as e:
                logger.debug(f"Request failed: {str(e)}")
                break
        
        return None
    
    def _filter_by_type(
        self,
        results: List[Dict],
        entity_type: str
    ) -> List[Dict]:
        """Filter search results by entity type"""
        # Type mapping (simplified)
        type_keywords = {
            'PER': ['human', 'person', 'politician', 'actor', 'singer'],
            'ORG': ['organization', 'company', 'institution', 'agency'],
            'LOC': ['country', 'city', 'location', 'place', 'region'],
        }
        
        keywords = type_keywords.get(entity_type, [])
        
        if not keywords:
            return results
        
        filtered = []
        for result in results:
            description = result.get('description', '').lower()
            if any(kw in description for kw in keywords):
                filtered.append(result)
        
        return filtered if filtered else results


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    linker = WikidataEntityLinker()
    
    print("\n" + "="*80)
    print("ENTITY LINKING TESTS")
    print("="*80)
    
    # Test 1: Link single entities
    print("\n→ Test 1: Single entity linking")
    
    test_entities = [
        ("Narendra Modi", "PER"),
        ("India", "LOC"),
        ("Apple Inc", "ORG"),
        ("New York", "LOC"),
        ("Barack Obama", "PER"),
        ("Microsoft", "ORG"),
        ("Paris", "LOC"),
    ]
    
    for entity_text, entity_type in test_entities:
        wikidata_id = linker.search_entity(entity_text, entity_type)
        
        if wikidata_id:
            info = linker.get_entity_info(wikidata_id)
            print(f"\n{entity_text} ({entity_type}):")
            print(f"  Wikidata ID: {wikidata_id}")
            if info:
                print(f"  Description: {info['description'][:60]}...")
                if 'hi' in info['labels']:
                    print(f"  Hindi: {info['labels']['hi']}")
        else:
            print(f"\n{entity_text}: Not found")
    
    # Test 2: Cross-lingual matching
    print("\n" + "="*80)
    print("→ Test 2: Cross-lingual matching")
    print("="*80)
    
    pairs = [
        ("Narendra Modi", "नरेंद्र मोदी"),
        ("India", "भारत"),
        ("Paris", "Paris"),
    ]
    
    for entity1, entity2 in pairs:
        match, wikidata_id = linker.match_entities_across_languages(entity1, entity2)
        print(f"\n'{entity1}' == '{entity2}':")
        print(f"  Match: {match}")
        if wikidata_id:
            print(f"  Wikidata ID: {wikidata_id}")
    
    # Test 3: Link from NER output
    print("\n" + "="*80)
    print("→ Test 3: Link NER entities")
    print("="*80)
    
    # Simulate NER output
    ner_entities = [
        {'word': 'Narendra Modi', 'entity_group': 'PER', 'score': 0.99},
        {'word': 'India', 'entity_group': 'LOC', 'score': 0.98},
        {'word': 'Apple', 'entity_group': 'ORG', 'score': 0.97},
    ]
    
    linked = linker.link_entities(ner_entities)
    
    for ent in linked:
        print(f"\n{ent['word']} ({ent['entity_group']}):")
        if 'wikidata_id' in ent:
            print(f"  Wikidata: {ent['wikidata_id']}")
            if 'wikidata_info' in ent:
                print(f"  Description: {ent['wikidata_info']['description'][:60]}...")
        else:
            print(f"  No Wikidata link found")
    
    # Test 4: Statistics
    print("\n" + "="*80)
    print("→ Statistics")
    print("="*80)
    
    stats = linker.get_statistics()
    print(f"\nSearches: {stats['searches']}")
    print(f"API calls: {stats['api_calls']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache size: {stats['cache_size']}/{stats['cache_maxsize']}")
    print(f"Failures: {stats['failures']}")
    
    print("\n" + "="*80)