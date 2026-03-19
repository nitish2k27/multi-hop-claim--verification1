"""
Named Entity Recognition (NER) Module
Extracts entities: PERSON, ORGANIZATION, LOCATION, DATE, etc.
"""

import logging
from typing import Dict, Any, List
from src.nlp.model_manager import ModelManager

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extract named entities from text
    
    Supports both trained and placeholder models
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize entity extractor
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.model_info = model_manager.load_ner_model()
        self.model = self.model_info['model']
        self.model_type = self.model_info['type']
        
        logger.info(f"EntityExtractor initialized (type: {self.model_type})")
    
    def extract(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract all entities from text
        
        Args:
            text: Input text
        
        Returns:
            {
                'PER': [{'text': 'Modi', 'score': 0.99, 'start': 0, 'end': 4}, ...],
                'ORG': [...],
                'LOC': [...],
                'DATE': [...],
                ...
            }
        """
        if not text or len(text) < 5:
            return {}
        
        # Run NER model
        raw_entities = self.model(text)
        
        # Group by entity type
        grouped = {}
        
        for entity in raw_entities:
            entity_type = entity['entity_group']
            
            if entity_type not in grouped:
                grouped[entity_type] = []
            
            grouped[entity_type].append({
                'text': entity['word'],
                'score': entity['score'],
                'start': entity['start'],
                'end': entity['end']
            })
        
        logger.debug(f"Extracted {len(raw_entities)} entities ({len(grouped)} types)")
        
        return grouped
    
    def extract_specific_type(self, text: str, entity_type: str) -> List[Dict]:
        """
        Extract only specific entity type
        
        Args:
            text: Input text
            entity_type: Entity type (e.g., 'PER', 'ORG', 'LOC')
        
        Returns:
            List of entities of that type
        """
        all_entities = self.extract(text)
        return all_entities.get(entity_type, [])
    
    def get_entity_summary(self, text: str) -> Dict[str, Any]:
        """
        Get summary of entities
        
        Returns:
            {
                'total_entities': int,
                'entity_types': ['PER', 'ORG', ...],
                'counts': {'PER': 3, 'ORG': 2, ...},
                'entities': {...}
            }
        """
        entities = self.extract(text)
        
        return {
            'total_entities': sum(len(ents) for ents in entities.values()),
            'entity_types': list(entities.keys()),
            'counts': {ent_type: len(ents) for ent_type, ents in entities.items()},
            'entities': entities
        }


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = ModelManager()
    extractor = EntityExtractor(manager)
    
    # Test
    test_text = "Narendra Modi is the Prime Minister of India. He visited New York in September 2024."
    
    print("\n" + "="*60)
    print("ENTITY EXTRACTION TEST")
    print("="*60)
    print(f"\nText: {test_text}\n")
    
    summary = extractor.get_entity_summary(test_text)
    
    print(f"Total entities: {summary['total_entities']}")
    print(f"Entity types: {summary['entity_types']}")
    print(f"Counts: {summary['counts']}")
    
    print("\nDetailed entities:")
    for ent_type, entities in summary['entities'].items():
        print(f"\n{ent_type}:")
        for ent in entities:
            print(f"  - {ent['text']} (score: {ent['score']:.3f})")