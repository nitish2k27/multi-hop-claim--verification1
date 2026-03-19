"""
Model Manager - Handles loading placeholder vs trained models
"""
 
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import pipeline
import logging
 
logger = logging.getLogger(__name__)
 
 
class ModelManager:
    """
    Manages NLP model loading with automatic fallback
    Supports both placeholder and trained models
    """
 
    def __init__(self, config_path: str = "configs/nlp_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
 
        self.nlp_config   = self.config['nlp_pipeline']
        self.models_config = self.nlp_config['models']
        self.device        = self.nlp_config['device']
 
        self._loaded_models = {}
 
        logger.info(f"ModelManager initialized (device: {self.device})")
        logger.info(f"Use trained models: {self.nlp_config['use_trained_models']}")
 
    def load_claim_detector(self) -> Dict[str, Any]:
        """Load claim detection model"""
        if 'claim_detector' in self._loaded_models:
            return self._loaded_models['claim_detector']
 
        model_config = self.models_config['claim_detector']
        use_trained  = model_config['use_trained'] and self.nlp_config['use_trained_models']
 
        if use_trained:
            model_path = model_config['trained_path']
 
            if os.path.exists(model_path):
                logger.info(f"Loading TRAINED claim detector from {model_path}")
 
                model = pipeline(
                    "text-classification",
                    model=model_path,
                    device=self.device
                )
 
                self._loaded_models['claim_detector'] = {
                    'model':      model,
                    'type':       'trained',
                    'task':       'binary_classification',
                    'model_path': model_path         # ← always include path
                }
 
                logger.info("✓ Trained claim detector loaded")
                return self._loaded_models['claim_detector']
            else:
                logger.warning(f"Trained model not found at {model_path}, using placeholder")
 
        # Placeholder fallback
        logger.info("Loading PLACEHOLDER claim detector (zero-shot)")
 
        model = pipeline(
            "zero-shot-classification",
            model=model_config['placeholder'],
            device=self.device
        )
 
        self._loaded_models['claim_detector'] = {
            'model':      model,
            'type':       'placeholder',
            'task':       'zero_shot',
            'model_path': None,
            'labels':     ['factual claim', 'opinion', 'question', 'statement']
        }
 
        logger.info("✓ Placeholder claim detector loaded")
        return self._loaded_models['claim_detector']
 
    def load_ner_model(self) -> Dict[str, Any]:
        """Load Named Entity Recognition model"""
        if 'ner' in self._loaded_models:
            return self._loaded_models['ner']
 
        model_config = self.models_config['ner']
        use_trained  = model_config['use_trained'] and self.nlp_config['use_trained_models']
 
        if use_trained:
            model_path = model_config['trained_path']
 
            if os.path.exists(model_path):
                logger.info(f"Loading TRAINED NER model from {model_path}")
 
                model = pipeline(
                    "ner",
                    model=model_path,
                    aggregation_strategy=model_config['aggregation_strategy'],
                    device=self.device
                )
 
                self._loaded_models['ner'] = {
                    'model':      model,
                    'type':       'trained',
                    'model_path': model_path
                }
 
                logger.info("✓ Trained NER model loaded")
                return self._loaded_models['ner']
            else:
                logger.warning(f"Trained NER not found at {model_path}, using placeholder")
 
        # Placeholder fallback
        logger.info("Loading PLACEHOLDER NER model")
 
        model = pipeline(
            "ner",
            model=model_config['placeholder'],
            aggregation_strategy=model_config['aggregation_strategy'],
            device=self.device
        )
 
        self._loaded_models['ner'] = {
            'model':      model,
            'type':       'placeholder',
            'model_path': None
        }
 
        logger.info("✓ Placeholder NER model loaded")
        return self._loaded_models['ner']
 
    def load_stance_detector(self) -> Dict[str, Any]:
        """Load stance detection model"""
        if 'stance_detector' in self._loaded_models:
            return self._loaded_models['stance_detector']
 
        model_config = self.models_config['stance_detector']
        use_trained  = model_config['use_trained'] and self.nlp_config['use_trained_models']
 
        if use_trained:
            model_path = model_config['trained_path']
 
            if os.path.exists(model_path):
                logger.info(f"Loading TRAINED stance detector from {model_path}")
 
                # NOTE: we do NOT load pipeline here for stance detector
                # stance.py loads tokenizer + model directly for proper
                # sentence-pair encoding — pipeline doesn't support text_pair
                self._loaded_models['stance_detector'] = {
                    'model':      None,       # stance.py loads this itself
                    'type':       'trained',
                    'task':       'stance_classification',
                    'model_path': model_path  # ← key fix — path passed correctly
                }
 
                logger.info("✓ Trained stance detector config loaded")
                return self._loaded_models['stance_detector']
            else:
                logger.warning(f"Trained stance model not found at {model_path}, using placeholder")
 
        # Placeholder fallback
        logger.info("Loading PLACEHOLDER stance detector (NLI model)")
 
        model = pipeline(
            "text-classification",
            model=model_config['placeholder'],
            device=self.device
        )
 
        self._loaded_models['stance_detector'] = {
            'model':      model,
            'type':       'placeholder',
            'task':       'nli',
            'model_path': None
        }
 
        logger.info("✓ Placeholder stance detector loaded")
        return self._loaded_models['stance_detector']
 
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model"""
        if model_name not in self._loaded_models:
            return {'loaded': False}
 
        info = self._loaded_models[model_name]
        return {
            'loaded':     True,
            'type':       info['type'],
            'task':       info.get('task', 'unknown'),
            'model_path': info.get('model_path')
        }
 
    def clear_cache(self):
        """Clear loaded models from memory"""
        self._loaded_models.clear()
        logger.info("Model cache cleared")
 
 
# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
 
    manager = ModelManager()
 
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
 
    claim_det = manager.load_claim_detector()
    print(f"Claim Detector: {manager.get_model_info('claim_detector')}")
 
    ner = manager.load_ner_model()
    print(f"NER: {manager.get_model_info('ner')}")
 
    stance = manager.load_stance_detector()
    print(f"Stance Detector: {manager.get_model_info('stance_detector')}")