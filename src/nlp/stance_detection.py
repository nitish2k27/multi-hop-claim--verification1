"""
Stance Detection Module
Determines if evidence SUPPORTS, REFUTES, or is NEUTRAL to claim
"""
 
import logging
import torch
from typing import Dict, Any, List
from src.nlp.model_manager import ModelManager
 
logger = logging.getLogger(__name__)
 
 
class StanceDetector:
    """
    Detect stance of evidence towards a claim
    
    Supports both trained and placeholder models
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize stance detector
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.model_info = model_manager.load_stance_detector()
        self.model = self.model_info['model']
        self.model_type = self.model_info['type']
        self.task = self.model_info['task']
        
        # Load tokenizer and raw model for trained model inference
        if self.model_type == 'trained':
            self._load_trained_components()
        
        logger.info(f"StanceDetector initialized (type: {self.model_type}, task: {self.task})")
    
    def _load_trained_components(self):
        """Load tokenizer and model directly for proper sentence-pair encoding"""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import json
        from pathlib import Path
        
        # Get model path from model_manager config
        model_path = self.model_info.get('model_path')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.raw_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.raw_model.eval()
        
        # Load label mapping
        label_file = Path(model_path) / "labels.json"
        if label_file.exists():
            with open(label_file, 'r') as f:
                label_info = json.load(f)
            self.id2label = {int(k): v for k, v in label_info['id2label'].items()}
        else:
            # Fallback — read from model config
            self.id2label = self.raw_model.config.id2label
        
        logger.info(f"Loaded tokenizer and model from {model_path}")
        logger.info(f"Labels: {self.id2label}")
    
    def detect(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Detect stance of evidence towards claim
        
        Args:
            claim: The claim being verified
            evidence: Evidence text
        
        Returns:
            {
                'stance': str,  # SUPPORTS, REFUTES, or NEUTRAL
                'confidence': float,
                'label_scores': dict
            }
        """
        if not claim or not evidence:
            return {
                'stance': 'NEUTRAL',
                'confidence': 0.0,
                'label_scores': {}
            }
        
        if self.model_type == 'trained':
            return self._detect_with_trained_model(claim, evidence)
        else:
            return self._detect_with_placeholder(claim, evidence)
    
    def _detect_with_trained_model(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Use trained stance detection model with proper sentence-pair encoding
        
        Format: [CLS] claim [SEP] evidence [SEP]
        """
        # Proper sentence-pair tokenization — NOT string concatenation
        inputs = self.tokenizer(
            claim,
            evidence,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            logits = self.raw_model(**inputs).logits
        
        # Get predicted label and confidence
        probs = torch.softmax(logits, dim=1)[0]
        predicted_id = probs.argmax().item()
        predicted_label = self.id2label[predicted_id]
        confidence = probs[predicted_id].item()
        
        # Map to standard stance labels
        label_map = {
            'SUPPORTS': 'SUPPORTS',
            'REFUTES': 'REFUTES',
            'NOT ENOUGH INFO': 'NEUTRAL',
            'NEUTRAL': 'NEUTRAL'
        }
        
        stance = label_map.get(predicted_label, 'NEUTRAL')
        
        # All label scores
        label_scores = {
            self.id2label[i]: probs[i].item()
            for i in range(len(self.id2label))
        }
        
        return {
            'stance': stance,
            'confidence': confidence,
            'label_scores': label_scores,
            'raw_label': predicted_label
        }
    
    def _detect_with_placeholder(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Use NLI model as placeholder
        
        NLI labels: entailment, contradiction, neutral
        Map to: SUPPORTS, REFUTES, NEUTRAL
        """
        # For NLI: evidence is premise, claim is hypothesis
        text_pair = f"{evidence} [SEP] {claim}"
        
        result = self.model(text_pair)[0]
        
        nli_to_stance = {
            'LABEL_0': 'SUPPORTS',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'REFUTES',
            'entailment': 'SUPPORTS',
            'neutral': 'NEUTRAL',
            'contradiction': 'REFUTES'
        }
        
        stance = nli_to_stance.get(result['label'], 'NEUTRAL')
        
        return {
            'stance': stance,
            'confidence': result['score'],
            'label_scores': {result['label']: result['score']},
            'note': 'Using NLI model as placeholder'
        }
    
    def detect_batch(self, claim: str, evidence_list: List[str]) -> List[Dict[str, Any]]:
        """
        Detect stance for multiple evidence pieces
        
        Args:
            claim: The claim
            evidence_list: List of evidence texts
        
        Returns:
            List of stance detection results
        """
        results = []
        
        for evidence in evidence_list:
            result = self.detect(claim, evidence)
            result['evidence'] = evidence
            results.append(result)
        
        logger.debug(f"Detected stance for {len(evidence_list)} evidence pieces")
        
        return results
    
    def aggregate_stances(self, stance_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate multiple stance detections
        
        Args:
            stance_results: List of stance detection results
        
        Returns:
            Aggregated stance information
        """
        supports = [r for r in stance_results if r['stance'] == 'SUPPORTS']
        refutes  = [r for r in stance_results if r['stance'] == 'REFUTES']
        neutral  = [r for r in stance_results if r['stance'] == 'NEUTRAL']
        
        support_score = sum(r['confidence'] for r in supports)
        refute_score  = sum(r['confidence'] for r in refutes)
        neutral_score = sum(r['confidence'] for r in neutral)
        
        total = support_score + refute_score + neutral_score
        
        if total > 0:
            support_pct = (support_score / total) * 100
            refute_pct  = (refute_score  / total) * 100
            neutral_pct = (neutral_score / total) * 100
        else:
            support_pct = refute_pct = neutral_pct = 0
        
        if support_score > refute_score and support_score > neutral_score:
            overall = 'SUPPORTS'
        elif refute_score > support_score and refute_score > neutral_score:
            overall = 'REFUTES'
        else:
            overall = 'CONFLICTING' if support_score > 0 and refute_score > 0 else 'NEUTRAL'
        
        return {
            'overall_stance': overall,
            'support_percentage': support_pct,
            'refute_percentage': refute_pct,
            'neutral_percentage': neutral_pct,
            'num_supports': len(supports),
            'num_refutes': len(refutes),
            'num_neutral': len(neutral),
            'confidence': abs(support_pct - refute_pct) if overall == 'CONFLICTING' else max(support_pct, refute_pct, neutral_pct)
        }