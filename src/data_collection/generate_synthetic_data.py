"""
Generate High-Quality Multilingual Synthetic Data
WITH RESUME CAPABILITY - Can continue if interrupted
"""

import google.generativeai as genai
import json
import time
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
from datetime import datetime
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResumableMultilingualGenerator:
    """
    Generate synthetic data with checkpoint/resume capability
    """
    
    LANGUAGES = {
        'en': {
            'name': 'English',
            'domains': [
                'economy', 'health', 'politics', 'science',
                'technology', 'environment', 'sports', 'education',
                'business', 'international_relations'
            ],
            'samples_per_domain': 200  # 2,000 per language
        },
        'hi': {
            'name': 'Hindi',
            'domains': [
                'indian_economy', 'healthcare', 'politics', 'science',
                'technology', 'environment', 'cricket', 'education',
                'bollywood', 'government_policy'
            ],
            'samples_per_domain': 200
        },
        'es': {
            'name': 'Spanish',
            'domains': [
                'economia', 'salud', 'politica', 'ciencia',
                'tecnologia', 'medio_ambiente', 'deportes', 'educacion',
                'cultura', 'negocios'
            ],
            'samples_per_domain': 200
        },
        'ar': {
            'name': 'Arabic',
            'domains': [
                'اقتصاد', 'صحة', 'سياسة', 'علوم',
                'تكنولوجيا', 'بيئة', 'رياضة', 'تعليم',
                'ثقافة', 'أعمال'
            ],
            'samples_per_domain': 200
        },
        'fr': {
            'name': 'French',
            'domains': [
                'économie', 'santé', 'politique', 'sciences',
                'technologie', 'environnement', 'sports', 'éducation',
                'culture', 'affaires'
            ],
            'samples_per_domain': 200
        },
        'zh': {
            'name': 'Chinese',
            'domains': [
                '经济', '健康', '政治', '科学',
                '科技', '环境', '体育', '教育',
                '文化', '商业'
            ],
            'samples_per_domain': 200
        }
    }
    
    def __init__(self, api_key: str):
        """Initialize generator"""
        genai.configure(api_key=api_key)
        
        # Use gemini-flash-latest (better free tier model)
        self.model = genai.GenerativeModel('gemini-flash-latest')
        
        # Rate limiting for FREE TIER (conservative approach)
        self.requests_per_minute = 2  # Very conservative for free tier
        self.delay = 60 / self.requests_per_minute  # 30 seconds between requests
        
        # Checkpoint directory
        self.checkpoint_dir = Path("data/synthetic/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ Resumable Generator initialized")
    
    def _get_checkpoint_path(self, language_code: str, domain: str) -> Path:
        """Get checkpoint file path for language-domain"""
        return self.checkpoint_dir / f"{language_code}_{domain}.json"
    
    def _load_checkpoint(self, language_code: str, domain: str) -> List[Dict]:
        """Load existing checkpoint if exists"""
        checkpoint_path = self._get_checkpoint_path(language_code, domain)
        
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"  ↻ Loaded {len(data)} existing examples from checkpoint")
                return data
        
        return []
    
    def _save_checkpoint(self, language_code: str, domain: str, examples: List[Dict]):
        """Save checkpoint"""
        checkpoint_path = self._get_checkpoint_path(language_code, domain)
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
    
    def _is_domain_complete(self, language_code: str, domain: str, target: int) -> bool:
        """Check if domain already has enough examples"""
        checkpoint = self._load_checkpoint(language_code, domain)
        return len(checkpoint) >= target
    
    def generate_for_language(
        self,
        language_code: str,
        output_dir: str = "data/synthetic/raw"
    ) -> Dict:
        """Generate data for specific language WITH RESUME"""
        
        if language_code not in self.LANGUAGES:
            raise ValueError(f"Unsupported language: {language_code}")
        
        lang_config = self.LANGUAGES[language_code]
        lang_name = lang_config['name']
        
        logger.info("="*80)
        logger.info(f"GENERATING DATA FOR: {lang_name} ({language_code})")
        logger.info("="*80)
        
        all_examples = []
        
        for domain in lang_config['domains']:
            target_samples = lang_config['samples_per_domain']
            
            # Check if already complete
            if self._is_domain_complete(language_code, domain, target_samples):
                logger.info(f"\n✓ Domain '{domain}' already complete, skipping")
                checkpoint = self._load_checkpoint(language_code, domain)
                all_examples.extend(checkpoint)
                continue
            
            logger.info(f"\nDomain: {domain}")
            
            # Load existing progress
            existing = self._load_checkpoint(language_code, domain)
            remaining = target_samples - len(existing)
            
            logger.info(f"  Progress: {len(existing)}/{target_samples} (generating {remaining} more)")
            
            # Generate remaining samples
            new_examples = self._generate_for_domain(
                language_code=language_code,
                language_name=lang_name,
                domain=domain,
                num_samples=remaining,
                start_index=len(existing)
            )
            
            # Combine with existing
            domain_examples = existing + new_examples
            
            # Save checkpoint
            self._save_checkpoint(language_code, domain, domain_examples)
            
            all_examples.extend(domain_examples)
            
            logger.info(f"  ✓ Generated {len(new_examples)} new examples")
            logger.info(f"  ✓ Total for domain: {len(domain_examples)}")
        
        # Save final file
        output_path = Path(output_dir) / f"fact_check_{language_code}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ SAVED FINAL: {output_path}")
        logger.info(f"  Total examples: {len(all_examples)}")
        
        # Clean up checkpoints for this language
        self._cleanup_checkpoints(language_code)
        
        return {
            'language': language_code,
            'total_examples': len(all_examples),
            'output_file': str(output_path)
        }
    
    def _cleanup_checkpoints(self, language_code: str):
        """Remove checkpoints after successful completion"""
        for checkpoint_file in self.checkpoint_dir.glob(f"{language_code}_*.json"):
            checkpoint_file.unlink()
        logger.info(f"  ✓ Cleaned up checkpoints for {language_code}")
    
    def _generate_for_domain(
        self,
        language_code: str,
        language_name: str,
        domain: str,
        num_samples: int,
        start_index: int = 0
    ) -> List[Dict]:
        """Generate samples for specific domain"""
        
        verdicts = ['TRUE', 'FALSE', 'MOSTLY TRUE', 'MOSTLY FALSE', 'CONFLICTING']
        examples = []
        
        progress = tqdm(
            range(num_samples),
            desc=f"  {domain}",
            initial=0,
            total=num_samples
        )
        
        for i in progress:
            verdict = verdicts[(start_index + i) % len(verdicts)]
            
            try:
                example = self._generate_single_example(
                    language_name=language_name,
                    domain=domain,
                    verdict=verdict
                )
                
                if example:
                    example['language'] = language_code
                    example['domain'] = domain
                    example['generated_at'] = datetime.now().isoformat()
                    examples.append(example)
                    
                    # Save checkpoint every 10 examples
                    if (i + 1) % 10 == 0:
                        temp_checkpoint = self._load_checkpoint(language_code, domain)
                        temp_checkpoint.extend(examples)
                        self._save_checkpoint(language_code, domain, temp_checkpoint)
                        examples = []  # Clear to avoid duplicates
                
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                continue
        
        # Save any remaining examples
        if examples:
            temp_checkpoint = self._load_checkpoint(language_code, domain)
            temp_checkpoint.extend(examples)
            self._save_checkpoint(language_code, domain, temp_checkpoint)
        
        return self._load_checkpoint(language_code, domain)
    
    def _generate_single_example(
        self,
        language_name: str,
        domain: str,
        verdict: str
    ) -> Dict:
        """Generate single example"""
        
        prompt = f"""You are a fact-checking expert creating training data in {language_name}.

Generate a realistic, high-quality fact-checking example about {domain}.

REQUIREMENTS:
1. CLAIM: Specific, verifiable, with numbers/dates/names
2. EVIDENCE: 3 pieces from realistic sources (Reuters, WHO, etc.)
3. EXPLANATION: 3-4 sentences explaining the verdict
4. Use ONLY {language_name} language

VERDICT: {verdict}

Format EXACTLY as:

CLAIM: [detailed claim in {language_name}]

EVIDENCE:
1. [According to [source], specific data in {language_name}]
2. [A [source] report states that... in {language_name}]
3. [Research found... in {language_name}]

VERDICT: {verdict}

EXPLANATION: [3-4 sentence explanation in {language_name}]

Generate now:"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            parsed = self._parse_response(text, verdict)
            return parsed
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                logger.warning(f"Rate limit hit, waiting 90 seconds...")
                time.sleep(90)  # Wait longer for rate limit reset
                return None
            elif "503" in error_str or "overloaded" in error_str.lower():
                logger.warning(f"Service overloaded, waiting 30 seconds...")
                time.sleep(30)
                return None
            else:
                logger.error(f"Generation error: {error_str}")
                return None
    
    def _parse_response(self, text: str, expected_verdict: str) -> Dict:
        """Parse response"""
        try:
            claim = self._extract_section(text, "CLAIM:", "EVIDENCE:")
            evidence_text = self._extract_section(text, "EVIDENCE:", "VERDICT:")
            verdict = self._extract_section(text, "VERDICT:", "EXPLANATION:")
            explanation = self._extract_section(text, "EXPLANATION:", None)
            
            evidence_pieces = []
            for line in evidence_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    evidence = line.split('.', 1)[-1].strip() if '.' in line else line
                    if evidence:
                        evidence_pieces.append(evidence)
            
            return {
                'claim': claim.strip(),
                'evidence': evidence_pieces,
                'verdict': verdict.strip(),
                'explanation': explanation.strip(),
                'source': 'synthetic_gemini'
            }
            
        except Exception as e:
            return None
    
    def _extract_section(self, text: str, start: str, end: str) -> str:
        """Extract section"""
        start_idx = text.find(start)
        if start_idx == -1:
            return ""
        
        start_idx += len(start)
        
        if end:
            end_idx = text.find(end, start_idx)
            if end_idx == -1:
                return text[start_idx:].strip()
            return text[start_idx:end_idx].strip()
        else:
            return text[start_idx:].strip()
    
    def generate_all_languages(self, output_dir: str = "data/synthetic/raw"):
        """Generate all languages with resume capability"""
        
        logger.info("\n" + "="*80)
        logger.info("RESUMABLE MULTILINGUAL DATA GENERATION")
        logger.info("="*80)
        logger.info("Checkpoints saved in: data/synthetic/checkpoints/")
        logger.info("Can resume if interrupted (Ctrl+C)")
        logger.info("="*80 + "\n")
        
        results = []
        
        for lang_code in self.LANGUAGES.keys():
            try:
                result = self.generate_for_language(lang_code, output_dir)
                results.append(result)
                
                logger.info(f"\n✓ Completed: {lang_code}")
                logger.info(f"  Samples: {result['total_examples']}")
                
            except KeyboardInterrupt:
                logger.warning("\n\n⚠ INTERRUPTED! Progress saved in checkpoints.")
                logger.warning("Run again to resume from where you stopped.")
                break
            except Exception as e:
                logger.error(f"Error for {lang_code}: {str(e)}")
                continue
        
        # Summary
        if results:
            logger.info("\n" + "="*80)
            logger.info("GENERATION SUMMARY")
            logger.info("="*80)
            
            total = sum(r['total_examples'] for r in results)
            logger.info(f"Total examples: {total}")
            
            for result in results:
                logger.info(f"  {result['language']}: {result['total_examples']}")
        
        return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    else:
        API_KEY = input("Enter Gemini API key: ")
    
    generator = ResumableMultilingualGenerator(api_key=API_KEY)
    
    try:
        generator.generate_all_languages()
        
        print("\n" + "="*80)
        print("✓ GENERATION COMPLETE!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted. Progress saved!")
        print("Run the same command again to resume.")