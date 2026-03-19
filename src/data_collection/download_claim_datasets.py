"""
Download FEVER dataset using official URLs
Based on the official fever.py loading script
"""

import os
import json
import logging
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FEVERDownloader:
    """
    Download FEVER v1.0 dataset from official source
    URLs from official fever.py script
    """
    
    # Official FEVER v1.0 URLs (from fever.py)
    BASE_URL = "https://fever.ai/download/fever"
    
    URLS = {
        'train': f"{BASE_URL}/train.jsonl",
        'labelled_dev': f"{BASE_URL}/shared_task_dev.jsonl",
        'paper_dev': f"{BASE_URL}/paper_dev.jsonl",
    }
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw" / "fever_official"
        self.processed_dir = self.output_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download file with progress bar"""
        
        filepath = self.raw_dir / filename
        
        # Check if already exists
        if filepath.exists():
            file_size = filepath.stat().st_size
            if file_size > 1000:  # At least 1KB
                logger.info(f"✓ {filename} already exists ({file_size:,} bytes)")
                return filepath
        
        logger.info(f"\nDownloading: {filename}")
        logger.info(f"From: {url}")
        
        try:
            # Stream download with progress
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download
            with open(filepath, 'wb') as f:
                if total_size == 0:
                    # No content-length header
                    f.write(response.content)
                    logger.info(f"✓ Downloaded {filename}")
                else:
                    # With progress bar
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            logger.info(f"✓ Downloaded {filename} ({filepath.stat().st_size:,} bytes)")
            return filepath
            
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed to download {filename}: {str(e)}")
            return None
    
    def load_jsonl(self, filepath: Path) -> list:
        """
        Load JSONL file
        
        FEVER format (from fever.py):
        {
            "id": 75397,
            "label": "SUPPORTS",  # or "REFUTES" or "NOT ENOUGH INFO"
            "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
            "evidence": [[[null, null, "Nikolaj_Coster-Waldau", 0], ...]]
        }
        """
        logger.info(f"\nLoading {filepath.name}...")
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc=f"Reading {filepath.name}"), 1):
                try:
                    example = json.loads(line.strip())
                    data.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: Invalid JSON")
                    continue
        
        logger.info(f"✓ Loaded {len(data):,} examples")
        
        # Show sample
        if data:
            logger.info(f"\nSample example:")
            sample = data[0]
            logger.info(f"  ID: {sample.get('id')}")
            logger.info(f"  Claim: {sample.get('claim', '')[:100]}...")
            logger.info(f"  Label: {sample.get('label', 'N/A')}")
        
        return data
    
    def download_all(self):
        """Download all FEVER files"""
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING FEVER v1.0 DATASET")
        logger.info("="*80)
        logger.info("\nOfficial source: https://fever.ai")
        logger.info("Total files to download: 3")
        logger.info("  - train.jsonl (~130MB)")
        logger.info("  - shared_task_dev.jsonl (~7MB)")
        logger.info("  - paper_dev.jsonl (~1MB)")
        logger.info("="*80)
        
        downloaded = {}
        
        for split_name, url in self.URLS.items():
            filename = url.split('/')[-1]
            filepath = self.download_file(url, filename)
            
            if filepath:
                downloaded[split_name] = filepath
            else:
                logger.warning(f"Skipping {split_name} due to download failure")
        
        if not downloaded:
            logger.error("\n✗ No files downloaded successfully")
            return None
        
        logger.info(f"\n✓ Successfully downloaded {len(downloaded)}/{len(self.URLS)} files")
        
        # Load data
        logger.info("\n" + "="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        
        fever_data = {}
        for split_name, filepath in downloaded.items():
            fever_data[split_name] = self.load_jsonl(filepath)
        
        # Show statistics
        total_examples = sum(len(examples) for examples in fever_data.values())
        
        logger.info("\n" + "="*80)
        logger.info("DATASET STATISTICS")
        logger.info("="*80)
        
        for split_name, examples in fever_data.items():
            logger.info(f"\n{split_name}: {len(examples):,} examples")
            
            # Count labels
            if examples:
                label_counts = {}
                for ex in examples:
                    label = ex.get('label', 'UNKNOWN')
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                logger.info("  Labels:")
                for label, count in sorted(label_counts.items()):
                    logger.info(f"    {label}: {count:,}")
        
        logger.info(f"\nTotal FEVER examples: {total_examples:,}")
        logger.info("="*80)
        
        return fever_data
    
    def process_for_claim_detection(self, fever_data):
        """Process FEVER for claim detection"""
        
        logger.info("\n" + "="*80)
        logger.info("PROCESSING FOR CLAIM DETECTION")
        logger.info("="*80)
        
        # ==========================================
        # STEP 1: Extract claims (positive examples)
        # ==========================================
        
        logger.info("\n→ Step 1: Extracting FEVER claims (positive examples)")
        
        claims = []
        
        for split_name, examples in fever_data.items():
            logger.info(f"  Processing {split_name}...")
            
            for ex in tqdm(examples, desc=f"  {split_name}"):
                claims.append({
                    'text': ex['claim'],
                    'label': 1,  # is_claim
                    'source': f'fever_{split_name}',
                    'fever_label': ex.get('label', 'UNKNOWN')
                })
        
        logger.info(f"\n✓ Extracted {len(claims):,} positive examples")
        
        positive_df = pd.DataFrame(claims)
        
        # Show FEVER label distribution
        logger.info("\nOriginal FEVER labels:")
        for label, count in positive_df['fever_label'].value_counts().items():
            logger.info(f"  {label}: {count:,}")
        
        # ==========================================
        # STEP 2: Create negative examples
        # ==========================================
        
        logger.info("\n→ Step 2: Creating negative examples (non-claims)")
        
        target_count = len(claims)
        logger.info(f"  Target: {target_count:,} negative examples")
        
        negative_examples = []
        
        # Templates for non-claims
        templates = {
            'question': [
                "What is {topic}?",
                "How does {topic} work?",
                "When did {event} happen?",
                "Where is {location}?",
                "Who is {person}?",
                "Why did {event} occur?",
                "Which {option} is better?",
                "Can you explain {topic}?",
            ],
            'opinion': [
                "I think {statement}",
                "It seems {statement}",
                "Perhaps {statement}",
                "In my opinion, {statement}",
                "It appears {statement}",
                "Probably {statement}",
            ],
            'instruction': [
                "Please check {topic}",
                "Review {document}",
                "Read about {topic}",
                "Consider {statement}",
                "Examine {topic}",
                "Verify {statement}",
            ],
            'generic': [
                "The article mentions {topic}",
                "According to sources, {statement}",
                "Information about {topic}",
                "Details on {topic}",
                "The report discusses {topic}",
            ]
        }
        
        # Placeholders
        placeholders = {
            'topic': ['GDP', 'economy', 'climate', 'policy', 'technology', 'healthcare'],
            'event': ['the election', 'the summit', 'the conference', 'the meeting'],
            'location': ['the capital', 'the region', 'the city'],
            'person': ['the president', 'the CEO', 'the official'],
            'statement': ['various factors exist', 'conditions vary', 'research continues'],
            'option': ['policy A or B', 'option 1 or 2'],
            'document': ['the report', 'the study', 'the analysis']
        }
        
        logger.info("  Generating synthetic non-claims...")
        
        all_templates = []
        for category_templates in templates.values():
            all_templates.extend(category_templates)
        
        for _ in tqdm(range(target_count), desc="  Generating"):
            template = random.choice(all_templates)
            
            # Fill placeholders
            text = template
            for placeholder_type, options in placeholders.items():
                if '{' + placeholder_type + '}' in text:
                    text = text.replace('{' + placeholder_type + '}', random.choice(options))
            
            negative_examples.append({
                'text': text,
                'label': 0,  # not_claim
                'source': 'synthetic'
            })
        
        logger.info(f"✓ Created {len(negative_examples):,} negative examples")
        
        # ==========================================
        # STEP 3: Combine
        # ==========================================
        
        logger.info("\n→ Step 3: Combining positive and negative")
        
        # Clean positive_df
        positive_clean = positive_df[['text', 'label', 'source']].copy()
        negative_df = pd.DataFrame(negative_examples)
        
        # Combine
        combined = pd.concat([positive_clean, negative_df], ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"\n✓ Combined dataset:")
        logger.info(f"  Total:      {len(combined):,}")
        logger.info(f"  Claims:     {len(combined[combined['label']==1]):,} ({len(combined[combined['label']==1])/len(combined)*100:.1f}%)")
        logger.info(f"  Not claims: {len(combined[combined['label']==0]):,} ({len(combined[combined['label']==0])/len(combined)*100:.1f}%)")
        
        return combined
    
    def create_splits(self, df):
        """Split into train/val/test"""
        
        logger.info("\n→ Step 4: Creating train/val/test splits")
        
        # 80/10/10
        train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
        
        logger.info(f"\n✓ Splits:")
        logger.info(f"  Train: {len(train):>8,} ({len(train)/len(df)*100:.1f}%)")
        logger.info(f"  Val:   {len(val):>8,} ({len(val)/len(df)*100:.1f}%)")
        logger.info(f"  Test:  {len(test):>8,} ({len(test)/len(df)*100:.1f}%)")
        
        return train, val, test
    
    def save_data(self, train, val, test):
        """Save processed data"""
        
        logger.info("\n→ Step 5: Saving datasets")
        
        train_path = self.processed_dir / "claim_detection_train.csv"
        val_path = self.processed_dir / "claim_detection_val.csv"
        test_path = self.processed_dir / "claim_detection_test.csv"
        
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)
        
        logger.info(f"\n✓ Saved:")
        logger.info(f"  {train_path}")
        logger.info(f"  {val_path}")
        logger.info(f"  {test_path}")
        
        # Summary
        print("\n" + "="*80)
        print("FINAL SUMMARY - REAL FEVER DATA")
        print("="*80)
        
        total = len(train) + len(val) + len(test)
        
        print(f"\nTotal examples: {total:,}")
        print(f"\nSplits:")
        print(f"  Train: {len(train):>8,} ({len(train)/total*100:.1f}%)")
        print(f"  Val:   {len(val):>8,} ({len(val)/total*100:.1f}%)")
        print(f"  Test:  {len(test):>8,} ({len(test)/total*100:.1f}%)")
        
        print(f"\nLabel distribution:")
        for name, df in [("Train", train), ("Val", val), ("Test", test)]:
            pos = len(df[df['label']==1])
            neg = len(df[df['label']==0])
            print(f"  {name}:")
            print(f"    Claims:     {pos:>8,} ({pos/len(df)*100:.1f}%)")
            print(f"    Not claims: {neg:>8,} ({neg/len(df)*100:.1f}%)")
        
        print("\n" + "="*80)
    
    def run(self):
        """Run complete pipeline"""
        
        logger.info("\n" + "="*80)
        logger.info("FEVER CLAIM DETECTION - OFFICIAL DATASET")
        logger.info("="*80)
        
        # Download
        fever_data = self.download_all()
        
        if not fever_data:
            logger.error("\n✗ Download failed")
            return False
        
        # Process
        combined = self.process_for_claim_detection(fever_data)
        
        # Split
        train, val, test = self.create_splits(combined)
        
        # Save
        self.save_data(train, val, test)
        
        logger.info("\n" + "="*80)
        logger.info("✓ COMPLETE - REAL FEVER DATA READY!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("  1. Verify: python scripts/verify_claim_data.py")
        logger.info("  2. Train:  python src/training/train_claim_detector.py")
        logger.info("="*80 + "\n")
        
        return True


if __name__ == "__main__":
    downloader = FEVERDownloader()
    success = downloader.run()
    
    if success:
        print("\n✓ Real FEVER dataset downloaded and processed!")
    else:
        print("\n✗ Failed to download FEVER dataset")