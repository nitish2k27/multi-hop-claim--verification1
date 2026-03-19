"""
Download FEVER-NLI dataset for Stance Detection
Uses pre-processed FEVER with evidence text already included
"""

import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StanceDatasetDownloader:
    """Download FEVER-NLI for stance detection"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_fever_nli(self):
        """
        Download FEVER-NLI dataset
        
        This is FEVER data pre-processed with actual evidence text
        No need to download 5GB Wikipedia dump!
        """
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING FEVER-NLI DATASET")
        logger.info("="*80)
        logger.info("\nDataset: pietrolesci/nli_fever")
        logger.info("Format: Claim + Evidence text (already paired)")
        logger.info("No Wikipedia download needed!")
        
        try:
            # Load from HuggingFace
            logger.info("\nLoading from HuggingFace...")
            
            dataset = load_dataset("pietrolesci/nli_fever")
            
            logger.info("\n✓ Dataset downloaded successfully!")
            logger.info(f"\nSplits:")
            logger.info(f"  Train: {len(dataset['train']):,} examples")
            logger.info(f"  Dev:   {len(dataset['dev']):,} examples")
            
            # Show sample
            sample = dataset['train'][0]
            logger.info(f"\nSample:")
            logger.info(f"  Claim: {sample['hypothesis'][:80]}...")
            logger.info(f"  Evidence: {sample['premise'][:80]}...")
            logger.info(f"  Label: {sample['label']}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"\n✗ Download failed: {str(e)}")
            return None
    
    def process_to_dataframe(self, dataset):
        """Convert to DataFrame and clean"""
        logger.info("\n" + "="*80)
        logger.info("PROCESSING TO DATAFRAME")
        logger.info("="*80)
        
        # Convert to pandas
        train_df = dataset['train'].to_pandas()
        dev_df = dataset['dev'].to_pandas()
        
        logger.info(f"\n✓ Converted to DataFrames")
        logger.info(f"  Train: {len(train_df):,}")
        logger.info(f"  Dev:   {len(dev_df):,}")
        
        # Label mapping
        # 0: entailment → SUPPORTS
        # 1: neutral → NOT ENOUGH INFO
        # 2: contradiction → REFUTES
        label_map = {
            0: 'SUPPORTS',
            1: 'NOT ENOUGH INFO',
            2: 'REFUTES'
        }
        
        logger.info(f"\nLabel mapping:")
        for num, text in label_map.items():
            logger.info(f"  {num} → {text}")
        
        # Process train
        train_df = train_df[['premise', 'hypothesis', 'label']].copy()
        train_df = train_df.rename(columns={
            'premise': 'evidence',
            'hypothesis': 'claim'
        })
        train_df['label'] = train_df['label'].map(label_map)
        
        # Process dev
        dev_df = dev_df[['premise', 'hypothesis', 'label']].copy()
        dev_df = dev_df.rename(columns={
            'premise': 'evidence',
            'hypothesis': 'claim'
        })
        dev_df['label'] = dev_df['label'].map(label_map)
        
        # Drop any nulls
        train_df = train_df.dropna()
        dev_df = dev_df.dropna()
        
        logger.info(f"\n✓ Processed and cleaned")
        logger.info(f"  Train: {len(train_df):,}")
        logger.info(f"  Dev:   {len(dev_df):,}")
        
        # Show label distribution
        logger.info(f"\nLabel distribution (Train):")
        for label, count in train_df['label'].value_counts().items():
            logger.info(f"  {label}: {count:,} ({count/len(train_df)*100:.1f}%)")
        
        return train_df, dev_df
    
    def create_splits(self, train_df, dev_df):
        """Create train/val/test splits"""
        logger.info("\n" + "="*80)
        logger.info("CREATING SPLITS")
        logger.info("="*80)
        
        # Split dev into val and test (50/50)
        val_df, test_df = train_test_split(
            dev_df,
            test_size=0.5,
            random_state=42,
            stratify=dev_df['label']
        )
        
        logger.info(f"\n✓ Splits created:")
        logger.info(f"  Train: {len(train_df):>8,} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df):>8,} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df):>8,} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_data(self, train, val, test):
        """Save processed data"""
        logger.info("\n" + "="*80)
        logger.info("SAVING PROCESSED DATA")
        logger.info("="*80)
        
        train_path = self.processed_dir / "stance_detection_train.csv"
        val_path = self.processed_dir / "stance_detection_val.csv"
        test_path = self.processed_dir / "stance_detection_test.csv"
        
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)
        
        logger.info(f"\n✓ Saved:")
        logger.info(f"  {train_path}")
        logger.info(f"  {val_path}")
        logger.info(f"  {test_path}")
        
        # Save label mapping
        label_info = {
            'labels': ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'],
            'num_labels': 3,
            'label2id': {
                'SUPPORTS': 0,
                'REFUTES': 1,
                'NOT ENOUGH INFO': 2
            },
            'id2label': {
                0: 'SUPPORTS',
                1: 'REFUTES',
                2: 'NOT ENOUGH INFO'
            }
        }
        
        label_file = self.processed_dir / "stance_labels.json"
        with open(label_file, 'w') as f:
            json.dump(label_info, f, indent=2)
        
        logger.info(f"  {label_file}")
        
        # Show samples
        logger.info("\n" + "="*80)
        logger.info("SAMPLE DATA")
        logger.info("="*80)
        
        for label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
            sample = train[train['label'] == label].iloc[0]
            logger.info(f"\n{label}:")
            logger.info(f"  Claim: {sample['claim'][:70]}...")
            logger.info(f"  Evidence: {sample['evidence'][:70]}...")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("DATASET SUMMARY")
        logger.info("="*80)
        
        total = len(train) + len(val) + len(test)
        
        logger.info(f"\nTotal examples: {total:,}")
        logger.info(f"\nSplits:")
        logger.info(f"  Train: {len(train):>8,}")
        logger.info(f"  Val:   {len(val):>8,}")
        logger.info(f"  Test:  {len(test):>8,}")
        
        logger.info(f"\nLabel distribution (Train):")
        for label, count in train['label'].value_counts().items():
            logger.info(f"  {label}: {count:>8,} ({count/len(train)*100:.1f}%)")
        
        logger.info("\n" + "="*80)
    
    def run(self):
        """Run complete pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STANCE DETECTION DATA PREPARATION")
        logger.info("="*80)
        logger.info("\nUsing: pietrolesci/nli_fever")
        logger.info("Benefits:")
        logger.info("  ✓ Evidence text already included")
        logger.info("  ✓ No 5GB Wikipedia download")
        logger.info("  ✓ Clean NLI format")
        logger.info("  ✓ Ready for training")
        logger.info("="*80)
        
        # Download
        dataset = self.download_fever_nli()
        
        if not dataset:
            logger.error("\n✗ Download failed")
            return False
        
        # Process
        train_df, dev_df = self.process_to_dataframe(dataset)
        
        # Split
        train, val, test = self.create_splits(train_df, dev_df)
        
        # Save
        self.save_data(train, val, test)
        
        logger.info("\n" + "="*80)
        logger.info("✓ STANCE DATA READY!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("  1. Upload CSVs to Kaggle dataset")
        logger.info("  2. Run Kaggle training notebook")
        logger.info("  3. Train for 2-3 hours")
        logger.info("="*80 + "\n")
        
        return True


if __name__ == "__main__":
    downloader = StanceDatasetDownloader()
    downloader.run()