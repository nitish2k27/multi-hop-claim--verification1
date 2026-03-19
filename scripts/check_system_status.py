"""
Pre-flight check: Verify all components are ready
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import json


def check_system_status():
    """Check if all system components are configured and ready"""
    
    print("\n" + "="*80)
    print("SYSTEM PRE-FLIGHT CHECK")
    print("="*80)
    
    issues = []
    warnings = []
    
    # ==========================================
    # 1. Check Config File
    # ==========================================
    
    print("\n-> Checking configuration...")
    
    config_path = project_root / "configs" / "nlp_config.yaml"
    
    if not config_path.exists():
        issues.append(f"Config file not found: {config_path}")
    else:
        print(f"  [OK] Config file found")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check models config
            models = config.get('nlp_pipeline', {}).get('models', {})
            
            # Claim detector
            claim_config = models.get('claim_detector', {})
            if claim_config.get('use_trained'):
                print(f"  [OK] Claim detector: use_trained = true")
            else:
                warnings.append("Claim detector: using placeholder (not trained)")
            
            # Stance detector
            stance_config = models.get('stance_detector', {})
            if stance_config.get('use_trained'):
                print(f"  [OK] Stance detector: use_trained = true")
            else:
                warnings.append("Stance detector: using placeholder (not trained)")
        
        except Exception as e:
            issues.append(f"Failed to read config: {e}")
    
    # ==========================================
    # 2. Check Model Files
    # ==========================================
    
    print("\n-> Checking model files...")
    
    models_dir = project_root / "models"
    
    # Claim detector
    claim_model_path = models_dir / "claim_detector" / "final"
    if claim_model_path.exists():
        required_files = ['config.json', 'tokenizer_config.json']
        missing = [f for f in required_files if not (claim_model_path / f).exists()]
        
        if missing:
            issues.append(f"Claim detector missing files: {missing}")
        else:
            # Check if model file exists (either pytorch_model.bin or model.safetensors)
            has_model = (claim_model_path / "pytorch_model.bin").exists() or \
                       (claim_model_path / "model.safetensors").exists()
            
            if has_model:
                print(f"  [OK] Claim detector model found: {claim_model_path}")
            else:
                issues.append("Claim detector: model weights not found (pytorch_model.bin or model.safetensors)")
    else:
        issues.append(f"Claim detector model not found: {claim_model_path}")
    
    # Stance detector
    stance_model_path = models_dir / "stance_detector" / "final"
    if stance_model_path.exists():
        required_files = ['config.json', 'tokenizer_config.json']
        missing = [f for f in required_files if not (stance_model_path / f).exists()]
        
        if missing:
            issues.append(f"Stance detector missing files: {missing}")
        else:
            # Check if model file exists
            has_model = (stance_model_path / "pytorch_model.bin").exists() or \
                       (stance_model_path / "model.safetensors").exists()
            
            if has_model:
                print(f"  [OK] Stance detector model found: {stance_model_path}")
            else:
                issues.append("Stance detector: model weights not found (pytorch_model.bin or model.safetensors)")
    else:
        warnings.append(f"Stance detector model not found: {stance_model_path}")
        warnings.append("  Set use_trained: false in config or train the model")
    
    # ==========================================
    # 3. Check NLTK Data
    # ==========================================
    
    print("\n-> Checking NLTK data...")
    
    try:
        import nltk
        
        required_nltk = ['punkt', 'punkt_tab', 'stopwords']
        
        for resource in required_nltk:
            try:
                if resource == 'stopwords':
                    nltk.data.find(f'corpora/{resource}')
                else:
                    nltk.data.find(f'tokenizers/{resource}')
                print(f"  [OK] NLTK {resource} found")
            except LookupError:
                issues.append(f"NLTK {resource} not found - run: python scripts/setup_nltk.py")
    
    except ImportError:
        warnings.append("NLTK not installed - some features may not work")
    
    # ==========================================
    # 4. Check Dependencies
    # ==========================================
    
    print("\n-> Checking dependencies...")
    
    required_packages = {
        'transformers': 'transformers',
        'torch': 'torch',
        'chromadb': 'chromadb',
        'sentence_transformers': 'sentence-transformers',
        'yaml': 'pyyaml',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        issues.append(f"Missing packages: {missing_packages}")
        issues.append(f"  Install with: pip install {' '.join(missing_packages)}")
    
    # ==========================================
    # 5. Check Data Directories
    # ==========================================
    
    print("\n-> Checking data directories...")
    
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    chroma_dir = data_dir / "chroma_db"
    
    if not processed_dir.exists():
        warnings.append(f"Processed data directory not found: {processed_dir}")
    else:
        print(f"  [OK] Processed data directory exists")
    
    if not chroma_dir.exists():
        warnings.append(f"ChromaDB directory not found: {chroma_dir}")
        warnings.append("  Run: python scripts/ingest_to_rag.py")
    else:
        print(f"  [OK] ChromaDB directory exists")
    
    # ==========================================
    # 6. Check Python Version
    # ==========================================
    
    print("\n-> Checking Python version...")
    
    py_version = sys.version_info
    if py_version.major >= 3 and py_version.minor >= 8:
        print(f"  [OK] Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        warnings.append(f"Python 3.8+ recommended, you have {py_version.major}.{py_version.minor}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if not issues and not warnings:
        print("\n[SUCCESS] ALL CHECKS PASSED!")
        print("\nYour system is ready to use.")
        return True
    
    if issues:
        print(f"\n[ERROR] CRITICAL ISSUES ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    if warnings:
        print(f"\n[WARNING] WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    print("\n" + "="*80)
    
    if issues:
        print("\n[ERROR] Please fix critical issues before running tests.")
        print("\nQuick fixes:")
        print("  1. Install NLTK data: python scripts/setup_nltk.py")
        print("  2. Install packages: pip install -r requirements.txt")
        print("  3. Train models or set use_trained: false in config")
        return False
    else:
        print("\n[WARNING] System will work but with limited functionality.")
        return True


if __name__ == "__main__":
    success = check_system_status()
    sys.exit(0 if success else 1)