#!/usr/bin/env python3
"""
Quick Start Script for NLP Pipeline UI
Run this from the project root to launch the interface
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Main startup function"""
    print("🚀 NLP Pipeline UI - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("ui").exists():
        print("❌ Please run this script from the project root directory")
        print("   (where src/ and ui/ folders are located)")
        return
    
    print("Choose interface:")
    print("1. Web UI (Streamlit) - Full featured")
    print("2. Simple Demo UI - Works without dependencies")
    print("3. Command Line Interface")
    print("4. Run Tests First")
    print("5. Batch Processing Demo")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        print("\n🌐 Starting Full Web UI...")
        print("This will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop")
        
        try:
            subprocess.run([sys.executable, "ui/run_ui.py"])
        except KeyboardInterrupt:
            print("\n👋 UI stopped")
            
    elif choice == "2":
        print("\n🌐 Starting Simple Demo UI...")
        print("This works without complex dependencies")
        print("Opening at http://localhost:8502")
        
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", "ui/simple_demo.py",
                "--server.port", "8502"
            ])
        except KeyboardInterrupt:
            print("\n👋 Demo stopped")
            
    elif choice == "3":
        print("\n💻 Starting CLI Demo...")
        try:
            subprocess.run([sys.executable, "ui/simple_cli_demo.py"])
        except KeyboardInterrupt:
            print("\n👋 CLI stopped")
            
    elif choice == "4":
        print("\n🧪 Running tests...")
        subprocess.run([sys.executable, "ui/test_ui.py"])
        
    elif choice == "5":
        print("\n🔄 Batch Processing Demo")
        print("Example commands:")
        print("  python ui/batch_processor.py texts 'The Earth is flat' 'Water boils at 100°C'")
        print("  python ui/batch_processor.py csv data/processed/news_articles_rag.csv content")
        print("  python ui/batch_processor.py folder data/uploads/test_user/20260403")
        
        demo_choice = input("\nRun demo with sample texts? (y/n): ").strip().lower()
        if demo_choice == 'y':
            subprocess.run([
                sys.executable, "ui/batch_processor.py", "texts",
                "The Earth is round and orbits the Sun",
                "Climate change is caused by human activities",
                "Vaccines are safe and effective"
            ])
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()