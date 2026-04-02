#!/usr/bin/env python3
"""
ChromaDB Structure Checker
Explains the actual ChromaDB storage structure in your project
"""

import sys
from pathlib import Path
import sqlite3
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def check_chromadb_structure():
    """Check and explain ChromaDB structure"""
    
    print("🔍 ChromaDB Structure Analysis")
    print("="*60)
    
    chroma_path = Path("data/chroma_db")
    
    if not chroma_path.exists():
        print("❌ ChromaDB folder not found at data/chroma_db")
        return
    
    print(f"📁 ChromaDB Location: {chroma_path}")
    
    # Check SQLite database
    sqlite_file = chroma_path / "chroma.sqlite3"
    if sqlite_file.exists():
        print(f"✅ Metadata Database: {sqlite_file}")
        
        try:
            # Connect to SQLite and check collections
            conn = sqlite3.connect(sqlite_file)
            cursor = conn.cursor()
            
            # Get collections
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"\n📊 Database Tables:")
            for table in tables:
                print(f"  - {table[0]}")
            
            # Try to get collections info
            try:
                cursor.execute("SELECT * FROM collections;")
                collections = cursor.fetchall()
                
                if collections:
                    print(f"\n📚 Collections Found:")
                    for collection in collections:
                        print(f"  - ID: {collection[0]}")
                        print(f"    Name: {collection[1] if len(collection) > 1 else 'Unknown'}")
                        print(f"    UUID: {collection[2] if len(collection) > 2 else 'Unknown'}")
                        print()
                else:
                    print("\n⚠️  No collections found in database")
                    
            except sqlite3.OperationalError as e:
                print(f"\n⚠️  Could not read collections table: {e}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error reading SQLite database: {e}")
    
    else:
        print("❌ ChromaDB metadata file (chroma.sqlite3) not found")
    
    # Check UUID folders
    uuid_folders = [d for d in chroma_path.iterdir() if d.is_dir()]
    
    if uuid_folders:
        print(f"\n📂 Collection UUID Folders:")
        for folder in uuid_folders:
            print(f"\n  📁 {folder.name}")
            
            # Check contents
            files = list(folder.iterdir())
            for file in files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"    - {file.name} ({size_mb:.2f} MB)")
            
            # Try to estimate document count from file sizes
            data_files = [f for f in files if f.name.startswith('data_level')]
            if data_files:
                total_size = sum(f.stat().st_size for f in data_files)
                estimated_docs = total_size // 1024  # Rough estimate
                print(f"    📊 Estimated documents: ~{estimated_docs}")
    
    else:
        print("\n⚠️  No collection UUID folders found")


def check_uploads_structure():
    """Check uploads folder structure"""
    
    print("\n" + "="*60)
    print("📤 Uploads Structure Analysis")
    print("="*60)
    
    uploads_path = Path("data/uploads")
    
    if not uploads_path.exists():
        print("❌ Uploads folder not found")
        return
    
    print(f"📁 Uploads Location: {uploads_path}")
    
    # Check user folders
    user_folders = [d for d in uploads_path.iterdir() if d.is_dir()]
    
    if user_folders:
        print(f"\n👥 User Folders:")
        
        for user_folder in user_folders:
            print(f"\n  📁 User: {user_folder.name}")
            
            # Check date folders
            date_folders = [d for d in user_folder.iterdir() if d.is_dir()]
            
            for date_folder in date_folders:
                print(f"    📅 Date: {date_folder.name}")
                
                # Check files
                files = [f for f in date_folder.iterdir() if f.is_file()]
                
                for file in files:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"      📄 {file.name} ({size_mb:.2f} MB)")
                    
                    # Show original filename if available
                    if len(file.name) > 50:  # Likely a hash
                        print(f"         (Hashed filename - original name stored in metadata)")
    
    else:
        print("⚠️  No user folders found in uploads")


def explain_chromadb_naming():
    """Explain ChromaDB naming convention"""
    
    print("\n" + "="*60)
    print("📖 ChromaDB Storage Explanation")
    print("="*60)
    
    explanation = """
🔍 How ChromaDB Storage Works:

1. COLLECTION NAMES vs STORAGE:
   - You create collections with names like: 'news_articles', 'uploaded_documents'
   - ChromaDB internally assigns UUIDs to each collection
   - Storage folders use UUIDs, not collection names

2. FOLDER STRUCTURE:
   data/chroma_db/
   ├── chroma.sqlite3                    # Metadata (collection names → UUIDs)
   └── {uuid}/                          # Collection data folder
       ├── data_level0.bin              # Vector embeddings (main data)
       ├── header.bin                   # Collection metadata
       ├── index_metadata.pickle        # HNSW index metadata
       ├── length.bin                   # Document lengths
       └── link_lists.bin               # HNSW graph connections

3. COLLECTION MAPPING:
   - chroma.sqlite3 contains the mapping: collection_name → uuid
   - When you search 'news_articles', ChromaDB looks up its UUID
   - Then reads data from the corresponding UUID folder

4. FILE CONTENTS:
   - data_level0.bin: Your document embeddings (vectors)
   - header.bin: Collection configuration
   - index_metadata.pickle: Search index metadata
   - link_lists.bin: HNSW graph for fast similarity search

5. UPLOADS STRUCTURE:
   data/uploads/{user_id}/{date}/{file_hash}.{extension}
   - user_id: Organizes by user (e.g., 'test_user', 'company_x')
   - date: YYYYMMDD format (e.g., '20260403')
   - file_hash: SHA256 hash prevents duplicates
   - Original filename stored in ChromaDB metadata
"""
    
    print(explanation)


def check_collection_mapping():
    """Try to map collection names to UUIDs"""
    
    print("\n" + "="*60)
    print("🗺️  Collection Name → UUID Mapping")
    print("="*60)
    
    try:
        from src.rag.vector_database import VectorDatabase
        
        vdb = VectorDatabase()
        collections = vdb.list_collections()
        
        if collections:
            print("📚 Active Collections:")
            for collection_name in collections:
                try:
                    collection = vdb.get_collection(collection_name)
                    stats = vdb.get_collection_stats(collection_name)
                    
                    print(f"\n  📖 {collection_name}")
                    print(f"     Documents: {stats['count']}")
                    print(f"     Status: {'✅ Active' if stats['exists'] else '❌ Missing'}")
                    
                    # Try to get collection ID/UUID
                    if hasattr(collection, 'id'):
                        print(f"     UUID: {collection.id}")
                    
                except Exception as e:
                    print(f"     ❌ Error: {e}")
        else:
            print("⚠️  No collections found")
            print("\nTo create collections:")
            print("1. Run: python src/rag/vector_database.py")
            print("2. Or use: python test_rag_comparison.py")
    
    except Exception as e:
        print(f"❌ Error checking collections: {e}")
        print("\nMake sure to install dependencies:")
        print("pip install chromadb sentence-transformers")


if __name__ == "__main__":
    print("🔍 ChromaDB & Uploads Structure Checker")
    print("This script analyzes your actual data folder structure")
    
    # Check ChromaDB structure
    check_chromadb_structure()
    
    # Check uploads structure
    check_uploads_structure()
    
    # Explain ChromaDB naming
    explain_chromadb_naming()
    
    # Check collection mapping
    check_collection_mapping()
    
    print("\n" + "="*60)
    print("✅ Analysis Complete")
    print("="*60)
    print("\n📝 Summary:")
    print("- ChromaDB uses UUID folders for storage (not collection names)")
    print("- Collection names are mapped to UUIDs in chroma.sqlite3")
    print("- Uploaded files are organized by user/date with hashed filenames")
    print("- Original filenames and metadata are stored in ChromaDB")