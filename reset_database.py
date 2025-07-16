#!/usr/bin/env python3
"""
Standalone script to reset/clear ChromaDB vector database
Run this script to completely clear your RAG system's vector store
"""

import os
import shutil
import chromadb

def reset_chromadb():
    """Reset the ChromaDB vector database completely"""
    print("🔄 Starting ChromaDB reset...")
    
    # Method 1: Try to delete the collection first
    try:
        print("📋 Attempting to delete collection 'rag_app'...")
        chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
        chroma_client.delete_collection(name="rag_app")
        print("✅ Collection 'rag_app' deleted successfully!")
    except Exception as e:
        if "does not exist" in str(e).lower():
            print("ℹ️ Collection 'rag_app' doesn't exist (already empty)")
        else:
            print(f"⚠️ Could not delete collection: {e}")
    
    # Method 2: Remove the entire ChromaDB directory
    try:
        if os.path.exists("./demo-rag-chroma"):
            print("🗑️ Removing ChromaDB directory...")
            shutil.rmtree("./demo-rag-chroma")
            print("✅ ChromaDB directory completely removed!")
        else:
            print("ℹ️ ChromaDB directory doesn't exist")
    except Exception as e:
        print(f"❌ Error removing directory: {e}")
    
    print("🎉 Vector database reset complete!")
    print("📝 You can now upload new documents without any old embeddings.")

def verify_reset():
    """Verify that the database is empty"""
    print("\n🔍 Verifying reset...")
    
    try:
        if not os.path.exists("./demo-rag-chroma"):
            print("✅ ChromaDB directory doesn't exist - database is empty")
            return True
        
        # Try to connect and check collection
        chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
        collections = chroma_client.list_collections()
        
        if not collections:
            print("✅ No collections found - database is empty")
            return True
        else:
            print(f"⚠️ Found {len(collections)} collection(s): {[c.name for c in collections]}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying reset: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🗄️ ChromaDB Vector Database Reset Tool")
    print("=" * 50)
    
    # Ask for confirmation
    response = input("\n⚠️ This will permanently delete ALL documents and embeddings.\nAre you sure? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        reset_chromadb()
        verify_reset()
    else:
        print("❌ Reset cancelled.")
    
    print("\n" + "=" * 50)
