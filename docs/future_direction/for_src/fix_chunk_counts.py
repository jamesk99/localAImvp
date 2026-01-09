"""
One-time script to fix chunk counts in tracking database.
Run this to update chunk counts for already-ingested documents.
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from config import TRACKING_DB_PATH, VECTOR_DB_DIR, COLLECTION_NAME
from document_tracker import DocumentTracker
import chromadb
from chromadb.config import Settings

def fix_chunk_counts():
    """Update chunk counts in tracking database from ChromaDB."""
    print("=" * 60)
    print("🔧 FIXING CHUNK COUNTS IN TRACKING DATABASE")
    print("=" * 60)
    
    # Connect to ChromaDB
    print("\n📊 Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
        total_chunks = chroma_collection.count()
        print(f"   Total chunks in ChromaDB: {total_chunks}")
    except Exception as e:
        print(f"❌ Error: Could not access ChromaDB collection: {e}")
        return
    
    # Get all items from ChromaDB
    print("\n📥 Retrieving chunk metadata...")
    all_items = chroma_collection.get(include=['metadatas'])
    
    if not all_items or not all_items['metadatas']:
        print("❌ No chunks found in ChromaDB")
        return
    
    # Count chunks per document
    doc_chunk_counts = {}
    for metadata in all_items['metadatas']:
        source_file = metadata.get('file_path')
        if source_file:
            doc_chunk_counts[source_file] = doc_chunk_counts.get(source_file, 0) + 1
    
    print(f"   Found {len(doc_chunk_counts)} unique documents")
    
    # Update tracking database
    print("\n💾 Updating tracking database...")
    tracker = DocumentTracker(TRACKING_DB_PATH)
    
    updated_count = 0
    for file_path_str, chunk_count in doc_chunk_counts.items():
        file_path = Path(file_path_str)
        
        # Check if document exists in tracking DB
        if tracker.is_document_ingested(file_path):
            # Update the chunk count
            tracker.mark_document_ingested(file_path, num_chunks=chunk_count)
            print(f"   ✓ {file_path.name}: {chunk_count} chunks")
            updated_count += 1
        else:
            print(f"   ⚠️  {file_path.name}: Not in tracking DB (skipping)")
    
    print("\n" + "=" * 60)
    print(f"✅ UPDATED {updated_count} DOCUMENTS")
    print("=" * 60)
    
    # Show updated stats
    stats = tracker.get_ingestion_stats()
    print(f"\n📊 Updated Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Total chunks:    {stats['total_chunks']}")
    print("=" * 60)


if __name__ == "__main__":
    fix_chunk_counts()
