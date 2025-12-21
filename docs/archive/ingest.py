# ingest.py
import os
import sys
from pathlib import Path
from typing import List
import chromadb
from chromadb.config import Settings
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings as LlamaSettings
import PyPDF2

# Import config
sys.path.append(os.path.dirname(__file__))
from config import (
    RAW_DOCS_DIR, VECTOR_DB_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, OLLAMA_BASE_URL
)


def load_documents() -> List[Document]:
    """Load all documents from the raw data directory."""
    documents = []
    
    raw_path = Path(RAW_DOCS_DIR)
    print(f"\n📂 Loading documents from: {raw_path}")
    
    # Get all files
    file_patterns = ["*.txt", "*.pdf", "*.md"]
    files = []
    for pattern in file_patterns:
        files.extend(raw_path.glob(pattern))
    
    if not files:
        print(f"⚠️  No documents found in {RAW_DOCS_DIR}")
        print(f"   Please add .txt, .pdf, or .md files to this directory.")
        return documents
    
    print(f"📄 Found {len(files)} files")
    
    for file_path in files:
        try:
            print(f"   Loading: {file_path.name}")
            
            if file_path.suffix.lower() == ".pdf":
                # Handle PDF files
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            else:
                # Handle text files (txt, md)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            if text.strip():
                doc = Document(
                    text=text,
                    metadata={
                        "filename": file_path.name,
                        "file_type": file_path.suffix,
                        "file_path": str(file_path)
                    }
                )
                documents.append(doc)
            else:
                print(f"   ⚠️  Skipped (empty): {file_path.name}")
                
        except Exception as e:
            print(f"   ❌ Error loading {file_path.name}: {e}")
    
    print(f"\n✅ Successfully loaded {len(documents)} documents")
    return documents


def create_vector_store():
    """Initialize ChromaDB vector store."""
    print("\n🔧 Initializing ChromaDB...")
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create collection
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"   Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    chroma_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print(f"✅ ChromaDB initialized at: {VECTOR_DB_DIR}")
    
    return vector_store


def ingest_documents():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("🚀 STARTING DOCUMENT INGESTION PIPELINE")
    print("=" * 60)
    
    # 1. Load documents
    documents = load_documents()
    if not documents:
        print("\n❌ No documents to ingest. Exiting.")
        return
    
    # 2. Configure embedding model
    print(f"\n🤖 Configuring embedding model: {EMBED_MODEL}")
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    LlamaSettings.embed_model = embed_model
    
    # 3. Create vector store
    vector_store = create_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 4. Configure chunking
    print(f"\n✂️  Configuring text splitter:")
    print(f"   Chunk size: {CHUNK_SIZE} tokens")
    print(f"   Chunk overlap: {CHUNK_OVERLAP} tokens")
    
    text_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # 5. Create index (this chunks, embeds, and stores)
    print(f"\n⚙️  Processing documents...")
    print(f"   - Splitting into chunks")
    print(f"   - Generating embeddings")
    print(f"   - Storing in vector database")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[text_splitter],
        show_progress=True
    )
    
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   Documents processed: {len(documents)}")
    print(f"   Vector store: {VECTOR_DB_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"\n💡 Next step: Run query.py to test retrieval")
    print("=" * 60)


if __name__ == "__main__":
    ingest_documents()