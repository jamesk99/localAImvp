# Local RAG System - Component Documentation

## Overview

This is a **local RAG (Retrieval-Augmented Generation)** system that runs entirely on your machine. It ingests documents, creates embeddings, stores them in a vector database, and allows you to query your documents using natural language with an LLM.

---

## Core Components

### 📁 **src/config.py**
**Purpose:** Central configuration file for all system parameters.

**What it does:**
- Defines paths for data directories (`data/raw`, `data/vectordb`, `data/tracking.db`)
- Sets Ollama connection parameters (`http://localhost:11434`)
- Configures RAG parameters (chunk size: 512, overlap: 50, top-k: 5)
- Specifies models (embedding: `nomic-embed-text`, LLM: `llama3.1:8b`)
- Creates necessary directories on import

**When it runs:** Automatically imported by other scripts - never run directly.

**Key settings:**
```python
CHUNK_SIZE = 512          # Tokens per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks
TOP_K = 5                 # Number of chunks to retrieve
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
```

---

### 📄 **src/document_tracker.py**
**Purpose:** SQLite-based tracking system to prevent duplicate ingestion.

**What it does:**
- Maintains a database (`data/tracking.db`) of ingested documents
- Calculates SHA-256 hash of each file to detect changes
- Stores metadata: file path, hash, size, timestamp, chunk count, status
- Provides methods to check if a document is already ingested
- Tracks statistics (total documents, total chunks, ingestion dates)
- Includes database integrity verification

**When it runs:** Automatically used by `ingest.py` and `db_manager.py` - never run directly.

**Key features:**
- Hash-based change detection (re-ingests if file modified)
- Context managers for safe database operations
- Absolute path resolution for portability
- Indexes on `file_hash` and `ingested_at` for fast queries

**Database schema:**
```sql
documents (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    num_chunks INTEGER,
    status TEXT DEFAULT 'completed'
)
```

---

### 📥 **src/ingest.py**
**Purpose:** Main ingestion pipeline - loads documents and creates embeddings.

**What it does:**
1. Loads documents from `data/raw/` (PDF, TXT, MD)
2. Checks `DocumentTracker` to skip already-ingested files
3. Connects to Ollama for embedding generation
4. Initializes ChromaDB vector store
5. Splits documents into chunks (512 tokens, 50 overlap)
6. Generates embeddings for each chunk
7. Stores embeddings in ChromaDB
8. Updates tracking database with accurate chunk counts

**When to run:** 
```bash
python src/ingest.py
```
Run this whenever you add new documents to `data/raw/`

**Output:**
- Shows which files are new vs. already ingested
- Displays progress bars for parsing and embedding
- Reports chunk counts per document
- Updates tracking database

**Error handling:**
- Checks Ollama connection before starting
- Skips empty files
- Handles PDF parsing errors gracefully

---

### 🔍 **src/query.py**
**Purpose:** Interactive query interface for RAG system.

**What it does:**
1. Loads the vector store from ChromaDB
2. Configures embedding model and LLM
3. Creates a query engine with retriever and similarity filtering
4. Provides interactive or single-query modes
5. Retrieves relevant chunks based on semantic similarity
6. Generates answers using the LLM with retrieved context
7. Shows sources with relevance scores

**When to run:**
```bash
# Interactive mode
python src/query.py

# Single query mode
python src/query.py "What is the AI Risk Management Framework?"
```

**Features:**
- Top-K retrieval (default: 5 chunks)
- Similarity threshold filtering (cutoff: 0.3)
- Source citations with relevance scores
- Chunk preview for verification

**Query flow:**
```
User Question → Embed Query → Vector Search → Retrieve Top-K Chunks 
→ Filter by Similarity → Send to LLM with Context → Generate Answer
```

---

### 🛠️ **src/db_manager.py**
**Purpose:** Database inspection and management utility.

**What it does:**
- View database statistics (total docs, chunks, dates)
- List all ingested documents with details
- Check for new documents in `data/raw/`
- Verify database integrity (SQLite PRAGMA check)
- Remove documents from tracking (doesn't delete files/embeddings)

**When to run:**
```bash
# View everything
python src/db_manager.py --all

# Just statistics
python src/db_manager.py --stats

# List documents
python src/db_manager.py --list

# Check for new files
python src/db_manager.py --check

# Verify integrity
python src/db_manager.py --verify

# Remove from tracking
python src/db_manager.py --remove filename.pdf
```

**Use cases:**
- Debugging ingestion issues
- Monitoring system status
- Checking what's been processed
- Verifying database health

---

## Data Components

### 📂 **data/raw/**
**Purpose:** Source document directory.

**What goes here:**
- PDF files (`.pdf`)
- Text files (`.txt`)
- Markdown files (`.md`)

**How it works:**
- Drop documents here
- Run `python src/ingest.py`
- System automatically processes new/modified files
- Original files are never modified or deleted

---

### 🗄️ **data/vectordb/**
**Purpose:** ChromaDB persistent storage for embeddings.

**What's stored:**
- Vector embeddings (768 dimensions for nomic-embed-text)
- Chunk text content
- Metadata (filename, file_path, file_type)
- Collection: `phase0_docs`

**Technical details:**
- Uses ChromaDB's persistent client
- Stores ~273 chunks for your 3 PDFs
- Each chunk has embedding + metadata
- Supports semantic similarity search

**Size:** Grows with number of documents (~1-2MB per 100 pages)

---

### 🗃️ **data/tracking.db**
**Purpose:** SQLite database for ingestion tracking.

**What's stored:**
- Document metadata (path, hash, size)
- Ingestion timestamps
- Chunk counts per document
- Processing status

**Why separate from ChromaDB:**
- Fast duplicate detection
- Efficient change tracking
- Simple SQL queries for stats
- Lightweight metadata storage

**Size:** ~20KB (minimal overhead)

---

## Supporting Files

### 📋 **requirements.txt**
**Purpose:** Python dependencies.

**Key packages:**
- `llama-index-core` - RAG orchestration framework
- `llama-index-embeddings-ollama` - Ollama embedding integration
- `llama-index-llms-ollama` - Ollama LLM integration
- `llama-index-vector-stores-chroma` - ChromaDB integration
- `chromadb` - Vector database
- `PyPDF2` - PDF parsing

**Installation:**
```bash
pip install -r requirements.txt
```

---

### 🚫 **.gitignore**
**Purpose:** Protect data files from version control.

**What's ignored:**
- `data/tracking.db` - Personal tracking database
- `data/vectordb/` - Large vector embeddings
- `data/raw/*.pdf|txt|md` - Your documents
- `logs/` - Log files
- `.venv/` - Virtual environment
- `__pycache__/` - Python cache

**Why:** Keeps your documents and embeddings private, reduces repo size.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERACTION                      │
│  python src/ingest.py  |  python src/query.py          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   PYTHON SCRIPTS (src/)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  ingest.py   │  │  query.py    │  │ db_manager.py│ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         ↓                  ↓                  ↓          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         document_tracker.py (SQLite)             │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   EXTERNAL SERVICES                      │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │   Ollama     │  │   ChromaDB   │                    │
│  │ (localhost)  │  │ (data/vectordb)                   │
│  │              │  │              │                    │
│  │ - Embeddings │  │ - Vectors    │                    │
│  │ - LLM        │  │ - Metadata   │                    │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    DATA STORAGE                          │
│  data/raw/          data/vectordb/      data/tracking.db│
│  (Documents)        (Embeddings)        (Metadata)      │
└─────────────────────────────────────────────────────────┘
```

---

## Workflow

### **Ingestion Workflow:**
1. User adds documents to `data/raw/`
2. Run `python src/ingest.py`
3. `document_tracker.py` checks if files are new/modified
4. New files are loaded and parsed
5. Text is split into 512-token chunks with 50-token overlap
6. Ollama generates embeddings for each chunk
7. ChromaDB stores embeddings + metadata
8. `tracking.db` is updated with file hash and chunk count

### **Query Workflow:**
1. User runs `python src/query.py`
2. User asks a question
3. Question is embedded using Ollama
4. ChromaDB performs vector similarity search
5. Top-5 most relevant chunks are retrieved
6. Chunks are filtered by similarity threshold (>0.3)
7. LLM generates answer using retrieved context
8. Answer and sources are displayed

---

## Key Design Decisions

### **Why SQLite + ChromaDB (two databases)?**
- **SQLite (tracking.db):** Fast metadata queries, duplicate detection, change tracking
- **ChromaDB (vectordb/):** Optimized for vector similarity search, stores embeddings

They complement each other - SQLite for metadata, ChromaDB for semantic search.

### **Why hash-based tracking?**
- Detects file modifications (even if filename unchanged)
- Prevents duplicate ingestion
- Enables incremental updates
- SHA-256 is fast and collision-resistant

### **Why 512-token chunks with 50-token overlap?**
- **512 tokens:** Balance between context and precision
- **50-token overlap:** Prevents information loss at chunk boundaries
- Adjustable in `config.py` for experimentation

### **Why local (not cloud)?**
- **Privacy:** Your documents never leave your machine
- **Cost:** No API fees
- **Control:** Full control over models and data
- **Speed:** No network latency (once models downloaded)

---

## Performance Metrics

**Current System (3 PDFs, 273 chunks):**
- Ingestion: ~15 seconds (parsing + embedding)
- Query latency: 2-5 seconds
- Embedding speed: ~20 chunks/second
- Database size: ~20KB (tracking) + ~2MB (vectors)

**Scalability:**
- Can handle 1000+ documents
- Linear scaling with document count
- Bottleneck: Ollama embedding generation
- Optimization: Batch processing, GPU acceleration

---

## Troubleshooting

### **"Cannot connect to Ollama"**
```bash
# Start Ollama
ollama serve

# Verify models are installed
ollama list
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

### **"No documents found"**
- Check files are in `data/raw/`
- Supported formats: `.pdf`, `.txt`, `.md`
- Verify file permissions

### **"Chunk count is 0"**
- This was a bug - now fixed
- Re-run ingestion after deleting databases
- Check `db_manager.py --verify` for integrity

### **Reset everything:**
```bash
Remove-Item data\tracking.db -Force
Remove-Item data\vectordb\* -Recurse -Force
python src/ingest.py
```

---

## Future Enhancements

See `FUTURE.md` for production roadmap including:
- PostgreSQL + pgvector migration
- 70B LLM support
- Semantic chunking
- Hybrid search (BM25 + vector)
- Reranking with ColBERT
- Agentic RAG
- Graph RAG
- Multimodal support

---

## Quick Reference

```bash
# Initial setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Start Ollama
ollama serve

# Ingest documents
python src/ingest.py

# Query system
python src/query.py

# Check status
python src/db_manager.py --all

# Reset system
Remove-Item data\tracking.db -Force
Remove-Item data\vectordb\* -Recurse -Force
```

---

**System Status:** ✅ Fully operational with 3 documents, 273 chunks indexed.
