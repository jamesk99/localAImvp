# PMP Project Requirements - Part 2: Technical Requirements

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERACTION                     │
│  CLI Interface (query.py, ingest.py, db_manager.py)     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│               APPLICATION LAYER (Python)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  ingest.py   │  │  query.py    │  │ db_manager.py│   │
│  │  - Parsing   │  │  - Retrieval │  │ - Admin      │   │
│  │  - Chunking  │  │  - Generation│  │ - Monitoring │   │
│  │  - Embedding │  │  - Formatting│  │ - Tracking   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │    document_tracker.py (Tracking Logic)          │   │
│  │    config.py (Configuration Management)          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATION LAYER                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │         LlamaIndex Framework                     │   │
│  │  - VectorStoreIndex                              │   │
│  │  - SentenceSplitter                              │   │
│  │  - RetrieverQueryEngine                          │   │
│  │  - SimilarityPostprocessor                       │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                EXTERNAL SERVICES                        │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │   Ollama     │  │   ChromaDB   │                     │
│  │ (localhost)  │  │ (Persistent) │                     │
│  │              │  │              │                     │
│  │ - LLM API    │  │ - Vector DB  │                     │
│  │ - Embeddings │  │ - Similarity │                     │
│  │ - Generation │  │ - Metadata   │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  DATA STORAGE LAYER                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  data/raw/   │  │data/vectordb/│  │tracking.db   │   │
│  │  Documents   │  │  Embeddings  │  │  Metadata    │   │
│  │  (PDF/TXT/MD)│  │  (ChromaDB)  │  │  (SQLite)    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Component Interaction Flow

**Ingestion Flow:**

```
User → ingest.py → DocumentTracker (check duplicates) → PyPDF2 (parse)
→ SentenceSplitter (chunk) → OllamaEmbedding (embed)
→ ChromaDB (store) → DocumentTracker (update) → User (report)
```

**Query Flow:**

```
User → query.py → OllamaEmbedding (embed query) → ChromaDB (retrieve)
→ SimilarityPostprocessor (filter) → Ollama LLM (generate) → User (response)
```

---

## 2. Technical Stack Requirements

### 2.1 Core Technologies

| Component   | Technology | Version | Purpose                 |
| ----------- | ---------- | ------- | ----------------------- |
| Language    | Python     | 3.10+   | Application development |
| Framework   | LlamaIndex | 0.10.0+ | RAG orchestration       |
| Vector DB   | ChromaDB   | 0.4.0+  | Embedding storage       |
| Metadata DB | SQLite     | 3.x     | Document tracking       |
| LLM Server  | Ollama     | Latest  | Model hosting           |
| PDF Parser  | PyPDF2     | 3.0.0+  | Document parsing        |

### 2.2 Python Package Requirements

**Core LlamaIndex Packages:**

- `llama-index-core>=0.10.0` - Core framework functionality
- `llama-index-embeddings-ollama>=0.1.0` - Ollama embedding integration
- `llama-index-llms-ollama>=0.1.0` - Ollama LLM integration
- `llama-index-vector-stores-chroma>=0.1.0` - ChromaDB integration

**Supporting Libraries:**

- `chromadb>=0.4.0` - Vector database
- `PyPDF2>=3.0.0` - PDF processing

### 2.3 Model Requirements

**Embedding Model:**

- **Name:** nomic-embed-text
- **Parameters:** 137M
- **Dimensions:** 768
- **Size:** ~274MB
- **Purpose:** Convert text to semantic vectors

**Primary LLM:**

- **Name:** llama3:latest (llama3.1:8b)
- **Parameters:** 8B
- **Context Window:** 8K tokens
- **Size:** ~4.7GB
- **Purpose:** Response generation
- **Model Configurability:** Default primary LLM is `llama3:latest` unless overridden at runtime.
- Primary and fallback models can be changed via environment variables (see Section 5.2), for example setting `LLM_MODEL=gemma3:4b`.

**Fallback LLM (Optional):**

- **Name:** deepseek-r1:latest
- **Purpose:** Backup if primary fails
- **Note:** Requires more RAM (~20GB+)

---

## 3. Infrastructure Requirements

### 3.1 Hardware Requirements

**Minimum Specifications:**

- **CPU:** Modern multi-core processor (4+ cores recommended)
- **RAM:** 16GB minimum (8GB for OS/apps, 8GB for models)
- **Storage:** 100GB free space
  - 50GB for Ollama models
  - 10GB for vector database (scales with document count)
  - 40GB for workspace and dependencies
- **OS:** Windows 10/11 (primary target)

**Recommended Specifications:**

- **CPU:** 8+ cores for faster processing
- **RAM:** 32GB for comfortable operation
- **Storage:** 200GB+ SSD for better I/O performance

### 3.2 Software Requirements

**Operating System:**

- Windows 10/11 (64-bit)
- PowerShell 5.1 or later

**Runtime Environment:**

- Python 3.10 or 3.11
- Pip package manager
- Python venv module

**External Services:**

- Ollama server running on localhost:11434
- No internet connectivity required after initial setup

### 3.3 Network Requirements

**Initial Setup:**

- Internet connection for:
  - Python package downloads
  - Ollama model downloads (~5-25GB total)
  - Ollama software installation

**Operation:**

- No internet required (fully local)
- Ollama communicates via localhost only

---

## 4. Data Architecture

### 4.1 Data Storage Structure

```
data/
├── raw/                    # Source documents (input)
│   ├── *.pdf              # PDF files
│   ├── *.txt              # Text files
│   └── *.md               # Markdown files
├── vectordb/              # ChromaDB persistent storage
│   └── chroma.sqlite3     # SQLite file with vectors + metadata
└── tracking.db            # Document tracking database
```

### 4.2 Document Tracking Database Schema

**Table: documents**

```sql
CREATE TABLE documents (
    file_path TEXT PRIMARY KEY,      -- Unique file identifier
    file_hash TEXT NOT NULL,         -- SHA-256 for change detection
    file_size INTEGER NOT NULL,      -- File size in bytes
    ingested_at TIMESTAMP NOT NULL,  -- Ingestion timestamp
    num_chunks INTEGER,              -- Number of chunks created
    status TEXT DEFAULT 'completed'  -- Processing status
);

-- Performance indexes
CREATE INDEX idx_file_hash ON documents(file_hash);
CREATE INDEX idx_ingested_at ON documents(ingested_at);
```

**Key Features:**

- Primary key on file_path prevents duplicates
- SHA-256 hash enables change detection
- Indexes optimize lookup performance
- Timestamp tracking for audit trail

### 4.3 Vector Database Schema

**ChromaDB Collection: phase0_docs**

**Stored Data Per Chunk:**

- **ID:** Auto-generated unique identifier
- **Embedding:** 768-dimensional float vector
- **Document:** Original text chunk content
- **Metadata:**
  - `filename`: Source file name
  - `file_type`: File extension (.pdf, .txt, .md)
  - `file_path`: Absolute path to source file

**Index Type:** Default ChromaDB indexing (HNSW-like)

### 4.4 Data Flow Diagrams

**Ingestion Data Flow:**

```
PDF/TXT/MD File
    ↓
PyPDF2 Extraction → Raw Text
    ↓
SentenceSplitter → Chunks (1024 tokens, 128 overlap)
    ↓
OllamaEmbedding → Vectors (768-dim)
    ↓
ChromaDB Storage → Persistent vectors + metadata
    ↓
DocumentTracker → tracking.db update (hash, chunks, timestamp)
```

**Query Data Flow:**

```
User Question (string)
    ↓
OllamaEmbedding → Query Vector (768-dim)
    ↓
ChromaDB Similarity Search → Top-K chunks (K=5)
    ↓
SimilarityPostprocessor → Filter by threshold (>0.5)
    ↓
Context Assembly → Prompt with retrieved chunks
    ↓
Ollama LLM → Generated Response
    ↓
Response Formatter → Answer + Sources + Scores
```

---

## 5. Configuration Management

### 5.1 Configuration File (config.py)

**Ollama Configuration:**

```python
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server endpoint
LLM_MODEL = "llama3:latest"                 # Primary LLM model
LLM_FALLBACK = "deepseek-r1:latest"         # Fallback model
EMBED_MODEL = "nomic-embed-text"            # Embedding model
```

**RAG Configuration:**

```python
CHUNK_SIZE = 1024          # Tokens per chunk (increased from 512)
CHUNK_OVERLAP = 128        # Token overlap between chunks (increased from 50)
TOP_K = 5                  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.5 # Minimum similarity score (increased from 0.3)
```

**Path Configuration:**

```python
DATA_DIR = "data/"                    # Base data directory
RAW_DOCS_DIR = "data/raw"            # Source documents
VECTOR_DB_DIR = "data/vectordb"      # Vector storage
TRACKING_DB_PATH = "data/tracking.db" # Tracking database
COLLECTION_NAME = "phase0_docs"       # ChromaDB collection
```

### 5.2 Environment Variables

**Currently Used:** Configuration values in `config.py` can be overridden via environment variables at runtime.  
**Ollama-Related Environment Variables:**

- `OLLAMA_BASE_URL` – Overrides Ollama server URL (default: `http://localhost:11434`).
- `LLM_MODEL` – Overrides primary LLM model name (default: `llama3:latest`, e.g., `gemma3:4b`).
- `LLM_FALLBACK` – Overrides fallback LLM model name (default: `deepseek-r1:latest`).
- `EMBED_MODEL` – Overrides embedding model name (default: `nomic-embed-text`).

**Example (PowerShell):**

- `$env:LLM_MODEL = "gemma3:4b"; python .\query.py`
  - Uses `gemma3:4b` as primary LLM for that session; omitting `LLM_MODEL` falls back to `llama3:latest`.

### 5.3 Configuration Changeability

**Low-Risk Changes (No Re-ingestion Required):**

- `TOP_K` - Affects retrieval only
- `SIMILARITY_THRESHOLD` - Affects filtering only
- `LLM_MODEL` - Can switch models freely

**High-Risk Changes (Requires Re-ingestion):**

- `CHUNK_SIZE` - Changes chunk boundaries
- `CHUNK_OVERLAP` - Changes chunk content
- `EMBED_MODEL` - Changes vector dimensions/semantics
- `COLLECTION_NAME` - Creates new vector store

---

## 6. Performance Requirements

### 6.1 Speed Requirements

| Operation            | Target        | Acceptable   | Maximum      |
| -------------------- | ------------- | ------------ | ------------ |
| Document Ingestion   | 5 docs/min    | 2 docs/min   | 10 min total |
| Query Latency (p50)  | 2 seconds     | 5 seconds    | 30 seconds   |
| Query Latency (p95)  | 5 seconds     | 15 seconds   | 60 seconds   |
| Embedding Generation | 20 chunks/sec | 5 chunks/sec | N/A          |
| Vector Retrieval     | <100ms        | <500ms       | 1 second     |

### 6.2 Throughput Requirements

- **Concurrent Queries:** 1 (single user)
- **Batch Ingestion:** Not required
- **Continuous Operation:** Not required (on-demand execution)

### 6.3 Scalability Requirements

**Phase 0 Targets:**

- Documents: 20-50 files
- Total Chunks: <1000
- Vector DB Size: <100MB
- Query Volume: <100 queries total

**Not Required for Phase 0:**

- Horizontal scaling
- Load balancing
- Caching layers
- Connection pooling

---

## 7. Security Requirements

### 7.1 Data Security

**Privacy Requirements:**

- All document processing occurs locally
- No data transmission to external services
- No telemetry or analytics collection
- Documents never leave local filesystem

**Access Control:**

- File system permissions only (OS-level)
- No authentication system required
- Single-user environment assumed

### 7.2 Code Security

**Dependencies:**

- Use pinned versions in requirements.txt
- All packages from PyPI official repository
- No custom or third-party package sources

**Secrets Management:**

- No API keys or credentials required
- No sensitive configuration

### 7.3 Data Integrity

**Hash-Based Verification:**

- SHA-256 hashing for file change detection
- Database integrity checks (SQLite PRAGMA)

**Backup Requirements:**

- User responsible for document backups
- No automated backup system

---

## 8. Reliability Requirements

### 8.1 Availability

- **Target Uptime:** Not applicable (on-demand tool)
- **Recovery Time:** Immediate (restart script)
- **Data Persistence:** Required across restarts

### 8.2 Error Handling

**Required Error Handling:**

- Ollama connection failures (clear error message)
- PDF parsing errors (skip file, continue processing)
- Empty document handling (skip with warning)
- Database corruption detection (integrity check command)
- Memory overflow protection (model fallback mechanism)

**Error Recovery:**

- Graceful degradation (fallback to alternative LLM)
- Skip problematic files during ingestion
- Continue processing after non-fatal errors

### 8.3 Data Consistency

**Requirements:**

- tracking.db must match actual ingested documents
- Vector count in ChromaDB must match tracking.db chunk counts
- File hash changes trigger re-ingestion

---

## 9. Compatibility Requirements

### 9.1 Platform Compatibility

**Primary Platform:**

- Windows 10/11 (64-bit)

**Path Handling:**

- Use `os.path.join()` for cross-platform paths
- Support absolute paths with Windows drive letters

### 9.2 Backward Compatibility

**Not Required:**

- Migration from previous versions
- Legacy data format support

**Version Control:**

- Document config.py changes
- Track requirements.txt updates

---

## 10. Technical Constraints

### 10.1 Technology Constraints

- **Python Only:** No compiled languages
- **Local Only:** No cloud service dependencies
- **Open Source:** No commercial software
- **Ollama Required:** No alternative LLM servers supported in Phase 0

### 10.2 Model Constraints

- **Model Size Limit:** <10GB per model (storage constraint)
- **Context Window:** 8K tokens maximum (llama3.1:8b limit)
- **Embedding Dimensions:** 768 (fixed by nomic-embed-text)

### 10.3 Operational Constraints

- **Single Process:** No multi-process architecture
- **Synchronous Operations:** No async ingestion (Phase 0)
- **Sequential Processing:** Documents processed one at a time

---

## 11. Integration Requirements

### 11.1 External Service Integration

**Ollama Integration:**

- **Protocol:** HTTP REST API
- **Endpoint:** http://localhost:11434
- **Timeout:** 180 seconds for LLM, 60 seconds for embeddings
- **Error Handling:** Retry with fallback model

**ChromaDB Integration:**

- **Mode:** Persistent client (disk storage)
- **Collection Management:** Get-or-create pattern
- **Telemetry:** Disabled (anonymized_telemetry=False)

### 11.2 Data Format Requirements

**Input Formats:**

- PDF: Any valid PDF with text layer
- TXT: UTF-8 encoded plain text
- MD: Standard Markdown syntax

**Output Formats:**

- Console text (stdout)
- UTF-8 encoding

---

## 12. Monitoring and Observability

### 12.1 Logging Requirements

**Console Logging:**

- Ingestion progress (files processed, chunks created)
- Query operations (retrieval, generation stages)
- Error messages with context
- Performance metrics (time elapsed)

**Not Required:**

- Persistent log files
- Structured logging (JSON)
- Log aggregation

### 12.2 Metrics Requirements

**Manual Metrics (Phase 0):**

- Documents ingested count
- Total chunks created
- Query response times (measured manually)
- Retrieval accuracy (measured against test set)

**Automated Metrics:**

- Database statistics via db_manager.py
- Chunk counts per document
- Ingestion timestamps

---

**Document Status:** APPROVED  
**Next Document:** PMP Project Requirements - Part 3: Functional Requirements
