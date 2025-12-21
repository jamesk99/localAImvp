# Phase 0 RAG Project

## Yes, info is stored persistently! ✅

After you run `ingest.py`, everything is saved to disk in `data/vectordb/`. This means:

- You only need to run `ingest.py` **once** (or when you add new documents)
- `query.py` loads from the saved database - no re-embedding needed
- The Chroma database persists across sessions
- If you restart your computer, the vectors are still there

---

## How Phase 0 Actually Works

### **The Storage Layer**

```markdown
data/
├── raw/               # Your original documents (input)
│   ├── doc1.pdf
│   ├── doc2.txt
│   └── notes.md
└── vectordb/          # Chroma database (persistent storage)
    └── chroma.sqlite3 # SQLite file with all vectors + metadata
```

```txt
data/
├── tracking.db          # SQLite tracking database (persistent)
│   └── documents table  # file_path (PK), file_hash, size, timestamp, chunks
├── vectordb/            # ChromaDB vector store (persistent)
│   └── [chroma files]   # Embeddings and metadata
└── raw/                 # Your source documents
    └── [your files]
```

```sql
CREATE TABLE documents (
    file_path TEXT PRIMARY KEY,      -- Unique identifier
    file_hash TEXT NOT NULL,         -- SHA-256 for change detection
    file_size INTEGER NOT NULL,      -- Quick size comparison
    ingested_at TIMESTAMP NOT NULL,  -- Audit trail
    num_chunks INTEGER,              -- Actual chunk count
    status TEXT DEFAULT 'completed'  -- Future: track failed ingests
)

-- Indexes for fast lookups
CREATE INDEX idx_file_hash ON documents(file_hash);
CREATE INDEX idx_ingested_at ON documents(ingested_at);
```

Integration with Vector Store:

```txt
Tracking DB (SQLite)          Vector DB (ChromaDB)
─────────────────────        ─────────────────────
file_path (PK)        ──┐    Collection: phase0_docs
file_hash             ──┼──> Embeddings + metadata
num_chunks            ──┘    Persistent to disk
ingested_at
```

When you run `ingest.py`:

1. Reads files from `data/raw/`
2. Splits into 512-token chunks
3. Sends each chunk to `nomic-embed-text` via Ollama → gets 768-dimensional vector
4. Stores vectors + original text + metadata in ChromaDB (saves to disk)

### **The Query Flow**

When you run `query.py`:

1. **Loads existing database** (no re-processing!)
2. You ask: "What are the key findings?"
3. Your question → `nomic-embed-text` → query vector (768d)
4. ChromaDB finds top 5 most similar chunk vectors (cosine similarity)
5. Retrieves the original text of those 5 chunks
6. Builds this prompt:

```markdown
Context:
    [chunk 1 text]
    [chunk 2 text]
    [chunk 3 text]
    [chunk 4 text]
    [chunk 5 text]
   
Question: What are the key findings?
   
Answer based only on the context above.
```

7. Sends to `llama3.1:8b` via Ollama
8. Returns answer

### **Why This Works**

**Semantic Search**: Vectors capture *meaning*, not just keywords

- "How do I reset my password?"
- vs "Password recovery steps"
- → These have similar vectors even with different words!

**Context Window Efficiency**: Instead of sending ALL your documents to the LLM (would exceed context limit), you only send the 5 most relevant chunks (~2500 tokens)

**No Re-work**: Embedding is expensive (time-wise). By storing vectors, you pay that cost once during ingestion, not on every query.

### **The Models' Roles**

**nomic-embed-text** (Embedding Model):

- Job: Convert text → numbers
- Input: "The sky is blue"
- Output: [0.23, -0.45, 0.67, ... ] (768 numbers)
- Used in: Both ingestion AND query time
- Why: Same model ensures query vector matches document vectors

**llama3.1:8b** (Language Model):

- Job: Read context + question → generate answer
- Only used: Query time
- Why: Good at reasoning and synthesis, but needs relevant context

---

## Yes! PowerShell can generate requirements.txt

```powershell
# Create requirements.txt
pipreqs . --force (to update and rewrite)
pipreqs . (to create new)
# above don't really work sometimes due to encoding issues and requires the below command instead:
pipreqs . --force --ignore .venv --encoding utf-8

```

---

## Key Concepts Summary

| Concept | What It Does | When It Runs |
|---------|-------------|--------------|
| **Embedding** | Text → Vector (numbers) | Ingestion + Query |
| **Chunking** | Big doc → Small pieces | Ingestion only |
| **Vector DB** | Stores vectors + text | Ingestion (write), Query (read) |
| **Retrieval** | Find similar chunks | Query only |
| **LLM** | Generate answer from context | Query only |

**The Magic**: Your question's vector is compared to all chunk vectors. Math finds which chunks are "closest" in meaning. Those chunks give the LLM the info it needs to answer accurately.

# 1. Install dependencies
pip install -r requirements.txt

# 2. Add documents to data/raw/
# (Put your PDFs, TXT, MD files there)

# 3. Run ingestion
python src/ingest.py

# 4. Run queries (interactive mode)
python src/query.py

# OR single query mode
python src/query.py "What is the main topic of these documents?"

# Your Instructions Now Work
bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Documents already in data/raw/ ✅
# (Your PDFs are already there)

# 3. Run ingestion
python src/ingest.py

# 4. Run queries (interactive mode)
python src/query.py
The app will automatically skip already-ingested documents on subsequent runs, so you can add new files to 
data/raw/
 and re-run python src/ingest.py to only process the new ones.