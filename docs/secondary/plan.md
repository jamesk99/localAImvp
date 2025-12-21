# Phase 0: Local RAG Dry Run - Complete Roadmap

## Objectives
- Validate the entire tech stack on your laptop
- Build a working end-to-end prototype
- Identify any gotchas before the real project
- Produce deliverables that prove feasibility

## Laptop-Appropriate Scope

**Model Size:** Stick to 7-8B models (will run on most modern laptops with 16GB+ RAM)

**Dataset:** Small but representative - 20-50 documents (PDFs, text files, web pages)

**Timeline:** 2-3 days of focused work

---

## Phase 0 Roadmap

### **Day 1: Environment Setup & Basic Pipeline (4-6 hours)**

**Hour 1-2: Installation & Setup**
- [ ] Install Ollama
- [ ] Pull `llama3.1:8b` model
- [ ] Pull `nomic-embed-text` model
- [ ] Create Python virtual environment
- [ ] Install core dependencies:
  ```bash
  pip install llama-index chromadb sentence-transformers pypdf python-dotenv
  ```
- [ ] Verify Ollama is working: `ollama run llama3.1:8b "Hello, world"`

**Hour 3-4: Document Ingestion Pipeline**
- [ ] Create project structure:
  ```
  phase0-rag/
  ├── data/raw/          # Your test documents
  ├── data/vectordb/     # Chroma storage
  ├── src/
  │   ├── ingest.py      # Document loading & indexing
  │   ├── query.py       # RAG query interface
  │   └── config.py      # Configuration
  └── notebooks/         # Testing & evaluation
  ```
- [ ] Write `ingest.py`: Load 20-50 test documents (PDFs or text)
- [ ] Chunk documents (512 tokens, 50 token overlap)
- [ ] Generate embeddings via `nomic-embed-text`
- [ ] Store in Chroma vector DB

**Hour 5-6: Basic Query Pipeline**
- [ ] Write `query.py`: 
  - Accept user question
  - Retrieve top 5 relevant chunks
  - Build prompt with context
  - Query Ollama LLM
  - Return response
- [ ] Test with 5-10 questions about your documents
- [ ] Verify responses are using retrieved context

**Deliverable:** Working prototype that ingests docs and answers questions

---

### **Day 2: Quality & Evaluation (3-4 hours)**

**Hour 1-2: Retrieval Quality Testing**
- [ ] Create test set: 10 questions with known answers in your docs
- [ ] Measure retrieval metrics:
  - Are correct chunks in top 5 results?
  - Calculate manual relevance scores
- [ ] Log retrieved chunks for each query
- [ ] Identify failure cases

**Hour 2-3: Response Quality Testing**
- [ ] Test same 10 questions end-to-end
- [ ] Evaluate responses:
  - Factual accuracy (vs source docs)
  - Hallucination detection
  - Relevance to question
- [ ] Document failure modes

**Hour 3-4: Basic Improvements**
- [ ] Experiment with:
  - Different chunk sizes (256, 512, 1024 tokens)
  - Different top-K values (3, 5, 10 chunks)
  - Different prompts for the LLM
- [ ] Pick best configuration based on your test set

**Deliverable:** Evaluation notebook with metrics and failure analysis

---

### **Day 3: Polish & Documentation (2-3 hours)**

**Hour 1: Simple UI (Optional but Recommended)**
- [ ] Create basic Streamlit or Gradio interface:
  - Upload documents
  - Ask questions
  - See retrieved chunks + response
- [ ] OR: Build simple CLI with command history

**Hour 2: Documentation**
- [ ] Write README with:
  - Setup instructions
  - How to run ingestion
  - How to query
  - Sample questions
  - Known limitations
- [ ] Document your findings:
  - What worked well?
  - What didn't work?
  - Performance observations
  - Memory usage

**Hour 3: Proof of Success**
- [ ] Record demo video (2-3 min):
  - Show document ingestion
  - Ask 3-5 questions
  - Show responses with retrieved context
- [ ] Screenshot key results
- [ ] List "lessons learned" for Phase 1

**Deliverable:** Demo-ready prototype + documentation

---

## Success Criteria for Phase 0

✅ **Must Have:**
1. Documents successfully ingested and indexed
2. Questions return relevant context from vector DB
3. LLM generates coherent responses using retrieved context
4. Can answer at least 7/10 test questions correctly
5. End-to-end pipeline takes <30 seconds per query

✅ **Nice to Have:**
1. Simple UI for easy demonstration
2. Comparison of 2-3 different configurations
3. Clear documentation of limitations

❌ **Red Flags (needs troubleshooting before Phase 1):**
- Retrieval returns irrelevant chunks consistently
- LLM hallucinates despite good context
- Memory issues or crashes
- Setup takes >2 hours (dependencies hell)

---

## Minimal Code Structure to Start

**config.py:**
```python
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
```

**Key Decision Points to Test:**
1. Does `nomic-embed-text` perform well for your domain?
2. Is 8B model sufficient or do you need larger?
3. Is basic RAG enough or do you need reranking?
4. What's the optimal chunk size for your documents?

---

## Output Artifacts

After Phase 0, you should have:
1. ✅ Working codebase (push to GitHub)
2. ✅ Test dataset with ground truth Q&A
3. ✅ Evaluation results (metrics + examples)
4. ✅ Demo video or screenshots
5. ✅ Lessons learned document
6. ✅ Go/No-Go decision for Phase 1

---

This dry run will take **9-13 total hours** of focused work. Once complete, you'll know exactly what works, what doesn't, and what to adjust before building on the AMD Ryzen AI system.

Should I drill into any specific section or help you prep your test dataset?