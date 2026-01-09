# PMP Project Requirements - Part 1: Overview & Scope

## Project Information

**Project Name:** Local RAG AI System - Phase 0 (Proof of Concept)  
**Project Code:** E6-PHASE0-DADPROJECT  
**Project Type:** Software Development - AI/ML System  
**Project Manager:** [To Be Assigned]  
**Date Created:** 2025  
**Document Version:** 1.0  

---

## 1. Executive Summary

### 1.1 Project Purpose
This project delivers a locally-hosted Retrieval-Augmented Generation (RAG) system that enables intelligent document querying using large language models (LLMs) without relying on cloud services. The system provides a proof-of-concept for validating the technical stack, identifying implementation challenges, and establishing feasibility for future production deployment.

### 1.2 Business Justification
- **Data Privacy:** Keep sensitive documents entirely on local infrastructure
- **Cost Control:** Eliminate recurring API fees from cloud-based AI services
- **Performance Validation:** Test RAG capabilities on consumer-grade hardware
- **Technology Assessment:** Evaluate LlamaIndex, Ollama, and ChromaDB integration
- **Risk Mitigation:** Identify technical limitations before production investment

### 1.3 Project Success Criteria
- [ ] Successfully ingest and index 20-50 documents
- [ ] Achieve >70% accuracy on test question set (7/10 questions)
- [ ] Query response time <30 seconds per question
- [ ] System runs on laptop hardware (16GB+ RAM)
- [ ] Complete documentation for setup and usage
- [ ] Go/No-Go decision supported by objective metrics

---

## 2. Project Scope

### 2.1 In-Scope

#### Core Functionality
- Document ingestion pipeline for PDF, TXT, and Markdown files
- Semantic embedding generation using local models
- Vector database storage with persistent state
- Natural language query interface
- Context-aware response generation
- Source citation and relevance scoring
- Duplicate detection and incremental updates
- Interactive and single-query modes

#### Technical Components
- Python-based implementation using LlamaIndex framework
- Ollama for local LLM and embedding model hosting
- ChromaDB for vector storage
- SQLite for document tracking and metadata
- PyPDF2 for PDF document parsing

#### Deliverables
- Working source code with modular architecture
- Configuration management system
- Document tracking database
- Vector embedding database
- Command-line query interface
- Comprehensive documentation (README, technical specs, usage guides)
- Test dataset with ground truth Q&A pairs
- Evaluation results and metrics
- Lessons learned report

### 2.2 Out-of-Scope (Future Phases)

#### Not Included in Phase 0
- Web-based user interface (Streamlit/Gradio)
- Multi-user support or authentication
- Production-grade database (PostgreSQL)
- Advanced retrieval strategies (reranking, hybrid search)
- Model fine-tuning or customization
- Distributed deployment
- API endpoints or REST services
- Advanced document parsing (tables, images, OCR)
- Multilingual support
- Real-time document monitoring
- Automated scheduled ingestion
- Performance optimization for large-scale datasets (1000+ documents)
- Cloud deployment options

### 2.3 Project Boundaries

#### Hardware Constraints
- **Target Environment:** Consumer laptop (16GB+ RAM recommended)
- **Model Size Limit:** 7-8B parameter models maximum
- **Storage:** Local disk storage (no network-attached storage)
- **Processing:** Single-machine, non-distributed architecture

#### Data Constraints
- **Document Count:** 20-50 documents for proof of concept
- **File Types:** PDF, TXT, MD only (no DOCX, PPTX, images)
- **Document Size:** Individual files <100MB recommended
- **Language:** English text only

#### Timeline Constraints
- **Development Duration:** 2-3 days of focused work (9-13 hours total)
- **Phase Completion:** Evaluated before proceeding to production planning

---

## 3. Project Objectives

### 3.1 Primary Objectives

1. **Validate Technical Stack**
   - Confirm Ollama can run 8B models on target hardware
   - Verify LlamaIndex integration with local models
   - Test ChromaDB persistence and retrieval performance
   - Validate end-to-end RAG pipeline functionality

2. **Establish Performance Baselines**
   - Measure ingestion speed (documents per minute)
   - Measure query latency (seconds per query)
   - Measure retrieval accuracy (relevant chunks in top-K)
   - Measure response quality (factual accuracy, hallucination rate)

3. **Identify Implementation Challenges**
   - Document dependency installation issues
   - Identify memory bottlenecks
   - Test error handling and recovery
   - Validate cross-platform compatibility (focus: Windows)

4. **Create Reusable Foundation**
   - Build modular, extensible codebase
   - Establish configuration management patterns
   - Create testing and evaluation frameworks
   - Document best practices and lessons learned

### 3.2 Secondary Objectives

1. **Knowledge Transfer**
   - Document system architecture for stakeholders
   - Create runbooks for common operations
   - Build troubleshooting guides

2. **Future Planning**
   - Identify requirements for production system
   - Estimate scaling characteristics
   - Define upgrade paths for models and infrastructure

---

## 4. Stakeholder Analysis

### 4.1 Primary Stakeholders

| Stakeholder | Role | Interest | Influence |
|-------------|------|----------|-----------|
| Project Owner | Decision Maker | ROI, feasibility, strategic alignment | High |
| Development Team | Implementation | Technical excellence, code quality | High |
| End Users | Consumers | Usability, accuracy, performance | Medium |
| IT Infrastructure | Support | Deployability, maintainability | Medium |

### 4.2 Communication Plan

- **Project Updates:** Daily during development phase
- **Technical Reviews:** After each major milestone
- **Demo Sessions:** Upon completion of query functionality
- **Final Presentation:** Go/No-Go decision meeting with metrics

---

## 5. Project Constraints

### 5.1 Technical Constraints

1. **Hardware Limitations**
   - Maximum 16GB RAM available for models and data
   - CPU-only inference (no GPU acceleration requirement)
   - Limited storage for model weights (~50GB available)

2. **Software Dependencies**
   - Python 3.10+ required
   - Ollama must be installable and runnable
   - Windows operating system compatibility required

3. **Model Constraints**
   - LLM: llama3:latest (8B parameters)
   - Embedding: nomic-embed-text (137M parameters, 768 dimensions)
   - Fallback LLM: deepseek-r1:latest (if available)

### 5.2 Resource Constraints

1. **Time**
   - Total development time: 9-13 hours
   - Timeline: 2-3 days
   - No extended debugging or optimization time

2. **Budget**
   - $0 for cloud services (local-only requirement)
   - Open-source tools only
   - No commercial licenses

3. **Personnel**
   - Single developer implementation
   - Limited availability for extended troubleshooting

### 5.3 Operational Constraints

1. **Data Privacy**
   - All processing must occur locally
   - No data transmission to external services
   - Documents remain on local filesystem

2. **Maintenance**
   - Self-contained system requiring minimal dependencies
   - No external service dependencies (except Ollama)

---

## 6. Assumptions and Dependencies

### 6.1 Assumptions

1. **Infrastructure**
   - Laptop meets minimum hardware specifications
   - Sufficient disk space available (>100GB free)
   - Internet connectivity available for initial setup and model downloads

2. **Technical**
   - Ollama can successfully run on Windows environment
   - PyPDF2 can parse target PDF documents adequately
   - 512-1024 token chunks provide sufficient context

3. **Data**
   - Test documents are representative of production use cases
   - Documents are text-based (not image-heavy scans)
   - English language content

4. **Operational**
   - Developer has Python development experience
   - Basic understanding of RAG concepts
   - Access to test documents for evaluation

### 6.2 Dependencies

1. **External Software**
   - Ollama server (localhost:11434)
   - Python 3.10+ installation
   - Pip package manager

2. **Python Packages**
   - llama-index-core >=0.10.0
   - llama-index-embeddings-ollama >=0.1.0
   - llama-index-llms-ollama >=0.1.0
   - llama-index-vector-stores-chroma >=0.1.0
   - chromadb >=0.4.0
   - PyPDF2 >=3.0.0

3. **Models (Downloaded via Ollama)**
   - llama3:latest (~4.7GB)
   - nomic-embed-text (~274MB)
   - Optional: deepseek-r1:latest (~20GB+)

4. **Data Assets**
   - Test document collection (20-50 files)
   - Ground truth question-answer pairs for evaluation

---

## 7. Project Milestones

### 7.1 Day 1: Environment Setup & Basic Pipeline (4-6 hours)

**Milestone 1.1: Environment Ready**
- Ollama installed and running
- Models downloaded and verified
- Python virtual environment configured
- Dependencies installed

**Milestone 1.2: Ingestion Pipeline Operational**
- Document loading functional
- Chunking strategy implemented
- Embeddings generated and stored
- ChromaDB persistence verified

**Milestone 1.3: Query Pipeline Functional**
- Basic retrieval working
- LLM integration complete
- Response generation verified
- Test queries returning results

### 7.2 Day 2: Quality & Evaluation (3-4 hours)

**Milestone 2.1: Evaluation Framework Ready**
- Test question set created (10 Q&A pairs)
- Retrieval metrics defined
- Logging infrastructure in place

**Milestone 2.2: Quality Metrics Captured**
- Retrieval accuracy measured
- Response quality evaluated
- Failure modes documented
- Configuration optimization tested

### 7.3 Day 3: Documentation & Demo (2-3 hours)

**Milestone 3.1: Documentation Complete**
- README with setup instructions
- Usage guide created
- Technical documentation finished
- Lessons learned documented

**Milestone 3.2: Demo Ready**
- Demo script prepared
- Example queries selected
- Results screenshots captured
- Go/No-Go recommendation prepared

---

## 8. High-Level Timeline

```
Week 1: Setup and Development
├── Day 1: Environment + Ingestion + Query (6 hrs)
├── Day 2: Testing + Evaluation (4 hrs)
└── Day 3: Documentation + Demo (3 hrs)

Total Duration: 3 days
Total Effort: 13 hours
```

---

## 9. Budget Summary

| Category | Estimated Cost |
|----------|---------------|
| Cloud Services | $0 (local only) |
| Software Licenses | $0 (open source) |
| Hardware | $0 (existing) |
| Personnel | [Internal resource] |
| **Total Project Cost** | **$0** |

---

## 10. Risk Summary

**Top 5 Risks:**
1. Model performance insufficient on laptop hardware
2. Ollama installation or compatibility issues on Windows
3. Document parsing failures with PDF files
4. Memory constraints during embedding generation
5. Retrieval returning irrelevant results

*Detailed risk analysis in Part 4: Quality & Risk Management*

---

**Document Status:** APPROVED  
**Next Document:** PMP Project Requirements - Part 2: Technical Requirements
